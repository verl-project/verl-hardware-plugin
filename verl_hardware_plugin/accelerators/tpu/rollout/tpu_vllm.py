# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and its multi-node layout, and changes only what TPU needs:

* One server actor per host of the replica, as upstream does on GPU: node 0 runs the engine and
  the HTTP API, the other hosts run vLLM headless. Each server holds no chips; vLLM's ``mp``
  executor starts one worker per local chip (or TensorCore on TPU 7x), and vllm-torchtpu's ``mp``
  multi-host backend (``TPU_MULTIHOST_BACKEND=mp``) joins the hosts of the replica into one TPU
  mesh through a TCPStore rendezvous at the replica's master address.
* ``enable_sleep_mode=False``: vllm-torchtpu has no sleep mode.
* With ``checkpoint_engine.backend=tpu-sync``, vLLM's workers load the tpu-sync worker extension
  (``tpu_sync_worker_extension.py``), which receives the trainer's weights over the network.
* On TPU 7x, TP=2 and TP=4 replicas share their host with other replicas (e.g. 16 x TP=2 on four hosts):
  each engine takes a slot of the host's devices and its own fixed ports (see ``subhost_engine_env``).

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import inspect
import logging
import os

import ray

from verl.single_controller.ray import RayResourcePool, ResourcePoolManager
from verl.utils.device import get_device_name
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    TPU_V7X_DEVICES_PER_HOST,
    TPU_V7X_SUBHOST_TOPOLOGY_MAP,
    is_tpu_v7x,
    ray_tpu_accelerator_name,
    single_host_tpu_topology,
)
from verl_hardware_plugin.accelerators.tpu.profilers.tpu_vllm_timeline import vLLMServerTimelineMixin
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (
    SUBHOST_MEGASCALE_PORT_ENV,
    patch_vllm_for_tpu,
)
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import TPU_SYNC_BACKEND, canonical_backend

logger = logging.getLogger(__name__)

# vLLM imports the worker extension by name in its worker processes.
TPU_SYNC_WORKER_EXTENSION_CLS = (
    "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension.vLLMTPUSyncWorkerExtension"
)

# Engines that share a host listen on fixed ports below Linux's ephemeral range (32768-60999). Otherwise
# vllm-torchtpu draws MASTER_PORT and the slice-builder ports from that range with portpicker, which several engines
# and Ray workers starting at once on one host can race for ("bind ... Address already in use"). The engine at
# device offset ``o`` takes the block that starts at ``base + 20 * o``.
SUBHOST_PORT_BASE_ENV = "VERL_TPU_ROLLOUT_SUBHOST_PORT_BASE"
DEFAULT_SUBHOST_PORT_BASE = 19000
_PORTS_PER_DEVICE = 20

# {node id: {engine: slot}} and {node id: devices per engine} of the engines that share a host. Every replica of a
# job is launched from this process (asyncio.gather), so the table sees all the engines of a host. A slot, once
# given, never moves, whatever order the replicas arrive in, and an engine launched again under the same name (a
# restarted replica) gets its slot back.
_HOST_SLOTS: dict[str, dict[str, int]] = {}
_HOST_ENGINE_DEVICES: dict[str, int] = {}


def allocate_host_slot(node_id: str, engine: str, devices_per_engine: int, devices_per_host: int) -> int:
    """The slot of ``engine`` on host ``node_id``: the one it holds, else the lowest free one.

    Slot ``s`` holds the host's devices ``[s * devices_per_engine, (s + 1) * devices_per_engine)``.

    Raises:
        ValueError: if ``devices_per_engine`` does not divide the host, differs from the size of the engines
            already on the host, or every slot of the host is taken.
    """
    if devices_per_engine <= 0 or devices_per_host % devices_per_engine:
        raise ValueError(f"Engines of {devices_per_engine} devices do not tile a host of {devices_per_host} devices")
    if _HOST_ENGINE_DEVICES.setdefault(node_id, devices_per_engine) != devices_per_engine:
        raise ValueError(
            f"Host {node_id} already runs engines of {_HOST_ENGINE_DEVICES[node_id]} devices; cannot add {engine!r} "
            f"with {devices_per_engine}"
        )
    slots = _HOST_SLOTS.setdefault(node_id, {})
    if engine not in slots:
        free = sorted(set(range(devices_per_host // devices_per_engine)) - set(slots.values()))
        if not free:
            raise ValueError(
                f"Host {node_id} has no free slot for {engine!r}: its {devices_per_host // devices_per_engine} "
                f"slots of {devices_per_engine} devices are taken by {slots}"
            )
        slots[engine] = free[0]
    return slots[engine]


def release_host_slot(node_id: str, engine: str) -> None:
    """Free the slot of ``engine`` on ``node_id``, e.g. for a replacement engine under another name."""
    slots = _HOST_SLOTS.get(node_id, {})
    slots.pop(engine, None)
    if not slots:
        reset_host_slots(node_id)


def reset_host_slots(node_id: str | None = None) -> None:
    """Forget the slots of ``node_id``, or of every host."""
    if node_id is None:
        _HOST_SLOTS.clear()
        _HOST_ENGINE_DEVICES.clear()
    else:
        _HOST_SLOTS.pop(node_id, None)
        _HOST_ENGINE_DEVICES.pop(node_id, None)


def subhost_engine_env(slot: int, devices_per_engine: int, replica_rank: int) -> dict[str, str]:
    """The environment of a vLLM engine in slot ``slot`` of a host it shares with other engines.

    Everything here is per engine; the per-worker part (MegaScale ports, libtpu log dir) is set in each worker by
    ``tpu_vllm_patches.patch_tpu_worker_subhost_env``, which ``SUBHOST_MEGASCALE_PORT_ENV`` switches on.
    """
    offset = slot * devices_per_engine
    block = int(os.environ.get(SUBHOST_PORT_BASE_ENV, DEFAULT_SUBHOST_PORT_BASE)) + _PORTS_PER_DEVICE * offset
    return {
        # vllm-torchtpu binds worker r to device offset + r: TPU_LOCAL_RANK_OFFSET in its spawn-time worker binding,
        # DEBUG_TPU_LOCAL_RANK_OFFSET where TPUWorker.init_device sets LOCAL_RANK. Neither is part of its compile
        # cache key.
        "TPU_LOCAL_RANK_OFFSET": str(offset),
        "DEBUG_TPU_LOCAL_RANK_OFFSET": str(offset),
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "1",
        # torch_tpu's rendezvous and slice builder on this engine's ports.
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(block + 200),
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{block + 300 + r}" for r in range(devices_per_engine)),
        # libtpu's MegaScale coordinator of this engine alone: the GKE TPU webhook's container-level value names the
        # pod, so every engine in the container would share it.
        "MEGASCALE_COORDINATOR_ADDRESS": f"127.0.0.1:{block}",
        SUBHOST_MEGASCALE_PORT_ENV: str(block),
        # Compile caches of their own: a cache keyed by the replica alone hands one TP size's executables to another
        # ("local device count is 0" when a TP=8 engine loaded TP=4 executables).
        "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_tp{devices_per_engine}_r{replica_rank}",
        "TORCHINDUCTOR_CACHE_DIR": f"/tmp/inductor_cache_tp{devices_per_engine}_r{replica_rank}",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": f"/tmp/tpu_tier3_tp{devices_per_engine}_r{replica_rank}",
    }


class TPUvLLMHttpServer(vLLMServerTimelineMixin, vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        super().__init__(*args, **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so this wins over the default.
        engine_kwargs["enable_sleep_mode"] = False

    def _get_worker_extension_cls(self) -> str:
        # The tpu-sync checkpoint engine transfers weights straight into vLLM's workers, which need the
        # methods of the tpu-sync worker extension. Other backends keep upstream's extension.
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if canonical_backend(getattr(checkpoint_engine, "backend", None)) == TPU_SYNC_BACKEND:
            return TPU_SYNC_WORKER_EXTENSION_CLS
        return super()._get_worker_extension_cls()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor per host, none of which holds a chip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    async def init_standalone(self, resource_pool: RayResourcePool | None = None):
        # verl builds a standalone replica's workers with bin_pack=False: Ray's best-effort PACK, which fills the
        # best-scored host and puts the remaining bundles on others, so it can split a replica over partly used hosts.
        # A replica on one host hands verl its pool with a STRICT_PACK placement group instead. A verl whose
        # init_standalone takes no pool (before verl-project/verl#7993) keeps PACK, and a split replica fails in
        # PlatformTPU.get_tpu_env_vars or TPUvLLMReplica.launch_servers.
        if (
            resource_pool is None
            and self.nnodes == 1
            and "resource_pool" in inspect.signature(super().init_standalone).parameters
        ):
            resource_pool = self._single_host_pool()
        if resource_pool is None:
            await super().init_standalone()
        else:
            await super().init_standalone(resource_pool=resource_pool)

    def _single_host_pool(self) -> RayResourcePool:
        """verl's standalone pool of this replica (same name and spec), with its placement group created now by
        STRICT_PACK: on one host, or pending until a host has room. verl's worker group reuses that placement group."""
        kind = "_reward" if self.is_reward_model else "_teacher" if self.is_teacher_model else ""
        name = f"rollout_pool{kind}_{self.replica_rank}{self.name_suffix}"
        manager = ResourcePoolManager(
            resource_pool_spec={name: [self.gpus_per_replica_node]}, mapping=None, max_colocate_count=2
        )
        manager.create_resource_pool()
        pool = manager.resource_pool_dict[name]
        pool.get_placement_groups(strategy="STRICT_PACK", device_name=get_device_name())
        return pool

    def _server_actor_name(self, node_rank: int = 0) -> str:
        kind = "_reward" if self.is_reward_model else "_teacher" if self.is_teacher_model else ""
        return f"{self._get_server_name_prefix()}server{kind}_{self.replica_rank}_{node_rank}{self.name_suffix}"

    def _get_server_env_vars(self, node_ids: list[str]) -> dict[str, str]:
        env_vars = {
            **super()._get_server_env_vars(node_ids),
            # vllm-torchtpu's default; set so that a job-level value cannot switch the replica to the Ray backend.
            "TPU_MULTIHOST_BACKEND": "mp",
        }
        if self.nnodes == 1:
            # A replica on one host (e.g. one of several replicas on a multi-host rollout slice) is its own mesh; it
            # must not inherit the slice's TORCH_TPU_TOPOLOGY. Multi-host replicas get theirs from vllm-torchtpu's
            # rendezvous. The TPU type comes from the replica's host, not from this process's environment, which
            # may be a CPU Ray head without TPU_ACCELERATOR_TYPE. On TPU 7x a TP=2 / TP=4 replica gets the mesh of
            # its one or two chips; sizes without a known mesh keep the inherited environment, as before.
            chips_per_host = self.gpus_per_replica_node
            pod_type = ray_tpu_accelerator_name(node_ids=list(node_ids))
            topology = single_host_tpu_topology(chips_per_host, pod_type=pod_type)
            if topology:
                env_vars["TORCH_TPU_TOPOLOGY"] = topology
            else:
                logger.warning(
                    "No single-host TPU mesh of %d chips (TPU type %r); the rollout replica keeps the inherited "
                    "TORCH_TPU_TOPOLOGY",
                    chips_per_host,
                    pod_type,
                )
            # Such a replica uses part of its host and shares it with other replicas (auto_assign_accelerator_type
            # spreads them over the rollout slices), so it takes a slot of the host's devices and ports of its own.
            if is_tpu_v7x(pod_type) and chips_per_host in TPU_V7X_SUBHOST_TOPOLOGY_MAP:
                node_id = node_ids[0]
                name = self._server_actor_name()
                slot = allocate_host_slot(node_id, name, chips_per_host, TPU_V7X_DEVICES_PER_HOST)
                env_vars = {**env_vars, **subhost_engine_env(slot, chips_per_host, self.replica_rank)}
                logger.info(
                    "Rollout engine %s takes slot %d (devices %d-%d) of host %s",
                    name,
                    slot,
                    slot * chips_per_host,
                    (slot + 1) * chips_per_host - 1,
                    node_id,
                )
        return env_vars

    async def launch_servers(self):
        if self.config.data_parallel_size > 1:
            raise NotImplementedError(
                "actor_rollout_ref.rollout.data_parallel_size="
                f"{self.config.data_parallel_size} is not supported on TPU. The TPU "
                "distributed runtime requires each compiled program to span the whole "
                "TPU mesh, but a single DP group only spans "
                f"tensor_model_parallel_size={self.config.tensor_model_parallel_size} "
                "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
                "then creates one rollout replica per tensor_model_parallel_size chips, "
                "which is equivalent to engine-internal data parallelism."
            )
        if self.nnodes == 1 and getattr(self, "workers", None):
            # Upstream launch_servers queries every worker's node id, then slices
            # worker_node_ids[::gpus_per_replica_node] (only worker 0 when nnodes == 1). Check every worker here so a
            # replica split across hosts by Ray's PACK strategy fails before any server starts: its engine runs on
            # worker 0's host and opens that host's devices.
            node_ids = await asyncio.gather(
                *[
                    worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                    for worker in self.workers
                ]
            )
            hosts = sorted(set(node_ids))
            if len(hosts) > 1:
                raise RuntimeError(
                    f"Rollout replica {self.replica_rank} must run on one host, but Ray placed its {len(node_ids)} "
                    f"workers on {len(hosts)} hosts ({', '.join(hosts)}): its engine would open devices of other "
                    "replicas. verl places rollout workers with Ray's best-effort PACK strategy (bin_pack=False), "
                    "which can split a replica over partly used hosts."
                )
        await super().launch_servers()
