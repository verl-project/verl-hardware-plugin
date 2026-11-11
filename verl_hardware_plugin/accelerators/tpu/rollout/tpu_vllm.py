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
  each engine takes a slot of the host's devices and its own fixed ports (see ``subhost_engine_env``), and all
  the engines of a host must come from one controller (``claim_host_for_subhost_engines``).

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import fcntl
import logging
import os
import socket
import uuid

import ray

from verl.plugin.platform import get_platform
from verl.utils.net_utils import is_valid_ipv6_address
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

# Engines that share a host listen on fixed ports. Otherwise vllm-torchtpu draws MASTER_PORT and the slice-builder
# ports from Linux's ephemeral range with portpicker, which several engines and Ray workers starting at once on one
# host can race for ("bind ... Address already in use"). The engine at device offset ``o`` takes the ports from
# ``base + 20 * o``, so a host's engines use ``[base, base + 20 * devices + 300)``: 21000-21459 by default, clear of
# Ray's default worker ports (``ray start --min-worker-port 10002 --max-worker-port 19999``) and of Linux's ephemeral
# ports (32768-60999). ``check_subhost_port_block`` enforces that.
SUBHOST_PORT_BASE_ENV = "VERL_TPU_ROLLOUT_SUBHOST_PORT_BASE"
DEFAULT_SUBHOST_PORT_BASE = 21000
_PORTS_PER_DEVICE = 20
RAY_DEFAULT_WORKER_PORTS = (10002, 19999)
EPHEMERAL_PORTS = (32768, 60999)

# A host's sub-host engines must all belong to one controller: each controller assigns slots on its own, so engines
# of two controllers would take the same devices and ports. The first engine of a host claims it for its controller
# (``claim_host_for_subhost_engines``), and engines of any other controller fail at startup.
SUBHOST_OWNER_ENV = "VERL_TPU_SUBHOST_OWNER"
SUBHOST_CLAIM_FILE = "/tmp/verl_tpu_subhost_rollout.lock"
_CONTROLLER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

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


def _subhost_port_base() -> int:
    return int(os.environ.get(SUBHOST_PORT_BASE_ENV, DEFAULT_SUBHOST_PORT_BASE))


def subhost_port_block(devices_per_host: int = TPU_V7X_DEVICES_PER_HOST) -> range:
    """Every port the engines that share one host may listen on (see ``subhost_engine_env``)."""
    base = _subhost_port_base()
    return range(base, base + _PORTS_PER_DEVICE * devices_per_host + 300)


def ray_worker_port_range() -> tuple[int, int]:
    """The ports Ray workers bind, when this process started the Ray node and knows them; else ``ray start``'s default.

    A process that connected to a running cluster cannot read the raylet's ``--min-worker-port`` /
    ``--max-worker-port``: with a custom range, keep ``VERL_TPU_ROLLOUT_SUBHOST_PORT_BASE`` clear of it by hand.
    """
    try:
        params = ray._private.worker._global_node._ray_params
        low, high = int(params.min_worker_port or 0), int(params.max_worker_port or 0)
        if 0 < low <= high:
            return low, high
    except Exception:
        pass
    return RAY_DEFAULT_WORKER_PORTS


def check_subhost_port_block(block: range) -> None:
    """Raise if ``block`` reaches into Ray's worker ports or Linux's ephemeral ports, or is not a valid port range."""
    if block.start < 1024 or block.stop > 65536:
        raise ValueError(f"Rollout engine ports {block.start}-{block.stop - 1} are outside 1024-65535")
    for name, (low, high) in (
        ("Ray's worker ports", ray_worker_port_range()),
        ("Linux's ephemeral ports", EPHEMERAL_PORTS),
    ):
        if block.start <= high and low < block.stop:
            raise ValueError(
                f"The ports of the rollout engines that share a host, {block.start}-{block.stop - 1}, overlap {name} "
                f"{low}-{high}. Set {SUBHOST_PORT_BASE_ENV} to a base that keeps them clear of both Ray's worker "
                f"ports and {EPHEMERAL_PORTS[0]}-{EPHEMERAL_PORTS[1]} (default {DEFAULT_SUBHOST_PORT_BASE})."
            )


def claim_host_for_subhost_engines(owner: str, path: str = SUBHOST_CLAIM_FILE) -> int:
    """Claim this host for the sub-host engines of controller ``owner``; return the descriptor that holds the claim.

    Engines of one controller share the claim: each holds a shared ``flock`` on ``path`` for the life of its process,
    and the file names the owner. An engine of another controller fails here rather than open the same devices and
    ports. A claim whose engines are all gone (the kernel drops their locks) is taken over.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(fd, fcntl.LOCK_SH)  # live engines hold the claim (this also waits out a takeover)
        else:
            os.ftruncate(fd, 0)
            os.pwrite(fd, owner.encode(), 0)
            fcntl.flock(fd, fcntl.LOCK_SH)
        holder = os.pread(fd, 4096, 0).decode()
        if holder != owner:
            raise RuntimeError(
                f"This host already runs TP=2 / TP=4 rollout engines of another controller ({holder}); this engine "
                f"belongs to {owner}. Engines that share a host must all come from one verl job's rollout, which "
                "assigns their devices and ports: give each job its own rollout hosts."
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def subhost_engine_env(slot: int, devices_per_engine: int, replica_rank: int) -> dict[str, str]:
    """The environment of a vLLM engine in slot ``slot`` of a host it shares with other engines.

    Everything here is per engine; the per-worker part (MegaScale ports, libtpu log dir) is set in each worker by
    ``tpu_vllm_patches.patch_tpu_worker_subhost_env``, which ``SUBHOST_MEGASCALE_PORT_ENV`` switches on.
    """
    offset = slot * devices_per_engine
    block = _subhost_port_base() + _PORTS_PER_DEVICE * offset
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
        # The server actor claims the host for this controller (see claim_host_for_subhost_engines).
        SUBHOST_OWNER_ENV: _CONTROLLER_ID,
        # Compile caches of their own: a cache keyed by the replica alone hands one TP size's executables to another
        # ("local device count is 0" when a TP=8 engine loaded TP=4 executables).
        "VLLM_CACHE_ROOT": f"/tmp/vllm_cache_tp{devices_per_engine}_r{replica_rank}",
        "TORCHINDUCTOR_CACHE_DIR": f"/tmp/inductor_cache_tp{devices_per_engine}_r{replica_rank}",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": f"/tmp/tpu_tier3_tp{devices_per_engine}_r{replica_rank}",
    }


async def launch_tpu_vllm_servers(replica: vLLMReplica) -> None:
    """Launch one vLLM rollout server actor per host of a TPU replica."""
    if replica.config.data_parallel_size > 1:
        raise NotImplementedError(
            "actor_rollout_ref.rollout.data_parallel_size="
            f"{replica.config.data_parallel_size} is not supported on TPU. The TPU "
            "distributed runtime requires each compiled program to span the whole "
            "TPU mesh, but a single DP group only spans "
            f"tensor_model_parallel_size={replica.config.tensor_model_parallel_size} "
            "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
            "then creates one rollout replica per tensor_model_parallel_size chips, "
            "which is equivalent to engine-internal data parallelism."
        )

    # As upstream, but the servers get no devices: verl's rollout workers hold no TPU (a chip belongs to one
    # process), so there are no accelerator ids to hand over. vLLM's own workers open the host's chips.
    chips_per_host = replica.gpus_per_replica_node
    node_ids = await asyncio.gather(
        *[
            worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
            for worker in replica.workers[::chips_per_host]
        ]
    )
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        # vllm-torchtpu's default; set so that a job-level value cannot switch the replica to the Ray backend.
        "TPU_MULTIHOST_BACKEND": "mp",
    }
    shares_host = False
    if replica.nnodes == 1:
        # A replica on one host (e.g. one of several replicas on a multi-host rollout slice) is its own mesh; it
        # must not inherit the slice's TORCH_TPU_TOPOLOGY. Multi-host replicas get theirs from vllm-torchtpu's
        # rendezvous. The TPU type comes from the replica's host, not from this process's environment, which
        # may be a CPU Ray head without TPU_ACCELERATOR_TYPE. On TPU 7x a TP=2 / TP=4 replica gets the mesh of
        # its one or two chips; sizes without a known mesh keep the inherited environment, as before.
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
        shares_host = is_tpu_v7x(pod_type) and chips_per_host in TPU_V7X_SUBHOST_TOPOLOGY_MAP
        if shares_host:
            check_subhost_port_block(subhost_port_block())
    prefix = replica._get_server_name_prefix()
    for node_rank, node_id in enumerate(node_ids):
        if replica.is_reward_model:
            name = f"{prefix}server_reward_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        elif replica.is_teacher_model:
            name = f"{prefix}server_teacher_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        else:
            name = f"{prefix}server_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        server_env_vars = env_vars
        if shares_host:
            slot = allocate_host_slot(node_id, name, chips_per_host, TPU_V7X_DEVICES_PER_HOST)
            server_env_vars = {**env_vars, **subhost_engine_env(slot, chips_per_host, replica.replica_rank)}
            logger.info(
                "Rollout engine %s takes slot %d (devices %d-%d) of host %s",
                name,
                slot,
                slot * chips_per_host,
                (slot + 1) * chips_per_host - 1,
                node_id,
            )
        server = replica.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": server_env_vars},
            name=name,
            max_concurrency=replica.config.ray_actor_max_concurrency,
        ).remote(
            config=replica.config,
            model_config=replica.model_config,
            rollout_mode=replica.rollout_mode,
            workers=replica.workers[node_rank * chips_per_host : (node_rank + 1) * chips_per_host],
            replica_rank=replica.replica_rank,
            node_rank=node_rank,
            gpus_per_node=chips_per_host,
            nnodes=replica.nnodes,
            cuda_visible_devices="",
        )
        replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await asyncio.gather(
        *[
            server.launch_server.remote(master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port)
            for server in replica.servers
        ]
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class TPUvLLMHttpServer(vLLMServerTimelineMixin, vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        owner = os.environ.get(SUBHOST_OWNER_ENV)
        if owner:
            # Held until this actor exits; the engine's processes are its children.
            self._subhost_host_claim = claim_host_for_subhost_engines(owner, SUBHOST_CLAIM_FILE)
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

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)
