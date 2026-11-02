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

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio

import ray

from verl.plugin.platform import get_platform
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.accelerators.tpu.profilers.tpu_vllm_timeline import vLLMServerTimelineMixin
from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import patch_vllm_for_tpu
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import TPU_SYNC_BACKEND, canonical_backend

# vLLM imports the worker extension by name in its worker processes.
TPU_SYNC_WORKER_EXTENSION_CLS = (
    "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension.vLLMTPUSyncWorkerExtension"
)


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
    prefix = replica._get_server_name_prefix()
    for node_rank, node_id in enumerate(node_ids):
        if replica.is_reward_model:
            name = f"{prefix}server_reward_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        elif replica.is_teacher_model:
            name = f"{prefix}server_teacher_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        else:
            name = f"{prefix}server_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        server = replica.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
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
