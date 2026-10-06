# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and changes only what TPU needs:

* One server actor per replica, on the first worker's node (``launch_tpu_vllm_servers``). vLLM
  then spans the replica's hosts through its Ray executor, instead of verl's one server per node.
* ``distributed_executor_backend=external_launcher`` and ``enable_sleep_mode=False``: the
  multi-host executor is selected by ``patch_vllm_for_tpu``, and vllm-torchtpu has no sleep
  mode.
* With ``checkpoint_engine.backend=raiden``, vLLM's workers load the Raiden worker extension
  (``tpu_raiden.py``), which receives the trainer's weights over the network.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.rollout.tpu_vllm_patches import patch_vllm_for_tpu

# vLLM imports the worker extension by name in its worker processes.
RAIDEN_WORKER_EXTENSION_CLS = "verl_hardware_plugin.rollout.tpu_raiden.vLLMRaidenWorkerExtension"


async def launch_tpu_vllm_servers(replica: vLLMReplica) -> None:
    """Launch the vLLM rollout server actor for a TPU replica."""
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

    # Upstream starts one server per node and gives it the node's devices. On TPU one server, on the
    # first worker's node, drives all of the replica's hosts and holds no chips: patch_vllm_for_tpu
    # gives each of vLLM's own workers a chip.
    node_id = await replica.workers[0].__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())

    prefix = replica._get_server_name_prefix()
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    pgs = replica.resource_pool.get_placement_groups(device_name=get_device_name())
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        # One engine spans all of the replica's hosts through Ray. patch_vllm_for_tpu builds on
        # vllm-torchtpu's Ray multi-host backend with the V1 executor, not the V2 one.
        "TPU_MULTIHOST_BACKEND": "ray",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
        # The initialize_ray_cluster patch in patch_vllm_for_tpu attaches vLLM to these placement groups.
        "VERL_TPU_PG_IDS": ",".join(pg.id.hex() for pg in pgs),
    }

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
        workers=replica.workers,
        replica_rank=replica.replica_rank,
        node_rank=0,
        gpus_per_node=replica.gpus_per_replica_node,
        nnodes=replica.nnodes,
        cuda_visible_devices="",
    )
    replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await replica.servers[0].launch_server.remote(
        master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class TPUvLLMHttpServer(vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def __init__(self, *args, **kwargs):
        # This actor builds the vLLM engine and starts its EngineCore process, so patch vLLM before
        # anything else. The patches carry themselves into the EngineCore and vLLM's workers.
        patch_vllm_for_tpu()
        super().__init__(*args, **kwargs)

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so these win over the defaults.
        # patch_vllm_for_tpu switches the engine to vLLM's Ray executor at config time.
        engine_kwargs["distributed_executor_backend"] = "external_launcher"
        engine_kwargs["enable_sleep_mode"] = False

    def _get_worker_extension_cls(self) -> str:
        # The raiden checkpoint engine transfers weights straight into vLLM's workers, which need the
        # methods of the Raiden worker extension. Other backends keep upstream's extension.
        checkpoint_engine = getattr(getattr(self, "config", None), "checkpoint_engine", None)
        if getattr(checkpoint_engine, "backend", None) == "raiden":
            return RAIDEN_WORKER_EXTENSION_CLS
        return super()._get_worker_extension_cls()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor drives all of the replica's hosts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)
