# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and changes only what TPU needs:

* One server actor per replica, on the first worker's node, with every worker's TPU
  environment forwarded (``launch_tpu_vllm_servers``). vLLM then spans the hosts through its
  Ray executor instead of verl's per-node ``mp`` launch.
* ``distributed_executor_backend=external_launcher`` and ``enable_sleep_mode=False``: the
  multi-host executor is selected by ``patch_vllm_for_tpu``, and vllm-torchtpu has no sleep
  mode.
* Standalone rollout workers do not claim a GPU resource.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import os

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name, get_resource_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.rollout.tpu_vllm_patches import patch_vllm_for_tpu


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource."""
    return get_resource_name() == "TPU"


async def get_tpu_server_launch_config(workers):
    """
    Asynchronously queries node ID, visible chips, and TPU specific environment
    variables from all TPU workers for launching the server actor on TPU.
    """
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    os.environ.get("TPU_VISIBLE_CHIPS", "0"),
                )
            )
            for worker in workers
        ]
    )

    worker_tpu_envs = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("TPU_")
                    or k.startswith("TORCH_TPU_")
                    or k
                    in (
                        "CLOUD_TPU_TASK_ID",
                        "CHIPS_PER_HOST",
                        "JAX_MEM_FRACTION",
                        "JAX_THREE_G_MEM_ALLOC_ON_FREE",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "LIBTPU_INIT_ARGS",
                        "TORCH_DYNAMO_RECOMPILE_LIMIT",
                        "SKIP_JAX_PRECOMPILE",
                        "VLLM_ENABLE_V1_MULTIPROCESSING",
                        # Keep the vLLM AOT compile cache disabled in the server
                        # process too: a reloaded artifact degrades the model to
                        # eager execution and reintroduces the unaligned-DUS
                        # crash (b/501165531). See patch_vllm_for_tpu().
                        "VLLM_DISABLE_COMPILE_CACHE",
                        "VERL_PLATFORM",
                        "XLA_FLAGS",
                    )
                }
            )
            for worker in workers
        ]
    )

    node_id = worker_infos[0][0]
    visible_chips = ",".join([info[1] for info in worker_infos])
    tpu_env_vars = worker_tpu_envs[0] if worker_tpu_envs else {}

    return node_id, visible_chips, tpu_env_vars


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

    node_id, visible_chips, tpu_env_vars = await get_tpu_server_launch_config(replica.workers)

    prefix = replica._get_server_name_prefix()
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        **tpu_env_vars,
        # One engine spans all of the replica's hosts through Ray. patch_vllm_for_tpu builds on
        # vllm-torchtpu's Ray multi-host backend with the V1 executor, not the V2 one.
        "TPU_MULTIHOST_BACKEND": "ray",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
    }
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
    # The initialize_ray_cluster patch in patch_vllm_for_tpu attaches vLLM to these placement groups.
    pgs = replica.resource_pool.get_placement_groups(device_name=get_device_name())
    env_vars["VERL_TPU_PG_IDS"] = ",".join(pg.id.hex() for pg in pgs)

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
        cuda_visible_devices=visible_chips,
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

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so these win over the defaults.
        # patch_vllm_for_tpu switches a multi-host engine to the Ray executor at config time.
        engine_kwargs["distributed_executor_backend"] = "external_launcher"
        engine_kwargs["enable_sleep_mode"] = False


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor per replica, no GPU resource claims."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    def rollout_worker_use_gpu(self) -> bool:
        return False

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)


# This module is imported by the driver (through the ``vllm`` replica loader) and by every
# TPUvLLMHttpServer actor, which is the process that spawns the vLLM EngineCore.
if is_tpu_vllm_run():
    patch_vllm_for_tpu()
