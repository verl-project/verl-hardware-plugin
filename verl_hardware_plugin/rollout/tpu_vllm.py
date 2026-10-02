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
import logging
import os

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name, get_resource_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.rollout.tpu_vllm_patches import patch_vllm_for_tpu

logger = logging.getLogger(__name__)


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource."""
    return get_resource_name() == "TPU"


def prepare_tpu_server_env() -> None:
    """Routes vllm-torchtpu to its Ray multi-host backend in the server process."""
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "0"
    # See patch_vllm_for_tpu: a reloaded empty AOT artifact runs the model eagerly (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    try:
        import vllm_torchtpu.envs as tpu_envs

        tpu_envs.TPU_MULTIHOST_BACKEND = "ray"
        if hasattr(tpu_envs, "__getattr__") and hasattr(tpu_envs.__getattr__, "cache_clear"):
            tpu_envs.__getattr__.cache_clear()
    except Exception as env_err:
        logger.warning(f"Failed to force TPU_MULTIHOST_BACKEND to ray: {env_err}")


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


def _tpu_preflight_log(message: str, tag: str = "TPU preflight") -> None:
    """Emit a preflight message via both logging and stdout."""
    logger.warning("[%s] %s", tag, message)
    print(f"[{tag}] {message}", flush=True)


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

    platform_env_vars = get_platform().rollout_env_vars()
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **platform_env_vars,
        **tpu_env_vars,
    }
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]

    flags_to_copy = set()
    resource_pool = getattr(replica, "resource_pool", None)
    if resource_pool is not None:
        pgs = resource_pool.get_placement_groups(device_name=get_device_name())
        if pgs:
            env_vars["VERL_TPU_PG_IDS"] = ",".join(pg.id.hex() for pg in pgs)
            flags_to_copy.add("VERL_TPU_PG_IDS")

    for flag_var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
        base_value = platform_env_vars.get(flag_var) or tpu_env_vars.get(flag_var)
        extra_value = os.environ.get(f"VERL_TPU_EXTRA_{flag_var}")
        resolved = " ".join(v for v in (base_value, extra_value) if v)
        _tpu_preflight_log(
            f"{flag_var}: base={base_value!r} extra={extra_value!r} -> engine={resolved or None!r}",
            tag="TPU env",
        )
        if resolved:
            env_vars[flag_var] = resolved
            flags_to_copy.add(flag_var)

    if flags_to_copy:
        copy_var = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
        existing = env_vars.get(copy_var) or os.environ.get(copy_var) or ""
        names = {tok.strip() for tok in existing.split(",") if tok.strip()}
        env_vars[copy_var] = ",".join(sorted(names | flags_to_copy))
        _tpu_preflight_log(f"{copy_var}={env_vars[copy_var]}", tag="TPU env")

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
        prepare_tpu_server_env()


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
