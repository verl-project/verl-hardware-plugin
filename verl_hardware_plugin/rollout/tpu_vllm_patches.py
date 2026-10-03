# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for multi-host TPU rollout.

``patch_vllm_for_tpu`` lets the vLLM engine of a ``TPUvLLMHttpServer`` run on all hosts of a TPU
slice through Ray. Each patch fills a gap in the pinned vLLM (v0.29.0) and vllm-torchtpu
(9faafb17) and can be dropped once upstream closes it:

* ``EngineArgs.create_engine_config``, ``initialize_ray_cluster`` and
  ``RayDistributedExecutor._init_workers_ray``: run the engine on vLLM's generic Ray executor,
  attached to the placement groups verl reserved for the replica (one per host), with one worker
  per chip that gets the TPU multi-host environment. vllm-torchtpu's own Ray executor can reuse
  only a single placement group. Upstream fix: let it run on several.
* ``RayDistributedExecutor._execute_dag``: run each step with plain Ray calls instead of a
  compiled Ray graph. The graph runs the model on a background thread, where vllm-torchtpu's
  raised torch.compile recompile limit does not apply, so the first request fails. Upstream fix:
  raise the limit on every thread.
* ``TPUWorker.reset_encoder_cache``: no-op. verl resets vLLM's caches after every weight update,
  and vllm-torchtpu's worker does not implement this call. Upstream fix: implement it.
* ``VLLM_DISABLE_COMPILE_CACHE=1``: a reloaded compile-cache artifact can make vLLM run the model
  eagerly, which crashes libtpu (b/501165531). Upstream fix: that bug.
* ``multiprocessing.process.BaseProcess.__init__`` and ``RayWorkerWrapper.__init__``: install
  these patches in the EngineCore process and in vLLM's worker actors, neither of which imports
  the plugin.
"""

import logging
import multiprocessing
import multiprocessing.process
import os
from collections import Counter

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl_hardware_plugin.platforms.platform_tpu import resolve_tpu_topology_bounds

logger = logging.getLogger(__name__)


def _local_ranks(worker_ips: list[str]) -> list[int]:
    """Index of each worker among the workers on its host, for workers in rank order."""
    return [worker_ips[:rank].count(ip) for rank, ip in enumerate(worker_ips)]


def _tpu_worker_envs(worker_ips: list[str], base_port: int) -> list[dict[str, str]]:
    """The TPU multi-host environment of each vLLM worker, for workers in rank order.

    Ranks on one host must be contiguous. A worker drives chip ``local_rank`` of its host and serves
    the slice builder on ``base_port + local_rank``. These are the variables libtpu and torch_tpu
    read to join the workers into one TPU mesh.
    """
    hosts = list(dict.fromkeys(worker_ips))
    local_ranks = _local_ranks(worker_ips)
    addresses = ",".join(
        f"{ip}:{base_port + local_rank}" for ip, local_rank in zip(worker_ips, local_ranks, strict=True)
    )
    topology, host_bounds, chips_per_host_bounds, _ = resolve_tpu_topology_bounds(len(worker_ips), len(hosts))
    return [
        {
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "TPU_PROCESS_ADDRESSES": addresses,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
            "CLOUD_TPU_TASK_ID": str(hosts.index(ip)),
            "TPU_WORKER_HOSTNAMES": ",".join(hosts),
            "TORCH_TPU_TOPOLOGY": topology,
            "TPU_HOST_BOUNDS": host_bounds,
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_per_host_bounds,
        }
        for ip, local_rank in zip(worker_ips, local_ranks, strict=True)
    ]


_PATCHES_APPLIED = False


class PickleableProcessWrapper:
    """Process target that runs ``patch_vllm_for_tpu`` in the child before ``target``.

    This is how the patches reach vLLM's EngineCore. Inside a Ray actor vLLM always uses the
    ``spawn`` start method, and the EngineCore target is ``EngineCoreProc.run_engine_core``, which
    vllm-torchtpu replaces with its own wrapper. A spawned child imports only the modules its pickled
    target names, so without this wrapper it would never import the plugin.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        if self.target is not None:
            return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """Wrap the target of every ``multiprocessing`` process started from now on in ``PickleableProcessWrapper``."""
    if getattr(multiprocessing.process.BaseProcess, "_tpu_patched", False):
        return

    original_init = multiprocessing.process.BaseProcess.__init__

    def patched_init(self, *args, **kwargs):
        target = kwargs.get("target", None)
        if target is None and len(args) > 1:
            target = args[1]

        if target is not None:
            wrapped_target = PickleableProcessWrapper(target)
            if "target" in kwargs:
                kwargs["target"] = wrapped_target
            elif len(args) > 1:
                args = list(args)
                args[1] = wrapped_target
                args = tuple(args)

        original_init(self, *args, **kwargs)

    multiprocessing.process.BaseProcess.__init__ = patched_init  # type: ignore[method-assign]
    multiprocessing.process.BaseProcess._tpu_patched = True  # type: ignore[attr-defined]


def patch_vllm_for_tpu() -> None:
    """
    Apply TPU-specific patches and workarounds to vLLM and vllm-torchtpu workers.
    Ensures correct topology routing, un-clashed TCP ports, and driver-worker environment
    synchronization on GKE TPU slices.
    """
    global _PATCHES_APPLIED

    # Environment side effects. Cheap and safe to repeat in every process that calls this.
    patch_multiprocessing_for_tpu()
    # vLLM can reload a degenerate AOT compile artifact (num_artifacts=0) that silently runs the
    # model eagerly; eager execution then hits the XLA:TPU unaligned-DUS CHECK (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    if _PATCHES_APPLIED:
        return

    # CPU unit tests and non-rollout processes may not have vllm or vllm_torchtpu installed.
    # Import all required vLLM / vllm-torchtpu symbols once here rather than at module import
    # time: if they are absent, leave _PATCHES_APPLIED=False so a rollout process can still
    # install the executor patches when vLLM is available.
    try:
        import vllm.v1.executor.ray_executor as v1_ray_executor
        import vllm.v1.executor.ray_utils as v1_ray_utils
        from vllm.engine.arg_utils import EngineArgs
        from vllm.platforms import current_platform
        from vllm.ray.ray_env import get_env_vars_to_copy
        from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
        from vllm.v1.executor.ray_utils import FutureWrapper, detach_zero_copy_from_model_runner_output
        from vllm_torchtpu import envs as vllm_torchtpu_envs
        from vllm_torchtpu.executors import ray_distributed_executor
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU executor patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    try:
        orig_create_engine_config = EngineArgs.create_engine_config

        def patched_create_engine_config(self, *args, **kwargs):
            vllm_config = orig_create_engine_config(self, *args, **kwargs)
            # vllm-torchtpu picked its own Ray executor, which takes a single placement group,
            # but verl creates one per host. Use vLLM's generic Ray executor: the patches below
            # attach it to verl's placement groups and adapt its worker start-up and dispatch to
            # TPU. It does not support async scheduling, which vLLM resolved to on for the
            # external_launcher backend that TPUvLLMHttpServer passes in.
            vllm_config.parallel_config.distributed_executor_backend = "ray"
            vllm_config.scheduler_config.async_scheduling = False
            return vllm_config

        EngineArgs.create_engine_config = patched_create_engine_config

        def dummy_reset_encoder_cache(*args, **kwargs):
            pass

        TPUWorker.reset_encoder_cache = dummy_reset_encoder_cache
        logger.info("Patched TPUWorker.reset_encoder_cache with no-op stub.")

        orig_init_ray_cluster = v1_ray_utils.initialize_ray_cluster

        def patched_initialize_ray_cluster(parallel_config, ray_address=None, *args, **kwargs):
            # Connect with the runtime_env vLLM captured from the server actor *before* any
            # other Ray call below: those would auto-init Ray with an empty runtime_env, vLLM
            # would then skip its own ray.init, and the RayWorkerWrapper actors would start
            # without the job's py_modules / working_dir.
            if not ray.is_initialized():
                ray.init(address=ray_address, runtime_env=getattr(parallel_config, "ray_runtime_env", None))
            pg_ids_str = os.environ.get("VERL_TPU_PG_IDS", "")
            if pg_ids_str:
                from ray._raylet import PlacementGroupID
                from ray.util.placement_group import PlacementGroup

                verl_pgs = [
                    PlacementGroup(PlacementGroupID.from_hex(pg_hex.strip()))
                    for pg_hex in pg_ids_str.split(",")
                    if pg_hex.strip()
                ]
                if verl_pgs:
                    parallel_config.placement_group = verl_pgs[0]
                    parallel_config._verl_tpu_placement_groups = verl_pgs
                    logger.info(
                        "Reusing %d verl placement group(s) for TPU rollout: %s",
                        len(verl_pgs),
                        [pg.id.hex() for pg in verl_pgs],
                    )
                    return
            if parallel_config.placement_group is None:
                parallel_config.placement_group = ray.util.get_current_placement_group()
            return orig_init_ray_cluster(parallel_config, ray_address, *args, **kwargs)

        v1_ray_executor.initialize_ray_cluster = patched_initialize_ray_cluster

        OriginalRayWorkerWrapper = ray_distributed_executor.RayWorkerWrapper
        original_wrapper_init = OriginalRayWorkerWrapper.__init__

        def patched_wrapper_init(self, *args, **kwargs):
            # Pooled vLLM Ray workers never import the plugin; this is how the patches reach them.
            patch_vllm_for_tpu()
            return original_wrapper_init(self, *args, **kwargs)

        OriginalRayWorkerWrapper.__init__ = patched_wrapper_init

        def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
            # One worker per TPU bundle of verl's placement groups.
            verl_pgs = getattr(self.parallel_config, "_verl_tpu_placement_groups", None) or [placement_group]
            bundles = [
                (pg, bundle_index)
                for pg in verl_pgs
                for bundle_index, bundle in enumerate(pg.bundle_specs)
                if bundle.get(current_platform.ray_device_key, 0)
            ][: self.parallel_config.world_size]
            self.workers = [
                ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: 1},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_capture_child_tasks=True,
                        placement_group_bundle_index=bundle_index,
                    ),
                    **ray_remote_kwargs,
                )(ray_distributed_executor.RayWorkerWrapper).remote(rpc_rank=rank)
                for rank, (pg, bundle_index) in enumerate(bundles)
            ]

            # Rank the workers as vLLM does: the driver's host first, because rank 0 serves the
            # torch.distributed store at the driver's address, then each host's workers together.
            driver_ip = get_ip()
            ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])
            workers_per_ip = Counter(ips)
            order = sorted(range(len(ips)), key=lambda i: (ips[i] != driver_ip, workers_per_ip[ips[i]], ips[i]))
            self.workers = [self.workers[i] for i in order]
            ips = [ips[i] for i in order]
            self.collective_rpc("adjust_rank", args=({created: rank for rank, created in enumerate(order)},))

            driver_env = {
                name: os.environ[name]
                for name in get_env_vars_to_copy(
                    exclude_vars=v1_ray_utils.WORKER_SPECIFIC_ENV_VARS,
                    additional_vars=set(current_platform.additional_env_vars),
                    destination="workers",
                )
                if name in os.environ
            }
            # torch_tpu's own rendezvous, separate from the torch.distributed one below.
            torch_tpu_master = {"MASTER_ADDR": ips[0], "MASTER_PORT": str(get_open_port())}
            self._env_vars_for_all_workers = [
                {**driver_env, **torch_tpu_master, **tpu_env}
                for tpu_env in _tpu_worker_envs(ips, vllm_torchtpu_envs.TORCH_TPU_BASE_PORT)
            ]
            self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))
            logger.info("TPU rollout process addresses: %s", self._env_vars_for_all_workers[0]["TPU_PROCESS_ADDRESSES"])

            distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())
            local_ranks = _local_ranks(ips)
            all_kwargs = [
                dict(
                    vllm_config=self.vllm_config,
                    local_rank=local_ranks[rank],
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=rank % self.parallel_config.tensor_parallel_size == 0,
                )
                for rank in range(len(self.workers))
            ]
            self.collective_rpc("init_worker", args=(all_kwargs,))
            self.collective_rpc("init_device")
            self.collective_rpc("load_model")

        def patched_execute_dag(self, scheduler_output, grammar_output, non_block=False):
            # Plain Ray calls instead of vLLM's compiled Ray graph. The graph runs the model on a
            # background thread, where torch.compile's recompile limit is still the default 8:
            # vllm-torchtpu raises it to 1024 on the main thread only (dynamo config overrides
            # are per thread), and the first request fails with FailOnRecompileLimitHit. verl uses
            # no KV connector, so rank 0's output is the step's result.
            refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
            if non_block:
                return FutureWrapper(refs[0])
            output = ray.get(refs)[0]
            detach_zero_copy_from_model_runner_output(output)
            return output

        # create_engine_config sets distributed_executor_backend="ray", which resolves to vLLM V1's
        # generic RayDistributedExecutor, so that is the class to patch.
        v1_ray_executor.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        v1_ray_executor.RayDistributedExecutor._execute_dag = patched_execute_dag

        _PATCHES_APPLIED = True

        logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")
    except Exception as e:
        logger.warning("Failed to apply TPU vLLM patches: %s", e, exc_info=True)
