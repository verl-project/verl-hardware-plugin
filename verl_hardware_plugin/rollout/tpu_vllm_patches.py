# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for multi-host TPU rollout.

``patch_vllm_for_tpu`` is what makes a vLLM TP engine that spans several TPU hosts start and run
under Ray. It is ported from verl's ``tpu-main`` branch
(``verl/workers/rollout/vllm_rollout/tpu_utils.py``). Every patch here works around a gap in
vLLM, vllm-torchtpu or Ray; ``docs/user_guide_tpu/rollout.md`` lists them with their upstream
owners. Each one should be deleted once its owner lands the fix.

Changes against the ``tpu-main`` version:

* ``load_weights_from_ray_registry`` is not installed here. The TPU checkpoint engine
  (``verl_hardware_plugin.engines.tpu_checkpoint_engine``) owns that worker method, and it
  reads a different registry actor.
* The function is idempotent. The first successful call installs the patches; later calls
  only re-apply the environment side effects. The ``tpu-main`` version re-wrapped every
  patched function on each call, and it is called again from several entry points.
* ``_orig_run_engine_core`` gets a module-level default, so ``_patched_run_engine_core``
  cannot raise ``NameError`` when it runs before any patch in its process.
* Patches that are no-ops on the pinned stack (vLLM v0.29.0, vllm-torchtpu 9faafb17) are
  dropped: the ``VLLM_USE_V1`` flag, the ``init_cached_hf_modules`` stub, the
  ``setup_device_if_necessary`` override, the torchtpu topology-map update, the shadowed
  bundle-index ``_init_workers_ray`` wrapper, and branches keyed on env vars nothing sets.
"""

import copy
import logging
import multiprocessing
import multiprocessing.process
import os
import time
from collections import defaultdict
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)

# Base port for the rollout slice builder mesh. Each local chip takes ``base + local_rank``.
# Distinct from the trainer's 8471 so a colocated trainer and rollout never collide.
TPU_ROLLOUT_BASE_PORT = 8070
DEFAULT_TPU_TOPOLOGY_MAP = {
    1: "1,1,1",
    2: "1,2,1",
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}


# TODO: consolidate with platform_tpu.resolve_tpu_topology_bounds once the TPU platform PR lands.
def _resolve_tpu_topology_bounds(total_chips: int, num_nodes: int) -> tuple[str, str, str, str]:
    """Dynamically resolves (topology, host_bounds, chips_per_host_bounds, chips_per_host) for TPU slices."""
    topology = os.environ.get("TORCH_TPU_TOPOLOGY") or DEFAULT_TPU_TOPOLOGY_MAP.get(total_chips, "1,1,1")
    inferred_chips_per_host = max(1, total_chips // max(1, num_nodes))
    chips_per_host = str(os.environ.get("VLLM_TPU_CHIPS_PER_HOST", inferred_chips_per_host))

    if total_chips <= 4:
        host_bounds = "1,1,1"
        chips_per_host_bounds = topology if num_nodes == 1 else "1,1,1"
    else:
        host_bounds = topology
        chips_per_host_bounds = "1,1,1"

    return topology, host_bounds, chips_per_host_bounds, chips_per_host


# Fallback imports for TPU vLLM platforms
try:
    from vllm_torchtpu.executors import ray_distributed_executor
except ImportError:
    ray_distributed_executor = None

try:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
except ImportError:
    EngineArgs = None
    AsyncEngineArgs = None

try:
    import vllm.envs as vllm_envs
except ImportError:
    vllm_envs = None

try:
    import vllm_torchtpu.envs as tpu_envs
except ImportError:
    tpu_envs = None

try:
    from vllm_torchtpu.worker.tpu_worker import TPUWorker
except ImportError:
    TPUWorker = None

try:
    from vllm.utils import get_ip
except ImportError:
    try:
        from vllm.utils.network_utils import get_ip
    except ImportError:
        get_ip = None

try:
    from vllm.v1.executor.ray_executor import RayWorkerMetaData
except ImportError:
    try:
        from vllm.v1.executor.ray_utils import RayWorkerMetaData
    except ImportError:

        class RayWorkerMetaData:  # type: ignore[no-redef]
            def __init__(self, worker, created_rank):
                self.worker = worker
                self.created_rank = created_rank
                self.adjusted_rank = None
                self.ip = None


try:
    from vllm.utils import get_open_port
except ImportError:
    try:
        from vllm_torchtpu.utils import get_open_port
    except ImportError:
        try:
            from vllm.utils.network_utils import get_open_port
        except ImportError:
            get_open_port = None

try:
    from vllm_torchtpu.platforms.tpu_platform import get_distributed_init_method
except ImportError:
    try:
        from vllm.utils.network_utils import get_distributed_init_method
    except ImportError:
        get_distributed_init_method = None

try:
    from vllm.platforms import current_platform
except ImportError:
    current_platform = None

# Resolved at call time by the patched _init_workers_ray.
try:
    from vllm_torchtpu.platforms.tpu_platform import get_env_vars_to_copy
except ImportError:
    try:
        from vllm.ray.ray_env import get_env_vars_to_copy
    except ImportError:
        get_env_vars_to_copy = None

_orig_run_engine_core: Any = None
_PATCHES_APPLIED = False


class PickleableProcessWrapper:
    """
    A pickle-compatible wrapper for multiprocessing target functions to ensure
    vLLM-on-TPU worker patches are automatically applied in the spawned processes.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        if self.target is not None:
            return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """
    Monkey-patch multiprocessing.process.BaseProcess to wrap target entrypoints
    with PickleableProcessWrapper, propagating vLLM-on-TPU patches to child processes.
    """
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


def _patched_run_engine_core(*args, **kwargs):
    try:
        patch_vllm_for_tpu()
    except Exception as e:
        print(f"[TPU DEBUG] Error re-applying patch in EngineCoreProc: {e}", flush=True)

    global _orig_run_engine_core
    if _orig_run_engine_core is None:
        import vllm.v1.engine.core as v1_core

        _orig_run_engine_core = getattr(v1_core.EngineCoreProc, "_unpatched_run_engine_core", None)

    if hasattr(_orig_run_engine_core, "__func__"):
        _orig_run_engine_core = _orig_run_engine_core.__func__

    if _orig_run_engine_core is not None:
        return _orig_run_engine_core(*args, **kwargs)
    raise RuntimeError("[TPU ERROR] _orig_run_engine_core could not be resolved in _patched_run_engine_core")


def patch_vllm_for_tpu() -> None:
    """
    Apply TPU-specific patches and workarounds to vLLM and torchtpu-vllm workers.
    Ensures correct topology routing, un-clashed TCP ports and driver-worker environment
    synchronization on GKE TPU v6e instances.
    """
    global _PATCHES_APPLIED

    # Environment side effects. Cheap and safe to repeat in every process that calls this.
    patch_multiprocessing_for_tpu()
    if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
        del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    # vLLM can reload a degenerate AOT compile artifact (num_artifacts=0) that silently runs the
    # model eagerly; eager execution then hits the XLA:TPU unaligned-DUS CHECK (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    if _PATCHES_APPLIED:
        return

    try:
        import torch
        import torch._dynamo
        import torch._ops
        import torch.compiler

        def _allow(op):
            if op is not None:
                for fn in (
                    getattr(torch.compiler, "allow_in_graph", None),
                    getattr(torch._dynamo, "allow_in_graph", None),
                ):
                    if fn:
                        try:
                            fn(op)
                        except Exception:
                            pass

        for ns_name in ["_c10d_functional", "c10d_functional"]:
            if hasattr(torch.ops, ns_name):
                ns = getattr(torch.ops, ns_name)
                _allow(ns)
                for name in dir(ns):
                    try:
                        attr = getattr(ns, name)
                        _allow(attr)
                        if hasattr(attr, "default"):
                            _allow(attr.default)
                    except Exception:
                        pass
    except Exception:
        pass

    def dummy_reset_encoder_cache(*args, **kwargs):
        pass

    try:
        orig_ray_init = ray.init

        def patched_ray_init(*args, **kwargs):
            if "runtime_env" in kwargs and kwargs["runtime_env"]:
                rt_env = kwargs["runtime_env"]
                if isinstance(rt_env, dict) and "worker_process_setup_hook" in rt_env:
                    rt_env = dict(rt_env)
                    del rt_env["worker_process_setup_hook"]
                    kwargs["runtime_env"] = rt_env
            if "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" in os.environ:
                del os.environ["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"]
            return orig_ray_init(*args, **kwargs)

        ray.init = patched_ray_init
    except Exception as e:
        logger.warning(f"Failed to patch ray.init to clear worker_process_setup_hook: {e}")

    try:
        if ray_distributed_executor is None:
            return

        try:
            if EngineArgs is not None:
                orig_create_engine_config = EngineArgs.create_engine_config

                def patched_create_engine_config(self, *args, **kwargs):
                    is_multi_host = (
                        self.tensor_parallel_size > 4
                        or int(os.environ.get("NNODES_ROLLOUT", "1")) > 1
                        or os.environ.get("TPU_MULTIHOST_BACKEND") == "ray"
                    )

                    if getattr(self, "data_parallel_size", 1) <= 1:
                        if hasattr(self, "data_parallel_external_lb"):
                            self.data_parallel_external_lb = False
                        if hasattr(self, "data_parallel_rank"):
                            self.data_parallel_rank = None
                        if hasattr(self, "data_parallel_size_local"):
                            self.data_parallel_size_local = None
                        if hasattr(self, "data_parallel_start_rank"):
                            self.data_parallel_start_rank = None
                        if hasattr(self, "data_parallel_hybrid_lb"):
                            self.data_parallel_hybrid_lb = False

                    if not is_multi_host:
                        if "TPU_MULTIHOST_BACKEND" in os.environ:
                            del os.environ["TPU_MULTIHOST_BACKEND"]
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = None
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = None

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)

                        logger.info("Single-host TPU rollout detected; using local executor backend.")
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = True
                            logger.info("Enabled async_scheduling on TPU for single-host rollout.")
                    else:
                        os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
                        if vllm_envs is not None and hasattr(vllm_envs, "TPU_MULTIHOST_BACKEND"):
                            vllm_envs.TPU_MULTIHOST_BACKEND = "ray"
                        if tpu_envs is not None and hasattr(tpu_envs, "TPU_MULTIHOST_BACKEND"):
                            tpu_envs.TPU_MULTIHOST_BACKEND = "ray"

                        vllm_config = orig_create_engine_config(self, *args, **kwargs)
                        vllm_config.parallel_config.distributed_executor_backend = "ray"
                        logger.info("Multi-host TPU rollout detected; using Ray distributed executor backend.")
                        if hasattr(vllm_config, "scheduler_config") and hasattr(
                            vllm_config.scheduler_config, "async_scheduling"
                        ):
                            vllm_config.scheduler_config.async_scheduling = False
                            logger.info("Disabled async_scheduling on TPU for multi-host Ray rollout.")
                    return vllm_config

                EngineArgs.create_engine_config = patched_create_engine_config
                if AsyncEngineArgs is not None:
                    AsyncEngineArgs.create_engine_config = patched_create_engine_config
        except Exception as e:
            logger.warning(f"Failed to patch EngineArgs.create_engine_config: {e}")

        try:
            if TPUWorker is not None:
                TPUWorker.reset_encoder_cache = dummy_reset_encoder_cache
                logger.info("Patched TPUWorker class with dummy_reset_encoder_cache.")
        except Exception as e:
            logger.warning(f"Failed to patch TPUWorker class directly: {e}")

        original_driver_environ_setitem = os.environ.__class__.__setitem__

        def patched_driver_environ_setitem(self, key, value):
            if key in (
                "TORCH_TPU_SLICEBUILDER_ADDRESSES",
                "TPU_PROCESS_ADDRESSES",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
                "TORCH_TPU_TOPOLOGY",
                "TPU_WORKER_HOSTNAMES",
            ):
                value = os.environ.get(key, value)
            elif key == "LIBTPU_INIT_ARGS":
                value = value.replace("--deepsea_chip_config_name=megachip_tccontrol", "")
            original_driver_environ_setitem(self, key, value)

        os.environ.__class__.__setitem__ = patched_driver_environ_setitem  # type: ignore[method-assign]

        try:
            import vllm.v1.executor.ray_utils as v1_ray_utils

            orig_avail_res = v1_ray_utils.available_resources_per_node

            def patched_avail_res(*args, **kwargs):
                res_map = orig_avail_res(*args, **kwargs)
                for node_id, res in res_map.items():
                    res["TPU"] = max(res.get("TPU", 0.0), 4.0)
                return res_map

            v1_ray_utils.available_resources_per_node = patched_avail_res

            orig_init_ray_cluster = v1_ray_utils.initialize_ray_cluster

            def patched_initialize_ray_cluster(parallel_config, ray_address=None, *args, **kwargs):
                # Connect with the runtime_env vLLM captured from the server actor *before* any
                # other Ray call below: those would auto-init Ray with an empty runtime_env, vLLM
                # would then skip its own ray.init, and the RayWorkerWrapper actors would start
                # without the job's py_modules / working_dir.
                if not ray.is_initialized():
                    ray.init(address=ray_address, runtime_env=getattr(parallel_config, "ray_runtime_env", None))
                if parallel_config.placement_group is None:
                    curr_pg = ray.util.get_current_placement_group()
                    if curr_pg is None:
                        try:
                            pgs = ray.util.placement_group_table()
                            for pg_id, pg_info in pgs.items():
                                state = (
                                    pg_info.get("state")
                                    if isinstance(pg_info, dict)
                                    else getattr(pg_info, "state", None)
                                )
                                if hasattr(state, "name"):
                                    state = state.name
                                name = (
                                    pg_info.get("name") if isinstance(pg_info, dict) else getattr(pg_info, "name", None)
                                )
                                if (
                                    str(state) in ("CREATED", "1")
                                    and name
                                    and ("rollout" in str(name) or "global" in str(name))
                                ):
                                    try:
                                        candidate_pg = ray.util.get_placement_group(name)
                                        if candidate_pg is not None:
                                            num_bundles = len(getattr(candidate_pg, "bundle_specs", []))
                                            if num_bundles >= parallel_config.world_size:
                                                curr_pg = candidate_pg
                                                break
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    parallel_config.placement_group = curr_pg
                try:
                    return orig_init_ray_cluster(parallel_config, ray_address, *args, **kwargs)
                except ValueError as e:
                    if "exceeds the total number of available" in str(e) or "placement group" in str(e):
                        logger.warning(f"Bypassed vLLM placement group size validation on multi-node TPU: {e}")
                        return
                    raise

            try:
                import vllm.v1.engine.core as v1_core

                if not getattr(v1_core, "_verl_tpu_patched", False):
                    v1_core._verl_tpu_patched = True
                    global _orig_run_engine_core
                    raw_fn = v1_core.EngineCoreProc.run_engine_core
                    if hasattr(raw_fn, "__func__"):
                        raw_fn = raw_fn.__func__
                    _orig_run_engine_core = raw_fn
                    v1_core.EngineCoreProc._unpatched_run_engine_core = raw_fn
                    v1_core.EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
                    v1_core.run_engine_core = _patched_run_engine_core
            except Exception as e3:
                logger.warning(f"Failed to patch v1_core.run_engine_core: {e3}")

            try:
                import vllm.v1.executor.ray_executor as v1_ray_executor

                v1_ray_executor.initialize_ray_cluster = patched_initialize_ray_cluster
            except Exception as e2:
                logger.warning(f"Failed to patch v1_ray_executor: {e2}")
        except Exception as e:
            logger.warning(f"Failed to patch v1_ray_utils: {e}")

        OriginalRayWorkerWrapper = ray_distributed_executor.RayWorkerWrapper
        original_init_worker = OriginalRayWorkerWrapper.init_worker
        original_wrapper_init = OriginalRayWorkerWrapper.__init__

        def patched_wrapper_init(self, *args, **kwargs):
            # Pooled vLLM Ray workers never import the plugin; this is how the patches reach them.
            patch_vllm_for_tpu()
            return original_wrapper_init(self, *args, **kwargs)

        OriginalRayWorkerWrapper.__init__ = patched_wrapper_init

        def patched_init_worker(self, *args, **kwargs):
            patch_vllm_for_tpu()

            try:
                import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils

                vllm_weight_utils.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass
            try:
                import vllm.model_executor.model_loader.dummy_loader as vllm_dummy_loader

                vllm_dummy_loader.initialize_dummy_weights = lambda *args, **kwargs: None
            except Exception:
                pass

            torch.set_grad_enabled(False)

            return original_init_worker(self, *args, **kwargs)

        OriginalRayWorkerWrapper.init_worker = patched_init_worker

        def patched_init_workers_ray(self, placement_group, **ray_remote_kwargs):
            import inspect

            RayWorkerWrapper_local = ray_distributed_executor.RayWorkerWrapper

            self.workers = []
            self.pp_tp_workers = []

            if self.parallel_config.ray_workers_use_nsight:
                ray_remote_kwargs = self._configure_ray_workers_use_nsight(ray_remote_kwargs)

            bundle_indices = []
            if vllm_envs is not None and vllm_envs.VLLM_RAY_BUNDLE_INDICES:
                bundle_indices = list(map(int, vllm_envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
                assert len(bundle_indices) == self.parallel_config.world_size, (
                    "VLLM_RAY_BUNDLE_INDICES must have the same size"
                    f" as the world size, but got {bundle_indices=} "
                    f"and {self.parallel_config.world_size=}"
                )
                assert len(set(bundle_indices)) == len(bundle_indices), (
                    f"VLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}"
                )
            else:
                for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                    if current_platform is not None and bundle.get(current_platform.ray_device_key, 0):
                        bundle_indices.append(bundle_id)

            worker_metadata = []
            driver_ip = get_ip() if get_ip is not None else ""
            num_tpu_per_worker = 1.0
            for rank, bundle_id in enumerate(bundle_indices):
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_tpu_per_worker}
                    if current_platform is not None
                    else {},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper_local).remote(rpc_rank=rank)
                worker_metadata.append(
                    RayWorkerMetaData(worker=worker, created_rank=rank) if RayWorkerMetaData is not None else None
                )

            worker_ips = ray.get([each.worker.get_node_ip.remote() for each in worker_metadata])

            for each, ip in zip(worker_metadata, worker_ips, strict=False):
                each.ip = ip

            logger.info(f"Initialized worker_metadata: {worker_metadata}")

            ip_counts = {}
            for ip in worker_ips:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

            def sort_by_driver_then_worker_ip(item):
                ip = item.ip
                return (0 if ip == driver_ip else 1, ip_counts[ip], ip)

            sorted_worker_metadata = sorted(worker_metadata, key=sort_by_driver_then_worker_ip)
            start_rank = 0
            for i, item in enumerate(sorted_worker_metadata):
                item.adjusted_rank = i + start_rank
            logger.info(f"Initialized sorted worker_metadata: {sorted_worker_metadata}")

            self.workers = [item.worker for item in sorted_worker_metadata]
            rerank_mapping = {item.created_rank: item.adjusted_rank for item in sorted_worker_metadata}
            self.collective_rpc("adjust_rank", args=(rerank_mapping,))

            worker_node_and_tpu_ids = []
            for worker in self.workers:
                if hasattr(worker, "get_node_and_gpu_ids"):
                    worker_node_and_tpu_ids.append(ray.get(worker.get_node_and_gpu_ids.remote()))
                else:
                    worker_node_and_tpu_ids.append(ray.get(worker.get_node_and_physical_gpu_ids.remote()))

            node_workers = defaultdict(list)
            node_tpus = defaultdict(list)

            for i, (node_id, tpu_ids) in enumerate(worker_node_and_tpu_ids):
                node_workers[node_id].append(i)
                tpu_ids = [int(x) for x in tpu_ids]
                node_tpus[node_id].extend(tpu_ids)
            for node_id, tpu_ids in node_tpus.items():
                node_tpus[node_id] = sorted(tpu_ids)
            logger.info(f"RayDistributedExecutor | node_workers={node_workers} | node_tpus={node_tpus}")

            all_ips = set(worker_ips + [driver_ip])
            n_ips = len(all_ips)
            n_nodes = len(node_workers)

            if n_nodes != n_ips:
                logger.warning(
                    f"Got {n_nodes} nodes but with {n_ips} IP addresses. "
                    "This is not a typical production setup whose "
                    "number of nodes and IPs is equal. This setup may "
                    "lead to unexpected behaviors."
                )

            unique_node_ids = list(node_workers.keys())
            num_nodes = len(unique_node_ids)

            sb_addresses = []
            base_port = int(os.environ.get("TORCH_TPU_BASE_PORT", TPU_ROLLOUT_BASE_PORT))
            for node_id in unique_node_ids:
                w_idx = node_workers[node_id][0]
                host_ip = sorted_worker_metadata[w_idx].ip
                chips_on_node = len(node_workers[node_id])
                for lr in range(chips_on_node):
                    sb_addresses.append(f"{host_ip}:{base_port + lr}")

            sb_addresses_str = ",".join(sb_addresses)
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
            logger.info(f"Constructed TORCH_TPU_SLICEBUILDER_ADDRESSES: {sb_addresses_str}")

            total_chips = len(self.workers)
            topology, host_bounds, chips_per_host_bounds, chips_per_host = _resolve_tpu_topology_bounds(
                total_chips=total_chips,
                num_nodes=num_nodes,
            )

            rank_0_node_id = unique_node_ids[0]
            rank_0_worker_index = node_workers[rank_0_node_id][0]
            master_addr = sorted_worker_metadata[rank_0_worker_index].ip
            master_port = str(get_open_port()) if get_open_port is not None else ""

            all_args_to_update_environment_variables = []
            for i in range(total_chips):
                node_id = worker_node_and_tpu_ids[i][0]
                node_rank = unique_node_ids.index(node_id)
                args = {
                    "NNODES": str(num_nodes),
                    "NODE_RANK": str(node_rank),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": master_port,
                    "TORCH_TPU_TOPOLOGY": topology,
                    "LOCAL_WORLD_SIZE": str(len(node_tpus[node_id])),
                    "TPU_NUM_HOSTS": str(num_nodes),
                }
                if "TORCH_TPU_XPROF_SESSION_ID" not in os.environ:
                    os.environ["TORCH_TPU_XPROF_SESSION_ID"] = str(time.time_ns())

                args["TORCH_TPU_XPROF_SESSION_ID"] = os.environ["TORCH_TPU_XPROF_SESSION_ID"]
                all_args_to_update_environment_variables.append(args)

            env_vars_to_copy_list = []
            if get_env_vars_to_copy is not None:
                exclude_vars = getattr(self, "WORKER_SPECIFIC_ENV_VARS", None)
                if exclude_vars is None:
                    from vllm.v1.executor import ray_utils as _ray_utils

                    exclude_vars = getattr(_ray_utils, "WORKER_SPECIFIC_ENV_VARS", set())
                env_vars_to_copy_list = get_env_vars_to_copy(
                    exclude_vars=exclude_vars,
                    additional_vars=set(current_platform.additional_env_vars)
                    if current_platform is not None
                    else set(),
                    destination="workers",
                )

            for i, args in enumerate(all_args_to_update_environment_variables):
                for name in env_vars_to_copy_list:
                    if name in os.environ:
                        args[name] = os.environ[name]
                logger.debug(f"RayDistributedExecutor | Worker {i} environment variables before patch: {args}")

            self._env_vars_for_all_workers = all_args_to_update_environment_variables

            try:
                unique_host_ips = [sorted_worker_metadata[node_workers[nid][0]].ip for nid in unique_node_ids]
                host_names_str = ",".join(unique_host_ips)
                for i, worker in enumerate(self.workers):
                    node_id = worker_node_and_tpu_ids[i][0]
                    host_idx = unique_node_ids.index(node_id)
                    local_chip_id = node_workers[node_id].index(i)
                    args = self._env_vars_for_all_workers[i]
                    args["RANK"] = str(i)
                    args["LOCAL_RANK"] = str(local_chip_id)
                    args["TPU_VISIBLE_CHIPS"] = str(local_chip_id)
                    args["TPU_PROCESS_PORT"] = str(base_port + local_chip_id)
                    args["CLOUD_TPU_TASK_ID"] = str(host_idx)
                    args["TPU_WORKER_HOSTNAMES"] = host_names_str
                    args["TPU_HOST_BOUNDS"] = host_bounds
                    args["TPU_CHIPS_PER_HOST_BOUNDS"] = chips_per_host_bounds
                    args["CHIPS_PER_HOST"] = chips_per_host
                    args["TORCH_TPU_TOPOLOGY"] = topology
                    args["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = sb_addresses_str
                    args["TPU_PROCESS_ADDRESSES"] = sb_addresses_str
                    if total_chips > 4 or num_nodes > 1:
                        args["TPU_MULTIHOST_BACKEND"] = "ray"

                    logger.info(
                        f"Configured TPU worker {i} (host {host_idx}, chip {local_chip_id}) env vars: "
                        f"TPU_VISIBLE_CHIPS={local_chip_id}, TPU_PROCESS_PORT={base_port + local_chip_id}, "
                        f"CLOUD_TPU_TASK_ID={host_idx}"
                    )
            except Exception as patch_err:
                logger.warning(f"Failed to inject TPU worker env vars: {patch_err}")

            self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))

            distributed_init_method = (
                get_distributed_init_method(driver_ip, get_open_port())
                if get_distributed_init_method is not None
                else ""
            )

            driver_node_id = ray.get_runtime_context().get_node_id()

            accepts_prev_worker_ip = True
            if TPUWorker is not None:
                try:
                    accepts_prev_worker_ip = "prev_worker_ip" in inspect.signature(TPUWorker.__init__).parameters
                except Exception:
                    pass

            accepts_assigned_physical_gpu_ids = False
            try:
                accepts_assigned_physical_gpu_ids = "assigned_physical_gpu_ids" in inspect.getsource(
                    original_init_worker
                )
            except Exception:
                pass

            all_kwargs = []
            for rank, (node_id, _) in enumerate(worker_node_and_tpu_ids):
                local_rank = node_workers[node_id].index(rank)
                ip = sorted_worker_metadata[rank].ip
                prev_ip = sorted_worker_metadata[rank - 1].ip if rank > 0 else ""

                worker_vllm_config = self.vllm_config

                if (
                    node_id != driver_node_id
                    and getattr(self.vllm_config, "model_config", None)
                    and getattr(self.vllm_config.model_config, "model_weights", None)
                ):
                    worker_vllm_config = copy.deepcopy(self.vllm_config)
                    worker_vllm_config.model_config.model = worker_vllm_config.model_config.model_weights
                    worker_vllm_config.model_config.model_weights = None

                kwargs = dict(
                    vllm_config=worker_vllm_config,
                    local_rank=local_rank,
                    rank=rank,
                    distributed_init_method=distributed_init_method,
                    is_driver_worker=(not self.parallel_config)
                    or (rank % self.parallel_config.tensor_parallel_size == 0),
                    ip=ip,
                )
                if accepts_prev_worker_ip:
                    kwargs["prev_worker_ip"] = prev_ip
                if accepts_assigned_physical_gpu_ids:
                    kwargs["assigned_physical_gpu_ids"] = sorted(node_tpus[node_id])
                all_kwargs.append(kwargs)
            self.collective_rpc("init_worker", args=(all_kwargs,))
            self.collective_rpc("init_device")
            if self.parallel_config.pipeline_parallel_size > 1:
                self.collective_rpc("initialize_pp_transfer_connect")
            self.collective_rpc("load_model")
            if hasattr(self, "pp_tp_workers"):
                self.pp_tp_workers = []
                pp_size = self.parallel_config.pipeline_parallel_size if self.parallel_config else 1
                tp_size = self.parallel_config.tensor_parallel_size if self.parallel_config else len(self.workers)
                for pp_rank in range(pp_size):
                    self.pp_tp_workers.append([])
                    for tp_rank in range(tp_size):
                        rank = (pp_rank * tp_size) + tp_rank
                        if rank < len(self.workers):
                            self.pp_tp_workers[pp_rank].append(self.workers[rank])

        def _detach_zero_copy(output):
            try:
                from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
            except ImportError:
                return
            detach_zero_copy_from_model_runner_output(output)

        def patched_execute_dag(
            self,
            scheduler_output,
            grammar_output,
            non_block: bool = False,
        ):
            refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
            if not self.has_connector:
                if not non_block:
                    all_results = ray.get(refs)
                    _detach_zero_copy(all_results[0])
                    return all_results[0]
                from vllm.v1.executor.ray_utils import FutureWrapper

                return FutureWrapper(refs[0])

            assert self.kv_output_aggregator is not None
            if not non_block:
                outputs = ray.get(refs)
                for output in outputs:
                    _detach_zero_copy(output)
                return self.kv_output_aggregator.aggregate(outputs)
            from vllm.v1.executor.ray_utils import FutureWrapper

            return FutureWrapper(refs, self.kv_output_aggregator)

        if ray_distributed_executor is not None:
            ray_distributed_executor.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        try:
            import vllm.executor.ray_distributed_executor as vllm_ray_dist_exec

            vllm_ray_dist_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
        except Exception:
            pass
        try:
            import vllm.v1.executor.ray_executor as vllm_v1_ray_exec

            vllm_v1_ray_exec.RayDistributedExecutor._init_workers_ray = patched_init_workers_ray
            vllm_v1_ray_exec.RayDistributedExecutor._execute_dag = patched_execute_dag
        except Exception:
            pass

        _PATCHES_APPLIED = True

        logger.info("Successfully applied all TPU patches and hacks to vLLM & torchtpu-vllm")
    except Exception as e:
        logger.warning(f"Failed to apply TPU patches: {e}")
