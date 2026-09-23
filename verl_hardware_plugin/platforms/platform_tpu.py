# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Google TPU platform implementation.

Registers the ``tpu`` platform and provides the device abstraction, the Ray resource
options, and the worker-environment plumbing a TPU process needs to join a slice. The
training engine, the generation engine and the weight-transfer path live elsewhere.

``torch_tpu`` mirrors part of the NVIDIA device API surface, so ``TPUDeviceModuleProxy``
adapts it to the memory-management calls verl expects. The platform class itself derives
from ``PlatformBase``: every device query, Ray resource option and environment variable is
defined here rather than inherited from another vendor's platform.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import ray
import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_tpu() -> bool:
    """Try to import torch_tpu, which registers the ``tpu_dist`` distributed backend.

    Returns True if the TPU runtime is usable after the attempt.
    """
    if hasattr(torch, "tpu"):
        return True
    try:
        import torch_tpu  # noqa: F401

        return hasattr(torch, "tpu")
    except Exception as e:
        logger.debug("The current machine has no torch.tpu, because: %s", e)
    return False


_ensure_torch_tpu()  # Attempt at module load time so availability checks are faster later

# Base port for the TPU distributed slice builder mesh. Each local rank takes ``base + local_rank``.
TPU_PROCESS_BASE_PORT = 8471

# TPU chip HBM capacities in bytes
HBM_BYTES_TPU_V5P = 95 * 1024 * 1024 * 1024  # 95 GB
HBM_BYTES_TPU_V6E = 32 * 1024 * 1024 * 1024  # 32 GB
HBM_BYTES_TPU_V7X = 192 * 1024 * 1024 * 1024  # 192 GB

TPU_HBM_BYTES_MAP = {
    "v5p": HBM_BYTES_TPU_V5P,
    "v6e": HBM_BYTES_TPU_V6E,
    "v7x": HBM_BYTES_TPU_V7X,
}

# Fallback HBM capacity when the chip generation cannot be determined.
HBM_BYTES_TPU_DEFAULT = HBM_BYTES_TPU_V6E

# TPU slice topology (an ``x,y,z`` chip mesh) keyed by the number of chips in the slice.
# Mirrors DEFAULT_TPU_TOPOLOGY_MAP in verl's rollout-side TPU utils so the trainer mesh and
# the rollout mesh agree on the physical layout of a given slice size.
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

# Pod types are consulted first: a pod type names the physical slice exactly, whereas the chip
# count is only the number of chips this job was given and can be a subset of the slice.
TPU_POD_TYPE_TOPOLOGY_MAP = {
    "v6e-4": "2,2,1",
    "v6e-8": "2,4,1",
    "v6e-16": "4,4,1",
    "v6e-32": "4,8,1",
}

# Slices at or below this many chips live on a single host, so the chips are addressed within
# the host bounds rather than across them.
TPU_SINGLE_HOST_MAX_CHIPS = 4


def resolve_tpu_topology_bounds(
    total_chips: int,
    num_nodes: int,
    pod_type: str = "",
) -> tuple[str, str, str, str]:
    """Resolve ``(topology, host_bounds, chips_per_host_bounds, chips_per_host)`` for a TPU slice.

    Resolution order for the topology is: the ``TORCH_TPU_TOPOLOGY`` escape hatch, then the pod
    type reported by Ray, then the chip count.

    Raises:
        ValueError: if the topology cannot be determined. Guessing here is worse than failing:
            a wrong mesh does not error, it silently trains on part of the slice.
    """
    topology = (
        os.environ.get("TORCH_TPU_TOPOLOGY")
        or TPU_POD_TYPE_TOPOLOGY_MAP.get(pod_type.lower())
        or DEFAULT_TPU_TOPOLOGY_MAP.get(total_chips)
    )
    if not topology:
        raise ValueError(
            f"Cannot determine the TPU topology for total_chips={total_chips} "
            f"(pod_type={pod_type!r}). Set TORCH_TPU_TOPOLOGY='x,y,z' to override. "
            f"Known chip counts: {sorted(DEFAULT_TPU_TOPOLOGY_MAP)}."
        )

    inferred_chips_per_host = max(1, total_chips // max(1, num_nodes))
    chips_per_host = str(os.environ.get("VERL_TPU_CHIPS_PER_HOST", inferred_chips_per_host))

    if total_chips <= TPU_SINGLE_HOST_MAX_CHIPS:
        host_bounds = "1,1,1"
        chips_per_host_bounds = topology if num_nodes == 1 else "1,1,1"
    else:
        host_bounds = topology
        chips_per_host_bounds = "1,1,1"

    return topology, host_bounds, chips_per_host_bounds, chips_per_host


def get_tpu_chip_hbm_bytes() -> int:
    """Detect the TPU chip generation from Ray node labels or env vars and return its HBM capacity."""
    tpu_type = ""

    # Query Ray cluster node labels for TPU resource type
    try:
        if ray.is_initialized():
            tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
            if tpu_nodes:
                labels = tpu_nodes[0].get("Labels", {})
                tpu_type = (labels.get("ray.io/accelerator-type") or labels.get("ray.io/tpu-pod-type") or "").lower()
    except Exception as e:
        logger.warning(f"Unable to query Ray node labels for TPU chip type: {e}")

    # Fallback to environment variables
    if not tpu_type:
        tpu_type = (
            os.environ.get("TPU_ACCELERATOR_TYPE")
            or os.environ.get("ACCELERATOR_TYPE")
            or os.environ.get("TPU_TYPE")
            or ""
        ).lower()

    for chip_gen, hbm_bytes in TPU_HBM_BYTES_MAP.items():
        if chip_gen in tpu_type:
            return hbm_bytes

    logger.warning(f"Unable to determine TPU chip HBM bytes for tpu_type='{tpu_type}'. Returning -1.")
    return -1


class DummyTpuDeviceModule:
    """Fallback device module for CPU-only nodes and driver processes.

    Provides no-op implementations for torch.tpu APIs on processes where torch_tpu is not imported
    or no TPU devices are attached.
    """

    def is_available(self) -> bool:
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 0

    def synchronize(self) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)


class TPUDeviceModuleProxy:
    """Proxy wrapper for torch.tpu to emulate the PyTorch NVIDIA memory management APIs.

    Provides default fallback implementations for memory tracking methods
    (e.g. memory_reserved, memory_allocated, get_device_properties) that are called throughout verl's
    codebase but not natively provided by torch_tpu.
    """

    def __init__(self, original_module):
        self.__dict__["_original_module"] = original_module

    def __getattr__(self, name):
        if name == "set_device":
            return self.set_device

        if hasattr(self._original_module, name):
            return getattr(self._original_module, name)

        if name == "memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_reserved":
            return lambda *args, **kwargs: 0
        elif name == "max_memory_allocated":
            return lambda *args, **kwargs: 0
        elif name == "reset_peak_memory_stats":
            return lambda *args, **kwargs: None
        elif name == "get_device_properties":

            class DummyDeviceProperties:
                def __init__(self, total_memory=HBM_BYTES_TPU_DEFAULT):
                    self.total_memory = total_memory
                    self.name = "Google TPU"
                    self.major = 1
                    self.minor = 0

            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: DummyDeviceProperties(total_memory=total_mem)
        elif name == "mem_get_info":
            hbm_bytes = get_tpu_chip_hbm_bytes()
            total_mem = hbm_bytes if hbm_bytes > 0 else HBM_BYTES_TPU_DEFAULT
            return lambda *args, **kwargs: (total_mem, total_mem)

        raise AttributeError(f"'TPUDeviceModuleProxy' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original_module, name, value)

    def is_available(self) -> bool:
        if hasattr(self._original_module, "is_available"):
            try:
                return self._original_module.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
                return False
        return False

    def set_device(self, device_index: Any) -> None:
        pass

    def current_device(self) -> int:
        if hasattr(self._original_module, "current_device"):
            try:
                return self._original_module.current_device()
            except Exception as e:
                logger.warning(f"torch.tpu.current_device() failed: {e}")
                return 0
        return 0

    def device_count(self) -> int:
        if hasattr(self._original_module, "device_count"):
            try:
                return self._original_module.device_count()
            except Exception as e:
                logger.warning(f"torch.tpu.device_count() failed: {e}")
                return 0
        return 0

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if hasattr(self._original_module, "synchronize"):
            try:
                self._original_module.synchronize()
            except Exception as e:
                logger.warning(f"torch.tpu.synchronize() failed: {e}")

    def manual_seed_all(self, seed: int) -> None:
        # torch_tpu does not always expose manual_seed_all; the global torch seed is the
        # right fallback because a TPU process owns exactly one chip.
        fn = getattr(self._original_module, "manual_seed_all", None)
        if fn is not None:
            fn(seed)
        else:
            torch.manual_seed(seed)

    def empty_cache(self) -> None:
        if hasattr(self._original_module, "_clear_cache"):
            try:
                self._original_module._clear_cache()
            except Exception as e:
                logger.warning(f"Failed to clear TPU cache: {e}")


def patch_ray_worker() -> None:
    """Ray ``worker_process_setup_hook`` for GKE TPU pods.

    Runs once in every Ray worker process before any task. It does two things:

    1. Pins ``VERL_PLATFORM=tpu`` so a worker that did not inherit the driver's environment still
       resolves the TPU platform.
    2. Makes Raylet's accelerator-id lookup non-fatal. Each GKE pod is given only its own slice of
       the host's TPU chips, so a host-level index lookup can run off the end of the visible list
       and raise ``IndexError``. Returning an empty list is correct here: verl assigns chips itself
       via ``TPU_VISIBLE_CHIPS``.
    """
    os.environ["VERL_PLATFORM"] = "tpu"

    try:
        original_func = ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource

        def patched_func(self, resource_name, resource_regex):
            try:
                return original_func(self, resource_name, resource_regex)
            except IndexError as e:
                logger.debug("Intercepted Ray accelerator lookup IndexError for resource %r: %s", resource_name, e)
                return []

        ray._private.worker.Worker.get_accelerator_ids_for_accelerator_resource = patched_func
    except Exception as e:
        logger.warning(f"Failed to apply Ray worker accelerator patch: {e}")


@PlatformRegistry.register(platform="tpu")
class PlatformTPU(PlatformBase):
    """Platform backend for Google TPUs.

    Answers device queries, builds the Ray resource options for TPU actors, and supplies
    the PJRT environment a worker needs. The parts of ``PlatformBase`` modelled on a CUDA
    device are answered explicitly here rather than inherited.
    """

    def __init__(self):
        super().__init__()
        original_tpu = getattr(torch, "tpu", DummyTpuDeviceModule())
        self._device_module = TPUDeviceModuleProxy(original_tpu)

    @property
    def vendor_name(self) -> str:
        return "google"

    @property
    def device_name(self) -> str:
        return "tpu"

    @property
    def device_module(self):
        return self._device_module

    def current_device(self) -> int:
        return self.device_module.current_device()

    def device_count(self) -> int:
        return self.device_module.device_count()

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        self.device_module.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        self.device_module.manual_seed_all(seed)

    def empty_cache(self) -> None:
        # Delegate when torch.tpu provides it. A TPU runtime that manages HBM itself has no
        # CUDA-style caching allocator to drain, in which case there is nothing to do.
        empty = getattr(self.device_module, "empty_cache", None)
        if empty is not None:
            empty()

    def set_allocator_settings(self, settings: str) -> None:
        # TPU exposes no configurable caching allocator equivalent to CUDA's.
        pass

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # TPU does not use the CUDA compute capability model.
        return (None, None)

    def is_available(self) -> bool:
        if hasattr(torch, "tpu"):
            try:
                return torch.tpu.is_available()
            except Exception as e:
                logger.warning(f"torch.tpu.is_available() check failed: {e}")
        return False

    def is_platform_available(self, use_smi_check=False) -> bool:
        if os.environ.get("VERL_PLATFORM") == "tpu":
            return True
        if "TPU_NAME" in os.environ or "TPU_VISIBLE_DEVICES" in os.environ:
            return True
        # The env vars come first because a CPU-only Ray driver process has no chip attached
        # yet still belongs to a TPU job. Falling back to the runtime lets a plain TPU VM,
        # where nothing sets TPU_NAME, detect itself instead of defaulting to CUDA.
        return self.is_available()

    def ray_resource_name(self) -> str:
        return "TPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        tpu_chips = int(num_gpus)
        return {"resources": {"TPU": tpu_chips}} if tpu_chips >= 1 else {}

    def communication_backend_name(self) -> str:
        return "tpu_dist"

    def visible_devices_envvar(self) -> str:
        # Deliberately CUDA_VISIBLE_DEVICES, not TPU_VISIBLE_CHIPS. verl assigns
        # os.environ[this key] when launching vLLM servers; pointing it at
        # TPU_VISIBLE_CHIPS would overwrite the chip index that get_worker_env_vars()
        # writes and ray_local_rank_override() reads. Changing this breaks rank mapping.
        return "CUDA_VISIBLE_DEVICES"

    def is_ipc_supported(self) -> bool:
        # TPU has no CUDA-style IPC handle. verl derives ``use_shm = not is_support_ipc()``,
        # so returning False routes weight transfer through shared memory instead.
        return False

    def cudart(self) -> Any:
        # There is no CUDA runtime on a TPU host; PlatformBase documents None for "not applicable".
        return None

    # No TPU profiler or NVTX equivalent. PlatformBase asks platforms without profiling
    # support to make these no-ops: yield immediately, and start/stop do nothing.
    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def ray_noset_envvars(self) -> list[str]:
        # CUDA_VISIBLE_DEVICES is listed because visible_devices_envvar() deliberately returns
        # it (see above), so Ray must be told not to manage that variable either.
        return [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def ray_local_rank_override(self) -> Optional[str]:
        # Ray does not enumerate TPU chips, so the index comes from TPU_VISIBLE_CHIPS, which
        # get_worker_env_vars() sets when the worker is created.
        return os.environ.get("TPU_VISIBLE_CHIPS", "0")

    def supports_colocated_worker_groups(self) -> bool:
        # A TPU chip belongs to one process and cannot be shared between colocated WorkerGroups.
        return False

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        """Install the GKE TPU worker setup hook and pin the platform inside Ray workers."""
        return {
            "runtime_env": {
                "worker_process_setup_hook": patch_ray_worker,
                "env_vars": {"VERL_PLATFORM": "tpu"},
            }
        }

    def auto_assign_accelerator_type(self, name_prefix: str, accelerator_type: Optional[str]) -> Optional[str]:
        """Pin a resource pool to a TPU slice on multi-slice clusters.

        Called unconditionally from ``verl.single_controller.ray.base.RayResourcePool.__init__``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        A KubeRay TPU cluster advertises one ``tpu-group-<n>`` custom resource per slice. Without
        an affinity label a pool can straddle two slices, which the TPU mesh cannot span, so pin
        it to the first slice.
        """
        if accelerator_type is not None:
            return accelerator_type

        try:
            if ray.is_initialized():
                tpu_slices = set()
                for node in ray.nodes():
                    if node.get("Alive"):
                        for res in node.get("Resources", {}):
                            if res.startswith("tpu-group-"):
                                tpu_slices.add(res)
                slices = sorted(tpu_slices)
                if slices:
                    return slices[0]
        except Exception as e:
            logger.debug("Could not auto-assign a TPU slice for %r: %s", name_prefix, e)

        return accelerator_type

    def configure_placement_group_bundle(
        self,
        bundle: dict,
        use_gpu: bool,
        device_name: str,
        name_prefix: str,
        accelerator_type: Optional[str] = None,
    ) -> None:
        """Shape a placement-group bundle for GKE TPU.

        Called unconditionally from ``verl.single_controller.ray.base.RayResourcePool``
        whenever ``device_name == "tpu"``; ``PlatformBase`` does not declare it, so the TPU
        platform has to provide it.

        The slice affinity is requested as a fractional amount so it acts as a label rather
        than a real reservation.
        """
        if use_gpu:
            bundle[device_name] = 1
        if accelerator_type is not None:
            bundle[accelerator_type] = 1e-4

    def get_tpu_env_vars(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        pgs: list,
    ) -> dict[str, str]:
        """Generate TPU-specific distributed environment variables for PJRT mesh initialization."""
        node_ip_map = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes() if node.get("Alive", False)}
        bundle_ips = []
        local_ip = ray.util.get_node_ip_address()
        clean_prefix = name_prefix.lower().split("_")[0] if name_prefix else ""
        matching_pgs = []

        # 1. Primary filter: Select placement group containing current worker's node IP
        for p in pgs:
            specs = ray._private.state.state.placement_group_table(p.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            pg_ips = [node_ip_map[node_id] for b_idx, node_id in sorted(bundles_map.items()) if node_id in node_ip_map]
            if local_ip in pg_ips:
                matching_pgs.append(p)

        # 2. Secondary fallback: Filter by clean_prefix if placement group names are explicitly set
        if not matching_pgs and clean_prefix:
            for p in pgs:
                p_name = ray._private.state.state.placement_group_table(p.id).get("name", "").lower()
                if clean_prefix in p_name:
                    matching_pgs.append(p)

        target_pgs = matching_pgs if matching_pgs else pgs

        for pg in target_pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            if specs.get("state") != "CREATED":
                continue
            bundles_map = specs.get("bundles_to_node_id", {})
            for b_idx in sorted(bundles_map.keys()):
                node_id = bundles_map[b_idx]
                if node_id in node_ip_map:
                    bundle_ips.append(node_ip_map[node_id])

        base_port = TPU_PROCESS_BASE_PORT
        sb_addresses = [f"{ip}:{base_port + (b_idx % local_world_size)}" for b_idx, ip in enumerate(bundle_ips)]

        # Extract unique worker hostnames preserving rank order
        unique_hostnames = list(dict.fromkeys(bundle_ips))

        env_vars = {
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_ADDRESSES": ",".join(sb_addresses),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "CLOUD_TPU_TASK_ID": str(rank // local_world_size),
            "TPU_WORKER_HOSTNAMES": ",".join(unique_hostnames),
            "TPU_VISIBLE_CHIPS": str(local_rank),
        }

        # Derive the mesh from the pod type Ray reports, falling back to the chip count.
        tpu_nodes = [node for node in ray.nodes() if "TPU" in node.get("Resources", {}) and node.get("Alive")]
        tpu_type = tpu_nodes[0].get("Labels", {}).get("ray.io/tpu-pod-type", "") if tpu_nodes else ""

        topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
            total_chips=world_size,
            num_nodes=len(unique_hostnames),
            pod_type=tpu_type,
        )

        env_vars.update(
            {
                "TORCH_TPU_TOPOLOGY": topology,
                "TPU_HOST_BOUNDS": host_bounds,
                "TPU_CHIPS_PER_HOST_BOUNDS": chips_per_host_bounds,
                "CHIPS_PER_HOST": chips_per_host,
            }
        )

        return env_vars

    def get_worker_env_vars(
        self,
        resource_pool,
        rank: int,
        world_size: int,
        local_rank: int,
        local_world_size: int,
        name_prefix: str,
        device_name: str,
    ) -> dict[str, str]:
        """Return platform-specific TPU environment variables for worker nodes."""
        env_vars = {}
        if "VERL_PLATFORM" in os.environ:
            env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]
        for var in self.ray_noset_envvars():
            env_vars[var] = "1"
        pgs = resource_pool.get_placement_groups(device_name=device_name)
        tpu_env = self.get_tpu_env_vars(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            name_prefix=name_prefix,
            pgs=pgs,
        )
        env_vars.update(tpu_env)
        return env_vars
