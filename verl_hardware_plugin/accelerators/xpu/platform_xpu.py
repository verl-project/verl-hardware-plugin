# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Intel XPU platform implementation.

Supports Intel Data Center GPU Max Series and similar devices via
torch.xpu and oneAPI/oneCCL (xccl) communication backend.

Key design decisions for Intel XPU:
- device_name: "xpu" (Intel's torch extension uses torch.xpu.*)
- vendor_name: "intel" (used for engine lookup key)
- communication_backend: "xccl" (oneAPI Collective Communications Library)
- ray_resource_name: "GPU" (reuses Ray's built-in GPU scheduling)
- is_ipc_supported: False (Intel XPU does not support CUDA-style IPC yet)
- visible_devices_envvar: "ZE_AFFINITY_MASK" (Intel Level Zero driver)

Prerequisites:
- intel-extension-for-pytorch must be installed
- Source oneAPI environment: source /opt/intel/oneapi/setvars.sh

Example usage:
    export VERL_PLATFORM=intel
    python -m verl.trainer.main --config config.yaml
"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry
from verl_hardware_plugin.accelerators.xpu.patches import (
    numa_affinity_patch,
    ray_device_pin_patch,
    reduce_avg_allreduce_patch,
)
from verl_hardware_plugin.accelerators.xpu.profilers import register_vtune

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Every monkeypatch this platform installs, applied from __init__ below. See
# each module's own docstring for what it patches and why.
_PATCHES = (reduce_avg_allreduce_patch, ray_device_pin_patch, register_vtune, numa_affinity_patch)


def _ensure_torch_xpu() -> bool:
    """Check if torch.xpu is available.

    Intel XPU support requires intel-extension-for-pytorch (IPEX).
    Importing IPEX registers the XPU backend with PyTorch, making
    torch.xpu.* APIs available.

    Returns:
        True if torch.xpu is available after attempting to import IPEX.

    Note:
        This pattern is common for vendor torch extensions:
        - Intel: import intel_extension_for_pytorch → torch.xpu
        - Huawei: import torch_npu → torch.npu
        - Cambricon: import torch_mlu → torch.mlu
    """
    if hasattr(torch, "xpu"):
        return True
    try:
        import intel_extension_for_pytorch  # noqa: F401

        return hasattr(torch, "xpu")
    except ImportError:
        return False


@PlatformRegistry.register(platform="intel")
class PlatformXPU(PlatformBase):
    """Platform backend for Intel XPU (Data Center GPU Max, Arc, etc.).

    Registration key: "intel"
    Engines for this platform should register with: device="xpu", vendor="intel"

    Note on xccl:
        xccl (oneAPI Collective Communications Library) does NOT support
        ReduceOp.AVG. The corresponding FSDP engine must enable
        force_sum_reduction to work around this limitation.
    """

    def __init__(self) -> None:
        """Apply this platform's monkeypatches (see ``_PATCHES`` above).

        Applied here rather than from the shared ``verl_hardware_plugin/__init__.py``
        so a patch only installs on hosts where XPU is actually in play, not on
        every host regardless of which platform verl selects -- that shared
        init runs once for every vendor. ``PlatformXPU()`` is also constructed
        as a losing detection probe on hosts where a different platform wins,
        so this constructor running does not mean XPU was selected: each
        patch's own ``apply()`` must gate its correctness at call time, not
        here. See each patch module's own docstring for how it does that.
        """
        for patch_module in _PATCHES:
            patch_module.apply()

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "xpu"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_xpu():
            raise RuntimeError("torch.xpu is not available. Install intel-extension-for-pytorch.")
        return torch.xpu

    def is_available(self) -> bool:
        if not _ensure_torch_xpu():
            return False
        return torch.xpu.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        if not _ensure_torch_xpu():
            return False
        return torch.xpu.is_available()

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.xpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.xpu.synchronize(device_index)
        else:
            torch.xpu.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # Intel XPU does not expose a configurable memory allocator interface
        # through torch.xpu. This is a no-op for now.
        pass

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # Intel XPU does not use the CUDA compute capability model.
        # Return (None, None) to indicate this concept is not applicable.
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        # xccl = oneAPI Collective Communications Library
        # Important: xccl does NOT support ReduceOp.AVG — engines must use
        # sum-based gradient reduction. See fsdp_xpu.py for the workaround.
        return "xccl"

    def visible_devices_envvar(self) -> str:
        # Intel Level Zero driver uses ZE_AFFINITY_MASK for device visibility
        # (analogous to CUDA_VISIBLE_DEVICES for NVIDIA)
        return "ZE_AFFINITY_MASK"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # Reuse Ray's built-in "GPU" resource. This simplifies cluster setup
        # since Ray auto-detects GPUs. If your hardware needs a custom resource,
        # return a custom name (e.g. "NPU", "MLU") and ensure Ray workers
        # advertise that resource.
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # Since we use the built-in "GPU" resource, use the standard num_gpus key
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        # Tell Ray NOT to auto-set ZE_AFFINITY_MASK — verl manages device
        # assignment manually to support multi-GPU-per-actor configurations
        return ["RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # Intel XPU does not support CUDA-style IPC tensor sharing.
        # When IPC is not supported, verl falls back to serialized tensor
        # transfer between processes (slower but universally compatible).
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # Intel XPU does not have an NVTX equivalent.
        # No-op: just yield immediately. If Intel adds profiling markers
        # in the future, implement them here.
        yield

    def profiler_start(self) -> None:
        # No-op: XPU profiling is handled externally via Intel VTune/Advisor
        pass

    def profiler_stop(self) -> None:
        # No-op: see profiler_start
        pass

    def profiler_markers(self):
        # Intel VTune (ITT) tracing markers. verl core asks the platform for
        # these first, before falling back to nvtx/mstx/generic markers -- see
        # verl/utils/profiler/__init__.py for how it discovers them.
        from verl_hardware_plugin.accelerators.xpu.profilers import itt_profile_xpu

        return (
            itt_profile_xpu.mark_start_range,
            itt_profile_xpu.mark_end_range,
            itt_profile_xpu.mark_annotate,
            itt_profile_xpu.marked_timer,
        )

    # ------------------------------------------------------------------
    # Model patches
    # ------------------------------------------------------------------

    def apply_model_patches(self, model_type: str) -> None:
        # No model patches needed for Intel XPU — standard PyTorch ops work.
        # If vendor-optimized ops are needed (e.g. fused RMSNorm), implement
        # monkey-patching logic here keyed on model_type.
        pass

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        # No special environment variables needed for vLLM rollout on Intel XPU.
        # If the vendor's vLLM backend needs env vars, add them here.
        return {}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        # No cupy-style raw comm-id primitive exists for XCCL; this shim
        # bootstraps a hybrid cpu:gloo,xpu:xccl torch.distributed process
        # group instead. See xccl_collective_xpu.py for why
        # get_unique_id()/NcclCommunicator() carry (host, port) rather than
        # a NCCL unique-id blob.
        from verl_hardware_plugin.accelerators.xpu import xccl_collective_xpu

        return xccl_collective_xpu

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        # Not applicable for Intel XPU (CUDA runtime is NVIDIA-specific).
        return None
