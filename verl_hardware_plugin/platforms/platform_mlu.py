# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cambricon MLU platform implementation.

Supports Cambricon MLU (Machine Learning Unit) devices via torch_mlu
and CNCL (Cambricon NCCL) communication backend.

Key design decisions for Cambricon MLU:
- device_name: "mlu" (Cambricon's torch extension uses torch.mlu.*)
- vendor_name: "cambricon" (used for engine lookup key)
- communication_backend: "cncl" (Cambricon's collective communication library)
- ray_resource_name: "MLU" (custom Ray resource — requires Ray workers to
  advertise this resource via --resources='{"MLU": N}')
- visible_devices_envvar: "MLU_VISIBLE_DEVICES" (Cambricon driver control)
- is_ipc_supported: False (not yet supported by Cambricon runtime)

Prerequisites:
- torch_mlu must be installed (provides torch.mlu.* API)
- Cambricon driver and runtime must be installed on the host

Example usage:
    export VERL_PLATFORM=cambricon
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

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _ensure_torch_mlu() -> bool:
    """Try to import torch_mlu so that torch.mlu becomes available.

    Cambricon's torch extension follows the same pattern as other vendor
    extensions (torch_npu for Huawei, intel_extension_for_pytorch for Intel):
    importing the package registers the device backend with PyTorch.

    Returns:
        True if torch.mlu is usable after the import attempt.
    """
    if hasattr(torch, "mlu"):
        return True
    try:
        import torch_mlu  # noqa: F401

        return hasattr(torch, "mlu")
    except ImportError:
        return False


@PlatformRegistry.register(platform="cambricon")
class PlatformMLU(PlatformBase):
    """Platform backend for Cambricon MLU.

    Registration key: "cambricon"
    Engines for this platform should register with: device="mlu", vendor="cambricon"

    Note on Ray resource:
        Unlike NVIDIA/Intel which use the built-in "GPU" resource, MLU uses a
        custom "MLU" resource. Ray workers must be started with:
            ray start --resources='{"MLU": 8}'
        This gives verl full control over device assignment without interfering
        with CUDA GPU scheduling.
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "mlu"

    @property
    def vendor_name(self) -> str:
        return "cambricon"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_mlu():
            raise RuntimeError("torch_mlu is not installed or torch.mlu is not available")
        return torch.mlu

    def is_available(self) -> bool:
        if not _ensure_torch_mlu():
            return False
        return torch.mlu.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        if not _ensure_torch_mlu():
            return False
        return torch.mlu.is_available()

    def current_device(self) -> int:
        return torch.mlu.current_device()

    def device_count(self) -> int:
        return torch.mlu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.mlu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.mlu.synchronize(device_index)
        else:
            torch.mlu.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.mlu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.mlu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.mlu.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning("torch_mlu does not support _set_allocator_settings")

    def empty_cache(self) -> None:
        torch.mlu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if hasattr(torch.mlu, "get_device_capability"):
            result = torch.mlu.get_device_capability(device_index)
            if result is None:
                return (None, None)
            return result
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        # CNCL = Cambricon NCCL — Cambricon's collective communication library
        # Compatible with torch.distributed process group initialization
        return "cncl"

    def visible_devices_envvar(self) -> str:
        # Cambricon driver uses MLU_VISIBLE_DEVICES to control device visibility
        # (analogous to CUDA_VISIBLE_DEVICES)
        return "MLU_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # Custom resource name — Ray workers must advertise "MLU" resources.
        # This avoids conflicting with CUDA "GPU" resources on heterogeneous clusters.
        return "MLU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # Custom resources use the "resources" dict (not the shorthand "num_gpus" key)
        return {"resources": {"MLU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        # Prevent Ray from auto-setting MLU_VISIBLE_DEVICES — verl manages this
        return ["RAY_EXPERIMENTAL_NOSET_MLU_VISIBLE_DEVICES"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Model patches
    # ------------------------------------------------------------------

    def apply_model_patches(self, model_type: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        return None

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
