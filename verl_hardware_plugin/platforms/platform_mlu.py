# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cambricon MLU platform implementation.

Supports Cambricon MLU (Machine Learning Unit) devices via torch_mlu
and CNCL (Cambricon NCCL) communication backend.
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
    """Try to import torch_mlu so that torch.mlu becomes available."""
    if hasattr(torch, "mlu"):
        return True
    try:
        import torch_mlu  # noqa: F401
        return hasattr(torch, "mlu")
    except ImportError:
        return False


@PlatformRegistry.register(platform="cambricon")
class PlatformMLU(PlatformBase):
    """Platform backend for Cambricon MLU."""

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

    def is_available(self, use_smi_check: bool = False) -> bool:
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

    def manual_seed(self, seed: int) -> None:
        torch.mlu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.mlu.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.mlu.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning("torch_mlu does not support _set_allocator_settings")

    def empty_cache(self) -> None:
        torch.mlu.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if hasattr(torch.mlu, "get_device_capability"):
            result = torch.mlu.get_device_capability(device_index)
            if result is None:
                return (None, None)
            return result
        return (None, None)

    def communication_backend_name(self) -> str:
        return "cncl"

    def visible_devices_envvar(self) -> str:
        return "MLU_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        # For MLU devices, we use GPU as resource name
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # For MLU devices, we use num_gpus because Ray clusters are typically
        # configured with GPU resources even when using MLU hardware
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_MLU_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return True

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None
