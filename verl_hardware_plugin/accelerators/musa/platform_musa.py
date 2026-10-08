# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Moore Threads MUSA platform implementation."""

import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_musa() -> bool:
    if hasattr(torch, "musa"):
        return True
    try:
        import torch_musa  # noqa: F401

        return hasattr(torch, "musa")
    except Exception as exc:
        logger.debug("torch.musa is unavailable: %s", exc)
        return False


@PlatformRegistry.register(platform="musa")
class PlatformMUSA(PlatformBase):
    """Platform backend for Moore Threads accelerators."""

    @property
    def device_name(self) -> str:
        return "musa"

    @property
    def vendor_name(self) -> str:
        return "moore_threads"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_musa():
            raise RuntimeError("torch_musa is not installed or torch.musa is unavailable")
        return torch.musa

    def is_available(self) -> bool:
        return _ensure_torch_musa() and torch.musa.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        return _ensure_torch_musa() and torch.musa.is_available()

    def current_device(self) -> int:
        return torch.musa.current_device()

    def device_count(self) -> int:
        return torch.musa.device_count()

    def set_device(self, device_index: int) -> None:
        torch.musa.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is None:
            torch.musa.synchronize()
        else:
            torch.musa.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        torch.musa.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.musa.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.musa.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning("torch.musa does not support _set_allocator_settings")

    def empty_cache(self) -> None:
        torch.musa.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if not self.is_available() or not hasattr(torch.musa, "get_device_capability"):
            return None, None
        result = torch.musa.get_device_capability(device_index)
        return result if result is not None else (None, None)

    def communication_backend_name(self) -> str:
        return "mccl"

    def visible_devices_envvar(self) -> str:
        return "MUSA_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return True

    @contextmanager
    def nvtx_range(self, msg: str):
        nvtx = getattr(torch.musa, "nvtx", None)
        range_fn = getattr(nvtx, "range", None)
        if range_fn is None:
            yield
        else:
            with range_fn(msg):
                yield

    def profiler_start(self) -> None:
        start = getattr(getattr(torch.musa, "profiler", None), "start", None)
        if start is not None:
            start()

    def profiler_stop(self) -> None:
        stop = getattr(getattr(torch.musa, "profiler", None), "stop", None)
        if stop is not None:
            stop()

    def cudart(self) -> Any:
        return None
