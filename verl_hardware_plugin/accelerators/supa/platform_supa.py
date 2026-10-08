# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SUPA platform implementation.

SUPA is CUDA-compatible and exposed to verl as a SUPA-backed platform.
torch_supa maps torch.supa calls to the SUPA runtime and rewrites NCCL
process-group setup to BCCL, so this platform uses device_name "supa" while
vendor_name "biren" for registry lookup and hardware identification.
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


def _ensure_torch_supa() -> bool:
    """Import torch_supa when needed so torch.supa becomes available."""

    if hasattr(torch, "supa"):
        return True
    try:
        import torch_supa  # noqa: F401  # imported for side effect (makes torch.supa available)

        return hasattr(torch, "supa")
    except ImportError as exc:
        logger.debug("torch.supa is unavailable: %s", exc)
        return False


@PlatformRegistry.register(platform="biren")
class PlatformSupa(PlatformBase):
    """Platform backend for Biren SUPA devices.

    Engines for this platform should register with: device="supa", vendor="biren".
    SUPA-compatible base engines can also be used through verl's fallback path.
    """

    @property
    def device_name(self) -> str:
        return "supa"

    @property
    def vendor_name(self) -> str:
        return "biren"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_supa():
            raise RuntimeError("torch_supa is not installed or torch.supa is not available")
        return torch.supa

    def is_available(self) -> bool:
        if not _ensure_torch_supa():
            return False
        return torch.supa.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        if not _ensure_torch_supa():
            return False
        if use_smi_check:
            return self.check_smi_command("brsmi")
        return torch.supa.is_available()

    def current_device(self) -> int:
        return torch.supa.current_device()

    def device_count(self) -> int:
        return torch.supa.device_count()

    def set_device(self, device_index: int) -> None:
        torch.supa.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.supa.synchronize(device_index)
        else:
            torch.supa.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.supa.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.supa.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.supa.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning("Failed to set CUDA allocator settings on SUPA")

    def empty_cache(self) -> None:
        torch.supa.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        if not self.is_available() or not hasattr(torch.supa, "get_device_capability"):
            return None, None
        return torch.supa.get_device_capability(device_index)

    def communication_backend_name(self) -> str:
        # torch_supa maps NCCL process-group setup to BCCL under the hood.
        return "bccl"

    def visible_devices_envvar(self) -> str:
        return "SUPA_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return True

    @contextmanager
    def nvtx_range(self, msg: str):
        nvtx = getattr(torch.supa, "nvtx", None)
        range_fn = getattr(nvtx, "range", None)
        if range_fn is None:
            yield
        else:
            with range_fn(msg):
                yield

    def profiler_start(self) -> None:
        start = getattr(getattr(torch.supa, "profiler", None), "start", None)
        if start is not None:
            start()

    def profiler_stop(self) -> None:
        stop = getattr(getattr(torch.supa, "profiler", None), "stop", None)
        if stop is not None:
            stop()

    def cudart(self) -> Any:
        return None
