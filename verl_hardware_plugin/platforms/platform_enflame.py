# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Enflame GCU platform implementation.

Supports Enflame GCU accelerators via torch_gcu and ECCL/FlagCX communication
backends.

Key design decisions for Enflame GCU:
- device_name: "gcu" (torch_gcu PyTorch API via torch.gcu.*)
- vendor_name: "enflame" (engine lookup vendor key)
- communication_backend: "flagcx" when USE_FLAGCX=1, otherwise "eccl"
- ray_resource_name: "GPU" (Ray maps ENFLAME workers to the built-in GPU resource)
- visible_devices_envvar: "TOPS_VISIBLE_DEVICES"
- is_ipc_supported: True (verl uses device-tensor/reduce_tensor path; Python SHM is unsupported on torch_gcu)
- ensure_initialized: loads torch_gcu and applies gcu runtime shims (ipc_collect, Stream)

Prerequisites:
- torch_gcu must be installed (provides torch.gcu.* API)
- Enflame driver and runtime must be installed on the host

Example usage:
    export VERL_PLATFORM=enflame
    export USE_FLAGCX=0
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


def _ensure_torch_gcu() -> bool:
    """Try to import torch_gcu so that torch.gcu becomes available."""
    if hasattr(torch, "gcu"):
        return True
    try:
        import torch_gcu  # noqa: F401

        return hasattr(torch, "gcu")
    except (ImportError, RuntimeError, AttributeError):
        return False


_gcu_runtime_patched = False


def _patch_gcu_runtime(gcu: ModuleType) -> None:
    """Apply torch.gcu compatibility shims required by verl/vLLM weight transfer."""
    global _gcu_runtime_patched
    if _gcu_runtime_patched:
        return
    _gcu_runtime_patched = True

    # torch_gcu stubs cuda.ipc_collect but not gcu.ipc_collect; verl cleanup calls it.
    if not hasattr(gcu, "ipc_collect"):
        gcu.ipc_collect = lambda: None

    stream_cls = getattr(gcu, "Stream", None)
    if stream_cls is not None and not hasattr(stream_cls, "cuda_stream"):
        if hasattr(stream_cls, "gcu_stream"):
            stream_cls.cuda_stream = property(lambda self: self.gcu_stream)


def _get_gcu_module() -> ModuleType:
    """Return the ``torch.gcu`` module, importing ``torch_gcu`` if needed."""
    if not _ensure_torch_gcu():
        raise RuntimeError("Enflame platform requires the 'torch_gcu' package. Please install it first.")
    gcu = torch.gcu
    _patch_gcu_runtime(gcu)
    return gcu


def _gcu_vllm_runtime_env_vars() -> dict[str, str]:
    """Env vars required for stable vLLM rollout on torch_gcu (verl 0.7.1 e2e parity)."""
    return {
        "NCCL_CUMEM_ENABLE": "0",
        "TORCH_ECCL_AVOID_RECORD_STREAMS": "1",
        "TORCHGCU_INDUCTOR_ENABLE": "0",
        "TORCHDYNAMO_DISABLE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }


@PlatformRegistry.register(platform="enflame")
class PlatformENFLAME(PlatformBase):
    """Platform backend for Enflame GCU accelerators.

    Registration key: "enflame" (``VERL_PLATFORM``)
    Engines for this platform should register with: device="gcu", vendor="enflame"

    Note: torch_gcu may patch torch.cuda to return True on GCU devices, so
    platform auto-detection must probe torch.gcu before CUDA.
    """

    @property
    def device_name(self) -> str:
        return "gcu"

    @property
    def vendor_name(self) -> str:
        return "enflame"

    @property
    def device_module(self) -> ModuleType:
        return _get_gcu_module()

    def is_available(self) -> bool:
        if not _ensure_torch_gcu():
            return False
        try:
            return torch.gcu.is_available()
        except (ImportError, RuntimeError, AttributeError):
            return False

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        """Determine if the current machine has Enflame GCU hardware.

        Must be probed before CUDA because torch_gcu may patch torch.cuda.
        """
        if not _ensure_torch_gcu():
            return False
        try:
            gcu = getattr(torch, "gcu", None)
            is_available = getattr(gcu, "is_available", None)
            return callable(is_available) and is_available()
        except (ImportError, RuntimeError, AttributeError):
            return False

    def current_device(self) -> int:
        return _get_gcu_module().current_device()

    def device_count(self) -> int:
        return _get_gcu_module().device_count()

    def set_device(self, device_index: int) -> None:
        _get_gcu_module().set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            _get_gcu_module().synchronize(device_index)
        else:
            _get_gcu_module().synchronize()

    def manual_seed(self, seed: int) -> None:
        _get_gcu_module().manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        _get_gcu_module().manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        gcu = _get_gcu_module()
        if hasattr(gcu, "memory") and hasattr(gcu.memory, "_set_allocator_settings"):
            gcu.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        _get_gcu_module().empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        gcu = _get_gcu_module()
        if hasattr(gcu, "get_device_capability"):
            return gcu.get_device_capability(device_index)
        return (None, None)

    def communication_backend_name(self) -> str:
        if os.getenv("USE_FLAGCX", "").lower() in ("1", "true"):
            return "flagcx"
        return "eccl"

    def visible_devices_envvar(self) -> str:
        return "TOPS_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        """Tell verl to avoid Python multiprocessing SHM for weight transfer.

        verl sets ``use_shm = not is_support_ipc()`` in vLLM rollout. torch_gcu
        does not support the SHM fallback; return True so verl uses on-device
        buffers with ``reduce_tensor`` instead.
        """
        return True

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on ENFLAME): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def apply_model_patches(self, model_type: str) -> None:
        pass

    def rollout_env_vars(self) -> dict[str, str]:
        return dict(_gcu_vllm_runtime_env_vars())

    def get_collective_module(self) -> Any:
        return None

    def cudart(self) -> Any:
        return None

    def ensure_initialized(self) -> None:
        """Eagerly load ``torch_gcu`` and apply runtime compatibility shims."""
        _get_gcu_module()
        logger.debug("torch_gcu initialised by PlatformENFLAME.ensure_initialized()")
