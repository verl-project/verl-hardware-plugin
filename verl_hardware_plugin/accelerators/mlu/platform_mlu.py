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

import importlib.metadata
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


# ---------------------------------------------------------------------------
# MLU runtime dependency check
# ---------------------------------------------------------------------------
# Verify that apex / fla / megatron-core / megatron-bridge are installed and
# are MLU builds. Internal MLU wheels carry a PEP 440 '+mlu' local version
# segment (e.g. 0.1+mlu0.16.0); that is the single detection marker. Any
# install lacking '+mlu' (upstream PyPI, or an MLU source fork built without
# the tag) -> warning.
# WARNING: warnings only, never raise — verl swallows plugin-load exceptions
# at debug level (verl/verl/__init__.py:75-76).
_MLU_REQUIRED: dict[str, tuple[str, str]] = {
    "apex": ("apex", "required by Megatron FusedAdam/FastLayerNorm"),
    "fla": ("flash-linear-attention", "required only for qwen3.5 linear attention + conv CP"),
    "megatron-core": ("megatron-core", "required by the Megatron engine"),
    "megatron-bridge": (
        "megatron-bridge",
        "required when use_mbridge=True (default); vanilla_mbridge uses legacy mbridge",
    ),
}


def check_mlu_runtime_dependencies() -> list[str]:
    """Check MLU runtime dependencies.

    Returns a list of warning strings (empty = all pass). Detection only, never raises.
    Each package is checked independently so one failure does not skip the rest.
    A package passes iff its installed version carries the '+mlu' local segment;
    anything else (missing, upstream, or an untagged source build) -> warning.
    """
    warns: list[str] = []
    for name, (dist, note) in _MLU_REQUIRED.items():
        try:
            installed = importlib.metadata.version(dist)
            if "+mlu" in installed:
                logger.info("[env-check] %s MLU build OK (installed='%s')", name, installed)
                continue
            warns.append(
                f"[env-check] {name} not an MLU build (installed='{installed}', no '+mlu' local segment). "
                f"{note} requires the MLU build."
            )
        except importlib.metadata.PackageNotFoundError:
            warns.append(f"[env-check] {name} not installed (dist='{dist}'). {note}")
        except Exception as e:
            warns.append(f"[env-check] {name} check error (dist='{dist}'): {e!r}")
    return warns


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
        # For MLU devices, we use GPU as resource name
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # For MLU devices, we use num_gpus because Ray clusters are typically
        # configured with GPU resources even when using MLU hardware
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        # Prevent Ray from auto-setting MLU_VISIBLE_DEVICES — verl manages this
        return ["RAY_EXPERIMENTAL_NOSET_MLU_VISIBLE_DEVICES"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        return True

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


# ---------------------------------------------------------------------------
# One-shot MLU environment check at startup (MLU platform only, never blocks load)
# ---------------------------------------------------------------------------
def _run_mlu_env_check_once() -> None:
    if not _ensure_torch_mlu():
        return  # non-MLU environment, skip
    try:
        _warns = check_mlu_runtime_dependencies()
    except Exception as e:  # never block plugin load (verl/__init__.py:75-76 swallows at debug)
        logger.warning("[env-check] skipped due to error: %s", e)
        return
    if _warns:
        logger.warning("[env-check] MLU runtime dependency check found %d issue(s):", len(_warns))
        for _w in _warns:
            logger.warning(_w)
    else:
        logger.warning("[env-check] MLU runtime dependency check passed")


_run_mlu_env_check_once()
