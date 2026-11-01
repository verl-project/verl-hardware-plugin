# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU device queries that work with every ``torch_tpu`` release.

``torch_tpu`` 0.2.0 makes the CUDA-style device functions on ``torch.tpu`` (``is_available``,
``device_count``, ``current_device``, ``set_device``, ``synchronize``, ``current_stream``,
``set_stream``, ``device``, ``get_rng_state``, ``initial_seed``) internal on PyTorch >= 2.11: called
from outside ``torch`` / ``torch_tpu`` they raise ``RuntimeError: ... is internal to TorchTPU``. The
public replacements are the device-generic ``torch.accelerator`` APIs.

Each helper uses ``torch.accelerator`` when the current accelerator is the TPU, and otherwise falls
back to the ``torch.tpu`` function, which older ``torch_tpu`` builds still expose publicly.

The memory statistics (``memory_allocated`` and the others below) exist only in ``torch.accelerator``:
``torch_tpu`` registers its PJRT allocator as the TPU's ``c10::DeviceAllocator``, and ``torch.tpu`` has
no memory functions.
"""

import logging
import warnings
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_accelerator_is_tpu_cache: Optional[bool] = None


def accelerator_is_tpu() -> bool:
    """Whether ``torch.accelerator`` is backed by the TPU in this process (cached once True)."""
    global _accelerator_is_tpu_cache
    if _accelerator_is_tpu_cache:
        return True
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return False
    try:
        current = accelerator.current_accelerator()
    except Exception:
        return False
    is_tpu = current is not None and current.type == "tpu"
    if is_tpu:
        _accelerator_is_tpu_cache = True
    return is_tpu


def _legacy(name: str):
    """The pre-0.2.0 ``torch.tpu.<name>`` function, or ``None``."""
    return getattr(getattr(torch, "tpu", None), name, None)


def is_available() -> bool:
    if accelerator_is_tpu():
        return bool(torch.accelerator.is_available())
    fn = _legacy("is_available")
    if fn is None:
        return False
    try:
        # On torch_tpu >= 0.2.0 this raises (or, with TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS=1, warns) in
        # processes without a TPU, e.g. the Ray driver; the answer there is simply "no TPU".
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool(fn())
    except Exception as e:
        logger.debug("torch.tpu.is_available() failed: %s", e)
        return False


def device_count() -> int:
    if accelerator_is_tpu():
        return torch.accelerator.device_count()
    fn = _legacy("device_count")
    return fn() if fn is not None else 0


def current_device() -> int:
    if accelerator_is_tpu():
        return torch.accelerator.current_device_index()
    fn = _legacy("current_device")
    return fn() if fn is not None else 0


def synchronize(device: Any = None) -> None:
    if accelerator_is_tpu():
        torch.accelerator.synchronize(device)
        return
    fn = _legacy("synchronize")
    if fn is not None:
        fn()


def current_stream(device: Any = None):
    if accelerator_is_tpu():
        return torch.accelerator.current_stream(device)
    return torch.tpu.current_stream(device)


def set_stream(stream: Any) -> None:
    if accelerator_is_tpu():
        torch.accelerator.set_stream(stream)
    else:
        torch.tpu.set_stream(stream)


def device(device_index: Any):
    """Context manager that makes ``device_index`` the current device (``torch.cuda.device`` style)."""
    if accelerator_is_tpu():
        return torch.accelerator.device_index(device_index)
    return torch.tpu.device(device_index)


def _default_generator(device: Any = None) -> torch.Generator:
    if device is None or device == "tpu":
        index = current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = torch.device(device).index
        index = current_device() if index is None else index
    return torch.tpu.default_generators[index]


def get_rng_state(device: Any = None) -> torch.Tensor:
    # torch.accelerator has no RNG API on PyTorch 2.13; the per-device default generator is the
    # public equivalent of torch.tpu.get_rng_state().
    if accelerator_is_tpu():
        return _default_generator(device).get_state()
    return torch.tpu.get_rng_state(device) if device is not None else torch.tpu.get_rng_state()


def initial_seed() -> int:
    if accelerator_is_tpu():
        return _default_generator().initial_seed()
    return torch.tpu.initial_seed()


def _tpu_runtime_initialized() -> bool:
    """Whether this process has started its TPU runtime (the PJRT client)."""
    is_initialized = getattr(getattr(torch, "tpu", None), "is_initialized", None)
    if is_initialized is None:
        return False
    try:
        # Experimental in torch_tpu 0.2.0; its one-time warning is noise in every worker's log.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool(is_initialized())
    except Exception as e:
        logger.debug("torch.tpu.is_initialized() failed: %s", e)
        return False


def _memory_fn(name: str):
    """``torch.accelerator.<name>`` if this process's TPU runtime is up, else ``None`` (the caller reports zero).

    Reading the allocator statistics before the runtime is up would start it (``PjrtBackend::EnsureInitialized``),
    and in a multi-process slice that waits for every other process: a metrics query must not do that. Processes
    without a TPU, such as the Ray driver or the rollout server actor, report zero as well.
    """
    if not accelerator_is_tpu() or not _tpu_runtime_initialized():
        return None
    return getattr(torch.accelerator, name, None)


def memory_allocated(device: Any = None) -> int:
    fn = _memory_fn("memory_allocated")
    return int(fn(device)) if fn is not None else 0


def max_memory_allocated(device: Any = None) -> int:
    fn = _memory_fn("max_memory_allocated")
    return int(fn(device)) if fn is not None else 0


def memory_reserved(device: Any = None) -> int:
    # PJRT's bytes_reserved: unlike CUDA's caching allocator this is not a superset of memory_allocated().
    fn = _memory_fn("memory_reserved")
    return int(fn(device)) if fn is not None else 0


def max_memory_reserved(device: Any = None) -> int:
    fn = _memory_fn("max_memory_reserved")
    return int(fn(device)) if fn is not None else 0


def memory_stats(device: Any = None) -> dict[str, Any]:
    fn = _memory_fn("memory_stats")
    return fn(device) if fn is not None else {}


def reset_peak_memory_stats(device: Any = None) -> None:
    fn = _memory_fn("reset_peak_memory_stats")
    if fn is not None:
        fn(device)


def mem_get_info(device: Any = None) -> tuple[int, int]:
    """``(free, total)`` bytes of this process's TPU device: ``torch.accelerator.get_memory_info``."""
    fn = _memory_fn("get_memory_info")
    if fn is None:
        return 0, 0
    free, total = fn(device)
    return int(free), int(total)
