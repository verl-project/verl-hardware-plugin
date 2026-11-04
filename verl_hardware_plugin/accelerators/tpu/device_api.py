# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU device queries via ``torch.accelerator`` (``torch_tpu`` >= 0.2.0).

``torch_tpu`` 0.2.0 makes the CUDA-style device functions on ``torch.tpu`` (``is_available``,
``device_count``, ``current_device``, ``set_device``, ``synchronize``, ``current_stream``,
``set_stream``, ``device``, ``get_rng_state``, ``initial_seed``) internal on PyTorch >= 2.11: called
from outside ``torch`` / ``torch_tpu`` they raise ``RuntimeError: ... is internal to TorchTPU``. The
public replacements are the device-generic ``torch.accelerator`` APIs.

On hosts without TPU devices (such as a CPU-only Ray head node), ``torch_tpu`` skips device backend
registration so ``torch.accelerator.current_accelerator()`` is ``None``; device queries return
``False``, ``0``, or no-op directly instead of falling back to internal ``torch.tpu`` functions.

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


def is_available() -> bool:
    return accelerator_is_tpu() and bool(torch.accelerator.is_available())


def device_count() -> int:
    return torch.accelerator.device_count() if accelerator_is_tpu() else 0


def current_device() -> int:
    return torch.accelerator.current_device_index() if accelerator_is_tpu() else 0


def synchronize(device: Any = None) -> None:
    if accelerator_is_tpu():
        torch.accelerator.synchronize(device)


def current_stream(device: Any = None):
    return torch.accelerator.current_stream(device)


def set_stream(stream: Any) -> None:
    torch.accelerator.set_stream(stream)


def device(device_index: Any):
    """Context manager that makes ``device_index`` the current device (``torch.cuda.device`` style)."""
    return torch.accelerator.device_index(device_index)


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
    return _default_generator(device).get_state()


def initial_seed() -> int:
    return _default_generator().initial_seed()


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
