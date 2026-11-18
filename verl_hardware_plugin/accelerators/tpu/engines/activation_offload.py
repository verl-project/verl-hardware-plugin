# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Host offload of the activations saved for backward (``VERL_TPU_ACTIVATION_OFFLOAD=1``).

With full (or splash+routing selective) activation checkpointing, what each compiled transformer block keeps for its
backward is essentially its input (``[T, D]`` bf16, 128 MiB per block at 16K tokens/rank for D=4096) plus the saved
Splash ``(out, lse)`` of the full-attention layers: ~7.5 GiB + ~3.8 GiB per micro-batch on Qwen3.5-397B at 16K tokens,
and twice that at 32K. These tensors are idle on the device from the block's forward to its backward. This is the
eager-engine counterpart of MaxText's ``decoder_layer_input=offload`` and of TorchTitan graph-trainer's
``tpu_host_offload`` pass (scan only).

A ``torch.autograd.graph.saved_tensors_hooks`` pair around the micro-batch forward copies every saved device tensor
of at least ``VERL_TPU_ACT_OFFLOAD_MIN_MB`` (default 16) and at most ``VERL_TPU_ACT_OFFLOAD_MAX_MB`` (default 1024)
MiB into a pinned host buffer (async D2H), and copies it back when the backward unpacks it (H2D). Parameters (and
tensors sharing their storage when the backend exposes it) and non-contiguous tensors are never offloaded. At most
``VERL_TPU_ACT_OFFLOAD_INFLIGHT`` (default 2) D2H copies are left in flight before the oldest is waited for, which
bounds how long a device tensor outlives its offload. Pinned buffers are pooled per (shape, dtype) and recycled by
:meth:`HostActivationOffload.end_micro_batch` once the micro-batch's backward has been synchronized.

PCIe cost per micro-batch is ~2x the offloaded bytes (~23 GiB at 16K tokens), small next to a ~25 s micro-batch.
Off by default; experimental (not yet validated on TPU).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

ACTIVATION_OFFLOAD_ENV = "VERL_TPU_ACTIVATION_OFFLOAD"
_MIN_MB_ENV = "VERL_TPU_ACT_OFFLOAD_MIN_MB"
_MAX_MB_ENV = "VERL_TPU_ACT_OFFLOAD_MAX_MB"
_INFLIGHT_ENV = "VERL_TPU_ACT_OFFLOAD_INFLIGHT"
_MiB = 1 << 20


def activation_offload_enabled() -> bool:
    return os.environ.get(ACTIVATION_OFFLOAD_ENV, "0") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class _Offloaded:
    __slots__ = ("buf", "device", "event", "key")

    def __init__(self, buf: torch.Tensor, device: torch.device, event: Any, key: tuple):
        self.buf, self.device, self.event, self.key = buf, device, event, key


def _record_event(device: torch.device) -> Any:
    if device.type != "tpu":
        return None
    event_cls = getattr(getattr(torch, "tpu", None), "Event", None)
    if event_cls is None:
        return None
    event = event_cls()
    event.record()
    return event


def _can_pin_memory() -> bool:
    """Whether ``torch.empty(..., pin_memory=True)`` has an active accelerator host allocator."""
    accel = getattr(torch, "accelerator", None)
    try:
        return bool(accel is not None and accel.is_available())
    except Exception:
        return False


def _storage_ptr(t: torch.Tensor) -> int | None:
    try:
        ptr = t.untyped_storage().data_ptr()
    except Exception:
        return None
    return ptr or None


class HostActivationOffload:
    """Pinned-host offload of saved activations for one engine (see the module docstring)."""

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        min_bytes: int | None = None,
        max_bytes: int | None = None,
        max_inflight: int | None = None,
        pin_memory: bool | None = None,
    ):
        self.min_bytes = int(min_bytes if min_bytes is not None else _env_float(_MIN_MB_ENV, 16) * _MiB)
        self.max_bytes = int(max_bytes if max_bytes is not None else _env_float(_MAX_MB_ENV, 1024) * _MiB)
        self.max_inflight = max(0, int(max_inflight if max_inflight is not None else _env_float(_INFLIGHT_ENV, 2)))
        self.pin_memory = _can_pin_memory() if pin_memory is None else pin_memory
        self._param_ids: set[int] = set()
        self._param_ptrs: set[int] = set()
        for p in params:
            for t in (p, p._local_tensor if isinstance(p, DTensor) else None):
                if t is None:
                    continue
                self._param_ids.add(id(t))
                ptr = _storage_ptr(t)
                if ptr is not None:
                    self._param_ptrs.add(ptr)
        self._free: dict[tuple, list[torch.Tensor]] = collections.defaultdict(list)
        self._used: list[tuple[tuple, torch.Tensor]] = []
        self._inflight: collections.deque[tuple[torch.Tensor, Any]] = collections.deque()
        self.offloaded_bytes = 0

    def _alloc(self, key: tuple, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        pool = self._free[key]
        if pool:
            return pool.pop()
        if self.pin_memory:
            try:
                return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
            except RuntimeError:
                self.pin_memory = False
        return torch.empty(shape, dtype=dtype, device="cpu")

    def _should_offload(self, t: torch.Tensor) -> bool:
        if not isinstance(t, torch.Tensor) or isinstance(t, DTensor) or t.device.type == "cpu":
            return False
        if isinstance(t, torch.nn.Parameter) or id(t) in self._param_ids:
            return False
        nbytes = t.numel() * t.element_size()
        if nbytes < self.min_bytes or nbytes > self.max_bytes or not t.is_contiguous():
            return False
        ptr = _storage_ptr(t)
        return ptr is None or ptr not in self._param_ptrs

    def pack(self, t: torch.Tensor) -> Any:
        if not self._should_offload(t):
            return t
        key = (tuple(t.shape), t.dtype)
        buf = self._alloc(key, t.shape, t.dtype)
        buf.copy_(t, non_blocking=t.device.type == "tpu")
        event = _record_event(t.device)
        # Keep the device tensor alive until its D2H copy is known to be done.
        self._inflight.append((t, event))
        while len(self._inflight) > self.max_inflight:
            _, old_event = self._inflight.popleft()
            if old_event is not None:
                old_event.synchronize()
        self._used.append((key, buf))
        self.offloaded_bytes += buf.numel() * buf.element_size()
        return _Offloaded(buf, t.device, event, key)

    def unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Offloaded):
            return packed
        if packed.event is not None:
            packed.event.synchronize()
        return packed.buf.to(packed.device, non_blocking=packed.device.type == "tpu")

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        """Offloads the tensors saved for backward while the context is active (wrap the micro-batch forward)."""
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield

    def end_micro_batch(self) -> None:
        """Recycles the pinned buffers; call only after the micro-batch's backward has been synchronized."""
        self._inflight.clear()
        for key, buf in self._used:
            self._free[key].append(buf)
        self._used.clear()

    def stats(self) -> dict[str, float]:
        pooled = sum(b.numel() * b.element_size() for bufs in self._free.values() for b in bufs)
        return {"offloaded_gib": self.offloaded_bytes / (1 << 30), "pooled_gib": pooled / (1 << 30)}
