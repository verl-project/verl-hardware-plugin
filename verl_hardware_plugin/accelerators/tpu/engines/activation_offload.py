# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Host offload of the activations saved for backward (verl's ``model.enable_activation_offload``, experimental).

``HostActivationOffload`` is a ``torch.autograd.graph.saved_tensors_hooks`` pair, like
``torch.autograd.graph.save_on_cpu(pin_memory=True)``: every tensor the model forward saves for backward is copied
into pinned host memory (D2H) and copied back to the device when the backward unpacks it (H2D). On top of
``save_on_cpu`` it adds what a training loop needs:

* **Positive selection.** Only device tensors of at least ``activation_offload_min_mb`` that do not alias a parameter
  are offloaded. The engine enters the hooks around the model forward only (``model_forward_step``), so what reaches
  them is what activation checkpointing decided to keep: the block inputs of full/selective AC, the ops selective AC
  saves (e.g. the Splash ``(out, lse)`` of ``ac_save_splash_and_routing``) and the saved tensors of ops outside the
  checkpointed blocks. The loss and log-prob computation is outside the hooks. Parameters are recognized by identity,
  view base and storage, re-read at every micro-batch (so parameter offload/onload and resharding are seen).
* **Selective-AC saves.** Eager selective AC (``torch.utils.checkpoint`` with a SAC ``context_fn``) keeps its saved
  tensors in its own cache, which bypasses outer hooks unless the checkpoint is called with
  ``respect_saved_tensors_hooks=True``. :func:`route_sac_saves_through_hooks` sets that on every
  ``CheckpointWrapper`` of the model. Under ``torch.compile`` the checkpointed block becomes one AOTAutograd function
  whose saved tensors (block input and the policy's saved ops) go through the outer hooks at run time anyway.
* **Dedup.** A tensor saved by several autograd nodes (or a view with the same storage, offset, shape, stride and
  version) gets one host copy; its device copy is refcounted on unpack and shared by every node that saved it.
* **No host stalls in forward.** The D2H copy is ``non_blocking`` into pinned memory; ``torch_tpu`` holds the source
  buffer until the transfer is done, so the device tensor can be dropped right away. An event recorded after the copy
  is waited for only before the host buffer is read back (in backward, by which point it has long completed).
* **Prefetch in backward.** Unpacking record ``i`` also starts the H2D copies of the ``activation_offload_prefetch``
  records saved just before it (backward visits blocks in reverse order), so their transfers overlap compute.
* **Bounded pinned pool.** Host buffers are pooled by byte size and recycled after each micro-batch's backward. Only
  the buffers the last micro-batch used are kept (so the pool never exceeds one micro-batch's offloaded bytes, however
  the shapes vary), optionally capped lower by ``activation_offload_host_cache_gb``; :meth:`release` (called on
  ``engine.to("cpu")``) frees them all. If a micro-batch raises, its buffers are dropped instead of pooled.

What is NOT validated yet: HBM savings, step time and transfer overlap on TPU, and whether ``torch_tpu``'s
``DEFER_AND_FUSE`` mode schedules the prefetched H2D copies early. Off by default.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import inspect
import logging
import os
import weakref
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import alloc_cpu_staging, record_tpu_event

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MiB = 1 << 20
_GiB = 1 << 30
DEFAULT_MIN_MB = 1.0
DEFAULT_PREFETCH = 1

_warned: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _storage_ptr(t: torch.Tensor) -> int | None:
    try:
        ptr = t.untyped_storage().data_ptr()
    except Exception:
        return None
    return ptr or None


class _PinnedPool:
    """Host buffers (flat ``uint8``, pinned when possible) pooled by exact byte size, at most ``cap_bytes`` free."""

    def __init__(self, cap_bytes: int | None):
        self.cap_bytes = cap_bytes
        self._free: dict[int, list[torch.Tensor]] = collections.defaultdict(list)
        self.free_bytes = 0
        self.live_bytes = 0
        self.peak_live_bytes = 0

    def take(self, nbytes: int) -> torch.Tensor:
        bufs = self._free.get(nbytes)
        if bufs:
            buf = bufs.pop()
            self.free_bytes -= nbytes
        else:
            buf = alloc_cpu_staging((nbytes,), torch.uint8)
        self.live_bytes += nbytes
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)
        return buf

    def give(self, buf: torch.Tensor, reuse: bool = True) -> None:
        nbytes = buf.numel()
        self.live_bytes -= nbytes
        if reuse and (self.cap_bytes is None or self.free_bytes + nbytes <= self.cap_bytes):
            self._free[nbytes].append(buf)
            self.free_bytes += nbytes

    def clear(self) -> None:
        self._free.clear()
        self.free_bytes = 0


class _Packed:
    """One offloaded tensor: its host copy and, while backward needs it, its device copy."""

    __slots__ = (
        "index",
        "host",
        "buf",
        "device",
        "spec",
        "event",
        "refs",
        "resident",
        "src",
        "root",
        "version",
        "skey",
    )

    def __init__(self, index, host, buf, device, spec, event, src, root, version, skey):
        self.index = index
        self.host = host  # [shape] dtype view of ``buf``
        self.buf = buf  # pooled flat uint8 buffer
        self.device = device
        self.spec = spec  # DTensorSpec when the saved tensor was a DTensor
        self.event = event  # D2H completion (TPU), None elsewhere
        self.refs = 1  # pack calls not yet matched by an unpack
        self.resident: torch.Tensor | None = None  # device copy shared by the unpacks
        self.src = src  # weakref to the saved tensor, for dedup by identity
        self.root = root  # weakref to its view base (owner of the storage), for dedup by storage
        self.version = version
        self.skey = skey


class HostActivationOffload:
    """Pinned-host offload of saved activations for one engine (see the module docstring).

    Args:
        params_fn: Returns the model parameters; re-read at every :meth:`hooks` entry.
        min_bytes: Smallest tensor offloaded (smaller copies are latency-bound and save little).
        host_cache_bytes: Most free pinned bytes kept pooled between micro-batches (default: what the last
            micro-batch used).
        prefetch: H2D copies started ahead of the backward's unpacks.
        device_types: Device types whose tensors are offloaded (``("tpu",)`` in the engine).
    """

    def __init__(
        self,
        params_fn: Callable[[], Iterable[torch.Tensor]],
        *,
        min_bytes: int = int(DEFAULT_MIN_MB * _MiB),
        host_cache_bytes: int | None = None,
        prefetch: int = DEFAULT_PREFETCH,
        device_types: Iterable[str] = ("tpu",),
    ):
        self._params_fn = params_fn
        self.min_bytes = int(min_bytes)
        self.prefetch = max(0, int(prefetch))
        self.device_types = frozenset(device_types)
        self._pool = _PinnedPool(None if host_cache_bytes is None else int(host_cache_bytes))
        self._param_ids: set[int] = set()
        self._param_ptrs: set[int] = set()
        self._records: list[_Packed] = []
        self._by_id: dict[int, _Packed] = {}
        self._by_storage: dict[tuple, _Packed] = {}
        self.packed_calls = 0
        self.dedup_hits = 0
        self.micro_batch_bytes = 0
        self.total_bytes = 0
        self._micro_batches = 0

    # ----- parameters -----------------------------------------------------------------------------------------

    def _refresh_params(self) -> None:
        self._param_ids.clear()
        self._param_ptrs.clear()
        for p in self._params_fn():
            for t in (p, p._local_tensor if isinstance(p, DTensor) else None):
                if t is None:
                    continue
                self._param_ids.add(id(t))
                ptr = _storage_ptr(t)
                if ptr is not None:
                    self._param_ptrs.add(ptr)

    def _is_param(self, t: torch.Tensor) -> bool:
        base = t
        while base is not None:
            if isinstance(base, torch.nn.Parameter) or id(base) in self._param_ids:
                return True
            base = base._base
        ptr = _storage_ptr(t)
        return ptr is not None and ptr in self._param_ptrs

    # ----- pack / unpack --------------------------------------------------------------------------------------

    def _lookup(self, t: torch.Tensor, skey: tuple | None) -> _Packed | None:
        rec = self._by_id.get(id(t))
        if rec is not None and rec.src() is t and rec.version == t._version:
            return rec
        if skey is not None:
            rec = self._by_storage.get(skey)
            if rec is not None and rec.root() is not None and rec.version == t._version:
                return rec
        return None

    def pack(self, t: torch.Tensor) -> Any:
        if not isinstance(t, torch.Tensor):
            return t
        self.packed_calls += 1
        if isinstance(t, DTensor):
            local, spec = t._local_tensor, t._spec
        else:
            local, spec = t, None
        if local.device.type not in self.device_types or local.layout != torch.strided:
            return t
        nbytes = local.numel() * local.element_size()
        if nbytes < self.min_bytes or self._is_param(local):
            return t
        ptr = _storage_ptr(local)
        skey = (
            (ptr, local.storage_offset(), tuple(local.shape), local.stride(), local.dtype, spec is None)
            if ptr is not None
            else None
        )
        rec = self._lookup(t, skey)
        if rec is not None:
            rec.refs += 1
            self.dedup_hits += 1
            return rec
        # A direct (async) D2H needs a contiguous source that starts at its storage and spans all of it.
        src = local
        if not src.is_contiguous() or src.storage_offset() != 0 or src.untyped_storage().nbytes() != nbytes:
            src = src.clone(memory_format=torch.contiguous_format)
        buf = self._pool.take(nbytes)
        host = buf.view(local.dtype).view(local.shape)
        host.copy_(src, non_blocking=True)
        event = record_tpu_event(local.device.type == "tpu")
        root = t._base if t._base is not None else t
        rec = _Packed(len(self._records), host, buf, local.device, spec, event, _weak(t), _weak(root), t._version, skey)
        self._records.append(rec)
        self._by_id[id(t)] = rec
        if skey is not None:
            self._by_storage[skey] = rec
        self.micro_batch_bytes += nbytes
        return rec

    def _h2d(self, rec: _Packed) -> torch.Tensor:
        if rec.resident is None:
            if rec.event is not None:
                rec.event.synchronize()  # the D2H was issued in forward; normally long complete
                rec.event = None
            local = rec.host.to(rec.device, non_blocking=True)
            rec.resident = local if rec.spec is None else DTensor(local, rec.spec, requires_grad=False)
        return rec.resident

    def unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed
        out = self._h2d(packed)
        packed.refs -= 1
        if packed.refs <= 0:
            packed.resident = None  # autograd now owns the device copy; a re-unpack copies from host again
        for j in range(packed.index - 1, max(-1, packed.index - 1 - self.prefetch), -1):
            rec = self._records[j]
            if rec.refs > 0:
                self._h2d(rec)
        return out

    # ----- lifecycle ------------------------------------------------------------------------------------------

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        """Offloads the tensors saved for backward while active (wrap the model forward of a training step)."""
        self._refresh_params()
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield

    def end_micro_batch(self, reuse_buffers: bool = True) -> None:
        """Returns the micro-batch's host buffers to the pool; call after its backward has been synchronized.

        Free buffers this micro-batch did not use are dropped first, so the pool tracks the current shapes. With
        ``reuse_buffers=False`` (the micro-batch raised, transfers may still be pending) its buffers are dropped too.
        """
        self._pool.clear()
        for rec in self._records:
            rec.resident = None
            self._pool.give(rec.buf, reuse=reuse_buffers)
        self._records.clear()
        self._by_id.clear()
        self._by_storage.clear()
        self.total_bytes += self.micro_batch_bytes
        self._micro_batches += 1
        if self._micro_batches == 1:
            if self.micro_batch_bytes:
                logger.warning("Activation offload: first micro-batch offloaded %s", self.stats())
            else:
                _warn_once(
                    "nothing",
                    "Activation offload is enabled but the first micro-batch offloaded nothing (%s); check "
                    "activation_offload_min_mb and that the forward saves device tensors.",
                    self.stats(),
                )
        else:
            logger.debug("Activation offload: %s", self.stats())
        self.micro_batch_bytes = 0
        self.packed_calls = 0
        self.dedup_hits = 0

    def release(self) -> None:
        """Frees every pooled pinned buffer (e.g. when the engine moves to CPU / sleeps)."""
        self._pool.clear()

    def stats(self) -> dict[str, float]:
        return {
            "micro_batch_gib": self.micro_batch_bytes / _GiB,
            "total_gib": self.total_bytes / _GiB,
            "saved_tensors": self.packed_calls,
            "dedup_hits": self.dedup_hits,
            "pool_free_gib": self._pool.free_bytes / _GiB,
            "pool_peak_live_gib": self._pool.peak_live_bytes / _GiB,
        }


def _dead_ref() -> None:
    return None


def _weak(t: torch.Tensor) -> Callable[[], torch.Tensor | None]:
    try:
        return weakref.ref(t)
    except TypeError:
        return _dead_ref


def route_sac_saves_through_hooks(modules: Iterable[torch.nn.Module]) -> int:
    """Makes eager selective AC pass its saved tensors through outer ``saved_tensors_hooks``.

    Sets ``respect_saved_tensors_hooks=True`` on the ``checkpoint`` call of every ``CheckpointWrapper`` (it only
    affects SAC regions; plain checkpoints recompute rather than save). Returns the number of wrappers updated, or 0
    with a warning when the installed torch has no such option (then SAC-saved tensors stay on the device in eager
    mode; compiled blocks are unaffected).
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    wrappers = [m for module in modules for m in module.modules() if isinstance(m, CheckpointWrapper)]
    if not wrappers:
        return 0
    if "respect_saved_tensors_hooks" not in inspect.signature(torch.utils.checkpoint.checkpoint).parameters:
        _warn_once(
            "sac",
            "torch.utils.checkpoint has no respect_saved_tensors_hooks (torch %s): tensors that eager selective "
            "activation checkpointing saves are not offloaded.",
            torch.__version__,
        )
        return 0
    updated = 0
    for m in wrappers:
        fn = m.checkpoint_fn
        if isinstance(fn, functools.partial) and "respect_saved_tensors_hooks" in fn.keywords:
            continue
        m.checkpoint_fn = functools.partial(fn, respect_saved_tensors_hooks=True)
        updated += 1
    return updated
