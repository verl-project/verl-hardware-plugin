# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Python garbage collection around the in-process engine.

The engine's decode steps run on the trainer's thread, inside the trainer's large heap. Three rules keep cyclic GC
pauses out of the decode steps: the heap the build leaves is collected and frozen once (as vLLM's EngineCore freezes its
own at construction), and again after the first training update; while the stepping pump runs, CPython's automatic
generation-2 collections are deferred and dynamo's post-compile ``gc.collect(1)`` is off for compiles inside the
pump's own task; a rank whose RSS grew by more than ``VERL_TPU_INPROC_GC_RSS_GIB`` since its last freeze collects and
freezes again at an aligned sync (no collective: every rank decides alone, at the same syncs).
"""

from __future__ import annotations

import contextlib
import gc
import os
import time
from typing import Any, Optional

GC_DEFER_THRESHOLD2 = 1_000_000  # gc threshold 2 while the pump steps
GC_RSS_EVERY = 10  # the RSS-growth freeze runs only at syncs with step % 10 == 0


def _dynamo_config() -> Any:
    try:
        import torch._dynamo.config as dc

        return dc
    except Exception:  # noqa: BLE001
        return None


def rss_bytes() -> Optional[int]:
    """This process's resident set size in bytes (/proc/self/statm); None where unavailable."""
    try:
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return None


def gib(b: Optional[int]) -> Optional[float]:
    return None if b is None else round(b / 2**30, 3)


class GcDefer:
    """While the pump steps: CPython's automatic generation-2 collections deferred (threshold 2 raised, the caller's
    value put back on exit unless someone changed it meanwhile), and ``torch._dynamo.config.run_gc_after_compile``
    off in the pump task's own context (a ContextVar in torch: no other task, and not the trainer, sees it)."""

    def __init__(self):
        self.saved2: Optional[int] = None
        self.pump_active = False
        self._dyn_prev: Optional[bool] = None

    def enter(self) -> None:
        self.pump_active = True
        if self._dyn_prev is None:
            dc = _dynamo_config()
            if dc is not None:
                try:
                    prev = bool(dc.run_gc_after_compile)
                    dc.run_gc_after_compile = False
                    self._dyn_prev = prev
                except Exception:  # noqa: BLE001
                    self._dyn_prev = None
        if self.saved2 is not None:
            return
        t0, t1, t2 = gc.get_threshold()
        self.saved2 = t2
        if t0 and t2 < GC_DEFER_THRESHOLD2:  # threshold 0 == 0: automatic collection is off already
            gc.set_threshold(t0, t1, GC_DEFER_THRESHOLD2)

    def exit(self) -> None:
        self.pump_active = False
        if self._dyn_prev is not None:
            prev, self._dyn_prev = self._dyn_prev, None
            dc = _dynamo_config()
            if dc is not None:
                with contextlib.suppress(Exception):
                    dc.run_gc_after_compile = prev
        if self.saved2 is None:
            return
        try:
            t0, t1, t2 = gc.get_threshold()
            if t2 == GC_DEFER_THRESHOLD2:
                gc.set_threshold(t0, t1, self.saved2)
        finally:
            self.saved2 = None


GC_DEFER = GcDefer()


def freeze_heap() -> tuple[str, float]:
    """vLLM's ``freeze_gc_heap`` (collect, then freeze the survivors), or the same three collections + freeze without
    it. Returns (how, seconds)."""
    t = time.perf_counter()
    try:
        from vllm.utils.gc_utils import freeze_gc_heap

        freeze_gc_heap()
        how = "vllm.utils.gc_utils.freeze_gc_heap"
    except Exception:  # noqa: BLE001
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        gc.freeze()
        how = "gc.collect(0..2) + gc.freeze"
    return how, time.perf_counter() - t


def rss_freeze_due(step: int, base: Optional[int], limit: int) -> tuple[bool, Optional[int], Optional[str]]:
    """Whether the RSS-growth freeze runs at this sync: (due, the RSS now, why). Only at aligned syncs; the first
    reading after a freeze only sets the baseline."""
    if not limit or step < GC_RSS_EVERY or step % GC_RSS_EVERY:
        return False, None, None
    now = rss_bytes()
    if now is None or base is None:
        return False, now, None
    grew = now - base
    if grew <= limit:
        return False, now, None
    return True, now, f"rss grew {grew / 2**30:.3f} GiB since the last freeze (> {limit / 2**30:.3f})"
