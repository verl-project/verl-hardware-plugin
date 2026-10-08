# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch torch.distributed.all_reduce(op=AVG) -> SUM + manual divide,
scoped to xccl process groups, for Intel XPU.

Why this is needed: Intel's oneCCL (xccl) implements ReduceOp.AVG on its
SYCL-kernel execution path, but not on its scheduler path. Which path a
given collective takes is selected internally by oneCCL and isn't
capability-aware, so a caller can't tell in advance whether an AVG request
will succeed or abort mid-collective (`average operation is not supported
for the scheduler path`). Forcing sum-based reduction + a manual divide is
the only way to guarantee correctness until that dispatch becomes
capability-aware -- fix tracked for oneCCL 2022.2 / torch 2.15, not yet
available on the stack this plugin targets. (A separate, narrower
double-division bug on very small messages, intel/torch-xpu-ops#3020, is
fixed on that same stack and is not what this patch works around.)

This is the one patch in this package worth calling "doable, but with a real
design smell" rather than "clean":

- verl/workers/engine_workers.py, verl/trainer/sft_trainer.py, and
  verl/utils/profiler/performance.py::reduce_timing each call
  `torch.distributed.all_reduce(x, op=ReduceOp.AVG, ...)` as a literal inline
  expression -- there is no per-site wrapper or dispatch object to patch, the
  way there is for attention padding.
- Patching `torch.distributed.all_reduce` itself is robust to import order
  (every call site does `dist.all_reduce(...)`, an attribute lookup on the
  `torch.distributed` module performed at call time, not at each file's own
  import time), so this does take effect regardless of load order.
- The cost: this still intercepts a public torch API process-wide -- once
  applied, every `dist.all_reduce(op=AVG)` call in the process is routed
  through this wrapper, XPU-related or not. It is scoped to xccl groups via
  `dist.get_backend(group) != "xccl"`, so a gloo/nccl group in the same
  process (e.g. a CPU-only Ray actor's coordination group alongside a GPU
  worker's xccl group) is untouched and keeps native `AVG` -- but anyone
  reading verl-core's source still can't see, from that source alone, that
  an xccl group's `ReduceOp.AVG` silently becomes SUM+divide. That is a
  legitimate argument for fixing this in verl-core instead
  (verl-project/verl#7917's `is_reduce_avg_supported()` hook covers exactly
  these 3 call sites) -- this patch exists to show it is *possible* without
  core changes, not to claim it is the better design.

async_op=True is intentionally unsupported: correctly dividing the result
requires the collective to have already completed, which async_op explicitly
defers. Call sites using async_op=True with op=AVG fall through to the
original behavior -- subject to the same oneCCL path-selection risk this
patch exists to avoid -- rather than silently producing a wrong answer with
no error.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def _xpu_available() -> bool:
    import torch

    return hasattr(torch, "xpu") and torch.xpu.is_available()


def apply() -> None:
    global _applied
    if _applied:
        return
    if not _xpu_available():
        return

    import torch.distributed as dist

    original_all_reduce = dist.all_reduce

    def _patched_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if op != dist.ReduceOp.AVG or async_op or dist.get_backend(group) != "xccl":
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

        world_size = dist.get_world_size(group=group)
        result = original_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
        tensor.div_(world_size)
        return result

    dist.all_reduce = _patched_all_reduce
    _applied = True
    logger.info("[verl_hardware_plugin] Patched torch.distributed.all_reduce(op=AVG) for xccl process groups")
