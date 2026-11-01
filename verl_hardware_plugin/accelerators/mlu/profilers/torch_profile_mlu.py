# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkey-patch verl's torch profiler to support Cambricon MLU.

Patched: ``TorchProfilerToolConfig`` accepts ``mlu`` and
``get_torch_profiler`` uses MLU activity when available.
"""

import functools
import logging
import os
from datetime import datetime, timezone

import torch

logger = logging.getLogger(__name__)

_original_get_torch_profiler = None
_original_post_init = None


def _patch_tool_config():
    """Add 'mlu' to allowed contents in TorchProfilerToolConfig."""
    from verl.utils.profiler.config import TorchProfilerToolConfig

    global _original_post_init
    if _original_post_init is not None:
        return

    _original_post_init = TorchProfilerToolConfig.__post_init__

    @functools.wraps(_original_post_init)
    def _patched_post_init(self):
        if not isinstance(self.contents, list):
            raise AssertionError(f"Profiler contents must be of type list, got {type(self.contents)}")

        # contents is a frozen-dataclass field -- direct assignment raises
        # FrozenInstanceError even from the class's own __post_init__. Delegate
        # to the saved original (with "mlu" filtered out) instead of
        # reimplementing its whitelist/token-range checks here, so any other
        # platform's contribution to the real __post_init__ (e.g. a plugin's
        # torch_profiler_content_name()) still takes effect instead of being
        # silently discarded.
        saved_contents = self.contents
        object.__setattr__(self, "contents", [c for c in saved_contents if c != "mlu"])
        try:
            _original_post_init(self)
        finally:
            object.__setattr__(self, "contents", saved_contents)

    TorchProfilerToolConfig.__post_init__ = _patched_post_init
    logger.info("[verl_hardware_plugin] Patched TorchProfilerToolConfig: +mlu")


def _patch_get_torch_profiler():
    """Replace get_torch_profiler with an MLU-aware version."""
    import verl.utils.profiler.torch_profile as tp

    global _original_get_torch_profiler
    if _original_get_torch_profiler is not None:
        return

    _original_get_torch_profiler = tp.get_torch_profiler

    def _mlu_get_torch_profiler(
        contents,
        save_path,
        role=None,
        save_file_prefix=None,
        rank=0,
        profile_step=None,
        schedule=None,
        name_mini_batch_window=False,
    ):
        if hasattr(tp, "build_trace_basename"):
            base_file_name = tp.build_trace_basename(
                rank=rank, role=role, save_file_prefix=save_file_prefix, profile_step=profile_step
            )
        else:
            ts = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d%H%M%S%f")[:-3]
            fname = f"prof_rank-{rank}_{os.getpid()}_{ts}"
            base_file_name = f"{save_file_prefix}_{fname}" if save_file_prefix else fname

        handler_state = {"count": 0}

        def _scheduled_mini_batch_range(step_num: int) -> tuple[int, int]:
            skip_first = int(schedule.get("skip_first", 0) or 0)
            wait = int(schedule.get("wait", 0) or 0)
            warmup = int(schedule.get("warmup", 0) or 0)
            active = max(int(schedule.get("active", 1) or 1), 1)

            last_mb = max(step_num - 1, 0)
            cycle_len = wait + warmup + active
            cycle = max(last_mb - skip_first, 0) // cycle_len if cycle_len else 0
            # Recording starts after this cycle's skipped, idle and warmup mini-batches.
            first_mb = skip_first + cycle * cycle_len + wait + warmup
            return min(first_mb, last_mb), last_mb

        def _trace_handler(prof):
            idx = handler_state["count"]
            handler_state["count"] += 1
            suffix = ""
            if schedule and name_mini_batch_window:
                step_num = getattr(prof, "step_num", None)
                if isinstance(step_num, int):
                    first_mb, last_mb = _scheduled_mini_batch_range(step_num)
                    suffix = f"_mb{first_mb}" if first_mb == last_mb else f"_mb{first_mb}-{last_mb}"
                else:
                    suffix = f"_part{idx}"
                # Always append callback counter to ensure unique filenames
                suffix = f"{suffix}_c{idx}"
            elif idx:
                suffix = f"_part{idx}"
            out_path = os.path.join(save_path, f"{base_file_name}{suffix}.json.gz")
            print(f"[Profiler] Saving trace to {out_path}")
            prof.export_chrome_trace(out_path)

        _contents = set(contents) if contents else set()
        activities = [torch.profiler.ProfilerActivity.CPU]

        if not _contents or "mlu" in _contents:
            if hasattr(torch.profiler.ProfilerActivity, "MLU"):
                activities.append(torch.profiler.ProfilerActivity.MLU)
            elif not _contents or "cuda" in _contents:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif "cuda" in _contents:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        profile_kwargs = dict(
            activities=activities,
            with_stack="stack" in _contents,
            record_shapes="shapes" in _contents,
            profile_memory="memory" in _contents,
            on_trace_ready=_trace_handler,
        )
        if schedule:
            profile_kwargs["schedule"] = torch.profiler.schedule(**schedule)

        prof = torch.profiler.profile(**profile_kwargs)
        if schedule:
            prof.record_steps = False

        return prof

    tp.get_torch_profiler = _mlu_get_torch_profiler
    logger.info("[verl_hardware_plugin] Patched get_torch_profiler: +MLU activity")
