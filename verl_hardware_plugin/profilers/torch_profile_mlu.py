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

        for content in self.contents:
            if content == "mlu":
                continue
            assert content in ["cuda", "cpu", "memory", "shapes", "stack"], (
                f"Profiler contents only supports mlu, cuda, cpu, memory, shapes, stack, but gets {content}"
            )

        start = getattr(self, "profile_token_start", None)
        stop = getattr(self, "profile_token_end", None)
        for name, value in (("profile_token_start", start), ("profile_token_end", stop)):
            if value is not None:
                assert isinstance(value, int), f"{name} must be int or None, got {type(value)}"
                assert value >= 0, f"{name} must be >= 0, got {value}"
        if start is not None and stop is not None:
            assert stop > start, f"profile_token_end must be > profile_token_start, got start={start}, stop={stop}"

    TorchProfilerToolConfig.__post_init__ = _patched_post_init
    logger.info("[verl_hardware_plugin] Patched TorchProfilerToolConfig: +mlu")


def _patch_get_torch_profiler():
    """Replace get_torch_profiler with an MLU-aware version."""
    import verl.utils.profiler.torch_profile as tp

    global _original_get_torch_profiler
    if _original_get_torch_profiler is not None:
        return

    _original_get_torch_profiler = tp.get_torch_profiler

    def _mlu_get_torch_profiler(contents, save_path, role=None, save_file_prefix=None, rank=0, schedule=None):
        save_dir = os.path.join(save_path, role) if role else save_path
        os.makedirs(save_dir, exist_ok=True)

        if hasattr(tp, "build_trace_basename"):
            base_file_name = tp.build_trace_basename(rank=rank, role=role, save_file_prefix=save_file_prefix)
        else:
            ts = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d%H%M%S%f")[:-3]
            fname = f"prof_rank-{rank}_{os.getpid()}_{ts}"
            base_file_name = f"{save_file_prefix}_{fname}" if save_file_prefix else fname

        handler_state = {"count": 0}

        def _trace_handler(prof):
            idx = handler_state["count"]
            handler_state["count"] += 1
            suffix = "" if idx == 0 else f"_cycle{idx}"
            out_path = os.path.join(save_dir, f"{base_file_name}{suffix}.json.gz")
            logger.info("[Profiler] Saving trace to %s", out_path)
            prof.export_chrome_trace(out_path)

        _contents = set(contents) if contents else set()
        activities = []
        if not _contents or "cpu" in _contents:
            activities.append(torch.profiler.ProfilerActivity.CPU)

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

        return torch.profiler.profile(**profile_kwargs)

    tp.get_torch_profiler = _mlu_get_torch_profiler
    logger.info("[verl_hardware_plugin] Patched get_torch_profiler: +MLU activity")
