# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Let verl's torch profiler (``global_profiler.tool=torch``) record TPU traces for XProf.

Without this patch, verl's torch profiler records only CPU events on TPU. The patch, applied when the plugin is
imported, does two things on a TPU host:

* tells verl that the ``tpu`` device is profiled as ``ProfilerActivity.PrivateUse1``;
* passes torch_tpu's ``TpuProfilerConfig`` to the profiler, so each traced rank writes
  ``<save_path>/.../<hostname>_<rank>.xplane.pb``. A failed chrome-trace export only logs a warning.

Off TPU, verl's profiler is unchanged. Example (trainer only, step 4, rank 0)::

    global_profiler.tool=torch global_profiler.steps="[4]" global_profiler.save_path=/tmp/profiles/<run> \\
    actor_rollout_ref.actor.profiler.enable=True actor_rollout_ref.actor.profiler.ranks="[0]" \\
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=1
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TPU_DEVICE = "tpu"
_patched = False


def _tpu_profiler_config_cls() -> Optional[type]:
    """torch_tpu's ``TpuProfilerConfig`` (importing it also registers the PrivateUse1 Kineto backend), or None."""
    for mod_name in ("torch_tpu._internal.profiler", "torch.tpu.profiler"):
        try:
            mod = __import__(mod_name, fromlist=["TpuProfilerConfig"])
        except Exception:  # noqa: BLE001 - absent off TPU
            continue
        cls = getattr(mod, "TpuProfilerConfig", None)
        if cls is not None:
            return cls
    return None


def _on_tpu() -> bool:
    try:
        from verl.utils.device import get_device_name

        return get_device_name() == TPU_DEVICE
    except Exception:  # noqa: BLE001
        return False


def _register_device_activity() -> None:
    import verl.utils.profiler.config as profiler_config
    import verl.utils.profiler.torch_profile as tp

    if not (hasattr(profiler_config, "DEVICE_ACTIVITIES") and hasattr(profiler_config, "DEVICE_CONTENTS")):
        return
    if TPU_DEVICE not in profiler_config.DEVICE_ACTIVITIES:
        profiler_config.DEVICE_ACTIVITIES[TPU_DEVICE] = torch.profiler.ProfilerActivity.PrivateUse1
        profiler_config.DEVICE_CONTENTS = frozenset(profiler_config.DEVICE_CONTENTS | {TPU_DEVICE})
    tp.DEVICE_ACTIVITIES = profiler_config.DEVICE_ACTIVITIES
    tp.DEVICE_CONTENTS = profiler_config.DEVICE_CONTENTS


def _safe_trace_handler(handler: Optional[Callable[[Any], Any]]) -> Optional[Callable[[Any], Any]]:
    if handler is None:
        return None

    @functools.wraps(handler)
    def _wrapped(prof: Any) -> Any:
        try:
            return handler(prof)
        except Exception as exc:  # noqa: BLE001 - the .xplane.pb is written by the TPU backend regardless
            logger.warning("TPU profiler: chrome-trace export failed (%s); the .xplane.pb trace is unaffected", exc)
            return None

    return _wrapped


@contextlib.contextmanager
def _profile_with_tpu_config(config: Any):
    """Temporarily make ``torch.profiler.profile`` add ``experimental_config`` and a fail-safe ``on_trace_ready``."""
    original = torch.profiler.profile

    def _profile(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("experimental_config", config)
        if "on_trace_ready" in kwargs:
            kwargs["on_trace_ready"] = _safe_trace_handler(kwargs["on_trace_ready"])
        return original(*args, **kwargs)

    torch.profiler.profile = _profile
    try:
        yield
    finally:
        torch.profiler.profile = original


def apply() -> bool:
    """Patch verl's torch profiler for TPU. Idempotent; returns whether the patch is (now) installed."""
    global _patched
    if _patched:
        return True
    try:
        import verl.utils.profiler.torch_profile as tp
    except Exception as exc:  # noqa: BLE001 - verl without the torch profiler module
        logger.debug("TPU profiler patch skipped: %s", exc)
        return False

    _register_device_activity()
    original_get_torch_profiler = tp.get_torch_profiler

    @functools.wraps(original_get_torch_profiler)
    def get_torch_profiler(*args: Any, **kwargs: Any) -> Any:
        config_cls = _tpu_profiler_config_cls() if _on_tpu() else None
        if config_cls is None:
            return original_get_torch_profiler(*args, **kwargs)
        # verl's callers (DistProfiler) pass every argument by keyword; rank defaults to 0 as in verl.
        save_path, rank = kwargs["save_path"], kwargs.get("rank", 0)
        config = config_cls(run_dir=save_path, worker_rank=str(rank))
        with _profile_with_tpu_config(config):
            prof = original_get_torch_profiler(*args, **kwargs)
        logger.warning("TPU profiler: XProf trace for rank %s -> %s/<host>_%s.xplane.pb", rank, save_path, rank)
        return prof

    tp.get_torch_profiler = get_torch_profiler
    _patched = True
    return True
