# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression test: MLU's profiler patch must not break other platforms.

verl-core's own ``Profiler.start()`` (``verl/utils/profiler/torch_profile.py``) calls
``get_torch_profiler(..., profile_step=..., schedule=...)`` on every profiled step, on
every platform. Before the delegate fix, MLU's hand-copied reimplementation of
``get_torch_profiler`` didn't know about ``profile_step``/``name_mini_batch_window`` and
TypeError'd on this call for every platform, not just MLU, the moment
``verl_hardware_plugin`` was installed -- confirmed live against real verl 0.9.1.
"""

import inspect

import pytest

import verl.utils.profiler.torch_profile as tp
from verl_hardware_plugin.accelerators.mlu.profilers import torch_profile_mlu


def _installed_get_torch_profiler_supports_new_kwargs() -> bool:
    """profile_step/name_mini_batch_window were added in verl-core #7408 (commit
    9097cc4). Older installs -- including the verl==0.9.0 PyPI wheel -- predate
    that and don't have them. The delegate fix forwards to whatever signature
    is actually installed rather than hardcoding one, so there's nothing to
    regression-test against an install that doesn't have these params yet.
    """
    sig = inspect.signature(tp.get_torch_profiler)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return "profile_step" in sig.parameters and "name_mini_batch_window" in sig.parameters


@pytest.mark.skipif(
    not _installed_get_torch_profiler_supports_new_kwargs(),
    reason=(
        "installed verl-core predates profile_step/name_mini_batch_window "
        "(verl-project/verl#7408, commit 9097cc4) -- pyproject.toml's declared "
        "floor (verl>=0.9.0) allows installs older than this"
    ),
)
def test_get_torch_profiler_accepts_new_kwargs_for_non_mlu_contents(tmp_path):
    torch_profile_mlu._patch_get_torch_profiler()

    prof = tp.get_torch_profiler(
        contents=["cpu"],
        save_path=str(tmp_path),
        role="actor",
        rank=0,
        profile_step=3,
        name_mini_batch_window=True,
    )

    assert prof is not None


def test_trace_filenames_unique_on_window_boundary_stop(tmp_path):
    """
    This test ensures filenames remain unique by verifying that two trace files
    are created with distinct "_c0" and "_c1" callback counter suffixes.
    """
    import os

    torch_profile_mlu._patch_get_torch_profiler()

    # Create a real torch profiler instance with schedule
    save_dir = tmp_path
    schedule = {"wait": 0, "warmup": 0, "active": 2, "repeat": 0}
    prof = tp.get_torch_profiler(
        contents=["cpu"],
        save_path=save_dir,
        role="actor",
        rank=0,
        profile_step=None,
        schedule=schedule,
        name_mini_batch_window=True,
        save_file_prefix="test_boundary",
    )

    assert prof is not None

    # Simulate exactly two mini-batches (the active window size)
    with prof:
        for _ in range(2):
            prof.step()

    # Check that two trace files were created with unique callback counter suffixes
    trace_files = sorted(f for f in os.listdir(save_dir) if f.startswith("test-boundary") and f.endswith(".json.gz"))

    assert len(trace_files) == 2, f"Expected 2 trace files, got {len(trace_files)}: {trace_files}"

    # Verify callback counter suffixes are unique (_c0 and _c1)
    assert "_c0" in trace_files[0], f"Expected _c0 suffix in {trace_files[0]}"
    assert "_c1" in trace_files[1], f"Expected _c1 suffix in {trace_files[1]}"
