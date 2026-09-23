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
from verl_hardware_plugin.profilers import torch_profile_mlu


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
        "floor (verl>=0.7.0) allows installs older than this"
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
