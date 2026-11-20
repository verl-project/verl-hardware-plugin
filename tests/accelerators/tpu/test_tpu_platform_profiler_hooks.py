# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for PlatformTPU's torch-profiler hooks (verl ``PlatformBase.torch_profiler_activity`` / ``_kwargs``)."""

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu import platform_tpu
from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU


class _FakeTpuProfilerConfig:
    def __init__(self, *, run_dir=None, worker_rank=None):
        self.run_dir = run_dir
        self.worker_rank = worker_rank


def test_with_torch_tpu_records_privateuse1_and_passes_the_tpu_config(monkeypatch):
    monkeypatch.setattr(platform_tpu, "tpu_profiler_config_cls", lambda: _FakeTpuProfilerConfig)
    platform = PlatformTPU()

    assert platform.torch_profiler_activity() == torch.profiler.ProfilerActivity.PrivateUse1
    kwargs = platform.torch_profiler_kwargs(save_path="/tmp/prof", rank=3)
    config = kwargs["experimental_config"]
    assert isinstance(config, _FakeTpuProfilerConfig)
    assert (config.run_dir, config.worker_rank) == ("/tmp/prof", "3")


def test_config_cls_is_none_without_torch_tpu(monkeypatch):
    import importlib

    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    assert platform_tpu.tpu_profiler_config_cls() is None


def test_without_torch_tpu_records_cpu_only(monkeypatch):
    monkeypatch.setattr(platform_tpu, "tpu_profiler_config_cls", lambda: None)
    platform = PlatformTPU()

    assert platform.torch_profiler_activity() is None
    assert platform.torch_profiler_kwargs(save_path="/tmp/prof", rank=0) == {}


def test_verl_get_torch_profiler_uses_the_hooks(monkeypatch, tmp_path):
    """End to end through verl's get_torch_profiler, when verl has the platform hooks."""
    from verl.plugin.platform.platform_base import PlatformBase

    if not hasattr(PlatformBase, "torch_profiler_kwargs"):
        pytest.skip("verl without PlatformBase torch-profiler hooks")
    import verl.utils.profiler.torch_profile as tp

    monkeypatch.setattr(platform_tpu, "tpu_profiler_config_cls", lambda: _FakeTpuProfilerConfig)
    platform = PlatformTPU()
    monkeypatch.setattr(tp, "get_platform", lambda: platform)
    seen = {}

    def fake_profile(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    tp.get_torch_profiler(contents=[], save_path=str(tmp_path), rank=2)

    assert torch.profiler.ProfilerActivity.PrivateUse1 in seen["activities"]
    assert seen["experimental_config"].worker_rank == "2"
