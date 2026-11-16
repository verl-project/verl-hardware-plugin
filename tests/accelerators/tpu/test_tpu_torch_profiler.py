# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU torch-profiler (XProf) patch of verl's ``get_torch_profiler``."""

import sys
import types

import pytest
import torch

pytest.importorskip("verl.utils.profiler.torch_profile")

import verl.utils.profiler.config as profiler_config  # noqa: E402
import verl.utils.profiler.torch_profile as tp  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.profilers import torch_profiler_tpu  # noqa: E402


class _FakeTpuProfilerConfig:
    def __init__(self, *, run_dir=None, worker_rank=None):
        self.run_dir = run_dir
        self.worker_rank = worker_rank


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(torch_profiler_tpu, "_patched", False)
    monkeypatch.setattr(tp, "get_torch_profiler", tp.get_torch_profiler)
    monkeypatch.setattr(profiler_config, "DEVICE_ACTIVITIES", dict(profiler_config.DEVICE_ACTIVITIES))
    monkeypatch.setattr(profiler_config, "DEVICE_CONTENTS", profiler_config.DEVICE_CONTENTS)
    monkeypatch.setattr(tp, "DEVICE_ACTIVITIES", profiler_config.DEVICE_ACTIVITIES)
    monkeypatch.setattr(tp, "DEVICE_CONTENTS", profiler_config.DEVICE_CONTENTS)
    assert torch_profiler_tpu.apply()
    return monkeypatch


def test_registers_tpu_as_privateuse1(patched):
    assert profiler_config.DEVICE_ACTIVITIES["tpu"] == torch.profiler.ProfilerActivity.PrivateUse1
    assert "tpu" in profiler_config.DEVICE_CONTENTS
    assert tp.DEVICE_ACTIVITIES is profiler_config.DEVICE_ACTIVITIES


def test_off_tpu_falls_through(patched, tmp_path):
    patched.setattr(torch_profiler_tpu, "_on_tpu", lambda: False)
    seen = {}
    original = torch.profiler.profile

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    patched.setattr(torch.profiler, "profile", spy)
    tp.get_torch_profiler(contents=[], save_path=str(tmp_path), rank=0)
    assert "experimental_config" not in seen


def test_on_tpu_adds_tpu_config_and_safe_handler(patched, tmp_path):
    patched.setattr(torch_profiler_tpu, "_on_tpu", lambda: True)
    patched.setattr(torch_profiler_tpu, "_tpu_profiler_config_cls", lambda: _FakeTpuProfilerConfig)
    seen = {}

    def fake_profile(*args, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(record_steps=True)

    patched.setattr(torch.profiler, "profile", fake_profile)
    tp.get_torch_profiler(contents=[], save_path=str(tmp_path), rank=3)
    cfg = seen["experimental_config"]
    assert isinstance(cfg, _FakeTpuProfilerConfig)
    assert cfg.run_dir == str(tmp_path) and cfg.worker_rank == "3"
    # verl's chrome-trace export failing must not fail the step.
    broken = types.SimpleNamespace(export_chrome_trace=lambda path: (_ for _ in ()).throw(RuntimeError("no kineto")))
    assert seen["on_trace_ready"](broken) is None
    # torch.profiler.profile is restored after the build.
    assert torch.profiler.profile is fake_profile


def test_apply_is_idempotent(patched):
    first = tp.get_torch_profiler
    assert torch_profiler_tpu.apply()
    assert tp.get_torch_profiler is first
    assert "torch_profiler_tpu" in sys.modules[torch_profiler_tpu.__name__].__name__


def _on_tpu_with_fake_profile(monkeypatch):
    """TPU host with a fake ``TpuProfilerConfig``; returns the kwargs ``torch.profiler.profile`` receives."""
    monkeypatch.setattr(torch_profiler_tpu, "_on_tpu", lambda: True)
    monkeypatch.setattr(torch_profiler_tpu, "_tpu_profiler_config_cls", lambda: _FakeTpuProfilerConfig)
    seen = {}

    def fake_profile(*args, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(record_steps=True)

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    return seen


def test_on_tpu_uses_verls_default_rank(patched, tmp_path):
    seen = _on_tpu_with_fake_profile(patched)
    tp.get_torch_profiler(contents=[], save_path=str(tmp_path))
    assert seen["experimental_config"].worker_rank == "0"


def test_works_under_another_plugins_bare_wrapper(monkeypatch, tmp_path):
    # Other backends' patches (e.g. MLU, installed on every host) wrap get_torch_profiler as (*args, **kwargs).
    monkeypatch.setattr(torch_profiler_tpu, "_patched", False)
    verl_fn = tp.get_torch_profiler
    monkeypatch.setattr(tp, "get_torch_profiler", lambda *args, **kwargs: verl_fn(*args, **kwargs))
    assert torch_profiler_tpu.apply()
    seen = _on_tpu_with_fake_profile(monkeypatch)
    tp.get_torch_profiler(contents=[], save_path=str(tmp_path), rank=2)
    assert seen["experimental_config"].run_dir == str(tmp_path)
    assert seen["experimental_config"].worker_rank == "2"


def test_profile_restored_when_building_the_profiler_fails(patched, tmp_path):
    patched.setattr(torch_profiler_tpu, "_on_tpu", lambda: True)
    patched.setattr(torch_profiler_tpu, "_tpu_profiler_config_cls", lambda: _FakeTpuProfilerConfig)

    def failing_profile(*args, **kwargs):
        raise RuntimeError("profiler init failed")

    patched.setattr(torch.profiler, "profile", failing_profile)
    with pytest.raises(RuntimeError, match="profiler init failed"):
        tp.get_torch_profiler(contents=[], save_path=str(tmp_path), rank=0)
    assert torch.profiler.profile is failing_profile


def test_config_cls_comes_from_the_import_alone(monkeypatch):
    def must_not_be_called():
        raise AssertionError("register_kineto_backend is a no-op in torch_tpu; the import registers the backend")

    fake = types.ModuleType("torch_tpu._internal.profiler")
    fake.TpuProfilerConfig = _FakeTpuProfilerConfig
    fake.register_kineto_backend = must_not_be_called
    for name in ("torch_tpu", "torch_tpu._internal"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torch_tpu._internal.profiler", fake)
    assert torch_profiler_tpu._tpu_profiler_config_cls() is _FakeTpuProfilerConfig


def test_config_cls_is_none_without_torch_tpu(monkeypatch):
    for name in ("torch_tpu", "torch_tpu._internal", "torch_tpu._internal.profiler", "torch.tpu", "torch.tpu.profiler"):
        monkeypatch.setitem(sys.modules, name, None)  # makes the import fail
    assert torch_profiler_tpu._tpu_profiler_config_cls() is None
