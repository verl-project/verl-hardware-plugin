# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU device queries go through torch.accelerator (torch_tpu >= 0.2.0)."""

import ast
import logging
import os
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

import verl_hardware_plugin.accelerators.tpu as tpu_pkg
from verl_hardware_plugin.accelerators.tpu import device_api
from verl_hardware_plugin.accelerators.tpu.platform_tpu import TPUDeviceModuleProxy


@pytest.fixture(autouse=True)
def _reset_cache():
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None
    yield
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None


def _internal(name):
    def fn(*args, **kwargs):
        raise RuntimeError(f"{name} is internal to TorchTPU and can only be called by the PyTorch framework")

    return fn


def _fake_accelerator(**overrides):
    acc = types.SimpleNamespace(
        # torch.device("tpu") needs torch_tpu, which the CPU test environment does not install.
        current_accelerator=lambda: types.SimpleNamespace(type="tpu"),
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device_index=lambda: 0,
        synchronize=mock.MagicMock(),
        current_stream=mock.MagicMock(return_value="stream0"),
        set_stream=mock.MagicMock(),
        device_index=mock.MagicMock(return_value="ctx"),
    )
    for k, v in overrides.items():
        setattr(acc, k, v)
    return acc


def test_uses_torch_accelerator_when_tpu_device_apis_are_internal():
    """torch_tpu >= 0.2.0: torch.tpu device functions raise, torch.accelerator serves them."""
    generator = mock.MagicMock()
    generator.get_state.return_value = torch.tensor([7])
    generator.initial_seed.return_value = 123
    legacy_tpu = types.SimpleNamespace(
        **{
            n: _internal(n)
            for n in ("is_available", "device_count", "current_device", "synchronize", "get_rng_state", "initial_seed")
        },
        is_initialized=lambda: True,
        default_generators=[generator],
    )
    acc = _fake_accelerator()
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", legacy_tpu, create=True):
        proxy = TPUDeviceModuleProxy(legacy_tpu)
        assert proxy.is_available() is True
        assert proxy.device_count() == 1
        assert proxy.current_device() == 0
        proxy.synchronize()
        acc.synchronize.assert_called_once_with(None)
        assert proxy.current_stream() == "stream0"
        proxy.set_stream("s")
        acc.set_stream.assert_called_once_with("s")
        assert proxy.device(0) == "ctx"
        assert torch.equal(proxy.get_rng_state(), torch.tensor([7]))
        assert proxy.initial_seed() == 123


def test_cpu_only_process_without_tpu_devices_never_calls_internal_torch_tpu_apis():
    """torch_tpu >= 0.2.0 on a CPU-only process: current_accelerator() is None and torch.tpu device APIs raise."""
    legacy_tpu = types.SimpleNamespace(
        **{n: _internal(n) for n in ("is_available", "device_count", "current_device", "synchronize")}
    )
    acc = _fake_accelerator(current_accelerator=lambda: None)
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", legacy_tpu, create=True):
        proxy = TPUDeviceModuleProxy(legacy_tpu)
        assert device_api.is_available() is False
        assert proxy.is_available() is False
        assert device_api.device_count() == 0
        assert proxy.device_count() == 0
        assert device_api.current_device() == 0
        assert proxy.current_device() == 0
        assert device_api.synchronize() is None
        assert proxy.synchronize() is None
        acc.synchronize.assert_not_called()


def _memory_accelerator():
    """torch.accelerator with torch_tpu's memory functions (the PJRT allocator statistics)."""
    return _fake_accelerator(
        memory_allocated=mock.MagicMock(return_value=3 << 30),
        max_memory_allocated=mock.MagicMock(return_value=5 << 30),
        memory_reserved=mock.MagicMock(return_value=1 << 30),
        max_memory_reserved=mock.MagicMock(return_value=2 << 30),
        memory_stats=mock.MagicMock(return_value={"allocated_bytes.all.current": 3 << 30}),
        reset_peak_memory_stats=mock.MagicMock(),
        get_memory_info=mock.MagicMock(return_value=(60 << 30, 95 << 30)),
    )


def test_memory_statistics_come_from_torch_accelerator():
    """torch.tpu has no memory functions; torch_tpu reports the PJRT allocator through torch.accelerator."""
    tpu = types.SimpleNamespace(is_initialized=lambda: True)
    acc = _memory_accelerator()
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", tpu, create=True):
        proxy = TPUDeviceModuleProxy(tpu)
        assert proxy.memory_allocated() == 3 << 30
        assert proxy.max_memory_allocated() == 5 << 30
        assert proxy.memory_reserved() == 1 << 30
        assert proxy.max_memory_reserved() == 2 << 30
        assert proxy.memory_stats() == {"allocated_bytes.all.current": 3 << 30}
        assert proxy.mem_get_info() == (60 << 30, 95 << 30)
        proxy.reset_peak_memory_stats()
        acc.reset_peak_memory_stats.assert_called_once_with(None)
        proxy.memory_allocated("tpu:0")
        acc.memory_allocated.assert_called_with("tpu:0")


@pytest.mark.parametrize(
    "current_accelerator, initialized",
    [
        (lambda: types.SimpleNamespace(type="tpu"), False),  # TPU process before its runtime is up
        (lambda: None, True),  # no TPU in this process (Ray driver, rollout server actor)
    ],
)
def test_memory_statistics_are_zero_until_the_tpu_runtime_is_up(current_accelerator, initialized):
    """Reading the statistics would start the PJRT client, which waits for the whole slice: report zero instead."""
    tpu = types.SimpleNamespace(is_initialized=lambda: initialized)
    acc = _memory_accelerator()
    acc.current_accelerator = current_accelerator
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", tpu, create=True):
        proxy = TPUDeviceModuleProxy(tpu)
        assert proxy.memory_allocated() == proxy.max_memory_allocated() == 0
        assert proxy.memory_reserved() == proxy.max_memory_reserved() == 0
        assert proxy.memory_stats() == {}
        assert proxy.mem_get_info() == (0, 0)
        proxy.reset_peak_memory_stats()
        proxy.synchronize()
    for name in ("memory_allocated", "memory_stats", "get_memory_info", "reset_peak_memory_stats", "synchronize"):
        getattr(acc, name).assert_not_called()


def test_empty_cache_never_evicts_the_compiled_executables():
    """torch.tpu._clear_cache() evicts every compiled executable; verl calls empty_cache() on every epoch."""
    tpu = types.SimpleNamespace(_clear_cache=mock.MagicMock(), is_initialized=lambda: True)
    acc = _fake_accelerator(empty_cache=mock.MagicMock())
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", tpu, create=True):
        proxy = TPUDeviceModuleProxy(tpu)
        assert proxy.empty_cache() is None
    tpu._clear_cache.assert_not_called()


def test_torch_tpu_device_functions_are_forwarded():
    """get_device_properties (HBM capacity), get_device_name and manual_seed_all are torch.tpu's own."""
    props = types.SimpleNamespace(name="TPU v6 lite", total_memory=32 << 30)
    tpu = types.SimpleNamespace(
        get_device_properties=mock.MagicMock(return_value=props),
        get_device_name=mock.MagicMock(return_value="TPU v6 lite"),
        manual_seed_all=mock.MagicMock(),
    )
    proxy = TPUDeviceModuleProxy(tpu)
    assert proxy.get_device_properties().total_memory == 32 << 30
    assert proxy.get_device_name() == "TPU v6 lite"
    proxy.manual_seed_all(7)
    tpu.manual_seed_all.assert_called_once_with(7)
    with pytest.raises(AttributeError):
        proxy.not_a_device_function  # noqa: B018


def test_platform_leaves_torch_tpu_untouched():
    """torchtitan patches torch.tpu with its own memory shims; a second patch would depend on import order."""
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU

    tpu = types.ModuleType("torch.tpu")
    with mock.patch.object(torch, "tpu", tpu, create=True):
        platform = PlatformTPU()
        assert platform.device_module._original_module is tpu
        platform.empty_cache()
    for name in ("empty_cache", "memory_stats", "reset_peak_memory_stats", "get_device_name", "get_device_properties"):
        assert not hasattr(tpu, name)


def test_synchronize_and_tpu_synchronize_contracts_and_no_internal_imports(caplog, monkeypatch):
    """synchronize() preserves PyTorch error semantics while tpu_synchronize() handles tensors and wait=False."""
    pkg_dir = Path(tpu_pkg.__file__).resolve().parent
    for py_file in sorted(pkg_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(py_file.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            assert not any(m == "torch_tpu._internal" or m.startswith("torch_tpu._internal.") for m in mods), py_file

    acc = _fake_accelerator()
    tpu_mod = types.SimpleNamespace(is_initialized=lambda: True)
    fake_event = mock.MagicMock()
    fake_event_cls = mock.MagicMock(return_value=fake_event)
    fake_tpu_tensor = mock.MagicMock(spec=torch.Tensor)
    fake_tpu_tensor.device = types.SimpleNamespace(type="tpu", index=1)

    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", tpu_mod, create=True),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        # 1. Standard device_api.synchronize(device) delegates to torch.accelerator.synchronize and propagates errors.
        device_api.synchronize("tpu:0")
        assert device_api._tpu_runtime_up_pid == os.getpid()
        acc.synchronize.assert_called_once_with("tpu:0")
        acc.synchronize.side_effect = RuntimeError("XLA sticky error")
        with pytest.raises(RuntimeError, match="XLA sticky error"):
            device_api.synchronize()
        acc.synchronize.side_effect = None
        acc.synchronize.reset_mock()

        # 2. tpu_synchronize: CPU tensor & CPU tensor sequence fast-exit without touching accelerator.
        device_api.tpu_synchronize(torch.ones(4), wait=True, strict=True)
        device_api.tpu_synchronize([torch.ones(2), torch.zeros(2)], wait=False, strict=True)
        acc.synchronize.assert_not_called()

        # 3. tpu_synchronize(wait=False): None records on current stream; explicit TPU tensor passes current_stream(dev)
        #    because c10::Event discards device.index when record() is called with no stream argument.
        device_api.tpu_synchronize(wait=False)
        fake_event_cls.assert_called_once_with()
        fake_event.record.assert_called_once_with()
        fake_event_cls.reset_mock()
        fake_event.reset_mock()

        device_api.tpu_synchronize([torch.ones(2), fake_tpu_tensor], wait=False)
        acc.current_stream.assert_called_once_with(fake_tpu_tensor.device)
        fake_event.record.assert_called_once_with("stream0")

        # 4. Fork invalidation: a child PID does not inherit the parent's cached PJRT-initialized state.
        tpu_mod.is_initialized = lambda: False
        monkeypatch.setattr(os, "getpid", lambda: -99999)
        device_api.tpu_synchronize(wait=True, strict=True)
        acc.synchronize.assert_not_called()

        # 5. strict vs lenient error handling on tpu_synchronize.
        tpu_mod.is_initialized = lambda: True
        fake_event.record.side_effect = RuntimeError("stream dispatch failed")
        with pytest.raises(RuntimeError, match="TPU synchronization failed: stream dispatch failed"):
            device_api.tpu_synchronize(wait=False, strict=True)
        with caplog.at_level(logging.WARNING, logger=device_api.__name__):
            device_api.tpu_synchronize(wait=False, strict=False)
        assert "Could not synchronize TPU device" in caplog.text


def test_placement_group_claims_tpu_slices_in_numeric_order():
    """Clusters with >=10 TPU slices claim tpu-group-0, tpu-group-1, ..., tpu-group-10 in numeric order."""
    import ray

    from verl_hardware_plugin.accelerators.tpu import platform_tpu as platform_tpu_mod
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU

    nodes = [{"Alive": True, "Resources": {f"tpu-group-{i}": 8.0}} for i in (10, 2, 0, 1, 9)]
    tpu = types.ModuleType("torch.tpu")
    with (
        mock.patch.object(torch, "tpu", tpu, create=True),
        mock.patch.object(ray, "is_initialized", return_value=True),
        mock.patch.object(ray, "nodes", return_value=nodes),
    ):
        platform = PlatformTPU()
        claimed = [platform.auto_assign_accelerator_type(f"pool_{k}", None) for k in range(7)]
    # Once every slice is claimed, further pools go round-robin over the slices after the first.
    assert claimed == [
        "tpu-group-0",
        "tpu-group-1",
        "tpu-group-2",
        "tpu-group-9",
        "tpu-group-10",
        "tpu-group-1",
        "tpu-group-2",
    ]
    assert platform_tpu_mod is not None


def test_get_worker_env_vars_merges_trainer_extra_libtpu_init_args(monkeypatch):
    """Trainer-only libtpu flags append onto LIBTPU_INIT_ARGS for RayWorkerGroup actors without touching rollout."""
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU

    pool = mock.MagicMock()
    pool.get_placement_groups.return_value = []
    platform = PlatformTPU()
    monkeypatch.setattr(platform, "get_tpu_env_vars", lambda **kwargs: {"TORCH_TPU_TOPOLOGY": "2,2,1,2"})

    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--xla_tpu_use_enhanced_launch_barrier=false")
    monkeypatch.delenv("VERL_TPU_TRAINER_LIBTPU_INIT_ARGS", raising=False)
    env_plain = platform.get_worker_env_vars(pool, 0, 8, 0, 8, "actor", "TPU")
    assert "LIBTPU_INIT_ARGS" not in env_plain

    monkeypatch.setenv(
        "VERL_TPU_TRAINER_LIBTPU_INIT_ARGS",
        "--xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sparse_core_collective_aggregator=true",
    )
    env_trainer = platform.get_worker_env_vars(pool, 0, 8, 0, 8, "actor", "TPU")
    assert env_trainer["LIBTPU_INIT_ARGS"] == (
        "--xla_tpu_use_enhanced_launch_barrier=false "
        "--xla_tpu_scoped_vmem_limit_kib=65472 --xla_tpu_enable_sparse_core_collective_aggregator=true"
    )
