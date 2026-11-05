# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU device queries go through torch.accelerator (torch_tpu >= 0.2.0)."""

import math
import types
from unittest import mock

import pytest
import torch

from verl.utils.flops_counter import get_device_flops
from verl_hardware_plugin.accelerators.tpu import device_api
from verl_hardware_plugin.accelerators.tpu.platform_tpu import TPUDeviceModuleProxy


@pytest.fixture(autouse=True)
def _reset_cache():
    device_api._accelerator_is_tpu_cache = None
    yield
    device_api._accelerator_is_tpu_cache = None


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
    for name in ("memory_allocated", "memory_stats", "get_memory_info", "reset_peak_memory_stats"):
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
        claimed = [platform.auto_assign_accelerator_type(f"pool_{k}", None) for k in range(6)]
    assert claimed == ["tpu-group-0", "tpu-group-1", "tpu-group-2", "tpu-group-9", "tpu-group-10", "tpu-group-10"]
    assert platform_tpu_mod is not None


@pytest.mark.parametrize(
    "device_name,expected_tflops",
    [
        ("TPU v7", 1153.5),
        ("TPU v7x", 1153.5),
        ("TPU7x", 1153.5),
        ("tpu7x", 1153.5),
        ("v7x", 1153.5),
        ("TPU v6 lite", 918.0),
        ("TPU v6e", 918.0),
        ("v6e", 918.0),
        ("TPU v5p", 459.0),
        ("v5p", 459.0),
        ("TPU v5 lite", 197.0),
        ("TPU v5e", 197.0),
        ("v5e", 197.0),
    ],
)
def test_tpu_device_flops_registered(device_name: str, expected_tflops: float):
    """TPU libtpu device_kind strings and GKE accelerator names resolve in verl's _DEVICE_FLOPS."""
    assert math.isclose(get_device_flops(unit="T", device_name=device_name), expected_tflops)
