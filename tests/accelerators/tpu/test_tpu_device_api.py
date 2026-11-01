# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU device queries go through torch.accelerator (torch_tpu >= 0.2.0) or the legacy torch.tpu API."""

import types
from unittest import mock

import pytest
import torch

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


def test_falls_back_to_legacy_torch_tpu_api():
    """Older torch_tpu: torch.accelerator is not TPU-backed, torch.tpu functions are public."""
    legacy_tpu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        current_device=lambda: 2,
        synchronize=mock.MagicMock(),
        get_rng_state=mock.MagicMock(return_value=torch.tensor([1])),
        initial_seed=lambda: 5,
    )
    acc = _fake_accelerator(current_accelerator=lambda: None)
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", legacy_tpu, create=True):
        proxy = TPUDeviceModuleProxy(legacy_tpu)
        assert proxy.is_available() is True
        assert proxy.device_count() == 4
        assert proxy.current_device() == 2
        proxy.synchronize()
        legacy_tpu.synchronize.assert_called_once_with()
        acc.synchronize.assert_not_called()
        assert torch.equal(proxy.get_rng_state(), torch.tensor([1]))
        assert proxy.initial_seed() == 5


def test_is_available_is_false_without_tpu_devices():
    """torch_tpu >= 0.2.0 on a CPU-only process: no TPU accelerator and torch.tpu.is_available raises."""
    legacy_tpu = types.SimpleNamespace(is_available=_internal("is_available"))
    acc = _fake_accelerator(current_accelerator=lambda: None)
    with mock.patch.object(torch, "accelerator", acc), mock.patch.object(torch, "tpu", legacy_tpu, create=True):
        assert device_api.is_available() is False


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
