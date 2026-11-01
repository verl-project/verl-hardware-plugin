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
