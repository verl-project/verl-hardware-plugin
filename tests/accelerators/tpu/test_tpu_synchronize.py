# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for public TPU synchronization (torch.accelerator.synchronize + torch.Event)."""

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
from verl_hardware_plugin.accelerators.tpu.engines import tpu_utils
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import tpu_synchronize


@pytest.fixture(autouse=True)
def _reset_device_api_cache():
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None
    yield
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None


def _fake_tpu_accelerator(**overrides):
    acc = types.SimpleNamespace(
        current_accelerator=lambda: types.SimpleNamespace(type="tpu"),
        is_available=lambda: True,
        device_count=lambda: 1,
        current_device_index=lambda: 0,
        synchronize=mock.MagicMock(),
    )
    for k, v in overrides.items():
        setattr(acc, k, v)
    return acc


def test_no_torch_tpu_internal_imports_in_tpu_plugin():
    """All TPU plugin modules must rely exclusively on public torch / torch.tpu / torch.accelerator APIs."""
    pkg_dir = Path(tpu_pkg.__file__).resolve().parent
    offenders: list[str] = []
    for py_file in sorted(pkg_dir.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch_tpu._internal" or alias.name.startswith("torch_tpu._internal."):
                        offenders.append(f"{py_file.relative_to(pkg_dir)}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module == "torch_tpu._internal" or node.module.startswith("torch_tpu._internal."):
                    offenders.append(f"{py_file.relative_to(pkg_dir)}:{node.lineno} from {node.module} import ...")
    assert offenders == []


def test_tpu_synchronize_wait_true_uses_torch_accelerator():
    """wait=True delegates to torch.accelerator.synchronize and does not allocate a torch.Event."""
    acc = _fake_tpu_accelerator()
    tpu_mod = types.SimpleNamespace(is_initialized=lambda: True)
    fake_event_cls = mock.MagicMock()

    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", tpu_mod, create=True),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        tpu_synchronize(wait=True)
        acc.synchronize.assert_called_once_with()
        fake_event_cls.assert_not_called()

        # Caching: subsequent calls reuse the per-PID runtime-initialized flag without re-querying is_initialized.
        assert device_api._tpu_runtime_up_pid == os.getpid()
        tpu_mod.is_initialized = mock.MagicMock(side_effect=AssertionError("should use cached runtime state"))
        tpu_synchronize()
        assert acc.synchronize.call_count == 2


def test_tpu_synchronize_wait_false_records_torch_event():
    """wait=False cuts and dispatches the active TPU stream via torch.Event().record() without blocking the host."""
    acc = _fake_tpu_accelerator()
    tpu_mod = types.SimpleNamespace(is_initialized=lambda: True)
    fake_event = mock.MagicMock()
    fake_event_cls = mock.MagicMock(return_value=fake_event)

    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", tpu_mod, create=True),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        tpu_synchronize(wait=False)
        fake_event_cls.assert_called_once_with()
        fake_event.record.assert_called_once_with()
        acc.synchronize.assert_not_called()


def test_tpu_synchronize_skips_cpu_tensors_off_tpu_hosts_and_uninitialized_runtime():
    """CPU tensors, CPU-only processes, and 0-TPU actors whose PJRT runtime is not initialized are no-ops."""
    acc = _fake_tpu_accelerator()
    tpu_mod = types.SimpleNamespace(is_initialized=lambda: True)
    fake_event_cls = mock.MagicMock()

    # 1. CPU tensor fast-exit even on an active TPU worker with strict=True.
    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", tpu_mod, create=True),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        tpu_synchronize(torch.ones(4), wait=True, strict=True)
        tpu_synchronize(torch.ones(4), wait=False, strict=True)
        acc.synchronize.assert_not_called()
        fake_event_cls.assert_not_called()

    # 2. CPU-only process (current_accelerator() is None): lenient calls are silent, strict=True raises.
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None
    cpu_only_acc = _fake_tpu_accelerator(current_accelerator=lambda: None)
    with (
        mock.patch.object(torch, "accelerator", cpu_only_acc),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        tpu_synchronize(wait=True)
        tpu_synchronize(wait=False)
        cpu_only_acc.synchronize.assert_not_called()
        fake_event_cls.assert_not_called()
        with pytest.raises(RuntimeError, match="TPU synchronization failed"):
            tpu_synchronize(strict=True)

    # 3. 0-TPU Ray actor on a TPU host before PJRT initialization (is_initialized() is False): must not start PJRT.
    device_api._accelerator_is_tpu_cache = None
    device_api._tpu_runtime_up_pid = None
    uninit_tpu = types.SimpleNamespace(is_initialized=lambda: False)
    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", uninit_tpu, create=True),
        mock.patch.object(torch, "Event", fake_event_cls),
    ):
        tpu_synchronize(wait=True)
        tpu_synchronize(wait=False)
        tpu_synchronize(strict=True)
        acc.synchronize.assert_not_called()
        fake_event_cls.assert_not_called()


def test_tpu_synchronize_wait_false_strict_and_warning(caplog):
    """Errors during async Event.record() respect strict=True vs warning logging."""
    acc = _fake_tpu_accelerator()
    tpu_mod = types.SimpleNamespace(is_initialized=lambda: True)
    failing_event = mock.MagicMock()
    failing_event.record.side_effect = RuntimeError("stream dispatch failed")

    with (
        mock.patch.object(torch, "accelerator", acc),
        mock.patch.object(torch, "tpu", tpu_mod, create=True),
        mock.patch.object(torch, "Event", return_value=failing_event),
    ):
        with pytest.raises(RuntimeError, match="TPU synchronization failed: stream dispatch failed"):
            tpu_synchronize(wait=False, strict=True)
        with caplog.at_level(logging.WARNING, logger=tpu_utils.__name__):
            tpu_synchronize(wait=False, strict=False)
        assert "Could not synchronize TPU device" in caplog.text
