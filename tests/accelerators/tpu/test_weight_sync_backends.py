# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU weight sync backend names and their deprecated aliases."""

import warnings

import pytest

from verl_hardware_plugin.accelerators.tpu.weight_sync import weight_sync_backends
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import (
    DEPRECATED_BACKEND_NAMES,
    RAY_PLASMA_STORE_BACKEND,
    TPU_SYNC_BACKEND,
    canonical_backend,
)


@pytest.fixture(autouse=True)
def _reset_warnings(monkeypatch):
    monkeypatch.setattr(weight_sync_backends, "_warned_deprecated_names", set())


def test_backend_names():
    assert (RAY_PLASMA_STORE_BACKEND, TPU_SYNC_BACKEND) == ("ray-plasma-store", "tpu-sync")
    assert DEPRECATED_BACKEND_NAMES == {"tpu": "ray-plasma-store"}


@pytest.mark.parametrize("backend", ["ray-plasma-store", "tpu-sync", "naive", "nccl", None])
def test_canonical_backend_keeps_current_and_other_names(backend):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert canonical_backend(backend) == backend


@pytest.mark.parametrize(("deprecated", "current"), [("tpu", "ray-plasma-store")])
def test_canonical_backend_maps_deprecated_names_and_warns_once(deprecated, current):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert canonical_backend(deprecated) == current
        assert canonical_backend(deprecated) == current
    assert [(w.category, str(w.message)) for w in caught] == [
        (
            FutureWarning,
            f"actor_rollout_ref.rollout.checkpoint_engine.backend={deprecated} is deprecated, use {current} instead.",
        )
    ]
