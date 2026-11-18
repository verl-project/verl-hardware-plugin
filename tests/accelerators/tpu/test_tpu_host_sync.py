# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the deferred micro-batch scalar reads (``engines/host_sync.py``)."""

import pytest
import torch

from verl.utils.metric import AggregationType, Metric
from verl_hardware_plugin.accelerators.tpu.engines import host_sync


def test_defer_metric_items_keeps_tensors_and_restores_append():
    original = Metric.append
    with host_sync.defer_metric_items():
        m = Metric(AggregationType.SUM, torch.tensor(2.5))
        m.append(1.0)
        assert isinstance(m.values[0], torch.Tensor)
        assert m.values[1] == 1.0
        with pytest.raises(ValueError):
            m.append(torch.ones(2))
    assert Metric.append is original
    assert Metric(AggregationType.SUM, torch.tensor(1.5)).values == [1.5]


def test_defer_metric_items_disabled_is_noop():
    with host_sync.defer_metric_items(enabled=False):
        assert Metric(AggregationType.MEAN, torch.tensor(3.0)).values == [3.0]


def test_materialize_host_scalars_replaces_loss_and_metrics():
    with host_sync.defer_metric_items():
        outputs = [
            {
                "model_output": {"log_probs": torch.ones(3)},
                "loss": torch.tensor(0.25),
                "metrics": {
                    "actor/pg_loss": Metric(AggregationType.SUM, torch.tensor(0.5)),
                    "actor/mismatch": Metric(AggregationType.SUM, torch.tensor(7, dtype=torch.int64)),
                    "plain": torch.tensor(1.0),
                    "listed": [torch.tensor(2.0), 3.0],
                    "kl_coef": 0.1,
                },
            },
            {"loss": torch.tensor(0.75), "metrics": {"actor/pg_loss": Metric(AggregationType.SUM, torch.tensor(1.5))}},
        ]
    host_sync.materialize_host_scalars(outputs)
    assert outputs[0]["loss"] == 0.25 and outputs[1]["loss"] == 0.75
    m0 = outputs[0]["metrics"]
    assert m0["actor/pg_loss"].values == [0.5]
    assert m0["actor/mismatch"].values == [7] and isinstance(m0["actor/mismatch"].values[0], int)
    assert m0["plain"] == 1.0 and m0["listed"] == [2.0, 3.0] and m0["kl_coef"] == 0.1
    assert torch.equal(outputs[0]["model_output"]["log_probs"], torch.ones(3))
    assert Metric.aggregate_dp([m0["actor/pg_loss"], outputs[1]["metrics"]["actor/pg_loss"]]) == pytest.approx(1.0)


def test_env_flags(monkeypatch):
    monkeypatch.delenv(host_sync.DEFER_METRIC_SYNC_ENV, raising=False)
    monkeypatch.delenv(host_sync.MB_SYNC_WAIT_ENV, raising=False)
    assert not host_sync.defer_metric_sync_enabled() and host_sync.micro_batch_sync_wait()
    monkeypatch.setenv(host_sync.DEFER_METRIC_SYNC_ENV, "1")
    monkeypatch.setenv(host_sync.MB_SYNC_WAIT_ENV, "0")
    assert host_sync.defer_metric_sync_enabled() and not host_sync.micro_batch_sync_wait()
