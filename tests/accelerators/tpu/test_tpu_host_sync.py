# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``engine_kwargs.tpu.defer_metric_sync`` (``engines/host_sync.py`` and ``forward_backward_batch``).

The integration tests run the real ``forward_backward_batch`` -> ``forward_step`` -> verl ``ppo_loss`` ->
``postprocess_batch_func`` path (only the model forward is replaced by a tiny differentiable stand-in) with the
option off and on, and check that the step's outputs and gradients are identical and that the option removes the
blocking ``.item()`` calls from the micro-batch loop.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from tests.accelerators.tpu.test_tpu_loss_inputs import _ppo_config, _ppo_micro_batch
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.engines import host_sync, torchtitan_tpu
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import TPU_PADDED_VALUES_ATTR, bucket_length
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options

_LAZY_VERL = host_sync.verl_defers_metric_reads()


@contextmanager
def _count_stacks():
    calls: list[int] = []
    real_stack = torch.stack

    def stack(tensors, *args, **kwargs):
        calls.append(len(tensors))
        return real_stack(tensors, *args, **kwargs)

    with mock.patch.object(torch, "stack", side_effect=stack):
        yield calls


@contextmanager
def _no_item():
    with mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected blocking .item()")):
        yield


# ---------------------------------------------------------------------------------------------------------------
# materialize_step_scalars
# ---------------------------------------------------------------------------------------------------------------


def _outputs() -> list[dict[str, Any]]:
    return [
        {
            "model_output": {"log_probs": torch.ones(3)},
            "loss": torch.tensor(0.25),
            "metrics": {
                "actor/pg_loss": Metric(AggregationType.SUM, torch.tensor(0.5)),
                "actor/mismatch": Metric(AggregationType.SUM, torch.tensor(7, dtype=torch.int64)),
                "plain": torch.tensor(1.0),
                "kl_coef": 0.1,
            },
        },
        {"loss": torch.tensor(0.75), "metrics": {"actor/pg_loss": Metric(AggregationType.SUM, torch.tensor(1.5))}},
        {"loss": 0.125, "metrics": {}},
    ]


def _check_materialized(outputs: list[dict[str, Any]]) -> None:
    assert [o["loss"] for o in outputs] == [0.25, 0.75, 0.125]
    m0 = outputs[0]["metrics"]
    assert m0["actor/pg_loss"].values == [0.5]
    assert m0["actor/mismatch"].values == [7] and isinstance(m0["actor/mismatch"].values[0], int)
    # Only what .item() converted with the option off: raw tensors and model outputs are left alone.
    assert isinstance(m0["plain"], torch.Tensor) and m0["kl_coef"] == 0.1
    assert torch.equal(outputs[0]["model_output"]["log_probs"], torch.ones(3))
    assert Metric.aggregate_dp([m0["actor/pg_loss"], outputs[1]["metrics"]["actor/pg_loss"]]) == pytest.approx(1.0)


def test_materialize_step_scalars_one_batched_read():
    outputs = _outputs()
    with _count_stacks() as stacks:
        host_sync.materialize_step_scalars(outputs)
    # Two float losses (+ two float and one int Metric values with a lazy verl): one stack per dtype at most.
    assert all(n >= 2 for n in stacks) and len(stacks) <= 2
    _check_materialized(outputs)


def test_materialize_step_scalars_without_verl_helper():
    """verl releases without ``materialize_scalars``: the plugin's own batched read of the losses."""
    outputs = _outputs()
    import verl.utils.metric as verl_metric

    with mock.patch.dict(verl_metric.__dict__), _count_stacks() as stacks:
        verl_metric.__dict__.pop("materialize_scalars", None)
        Metric.materialize_all(_all_metrics(outputs)) if _LAZY_VERL else None
        host_sync.materialize_step_scalars(outputs)
    assert stacks[-1] == 2  # the two tensor losses
    _check_materialized(outputs)


def _all_metrics(outputs):
    return [m for o in outputs for m in o["metrics"].values() if isinstance(m, Metric)]


def test_scalars_to_host_fallback_matches_item():
    tensors = [torch.tensor(0.1, dtype=torch.bfloat16), torch.tensor(0.3, dtype=torch.bfloat16), torch.tensor(True)]
    tensors += [torch.tensor(-3, dtype=torch.int64), torch.tensor([2.0]), torch.tensor(5.0)]
    with _count_stacks() as stacks:
        assert host_sync._scalars_to_host_fallback(tensors) == [t.item() for t in tensors]
    assert sorted(stacks) == [1, 1, 2, 2]  # bf16 x2, bool, int64, float32 x2: padded to powers of two


def test_warns_once_only_for_legacy_verl(caplog):
    with mock.patch.object(host_sync, "_warned_legacy_verl", False):
        host_sync.warn_if_metrics_not_deferred()
        host_sync.warn_if_metrics_not_deferred()
    warnings = [r for r in caplog.records if "defer_metric_sync" in r.getMessage()]
    assert len(warnings) == (0 if _LAZY_VERL else 1)


# ---------------------------------------------------------------------------------------------------------------
# TPUEngineOptions.defer_metric_sync
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [(None, None), (True, True), ("yes", True), ("off", False), (0, False)])
def test_defer_metric_sync_option(raw, expected):
    cfg = SimpleNamespace(engine_kwargs={"tpu": {} if raw is None else {"defer_metric_sync": raw}})
    assert get_tpu_engine_options(cfg).defer_metric_sync is expected


def test_defer_metric_sync_option_rejects_typos():
    with pytest.raises(ValueError, match="defer_metric_sync"):
        get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"defer_metric_sync": "maybe"}}))
    with pytest.raises(ValueError):
        get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"defer_metric_syncs": True}}))


# ---------------------------------------------------------------------------------------------------------------
# forward_backward_batch, option off vs on
# ---------------------------------------------------------------------------------------------------------------

_MICRO_BATCH_LENGTHS = [[(3, 2), (6, 9), (4, 1)], [(5, 7), (9, 11)], [(2, 4)]]


def _engine(defer: Any, weight: torch.nn.Parameter):
    engine = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        tensor_parallel_size=1,
        context_parallel_size=1,
        engine_kwargs={"tpu": {} if defer is None else {"defer_metric_sync": defer}},
    )
    engine.parallel_dims = SimpleNamespace(cp_enabled=False, tp_enabled=False)
    engine._use_simple_fsdp = False
    engine._simple_fsdp_loss_scale = None
    engine._tpu_eager_mode = None
    engine.trainer = SimpleNamespace(train_context=nullcontext)
    engine.get_data_parallel_group = lambda: None
    engine.get_data_parallel_size = lambda: 1
    engine.prepare_model_inputs = lambda micro_batch: (micro_batch["input_ids"], {}, {}, {})
    engine.model_forward_step = lambda **kwargs: None

    def prepare_model_outputs(logits, output_args, micro_batch):
        # Bucket-padded device log-probs + their trimmed view, as the real prepare_model_outputs returns them; a
        # function of ``weight`` so the backward produces a gradient to compare.
        seq_lens = micro_batch["input_ids"].offsets().diff()
        cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int64), seq_lens.cumsum(0)])
        total = int(cu_seqlens[-1])
        gen = torch.Generator().manual_seed(total)
        padded = -torch.rand(bucket_length(total, 16), generator=gen) * weight
        nested = torch.nested.nested_tensor_from_jagged(padded.detach()[:total], cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, padded)
        return {"log_probs": nested}

    engine.prepare_model_outputs = prepare_model_outputs
    return engine


def _micro_batches():
    out = []
    for seed, lengths in enumerate(_MICRO_BATCH_LENGTHS):
        mb = _ppo_micro_batch(lengths, max_response_len=bucket_length(max(r for _, r in lengths), 16), seed=seed)
        tu.assign_non_tensor(mb, batch_num_tokens=64, dp_size=1)
        out.append(mb)
    return out


def _run_step(defer: Any) -> tuple[dict[str, Any], torch.Tensor, int, list[Any]]:
    """One ``forward_backward_batch``; returns its output, the gradient, the ``.item()`` reads of forward results
    (floating-point / bool) inside the micro-batch loop, and the type of each micro-batch's ``loss`` before the
    read-back."""
    weight = torch.nn.Parameter(torch.tensor(1.0))
    engine = _engine(defer, weight)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, use_dynamic_bsz=False)
    loss_function = partial(ppo_loss, config=_ppo_config())

    item_calls = 0
    real_item = torch.Tensor.item

    def counting_item(self):
        # Integer reads in the loop are of host-side sequence metadata (nested-tensor offsets of the micro-batch and
        # of the static loss layout, which stay on the CPU on TPU too): they never wait for the device.
        nonlocal item_calls
        if self.is_floating_point() or self.dtype == torch.bool:
            item_calls += 1
        return real_item(self)

    real_run = engine._run_micro_batches
    loss_types: list[Any] = []

    def run_counted(*args, **kwargs):
        with mock.patch.object(torch.Tensor, "item", counting_item):
            output_lst = real_run(*args, **kwargs)
        loss_types.extend(type(o["loss"]) for o in output_lst)
        return output_lst

    engine._run_micro_batches = run_counted
    with (
        mock.patch.object(torchtitan_tpu, "compute_global_batch_num_tokens", return_value=64),
        mock.patch.object(torchtitan_tpu, "prepare_micro_batches", return_value=(_micro_batches(), None)),
        mock.patch.object(torchtitan_tpu, "get_device_name", return_value="cpu"),
        mock.patch.object(torchtitan_tpu, "tpu_eager_mode_context", lambda mode: nullcontext()),
        mock.patch.object(torchtitan_tpu, "tpu_synchronize", lambda *a, **k: None),
        mock.patch.object(torchtitan_tpu, "spmd_no_typecheck_context", nullcontext),
    ):
        out = engine.forward_backward_batch(data, loss_function=loss_function, forward_only=False)
    assert engine._defer_scalar_reads is False  # never leaks out of the loop
    assert weight.grad is not None
    return out, weight.grad, item_calls, loss_types


def test_forward_backward_batch_defer_metric_sync_on_off_parity():
    off, off_grad, off_items, off_types = _run_step(None)
    explicit_off, _, explicit_off_items, _ = _run_step(False)
    on, on_grad, on_items, on_types = _run_step(True)

    n = len(_MICRO_BATCH_LENGTHS)
    # Off (default or explicit): a Python float loss per micro-batch, as before.
    assert off_types == [float] * n and explicit_off_items == off_items
    # On: each micro-batch's loss stays a tensor until after the loop...
    assert on_types == [torch.Tensor] * n
    # ... which removes the per-micro-batch loss .item(); with a lazy-Metric verl every host read leaves the loop.
    assert off_items - on_items == n
    if _LAZY_VERL:
        assert on_items == 0

    # Same step: losses, every metric value and the gradient are identical (same tensors, read later).
    assert on["loss"] == off["loss"] and all(isinstance(v, float) for v in on["loss"])
    assert set(on["metrics"]) == set(off["metrics"])
    for key, off_val in off["metrics"].items():
        on_val = on["metrics"][key]
        if isinstance(off_val, Metric):
            assert on_val.values == off_val.values, key
            assert all(not isinstance(v, torch.Tensor) for v in on_val.values), key
        else:
            assert on_val == off_val, key
    assert torch.equal(on_grad, off_grad)
    for key in ("log_probs",):
        assert torch.equal(on["model_output"][key].values(), off["model_output"][key].values())


@pytest.mark.skipif(not _LAZY_VERL, reason="needs verl with lazy Metric values (verl-project/verl#8776)")
def test_forward_backward_batch_defer_metric_sync_has_no_host_read_in_the_loop():
    _, _, on_items, _ = _run_step(True)
    assert on_items == 0
