# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests (CPU) for the adapter between the TPU engine's outputs and verl's SFT / PPO losses."""

import functools
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import TPU_PADDED_VALUES_ATTR, bucket_length
from verl_hardware_plugin.accelerators.tpu.losses import loss_inputs
from verl_hardware_plugin.accelerators.tpu.losses.loss_inputs import prepare_tpu_loss_inputs


def test_prepare_tpu_loss_inputs_matches_unmodified_sft_loss():
    # 2 packed samples of lengths 3 and 2 (orig_seq_len = 5), padded to bucket 8
    padded_log_probs = torch.tensor(
        [-1.0, -2.0, -3.0, -4.0, -5.0, -100.0, -200.0, -300.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    unpadded_log_probs = padded_log_probs.detach().clone()[:5]
    nested_log_probs = torch.nested.nested_tensor_from_jagged(unpadded_log_probs, cu_seqlens)
    setattr(nested_log_probs, TPU_PADDED_VALUES_ATTR, padded_log_probs)

    loss_mask = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 1]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({"loss_mask": loss_mask}, batch_size=[2])
    tu.assign_non_tensor(
        micro_batch,
        pad_mode=DatasetPadMode.NO_PADDING,
        dp_size=2,
        batch_num_tokens=6,
    )

    model_output = {"log_probs": nested_log_probs}
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = sft_loss(config=None, model_output=loss_model_output, data=loss_data)
    assert metrics == {}
    loss.backward()

    # Rolled unpadded mask is [1, 1, 0, 1, 0], padded with 3 zeros -> [1, 1, 0, 1, 0, 0, 0, 0].
    # Selected log_probs are -1.0, -2.0, -4.0 -> sum = -7.0 -> loss = -(-7.0) / 6 * 2 = 7/3.
    assert loss.item() == pytest.approx(7.0 / 3.0)
    assert padded_log_probs.grad is not None
    expected_grad = torch.tensor([-1.0 / 3.0, -1.0 / 3.0, 0.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(padded_log_probs.grad, expected_grad)


def _ppo_micro_batch(lengths, max_response_len=None, seed=0, response_masks=None):
    """Packed PPO micro-batch as `TorchTitanTPUEngineWithLMHead` builds it: nested prompts/responses on CPU.

    ``response_masks`` (one list per row) marks the valid response tokens; all valid by default.
    """
    gen = torch.Generator().manual_seed(seed)

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    prompt_lens = [p for p, _ in lengths]
    response_lens = [r for _, r in lengths]
    micro_batch = TensorDict(
        {
            "prompts": nested([torch.randint(0, 100, (p,), generator=gen) for p in prompt_lens]),
            "responses": nested([torch.randint(0, 100, (r,), generator=gen) for r in response_lens]),
            "input_ids": nested([torch.randint(0, 100, (p + r,), generator=gen) for p, r in lengths]),
            "response_mask": nested(
                [torch.ones(r, dtype=torch.int64) for r in response_lens]
                if response_masks is None
                else [torch.tensor(m, dtype=torch.int64) for m in response_masks]
            ),
            "old_log_probs": nested([-torch.rand(r, generator=gen) for r in response_lens]),
            "advantages": nested([torch.randn(r, generator=gen) for r in response_lens]),
            "ref_log_prob": nested([-torch.rand(r, generator=gen) for r in response_lens]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(micro_batch, dp_size=2, batch_num_tokens=sum(response_lens) * 2, global_batch_size=8)
    if max_response_len is not None:
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_response_len)
    return micro_batch


def _ppo_model_output(lengths, padded_len, with_entropy, seed=0):
    """Bucket-padded device outputs + their trimmed CPU views, as `prepare_model_outputs` returns them."""
    gen = torch.Generator().manual_seed(seed + 1)
    total = sum(p + r for p, r in lengths)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    leaves = {"log_probs": -torch.rand(padded_len, generator=gen)}
    if with_entropy:
        leaves["entropy"] = torch.rand(padded_len, generator=gen)
    model_output = {}
    for key, leaf in leaves.items():
        leaf.requires_grad_(True)
        nested = torch.nested.nested_tensor_from_jagged(leaf.detach()[:total], cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, leaf)
        model_output[key] = nested
    return model_output, leaves, cu_seqlens


def _metric_value(metric):
    """A loss metric's value: verl ``Metric`` objects aggregate, plain numbers pass through."""
    return metric.aggregate() if hasattr(metric, "aggregate") else float(metric)


def _ppo_config():
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
    )


# Same buckets (response bucket 16, packed bucket 48), different mask contents and numbers of valid tokens.
_MASK_CASES = [
    ([(3, 2), (6, 9), (4, 1)], None),
    ([(3, 2), (6, 9), (4, 1)], [[1, 0], [1, 1, 0, 1, 0, 1, 1, 0, 1], [1]]),
    ([(3, 2), (6, 9), (4, 1)], [[0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1], [1]]),
    ([(5, 7), (9, 11)], [[0, 0, 1, 0, 0, 0, 1], [1] * 11]),
]


@pytest.mark.parametrize(
    "lengths,response_masks", [([(5, 7)], None), ([(3, 2), (6, 9), (4, 1)], None), *_MASK_CASES[1:]]
)
@pytest.mark.parametrize("with_entropy", [False, True])
def test_prepare_tpu_loss_inputs_ppo_matches_unmodified_ppo_loss(lengths, response_masks, with_entropy):
    padded_len = bucket_length(sum(p + r for p, r in lengths), 16)
    response_bucket = bucket_length(max(r for _, r in lengths), 16)

    # Reference: core ppo_loss on exact (unbucketed) packed outputs, as on GPU.
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len, with_entropy)
    total = int(cu_seqlens[-1])
    reference_output = {
        key: torch.nested.nested_tensor_from_jagged(leaf[:total], cu_seqlens) for key, leaf in leaves.items()
    }
    ref_loss, ref_metrics = ppo_loss(
        _ppo_config(), reference_output, _ppo_micro_batch(lengths, response_masks=response_masks)
    )
    ref_loss.backward()
    ref_grads = {key: leaf.grad.clone() for key, leaf in leaves.items()}
    for leaf in leaves.values():
        leaf.grad = None

    # TPU adapter: bucket-padded outputs, static layout, same unmodified core loss.
    micro_batch = _ppo_micro_batch(lengths, max_response_len=response_bucket, response_masks=response_masks)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = ppo_loss(_ppo_config(), loss_model_output, loss_data)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-6)
    for key, leaf in leaves.items():
        torch.testing.assert_close(leaf.grad, ref_grads[key], rtol=0, atol=1e-7)
    assert set(metrics) == set(ref_metrics)
    for key, ref_metric in ref_metrics.items():
        assert _metric_value(metrics[key]) == pytest.approx(_metric_value(ref_metric), rel=1e-6, abs=1e-7), key
    # The source micro-batch keeps its real sequence metadata.
    assert micro_batch["responses"].offsets().diff().tolist() == [r for _, r in lengths]


def test_prepare_tpu_loss_inputs_ppo_shapes_depend_only_on_buckets():
    """Different exact lengths in the same buckets must give the loss identical shapes (no XLA recompiles)."""

    def loss_inputs(lengths):
        model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
        micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
        return prepare_tpu_loss_inputs(model_output, micro_batch)

    out_a, data_a = loss_inputs([(5, 7), (9, 11)])
    out_b, data_b = loss_inputs([(12, 16), (3, 2)])
    for key in ("log_probs", "entropy"):
        assert out_a[key].values().shape == out_b[key].values().shape == torch.Size([2 * (16 + 1)])
    for key in ("prompts", "responses"):
        assert data_a[key].offsets().tolist() == data_b[key].offsets().tolist()
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert data_a[key].shape == data_b[key].shape == torch.Size([2, 16])
    assert tu.get_non_tensor_data(data_a, "max_response_len", None) == 16


def test_prepare_tpu_loss_inputs_ppo_falls_back_to_trimmed_outputs():
    """A packed output without its bucket-padded device tensor keeps the exact-length (trimmed) path."""
    lengths = [(3, 2), (4, 5)]
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len=32, with_entropy=False)
    model_output["values"] = torch.nested.nested_tensor_from_jagged(torch.zeros(int(cu_seqlens[-1])), cu_seqlens)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, _ppo_micro_batch(lengths, 16))
    assert loss_model_output["log_probs"].values().shape == torch.Size([int(cu_seqlens[-1])])
    assert loss_data["responses"].offsets().diff().tolist() == [2, 5]


class _DataDependentShapeGuard(TorchDispatchMode):
    """Fails on ops whose output shape depends on tensor values: boolean-mask indexing, dynamic ``nonzero`` and
    ``masked_select``. On TPU every new output size compiles new programs, once per micro-batch when the size follows
    its valid tokens (verl-project/verl#8663). Integer indexing and gather keep static shapes and stay allowed."""

    _DYNAMIC_OUTPUT = {torch.ops.aten.nonzero.default, torch.ops.aten.masked_select.default}
    _INDEXING = {
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
        torch.ops.aten._index_put_impl_.default,
    }

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in self._DYNAMIC_OUTPUT:
            raise AssertionError(f"data-dependent output shape in the TPU loss path: {func}")
        if func in self._INDEXING:
            indices = args[1] if len(args) > 1 else kwargs.get("indices", ())
            if any(isinstance(i, torch.Tensor) and i.dtype in (torch.bool, torch.uint8) for i in indices or ()):
                raise AssertionError(f"boolean-mask indexing in the TPU loss path: {func}")
        return func(*args, **kwargs)


class _DenseShapeRecorder(TorchDispatchMode):
    """Records each op and the shapes of the dense (non-nested) tensors it returns."""

    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        shapes = tuple(tuple(t.shape) for t in tree_leaves(out) if isinstance(t, torch.Tensor) and not t.is_nested)
        self.records.append((str(func), shapes))
        return out


def _tpu_ppo_loss_inputs(lengths, response_masks):
    model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
    micro_batch = _ppo_micro_batch(lengths, max_response_len=16, response_masks=response_masks)
    return prepare_tpu_loss_inputs(model_output, micro_batch)


@pytest.mark.parametrize("lengths,response_masks", _MASK_CASES)
def test_ppo_loss_on_tpu_inputs_has_no_data_dependent_shapes(lengths, response_masks):
    loss_model_output, loss_data = _tpu_ppo_loss_inputs(lengths, response_masks)
    with _DataDependentShapeGuard():
        ppo_loss(_ppo_config(), loss_model_output, loss_data)


def test_ppo_loss_shapes_on_tpu_inputs_do_not_depend_on_mask_contents():
    """Same buckets, different mask contents and valid-token counts: every dense intermediate keeps its shape."""
    traces = []
    for lengths, response_masks in _MASK_CASES[:3]:
        loss_model_output, loss_data = _tpu_ppo_loss_inputs(lengths, response_masks)
        with _DenseShapeRecorder() as recorder:
            ppo_loss(_ppo_config(), loss_model_output, loss_data)
        traces.append(recorder.records)
    assert traces[0] == traces[1] == traces[2]


def test_core_under_test_emits_the_expected_loss_metrics():
    """The core-SHA CI job names metrics in TPU_LOSS_EXPECT_METRICS so the checks above provably run them."""
    expected = [m for m in os.environ.get("TPU_LOSS_EXPECT_METRICS", "").split(",") if m]
    if not expected:
        pytest.skip("TPU_LOSS_EXPECT_METRICS is not set (the core-SHA CI job sets it)")
    loss_model_output, loss_data = _tpu_ppo_loss_inputs(*_MASK_CASES[1])
    _, metrics = ppo_loss(_ppo_config(), loss_model_output, loss_data)
    missing = [m for m in expected if m not in metrics]
    assert not missing, f"the verl core under test does not emit {missing}"


# Fully masked real row (row 2): bucketing must not change how such rows are counted.
_ROW_CASE = ([(3, 2), (6, 9), (4, 1)], [[1, 0], [1, 1, 0, 1, 0, 1, 1, 0, 1], [0]])
_AGG_MODES = ["token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"]


def _loss_mode_config(loss_mode, loss_agg_mode, rollout_is=None):
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.workers.config.actor import PolicyLossConfig

    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode=loss_agg_mode,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
        policy_loss=PolicyLossConfig(
            loss_mode=loss_mode,
            dro_beta=0.5,
            # Large enough that clip_cov / kl_cov select some tokens in the tiny test batch.
            clip_cov_ratio=0.2,
            kl_cov_ratio=0.2,
            rollout_correction=RolloutCorrectionConfig(rollout_is=rollout_is),
        ),
    )


def _run_ppo_loss(config, padded_num_seqs, lengths=_ROW_CASE[0], response_masks=_ROW_CASE[1], loss_scalars=None):
    """verl's ppo_loss on the TPU loss inputs, optionally row-padded to ``padded_num_seqs``: loss, grads, metrics."""
    model_output, leaves, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
    micro_batch = _ppo_micro_batch(lengths, max_response_len=16, response_masks=response_masks)
    if padded_num_seqs:
        tu.assign_non_tensor_data(micro_batch, "tpu_padded_num_seqs", padded_num_seqs)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch, loss_scalars)
    torch.manual_seed(0)  # clip_cov samples the clipped tokens with randperm
    loss, metrics = ppo_loss(config, loss_model_output, loss_data)
    loss.backward()
    return loss, {k: v.grad for k, v in leaves.items()}, metrics, loss_data


def _registered_policy_losses():
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

    return sorted(POLICY_LOSS_REGISTRY)


def _assert_same_results(a, b):
    loss, grads, metrics, _ = a
    ref_loss, ref_grads, ref_metrics, _ = b
    torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
    for key, grad in ref_grads.items():
        torch.testing.assert_close(grads[key], grad, rtol=1e-6, atol=1e-7)
    assert set(metrics) == set(ref_metrics)
    for key, ref_metric in ref_metrics.items():
        assert _metric_value(metrics[key]) == pytest.approx(_metric_value(ref_metric), rel=1e-6, abs=1e-7), key


def test_every_registered_policy_loss_is_classified_for_row_padding():
    """A policy loss verl adds later must be checked (and listed) before its rows are padded."""
    classified = loss_inputs.ROW_PADDING_INVARIANT_POLICY_LOSSES | loss_inputs.ROW_PADDING_VARIANT_POLICY_LOSSES
    assert not loss_inputs.ROW_PADDING_INVARIANT_POLICY_LOSSES & loss_inputs.ROW_PADDING_VARIANT_POLICY_LOSSES
    assert set(_registered_policy_losses()) <= classified


@pytest.mark.parametrize("loss_agg_mode", _AGG_MODES)
@pytest.mark.parametrize("loss_mode", sorted(loss_inputs.ROW_PADDING_INVARIANT_POLICY_LOSSES))
def test_ppo_loss_modes_bucketed_rows_match_unbucketed(loss_mode, loss_agg_mode):
    """Every policy loss bucketing is enabled for gives the same loss, grads and metrics with 3 rows padded to 8."""
    if loss_mode not in _registered_policy_losses():
        pytest.skip(f"{loss_mode} is not registered by the verl under test")
    config = _loss_mode_config(loss_mode, loss_agg_mode)
    assert loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=config))
    ref = _run_ppo_loss(config, 0)
    out = _run_ppo_loss(config, 8)
    assert ref[3]["response_mask"].shape == torch.Size([3, 16])
    assert out[3]["response_mask"].shape == torch.Size([8, 16])
    assert out[3]["prompts"].offsets().tolist() == list(range(9))
    _assert_same_results(out, ref)


def test_geo_mean_rows_are_not_padded_because_pad_rows_rescale_it():
    """geo_mean averages over every row: pad rows would scale the loss by rows / padded rows, so it keeps its rows."""
    config = _loss_mode_config("geo_mean", "token-mean")
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=config))
    ref_pg_loss = _metric_value(_run_ppo_loss(config, 0)[2]["actor/pg_loss"])
    padded_pg_loss = _metric_value(_run_ppo_loss(config, 4)[2]["actor/pg_loss"])  # if the engine padded its rows
    assert padded_pg_loss == pytest.approx(ref_pg_loss * 3 / 4, rel=1e-5)


@pytest.mark.parametrize("rollout_is", [None, "token", "sequence"])
def test_bypass_mode_rows_are_not_padded_because_rollout_metrics_count_pad_rows(rollout_is):
    """bypass_mode's sequence-level rollout-correction metrics average over every row."""
    config = _loss_mode_config("bypass_mode", "token-mean", rollout_is=rollout_is)
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=config))
    if rollout_is is not None:
        # ``rollout_corr/rollout_is_seq_mean`` averages ``seq_mean_weights`` across all batch rows (including zeroed
        # pad rows) in both released verl and pinned main; ``rollout_corr/training_ppl`` also did so prior to
        # ``_valid_row_mean`` in newer verl main.
        ref_metrics = _run_ppo_loss(config, 0)[2]
        padded_metrics = _run_ppo_loss(config, 4)[2]  # if the engine padded its rows
        name = "rollout_corr/rollout_is_seq_mean"
        assert _metric_value(padded_metrics[name]) == pytest.approx(_metric_value(ref_metrics[name]) * 3 / 4, rel=1e-5)


def test_loss_rows_can_be_padded_only_for_known_row_invariant_ppo_losses():
    vanilla = _loss_mode_config("vanilla", "seq-mean-token-sum-norm")
    assert loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=vanilla))
    assert loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, vanilla))
    # Unknown losses, other loss functions and custom policy losses keep their rows.
    custom = SimpleNamespace(loss_agg_mode="token-mean", policy_loss={"loss_mode": "my_custom_loss"})
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=custom))
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(sft_loss, config=vanilla))
    assert not loss_inputs.loss_rows_can_be_padded(lambda **kwargs: None)
    assert not loss_inputs.loss_rows_can_be_padded(None)


def test_prepare_tpu_loss_inputs_ppo_bucketed_rows_share_shapes_across_batch_sizes():
    """2- and 3-row micro-batches in the same row bucket give the loss identical shapes (no XLA recompiles)."""

    def loss_inputs_for(lengths):
        model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
        micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
        tu.assign_non_tensor_data(micro_batch, "tpu_padded_num_seqs", 4)
        return prepare_tpu_loss_inputs(model_output, micro_batch)

    out_a, data_a = loss_inputs_for([(5, 7), (9, 11)])
    out_b, data_b = loss_inputs_for([(3, 2), (6, 9), (4, 1)])
    for key in ("log_probs", "entropy"):
        assert out_a[key].values().shape == out_b[key].values().shape == torch.Size([4 * (16 + 1)])
    for key in ("prompts", "responses"):
        assert data_a[key].offsets().tolist() == data_b[key].offsets().tolist()
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert data_a[key].shape == data_b[key].shape == torch.Size([4, 16])
    # Scalar host metadata survives on the row-padded loss data.
    assert tu.get_non_tensor_data(data_a, "dp_size", None) == 2
    assert tu.get_non_tensor_data(data_a, "global_batch_size", None) == 8


def test_prepare_tpu_loss_inputs_ppo_pads_every_per_row_field():
    """Row padding keeps every key of the micro-batch (custom losses may read any of them), padded to the new rows."""
    lengths = [(3, 2), (6, 9), (4, 1)]
    model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=False)
    micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
    micro_batch["extra_per_row"] = torch.arange(6.0).reshape(3, 2)
    micro_batch["nested_td"] = TensorDict({"x": torch.ones(3)}, batch_size=[3])
    tu.assign_non_tensor(micro_batch, uid=["a", "b", "c"])
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_num_seqs", 4)
    _, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)

    assert set(micro_batch.keys()) <= set(loss_data.keys())
    assert loss_data.batch_size == torch.Size([4])
    assert torch.equal(loss_data["extra_per_row"], torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [0.0, 0.0]]))
    assert torch.equal(loss_data["nested_td"]["x"], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert loss_data["input_ids"].offsets().diff().tolist() == [5, 15, 5, 0]
    assert torch.equal(loss_data["input_ids"].values(), micro_batch["input_ids"].values())
    assert loss_data.get("uid").tolist() == ["a", "b", "c", None]
    assert tu.get_non_tensor_data(loss_data, "dp_size", None) == 2


def test_pad_tensordict_rows_rejects_unpaddable_fields():
    td = TensorDict({"strided": torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])}, batch_size=[2])
    with pytest.raises(TypeError, match="nested tensor"):
        loss_inputs.pad_tensordict_rows(td, 4)
    same = loss_inputs.pad_tensordict_rows(td, 2)
    assert same.batch_size == torch.Size([2]) and same.device is None


def test_prepare_tpu_loss_inputs_keeps_rows_when_a_field_cannot_be_padded():
    """A field type with no row padding keeps the exact row count (and still gives verl's loss)."""
    lengths = [(3, 2), (6, 9), (4, 1)]
    model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=False)
    micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
    micro_batch["strided"] = torch.nested.nested_tensor([torch.ones(2), torch.ones(3), torch.ones(1)])
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_num_seqs", 4)
    _, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    assert loss_data["response_mask"].shape == torch.Size([3, 16])


def test_device_loss_scalars_is_empty_off_tpu():
    micro_batch = _ppo_micro_batch([(3, 2)])
    with mock.patch.object(loss_inputs, "accelerator_is_tpu", return_value=False):
        assert loss_inputs.device_loss_scalars(micro_batch, "cpu") == {}


def _tpu_loss_scalars(micro_batch):
    """``device_loss_scalars`` as on TPU (the device type mocked, CPU tensors); the device is resolved lazily."""
    device = mock.Mock(return_value=torch.device("cpu"))
    with mock.patch.object(loss_inputs, "accelerator_is_tpu", return_value=True):
        scalars = loss_inputs.device_loss_scalars(micro_batch, device)
    device.assert_called_once_with()
    return scalars


def test_device_loss_scalars_on_tpu_is_a_0d_float32_batch_num_tokens():
    micro_batch = _ppo_micro_batch([(3, 2), (6, 9)])
    scalars = _tpu_loss_scalars(micro_batch)
    assert set(scalars) == {"batch_num_tokens"}
    value = scalars["batch_num_tokens"]
    assert value.shape == torch.Size([]) and value.dtype == torch.float32
    assert float(value) == tu.get_non_tensor_data(micro_batch, "batch_num_tokens", None) == 22


@pytest.mark.parametrize("padded_num_seqs", [0, 8])
def test_ppo_loss_with_device_batch_num_tokens_matches_host_number(padded_num_seqs):
    """The TPU path (0-D batch_num_tokens tensor, with and without row padding) gives the host-number loss."""
    config = _loss_mode_config("vanilla", "token-mean")
    ref = _run_ppo_loss(config, 0)
    scalars = _tpu_loss_scalars(_ppo_micro_batch(_ROW_CASE[0], response_masks=_ROW_CASE[1]))
    out = _run_ppo_loss(config, padded_num_seqs, loss_scalars=scalars)
    assert isinstance(out[3], loss_inputs._TPULossDataProxy)
    assert out[3]["batch_num_tokens"] is scalars["batch_num_tokens"]
    # ppo_loss hands the tensor on to agg_loss through the config's global batch info.
    assert config.global_batch_info["batch_num_tokens"] is scalars["batch_num_tokens"]
    _assert_same_results(out, ref)


def test_device_batch_num_tokens_promotes_a_bf16_masked_sum_to_float32():
    """``masked_sum(bf16) / 0-D float32`` is float32, where ``/ python float`` stayed bf16 (same value, more bits)."""
    from verl.trainer.ppo.core_algos import agg_loss

    loss_mat = torch.tensor([[0.3, -1.7, 2.1]], dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 0]])
    host = agg_loss(loss_mat, mask, "token-mean", dp_size=2, batch_num_tokens=7.0)
    device = agg_loss(loss_mat, mask, "token-mean", dp_size=2, batch_num_tokens=torch.tensor(7.0))
    assert host.dtype == torch.bfloat16 and device.dtype == torch.float32
    torch.testing.assert_close(device, host.float(), rtol=1e-2, atol=1e-3)


def test_rollout_consistency_metrics_with_device_batch_num_tokens():
    """The engine's extra PPO metrics run on the proxied (TPU) loss data and match the host-number path."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import _add_rollout_consistency_metrics

    results = []
    for scalars in ({}, _tpu_loss_scalars(_ppo_micro_batch(_ROW_CASE[0], response_masks=_ROW_CASE[1]))):
        config = _loss_mode_config("vanilla", "token-mean")
        loss_fn = functools.partial(ppo_loss, config=config)
        model_output, _, _ = _ppo_model_output(_ROW_CASE[0], padded_len=48, with_entropy=False)
        micro_batch = _ppo_micro_batch(_ROW_CASE[0], max_response_len=16, response_masks=_ROW_CASE[1])
        tu.assign_non_tensor_data(micro_batch, "tpu_padded_num_seqs", 4)
        loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch, scalars)
        _, metrics = loss_fn(model_output=loss_model_output, data=loss_data)
        _add_rollout_consistency_metrics(loss_fn, loss_model_output, loss_data, metrics)
        results.append(metrics)
    for name in ("actor/ppo_kl_token_mean", "actor/ppo_k3_token_mean", "actor/logprob_gap_abs_token_mean"):
        assert _metric_value(results[1][name]) == pytest.approx(_metric_value(results[0][name]), rel=1e-6), name


def test_sft_and_critic_losses_get_the_device_loss_scalars():
    """The shared loss scalars reach verl's sft_loss and value_loss too (one helper for every loss path)."""
    from verl.workers.config import CriticConfig
    from verl.workers.utils.losses import value_loss

    # SFT: bucket-padded log-probs with a loss mask.
    padded = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, 0.0, 0.0, 0.0])
    nested = torch.nested.nested_tensor_from_jagged(padded[:5], torch.tensor([0, 3, 5]))
    setattr(nested, TPU_PADDED_VALUES_ATTR, padded)
    loss_mask = torch.nested.nested_tensor([torch.tensor([0, 1, 1]), torch.tensor([0, 1])], layout=torch.jagged)
    sft_batch = TensorDict({"loss_mask": loss_mask}, batch_size=[2])
    tu.assign_non_tensor(sft_batch, pad_mode=DatasetPadMode.NO_PADDING, dp_size=2, batch_num_tokens=6)
    scalars = _tpu_loss_scalars(sft_batch)
    out, data = prepare_tpu_loss_inputs({"log_probs": nested}, sft_batch, scalars)
    assert data["batch_num_tokens"] is scalars["batch_num_tokens"]
    loss, _ = sft_loss(config=None, model_output=out, data=data)
    assert loss.dtype == torch.float32 and loss.item() == pytest.approx(7.0 / 3.0)

    # Critic: no log-probs, so only the scalars are overridden.
    lengths = [(3, 2), (4, 3)]
    critic_batch = _ppo_micro_batch(lengths)
    critic_batch["values"] = critic_batch["old_log_probs"]
    critic_batch["returns"] = critic_batch["advantages"]
    cu_seqlens = torch.tensor([0, 5, 12])
    values = torch.nested.nested_tensor_from_jagged(torch.linspace(-1, 1, 12), cu_seqlens)
    config = CriticConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=2, rollout_n=1)
    ref_loss, _ = value_loss(config, {"values": values}, critic_batch)
    out, data = prepare_tpu_loss_inputs({"values": values}, critic_batch, _tpu_loss_scalars(critic_batch))
    assert isinstance(data["batch_num_tokens"], torch.Tensor)
    loss, _ = value_loss(config, out, data)
    torch.testing.assert_close(loss, ref_loss.float(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mrope_planes", [3, 4])
def test_prepare_tpu_loss_inputs_ppo_pads_3d_mrope_jagged_position_ids(mrope_planes):
    """3-D mRoPE ``position_ids`` (``values`` shaped ``(3|4, total_tokens)``, ``jagged_dim=2``) pad cleanly."""
    lengths = [(3, 2), (6, 9), (4, 1)]
    seq_lens = [p + r for p, r in lengths]
    total_tokens = sum(seq_lens)
    offsets = torch.tensor([0, *seq_lens], dtype=torch.int64).cumsum(0)
    mrope_values = torch.arange(mrope_planes * total_tokens, dtype=torch.int64).reshape(mrope_planes, total_tokens)
    mrope_pos = torch.nested.nested_tensor_from_jagged(mrope_values, offsets, jagged_dim=2)
    assert mrope_pos._ragged_idx == 2

    def make_batch(padded_rows):
        mb = _ppo_micro_batch(lengths, max_response_len=16, response_masks=_ROW_CASE[1])
        mb["position_ids"] = mrope_pos
        if padded_rows:
            tu.assign_non_tensor_data(mb, "tpu_padded_num_seqs", padded_rows)
        return mb

    assert loss_inputs.can_pad_tensordict_rows(make_batch(0))
    config = _loss_mode_config("vanilla", "token-mean")

    ref_output, ref_leaves, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
    ref_loss_out, ref_data = prepare_tpu_loss_inputs(ref_output, make_batch(0))
    ref_loss, ref_metrics = ppo_loss(config, ref_loss_out, ref_data)
    ref_loss.backward()

    pad_output, pad_leaves, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
    pad_loss_out, pad_data = prepare_tpu_loss_inputs(pad_output, make_batch(8))
    assert pad_data.batch_size == torch.Size([8])
    padded_pos = pad_data["position_ids"]
    assert padded_pos.layout == torch.jagged and padded_pos._ragged_idx == 2
    assert torch.equal(padded_pos.values(), mrope_values)
    assert [row.shape for row in padded_pos.unbind()] == [
        *(torch.Size([mrope_planes, n]) for n in seq_lens),
        *([torch.Size([mrope_planes, 0])] * 5),
    ]
    pad_loss, pad_metrics = ppo_loss(config, pad_loss_out, pad_data)
    pad_loss.backward()

    _assert_same_results(
        (pad_loss, {k: v.grad for k, v in pad_leaves.items()}, pad_metrics, pad_data),
        (ref_loss, {k: v.grad for k, v in ref_leaves.items()}, ref_metrics, ref_data),
    )


def test_cross_rank_unpaddable_fallback_agreement_and_rate_limited_warning(caplog, monkeypatch):
    """If any rank has an unpaddable field, every rank disables ``tpu_padded_num_seqs`` together (no divergence)."""
    import logging

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_utils

    monkeypatch.setattr(tpu_utils, "_WARNED_UNPADDABLE", set())
    monkeypatch.setattr(loss_inputs, "_WARNED_UNPADDABLE", set())

    # Rank 0 has batch_size == 4 (matching bucket_num_sequences(4) == 4) with a paddable batch;
    # Rank 1 has batch_size == 3 with an unpaddable strided nested tensor.
    # Previously Rank 0 kept tpu_padded_num_seqs (or static_num_seqs=8 -> 8) while Rank 1 caught TypeError -> 3.
    rank0_lengths = [(3, 2), (4, 3), (2, 2), (5, 1)]
    rank1_lengths = [(3, 2), (6, 9), (4, 1)]
    rank0_mbs = [_ppo_micro_batch(rank0_lengths, max_response_len=16)]
    rank1_mb = _ppo_micro_batch(rank1_lengths, max_response_len=16)
    rank1_mb["strided"] = torch.nested.nested_tensor([torch.ones(2), torch.ones(3), torch.ones(1)])
    rank1_mbs = [rank1_mb]

    assert loss_inputs.can_pad_tensordict_rows(rank0_mbs[0])
    assert not loss_inputs.can_pad_tensordict_rows(rank1_mbs[0])

    # Compute each rank's local reduction vector, then take elementwise MAX across both ranks as all_reduce(MAX) does.
    def run_two_rank_align(fn_name):
        captured = {}
        for rank, mbs in ((0, [rank0_mbs[0].copy()]), (1, [rank1_mbs[0].copy()])):
            tu.assign_non_tensor_data(mbs[0], "tpu_padded_num_seqs", 8)  # stale value must also be cleared

            def capture_only(tensor, op=None, rank=rank):
                captured[rank] = tensor.clone()

            with (
                mock.patch.object(torch.distributed, "is_initialized", return_value=True),
                mock.patch.object(torch.distributed, "get_world_size", return_value=2),
                mock.patch.object(torch.distributed, "all_reduce", side_effect=capture_only),
            ):
                if fn_name == "shapes":
                    tpu_utils.align_micro_batch_shapes_across_ranks(
                        mbs, bucket_size=16, num_seqs_bucketing=True, static_num_seqs=8
                    )
                else:
                    tpu_utils.align_micro_batch_num_seqs(mbs, static_num_seqs=8)

        global_max = torch.maximum(captured[0], captured[1])
        results = []
        for mbs in ([rank0_mbs[0].copy()], [rank1_mbs[0].copy()]):
            tu.assign_non_tensor_data(mbs[0], "tpu_padded_num_seqs", 8)
            with (
                mock.patch.object(torch.distributed, "is_initialized", return_value=True),
                mock.patch.object(torch.distributed, "get_world_size", return_value=2),
                mock.patch.object(torch.distributed, "all_reduce", side_effect=lambda t, op=None: t.copy_(global_max)),
            ):
                if fn_name == "shapes":
                    tpu_utils.align_micro_batch_shapes_across_ranks(
                        mbs, bucket_size=16, num_seqs_bucketing=True, static_num_seqs=8
                    )
                else:
                    tpu_utils.align_micro_batch_num_seqs(mbs, static_num_seqs=8)
            results.append(mbs[0])
        return results

    with caplog.at_level(logging.WARNING, logger=tpu_utils.__name__):
        r0_shapes, r1_shapes = run_two_rank_align("shapes")
        r0_seqs, r1_seqs = run_two_rank_align("num_seqs")

    for mb in (r0_shapes, r1_shapes, r0_seqs, r1_seqs):
        assert tu.get_non_tensor_data(mb, "tpu_padded_num_seqs", None) is None
    # Rate-limited: one warning for ("strided", ...) on rank 1 and one for ("<remote_rank>", ...) on rank 0.
    tpu_warnings = [r.getMessage() for r in caplog.records if "Disabling PPO loss row bucketing" in r.getMessage()]
    assert len(tpu_warnings) == 2

    # Both ranks keep their exact batch_size (4 and 3) in prepare_tpu_loss_inputs rather than rank 0 padding to 8.
    out0, _, _ = _ppo_model_output(rank0_lengths, padded_len=48, with_entropy=False)
    out1, _, _ = _ppo_model_output(rank1_lengths, padded_len=48, with_entropy=False)
    _, data0 = prepare_tpu_loss_inputs(out0, r0_shapes)
    _, data1 = prepare_tpu_loss_inputs(out1, r1_shapes)
    assert data0["response_mask"].shape == torch.Size([4, 16])
    assert data1["response_mask"].shape == torch.Size([3, 16])

    # Defense-in-depth fallback in loss_inputs is also rate-limited via _WARNED_UNPADDABLE.
    caplog.clear()
    direct_mb = rank1_mbs[0].copy()
    tu.assign_non_tensor_data(direct_mb, "tpu_padded_num_seqs", 8)
    with caplog.at_level(logging.WARNING, logger=loss_inputs.__name__):
        prepare_tpu_loss_inputs(out1, direct_mb)
        prepare_tpu_loss_inputs(out1, direct_mb)
    fallback_warnings = [r.getMessage() for r in caplog.records if "Not padding the PPO loss rows" in r.getMessage()]
    assert len(fallback_warnings) == 1


def test_device_loss_scalars_transfers_non_blocking():
    """``device_loss_scalars`` passes ``non_blocking=True`` when copying scalars to the accelerator."""
    seen_non_blocking = []
    orig_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        if "device" in kwargs or (args and isinstance(args[0], (torch.device, str))):
            seen_non_blocking.append(kwargs.get("non_blocking", False))
        return orig_to(self, *args, **kwargs)

    for raw_tokens in (22, torch.tensor(22)):
        mb = _ppo_micro_batch([(3, 2)])
        tu.assign_non_tensor_data(mb, "batch_num_tokens", raw_tokens)
        with (
            mock.patch.object(loss_inputs, "accelerator_is_tpu", return_value=True),
            mock.patch.object(torch.Tensor, "to", spy_to),
        ):
            scalars = loss_inputs.device_loss_scalars(mb, torch.device("cpu"))
        assert float(scalars["batch_num_tokens"]) == 22.0
    assert seen_non_blocking and all(seen_non_blocking)


def test_tpu_loss_data_proxy_completeness():
    """``_TPULossDataProxy`` implements ``len``, ``iter``, ``keys``, ``in``, and ``select`` over base + overrides."""
    base = _ppo_micro_batch([(3, 2), (4, 5)], max_response_len=16)
    base["dense_field"] = torch.arange(6.0).reshape(2, 3)
    override_tokens = torch.tensor(42.0)
    extra_scalar = torch.tensor(7.0)
    proxy = loss_inputs._TPULossDataProxy(base, {"batch_num_tokens": override_tokens, "extra_scalar": extra_scalar})

    assert len(proxy) == len(base) == 2
    expected_keys = [*base.keys(), "extra_scalar"]
    assert list(proxy) == expected_keys
    assert list(proxy.keys()) == expected_keys
    assert "batch_num_tokens" in proxy
    assert "extra_scalar" in proxy
    assert "prompts" in proxy
    assert "nonexistent_key" not in proxy
    assert proxy["batch_num_tokens"] is override_tokens
    assert proxy["extra_scalar"] is extra_scalar
    assert proxy.select("dense_field")[:1].batch_size == torch.Size([1])

    selected = proxy.select("prompts", "batch_num_tokens", "extra_scalar")
    assert isinstance(selected, loss_inputs._TPULossDataProxy)
    assert len(selected) == 2
    assert list(selected.keys()) == ["prompts", "batch_num_tokens", "extra_scalar"]
    assert selected["batch_num_tokens"] is override_tokens
    assert selected["extra_scalar"] is extra_scalar
    assert torch.equal(selected["prompts"].values(), base["prompts"].values())

    selected_non_strict = proxy.select("prompts", "missing_key", strict=False)
    assert list(selected_non_strict.keys()) == ["prompts"]
    with pytest.raises(KeyError):
        proxy.select("prompts", "missing_key")
    with pytest.raises(ValueError, match="read-only"):
        proxy.select("prompts", inplace=True)


@pytest.mark.parametrize("raw_zero", [0, 0.0, torch.tensor(0), torch.tensor(0.0)])
def test_zero_batch_num_tokens_clamped_produces_finite_zero_losses_and_grads(raw_zero):
    """When ``batch_num_tokens == 0`` globally, clamping to ``1.0`` gives finite ``0.0`` loss/grads (no ``NaN``)."""
    from verl.workers.config import CriticConfig
    from verl.workers.utils.losses import value_loss
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import _add_rollout_consistency_metrics

    lengths = [(3, 2), (4, 3)]
    zero_masks = [[0, 0], [0, 0, 0]]

    # 1. PPO (pg + kl + entropy terms) + rollout consistency metrics
    ppo_mb = _ppo_micro_batch(lengths, max_response_len=16, response_masks=zero_masks)
    tu.assign_non_tensor_data(ppo_mb, "batch_num_tokens", raw_zero)
    scalars = _tpu_loss_scalars(ppo_mb)
    assert float(scalars["batch_num_tokens"]) == 1.0
    model_output, leaves, _ = _ppo_model_output(lengths, padded_len=32, with_entropy=True)
    config = _loss_mode_config("vanilla", "token-mean")
    loss_fn = functools.partial(ppo_loss, config=config)
    loss_out, loss_data = prepare_tpu_loss_inputs(model_output, ppo_mb, scalars)
    loss, metrics = loss_fn(model_output=loss_out, data=loss_data)
    _add_rollout_consistency_metrics(loss_fn, loss_out, loss_data, metrics)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() == 0.0
    for leaf in leaves.values():
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all() and leaf.grad.abs().max().item() == 0.0
    for name in ("actor/ppo_kl_token_mean", "actor/ppo_k3_token_mean", "actor/logprob_gap_abs_token_mean"):
        val = _metric_value(metrics[name])
        assert val == 0.0

    # 2. SFT
    sft_leaf = torch.tensor([-1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    sft_nested = torch.nested.nested_tensor_from_jagged(sft_leaf.detach()[:4], torch.tensor([0, 2, 4]))
    setattr(sft_nested, TPU_PADDED_VALUES_ATTR, sft_leaf)
    sft_mask = torch.nested.nested_tensor([torch.zeros(2, dtype=torch.int64)] * 2, layout=torch.jagged)
    sft_mb = TensorDict({"loss_mask": sft_mask}, batch_size=[2])
    tu.assign_non_tensor(sft_mb, pad_mode=DatasetPadMode.NO_PADDING, dp_size=2, batch_num_tokens=raw_zero)
    sft_out, sft_data = prepare_tpu_loss_inputs({"log_probs": sft_nested}, sft_mb, _tpu_loss_scalars(sft_mb))
    sft_l, _ = sft_loss(config=None, model_output=sft_out, data=sft_data)
    sft_l.backward()
    assert torch.isfinite(sft_l) and sft_l.item() == 0.0
    assert sft_leaf.grad is not None and torch.isfinite(sft_leaf.grad).all() and sft_leaf.grad.abs().max().item() == 0.0

    # 3. Critic value_loss
    critic_mb = _ppo_micro_batch(lengths, response_masks=zero_masks)
    tu.assign_non_tensor_data(critic_mb, "batch_num_tokens", raw_zero)
    critic_mb["values"] = critic_mb["old_log_probs"]
    critic_mb["returns"] = critic_mb["advantages"]
    val_leaf = torch.linspace(-1, 1, 12, requires_grad=True)
    values = torch.nested.nested_tensor_from_jagged(val_leaf, torch.tensor([0, 5, 12]))
    critic_cfg = CriticConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=2, rollout_n=1)
    c_out, c_data = prepare_tpu_loss_inputs({"values": values}, critic_mb, _tpu_loss_scalars(critic_mb))
    c_loss, _ = value_loss(critic_cfg, c_out, c_data)
    c_loss.backward()
    assert torch.isfinite(c_loss) and c_loss.item() == 0.0
    assert val_leaf.grad is not None and torch.isfinite(val_leaf.grad).all() and val_leaf.grad.abs().max().item() == 0.0


def test_loss_rows_can_be_padded_defensive_config_shapes():
    """Mappings, dataclasses/objects without ``.get()``, ``policy_loss=None``, and incomplete configs never crash."""
    import dataclasses

    @dataclasses.dataclass
    class _PolicyCfg:
        loss_mode: str = "sapo"

    @dataclasses.dataclass
    class _ActorCfg:
        loss_agg_mode: str = "token-mean"
        policy_loss: object = dataclasses.field(default_factory=_PolicyCfg)

    assert loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=_ActorCfg()))
    assert loss_inputs.loss_rows_can_be_padded(
        functools.partial(ppo_loss, config={"loss_agg_mode": "token-mean", "policy_loss": {"loss_mode": "gspo"}})
    )
    assert loss_inputs.loss_rows_can_be_padded(
        functools.partial(ppo_loss, config={"loss_agg_mode": "token-mean", "policy_loss": None})
    )
    assert not loss_inputs.loss_rows_can_be_padded(
        functools.partial(ppo_loss, config={"loss_agg_mode": "token-mean", "policy_loss": _PolicyCfg("geo_mean")})
    )
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config={}))
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=SimpleNamespace()))
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=object()))
    assert not loss_inputs.loss_rows_can_be_padded(functools.partial(ppo_loss, config=None))
