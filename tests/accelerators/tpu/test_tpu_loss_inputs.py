# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests (CPU) for the adapter between the TPU engine's outputs and verl's SFT / PPO losses."""

import math

import pytest
import torch
from tensordict import TensorDict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakTensorKeyDictionary

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_NUM_SEQS_KEY,
    TPU_PADDED_VALUES_ATTR,
    bucket_length,
)
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


def _ppo_micro_batch(lengths, max_response_len=None, seed=0):
    """Packed PPO micro-batch as `TorchTitanTPUEngineWithLMHead` builds it: nested prompts/responses on CPU."""
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
            "response_mask": nested([torch.ones(r, dtype=torch.int64) for r in response_lens]),
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


@pytest.mark.parametrize("lengths", [[(5, 7)], [(3, 2), (6, 9), (4, 1)]])
@pytest.mark.parametrize("with_entropy", [False, True])
def test_prepare_tpu_loss_inputs_ppo_matches_unmodified_ppo_loss(lengths, with_entropy):
    padded_len = bucket_length(sum(p + r for p, r in lengths), 16)
    response_bucket = bucket_length(max(r for _, r in lengths), 16)

    # Reference: core ppo_loss on exact (unbucketed) packed outputs, as on GPU.
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len, with_entropy)
    total = int(cu_seqlens[-1])
    reference_output = {
        key: torch.nested.nested_tensor_from_jagged(leaf[:total], cu_seqlens) for key, leaf in leaves.items()
    }
    ref_loss, ref_metrics = ppo_loss(_ppo_config(), reference_output, _ppo_micro_batch(lengths))
    ref_loss.backward()
    ref_grads = {key: leaf.grad.clone() for key, leaf in leaves.items()}
    for leaf in leaves.values():
        leaf.grad = None

    # TPU adapter: bucket-padded outputs, static layout, same unmodified core loss.
    micro_batch = _ppo_micro_batch(lengths, max_response_len=response_bucket)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = ppo_loss(_ppo_config(), loss_model_output, loss_data)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-6)
    for key, leaf in leaves.items():
        torch.testing.assert_close(leaf.grad, ref_grads[key], rtol=0, atol=1e-7)
    assert set(metrics) == set(ref_metrics)
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


# ---------------------------------------------------------------------------------------------------------------------
# static_micro_batch_seqs: PPO loss inputs padded to at least N rows (``tpu_padded_num_seqs``), verl's unmodified loss
# ---------------------------------------------------------------------------------------------------------------------

LENGTHS = [(5, 7), (3, 2), (6, 9), (4, 1), (7, 13)]  # (prompt, response) tokens of the packed sequences


def _rows_case(lengths, padded_len, seed=0, dp_size=2, zero_mask_row=None):
    """A packed PPO micro-batch and its bucket-padded device leaves (log-probs, entropy).

    The response log-probs stay within 0.4 of the old ones, so the PPO ratios vary around 1 and some are clipped.
    With ``dp_size > 1`` the micro-batch carries global token / sequence counts, as verl sets them under data
    parallelism; with ``dp_size == 1`` the loss normalizes by the micro-batch's own counts.
    """
    gen = torch.Generator().manual_seed(seed)

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    masks = [torch.ones(r, dtype=torch.int64) for _, r in lengths]
    if zero_mask_row is not None:
        masks[zero_mask_row] = torch.zeros_like(masks[zero_mask_row])
    old_log_probs = [-0.5 - torch.rand(r, generator=gen) for _, r in lengths]
    micro_batch = TensorDict(
        {
            "prompts": nested([torch.randint(0, 100, (p,), generator=gen) for p, _ in lengths]),
            "responses": nested([torch.randint(0, 100, (r,), generator=gen) for _, r in lengths]),
            "input_ids": nested([torch.randint(0, 100, (p + r,), generator=gen) for p, r in lengths]),
            "response_mask": nested(masks),
            "old_log_probs": nested(old_log_probs),
            "advantages": nested([torch.randn(r, generator=gen) for _, r in lengths]),
            "ref_log_prob": nested([-0.5 - torch.rand(r, generator=gen) for _, r in lengths]),
        },
        batch_size=[len(lengths)],
    )
    num_tokens = sum(int(mask.sum()) for mask in masks)
    if dp_size > 1:
        tu.assign_non_tensor(micro_batch, dp_size=dp_size, batch_num_tokens=3 * num_tokens, global_batch_size=48)
    else:
        tu.assign_non_tensor(micro_batch, dp_size=1, batch_num_tokens=None, global_batch_size=None)

    log_probs = -0.5 - torch.rand(padded_len, generator=gen)
    offset = 0
    for (p, r), old in zip(lengths, old_log_probs, strict=True):
        log_probs[offset + p - 1 : offset + p - 1 + r] = old + 0.8 * (torch.rand(r, generator=gen) - 0.5)
        offset += p + r
    leaves = {"log_probs": log_probs, "entropy": torch.rand(padded_len, generator=gen)}
    for leaf in leaves.values():
        leaf.requires_grad_(True)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    return micro_batch, leaves, cu_seqlens


def _rows_model_output(leaves, cu_seqlens):
    """The engine's model outputs: trimmed host views that carry their bucket-padded device leaves."""
    total = int(cu_seqlens[-1])
    model_output = {}
    for key, leaf in leaves.items():
        nested = torch.nested.nested_tensor_from_jagged(leaf.detach()[:total].clone(), cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, leaf)
        model_output[key] = nested
    return model_output


def _rows_loss_inputs(micro_batch, model_output, num_rows=0, width=None):
    """``prepare_tpu_loss_inputs`` as the engine calls it, optionally with ``tpu_padded_num_seqs`` (N) set."""
    response_lens = micro_batch["responses"].offsets().diff()
    tu.assign_non_tensor_data(micro_batch, "max_response_len", bucket_length(int(response_lens.max()), 16))
    if width:
        tu.assign_non_tensor_data(micro_batch, "tpu_max_response_len", width)
    if num_rows:
        tu.assign_non_tensor_data(micro_batch, TPU_PADDED_NUM_SEQS_KEY, num_rows)
    return prepare_tpu_loss_inputs(model_output, micro_batch)


def _plain_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if hasattr(value, "aggregate"):
            value = value.aggregate()
        out[key] = float(value.item() if hasattr(value, "item") else value)
    return out


def _run_rows(config, lengths=LENGTHS, padded_len=96, *, num_rows=0, width=None, exact=False, **case_kwargs):
    """Loss, gradients of the device leaves, metrics and loss inputs of one micro-batch."""
    micro_batch, leaves, cu_seqlens = _rows_case(lengths, padded_len, **case_kwargs)
    if exact:  # verl's own path on exact (unbucketed) packed outputs, as on GPU
        total = int(cu_seqlens[-1])
        loss_output = {k: torch.nested.nested_tensor_from_jagged(v[:total], cu_seqlens) for k, v in leaves.items()}
        loss_data = micro_batch
    else:
        loss_output, loss_data = _rows_loss_inputs(micro_batch, _rows_model_output(leaves, cu_seqlens), num_rows, width)
    loss, metrics = ppo_loss(config, loss_output, loss_data)
    loss.backward()
    grads = {key: leaf.grad.detach().clone() for key, leaf in leaves.items()}
    return loss.detach(), grads, _plain_metrics(metrics), loss_output, loss_data


def _rows_config(loss_agg_mode="token-mean", **kwargs):
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode=loss_agg_mode,
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
        **kwargs,
    )


def _bypass_config():
    """Rollout correction in bypass mode with token-level IS weights, as common for TPU rollouts."""
    correction = RolloutCorrectionConfig(
        rollout_is="token",
        rollout_is_threshold=3.0,
        rollout_is_batch_normalize=False,
        rollout_rs=None,
        bypass_mode=True,
        loss_type="ppo_clip",
    )
    return ActorConfig(
        strategy="fsdp",
        rollout_n=16,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=1.0e9,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=correction),
    )


def _assert_same_loss(a, b, metric_keys=None):
    torch.testing.assert_close(a[0], b[0], rtol=1e-6, atol=1e-6)
    assert set(a[1]) == set(b[1])
    for key in a[1]:
        torch.testing.assert_close(a[1][key], b[1][key], rtol=0, atol=1e-7)
    assert set(a[2]) == set(b[2])
    for key in a[2] if metric_keys is None else metric_keys:
        assert math.isclose(a[2][key], b[2][key], rel_tol=1e-5, abs_tol=1e-6), (key, a[2][key], b[2][key])


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize(
    "loss_agg_mode, kwargs",
    [
        ("token-mean", {}),
        ("token-sum", {}),
        ("seq-mean-token-sum", {}),
        ("seq-mean-token-mean", {}),
        ("seq-mean-token-sum-norm", {"loss_scale_factor": 64}),
    ],
)
def test_static_micro_batch_seqs_rows_give_the_same_loss_gradients_and_metrics(loss_agg_mode, kwargs, dp_size):
    """Empty padding rows (N=12 for 5 sequences, with a wider response bucket) change nothing the loss computes."""
    config = _rows_config(loss_agg_mode, **kwargs)
    reference = _run_rows(config, dp_size=dp_size)
    padded = _run_rows(config, num_rows=12, width=32, dp_size=dp_size)
    assert padded[4]["response_mask"].shape == (12, 32)
    _assert_same_loss(reference, padded)
    assert padded[2]["actor/pg_clipfrac"] > 0  # the clipped branch of the PPO loss is exercised
    _assert_same_loss(_run_rows(config, exact=True, dp_size=dp_size), padded)


def test_static_micro_batch_seqs_rows_with_rollout_correction():
    """Bypass-mode rollout correction: the same loss and gradients; only rollout-correction diagnostics may differ.

    Per-sequence rollout-correction statistics that average over every loss row also count the padding rows: the
    sequence-level IS statistics, and in verl 0.9.1 also the off-policy perplexity and chi-square metrics. The loss,
    the gradients and every other metric are unchanged.
    """
    reference = _run_rows(_bypass_config())
    padded = _run_rows(_bypass_config(), num_rows=12, width=32)
    assert any(key.startswith("rollout_corr/") for key in padded[2])
    _assert_same_loss(
        reference, padded, metric_keys=[key for key in reference[2] if not key.startswith("rollout_corr/")]
    )
    differing = {
        key for key in reference[2] if not math.isclose(reference[2][key], padded[2][key], rel_tol=1e-5, abs_tol=1e-6)
    }
    assert all(key.startswith("rollout_corr/") for key in differing), sorted(differing)


def test_static_micro_batch_seqs_shapes_do_not_depend_on_the_sequence_count():
    a = _run_rows(_rows_config(), [(5, 7), (9, 11)], 64, num_rows=8, width=32, seed=1)
    b = _run_rows(_rows_config(), [(12, 16), (3, 2), (4, 4), (2, 30), (6, 1)], 96, num_rows=8, width=32, seed=2)
    for key in ("log_probs", "entropy"):
        assert a[3][key].values().shape == b[3][key].values().shape == torch.Size([8 * (32 + 1)])
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert tuple(a[4][key].shape) == tuple(b[4][key].shape) == (8, 32)
    for key in ("prompts", "responses"):
        assert a[4][key].offsets().tolist() == b[4][key].offsets().tolist()
    assert tu.get_non_tensor_data(a[4], "max_response_len", None) == 32
    assert tu.get_non_tensor_data(a[4], "dp_size", None) == 2  # scalar metadata kept
    assert "input_ids" not in a[4].keys()  # per-sequence entries left out


class _DeviceOpTrace(TorchDispatchMode):
    """Records the ops on (CPU stand-ins for) device tensors, with their input and output shapes.

    A tensor is a device tensor if the test marks it, or if an op with a device input produced it; ops on host tensors
    only (e.g. the gather indices that ``prepare_tpu_loss_inputs`` builds on CPU) are not recorded. Ops whose output
    size depends on tensor values (boolean-mask indexing, ``nonzero``), and every op computed from such an output, go
    to ``data_dependent`` instead of ``static``.
    """

    def __init__(self):
        super().__init__()
        self.device = WeakTensorKeyDictionary()  # device tensor -> derived from a data-dependent size
        self.static = []
        self.data_dependent = []

    def mark(self, *tensors):
        for tensor in tensors:
            self.device[tensor] = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        inputs = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        if not any(t in self.device for t in inputs):
            return out
        name = str(func)
        dependent = any(self.device.get(t, False) for t in inputs)
        dependent |= name.startswith(("aten.nonzero.", "aten.masked_select."))
        dependent |= name.startswith("aten.index.") and any(t.dtype == torch.bool for t in inputs[1:])
        outputs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        for tensor in outputs:
            self.device[tensor] = dependent
        record = (name, [tuple(t.shape) for t in inputs], [tuple(t.shape) for t in outputs])
        (self.data_dependent if dependent else self.static).append(record)
        return out


def _trace_loss_and_backward(config, lengths, num_rows, seed):
    """Device-op trace of prepare_tpu_loss_inputs + verl's ppo_loss + backward for one micro-batch."""
    micro_batch, leaves, cu_seqlens = _rows_case(lengths, padded_len=96, seed=seed)
    model_output = _rows_model_output(leaves, cu_seqlens)
    with _DeviceOpTrace() as trace:
        trace.mark(*leaves.values())
        loss_output, loss_data = _rows_loss_inputs(micro_batch, model_output, num_rows, width=32)
        # The PPO fields are built on the host and moved to the device (a no-op on CPU): mark them as device tensors.
        trace.mark(*(loss_data[key] for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob")))
        loss, _ = ppo_loss(config, loss_output, loss_data)
        loss.backward()
    return trace


def test_static_micro_batch_seqs_device_ops_do_not_depend_on_the_sequence_count():
    """Micro-batches with different sequence and valid-token counts give the same device ops and shapes with N set.

    This covers the forward loss, its metrics and the backward. An op whose output size depends on tensor values
    (verl versions whose ppo_loss counts ``actor/old_logprobs_mismatch_count`` with a boolean mask have one) cannot be
    made static by row padding; such ops must stay off the autograd graph, and are compared separately.
    """
    config = _rows_config()
    a = _trace_loss_and_backward(config, [(5, 7), (9, 11), (4, 3)], num_rows=8, seed=1)
    b = _trace_loss_and_backward(config, [(12, 16), (3, 2), (4, 4), (2, 30), (6, 1)], num_rows=8, seed=2)
    assert len(a.static) > 50  # the trace sees the loss, the metrics and the backward
    assert a.static == b.static
    assert [record[0] for record in a.data_dependent] == [record[0] for record in b.data_dependent]
    # Without N, the row count of the loss inputs follows the sequence count.
    a = _trace_loss_and_backward(config, [(5, 7), (9, 11), (4, 3)], num_rows=0, seed=1)
    b = _trace_loss_and_backward(config, [(12, 16), (3, 2), (4, 4), (2, 30), (6, 1)], num_rows=0, seed=2)
    assert a.static != b.static


def test_static_micro_batch_seqs_below_the_sequence_count_keeps_the_rows():
    """N below the sequence count keeps the exact rows; N equal to it changes nothing."""
    reference = _run_rows(_rows_config())
    for num_rows in (3, len(LENGTHS)):
        result = _run_rows(_rows_config(), num_rows=num_rows)
        assert result[4]["response_mask"].shape[0] == len(LENGTHS)
        _assert_same_loss(reference, result)


def test_static_micro_batch_seqs_pads_around_a_sequence_without_valid_tokens():
    """A real sequence whose response mask is all zero is padded like any other, with the same loss and gradients."""
    reference = _run_rows(_rows_config(), zero_mask_row=1)
    padded = _run_rows(_rows_config(), num_rows=12, width=32, zero_mask_row=1)
    assert padded[4]["response_mask"].shape == (12, 32)
    _assert_same_loss(reference, padded)


def test_static_micro_batch_seqs_leaves_sft_and_trimmed_outputs_alone():
    """Only the static PPO layout reads ``tpu_padded_num_seqs``: SFT and the trimmed fallback keep their inputs."""
    padded_log_probs = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -100.0, -200.0, -300.0], requires_grad=True)
    cu_seqlens = torch.tensor([0, 3, 5])
    nested_log_probs = torch.nested.nested_tensor_from_jagged(padded_log_probs.detach()[:5], cu_seqlens)
    setattr(nested_log_probs, TPU_PADDED_VALUES_ATTR, padded_log_probs)
    loss_mask = torch.nested.nested_tensor([torch.tensor([0, 1, 1]), torch.tensor([0, 1])], layout=torch.jagged)
    micro_batch = TensorDict({"loss_mask": loss_mask}, batch_size=[2])
    tu.assign_non_tensor(micro_batch, pad_mode=DatasetPadMode.NO_PADDING, dp_size=2, batch_num_tokens=6)
    tu.assign_non_tensor_data(micro_batch, TPU_PADDED_NUM_SEQS_KEY, 8)
    loss_model_output, loss_data = prepare_tpu_loss_inputs({"log_probs": nested_log_probs}, micro_batch)
    loss, _ = sft_loss(config=None, model_output=loss_model_output, data=loss_data)
    assert loss.item() == pytest.approx(7.0 / 3.0)

    # A packed output without its bucket-padded device tensor: the trimmed path, exact rows.
    lengths = [(3, 2), (4, 5)]
    model_output, _, cu_seqlens = _ppo_model_output(lengths, padded_len=32, with_entropy=False)
    model_output["values"] = torch.nested.nested_tensor_from_jagged(torch.zeros(int(cu_seqlens[-1])), cu_seqlens)
    micro_batch = _ppo_micro_batch(lengths, 16)
    tu.assign_non_tensor_data(micro_batch, TPU_PADDED_NUM_SEQS_KEY, 8)
    _, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    assert loss_data["response_mask"].shape[0] == len(lengths)
