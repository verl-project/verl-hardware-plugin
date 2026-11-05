# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for fixed-shape micro-batches (``engines/tpu_fixed_shapes.py``) and their PPO loss inputs."""

import contextlib
import json
import logging
import math
import random
import threading
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig
from verl.workers.config.actor import PolicyLossConfig
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu import options as tpu_options
from verl_hardware_plugin.accelerators.tpu.engines import tpu_fixed_shapes as fs
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    bucket_length,
    pad_packed_inputs_for_tpu,
)
from verl_hardware_plugin.accelerators.tpu.losses.loss_inputs import prepare_tpu_loss_inputs
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options, parse_fixed_shape_rows

ROWS = "96:160:256"  # packed-row lengths of the engine-level tests (sequence bucket 32)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for env_name in tpu_options._ENGINE_LEGACY_ENV_VARS.values():
        monkeypatch.delenv(env_name, raising=False)


# ---------------------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------------------


def _engine_config(rows=ROWS, max_seqs=8, response_len=48, **extra):
    tpu = {"seq_bucket_size": 32, **extra}
    if rows is not None:
        tpu.update(fixed_shape_rows=rows, fixed_shape_max_seqs=max_seqs, fixed_shape_response_len=response_len)
    return SimpleNamespace(engine_kwargs={"tpu": tpu})


def test_fixed_shape_rows_are_parsed():
    assert parse_fixed_shape_rows("16384:8192:12288") == (8192, 12288, 16384)
    assert parse_fixed_shape_rows("8192, 16384") == (8192, 16384)
    assert parse_fixed_shape_rows("[8192, 16384, 8192]") == (8192, 16384)
    assert parse_fixed_shape_rows([16384, 8192]) == (8192, 16384)
    assert parse_fixed_shape_rows(16384) == (16384,)
    for bad in ("", "0:8192", "8192:abc", "8192.5", [True], [8192.0], None):
        with pytest.raises(ValueError):
            parse_fixed_shape_rows(bad)


def test_fixed_shape_options_are_off_by_default():
    options = get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"seq_bucket_size": 32}}))
    assert (
        options.fixed_shape_rows,
        options.fixed_shape_max_seqs,
        options.fixed_shape_response_len,
        options.fixed_shape_prewarm,
    ) == (None, None, None, None)
    assert fs.FixedShapeConfig.from_engine_config(SimpleNamespace(engine_kwargs={})) is None


def test_fixed_shape_options_are_typed_and_validated():
    options = get_tpu_engine_options(_engine_config(rows="256,96, 160", max_seqs="8", fixed_shape_prewarm="false"))
    assert options.fixed_shape_rows == "96:160:256"
    assert options.fixed_shape_max_seqs == 8
    assert options.fixed_shape_response_len == 48
    assert options.fixed_shape_prewarm is False
    with pytest.raises(ValueError, match="fixed_shape_response_len is required"):
        get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"fixed_shape_rows": ROWS}}))
    with pytest.raises(ValueError, match="need fixed_shape_rows"):
        get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"fixed_shape_max_seqs": 8}}))
    with pytest.raises(ValueError):
        get_tpu_engine_options(_engine_config(max_seqs=True))
    with pytest.raises(ValueError):
        get_tpu_engine_options(_engine_config(rows="96:-160"))
    with pytest.raises(ValueError, match="Unknown"):
        get_tpu_engine_options(_engine_config(fixed_shape_typo=1))


def test_fixed_shape_config_from_engine_config():
    config = fs.FixedShapeConfig.from_engine_config(_engine_config())
    assert (config.rows, config.max_seqs, config.response_len, config.prewarm) == ((96, 160, 256), 8, 64, True)
    assert config.costs == {row: fs.default_cost(row) for row in config.rows}
    assert config.cost_source == "default"
    assert fs.FixedShapeConfig.from_engine_config(_engine_config(max_seqs=None)).max_seqs == fs.DEFAULT_MAX_SEQS
    with pytest.raises(ValueError, match="not multiples of the sequence bucket"):
        fs.FixedShapeConfig.from_engine_config(_engine_config(rows="100:256"))


# ---------------------------------------------------------------------------------------------------------------------
# Layout planner
# ---------------------------------------------------------------------------------------------------------------------


def _lockstep_plans(rank_lens, rows, max_seqs, costs):
    """``plan_layout`` on one thread per rank, with an in-process element-wise MAX all-reduce."""
    n = len(rank_lens)
    barrier, slots, reduced, plans, errors = threading.Barrier(n), [None] * n, [None], [None] * n, []

    def make_allreduce(rank):
        def allreduce_max(values):
            slots[rank] = list(values)
            if barrier.wait() == 0:
                reduced[0] = [max(column) for column in zip(*slots, strict=True)]
            barrier.wait()
            out = list(reduced[0])
            barrier.wait()
            return out

        return allreduce_max

    def run(rank):
        try:
            plans[rank] = fs.plan_layout(rank_lens[rank], rows, max_seqs, costs, make_allreduce(rank))
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return plans


def _check_plan(lens, plan, max_seqs):
    assert sorted(i for bins in plan.bins for i in bins) == list(range(len(lens)))
    assert len(plan.bins) == len(plan.caps)
    assert list(plan.caps) == sorted(plan.caps, reverse=True)
    for cap, bins in zip(plan.caps, plan.bins, strict=True):
        assert 1 <= len(bins) <= max_seqs
        assert sum(lens[i] for i in bins) <= cap


def _random_lens(rng, n, cap=4352):
    """Prompt + response lengths of RL rollouts: short prompts, mostly short responses, some at the maximum."""
    lens = []
    for _ in range(n):
        prompt = rng.randint(60, 240)
        response = 4096 if rng.random() < 0.12 else min(4096, int(rng.expovariate(1 / 700)) + 1)
        lens.append(min(cap, prompt + response))
    return lens


# Synthetic per-row costs, super-linear in the row length: two short rows can be cheaper than one long row.
COSTS = {8192: 1.0, 12288: 1.75, 14336: 2.25, 16384: 3.0}


def test_pack_invariants_on_random_inputs():
    rng = random.Random(0)
    packed = 0
    for _ in range(300):
        n = rng.randint(1, 80)
        lens = _random_lens(rng, n)
        caps = sorted((rng.choice([4352, 8192, 12288, 16384]) for _ in range(rng.randint(1, min(n, 12)))), reverse=True)
        max_seqs = rng.choice([8, 16, 32, 64])
        bins = fs.pack(lens, caps, max_seqs)
        if bins is not None:
            packed += 1
            _check_plan(lens, fs.Plan(tuple(caps), bins, 0.0, 0, 0), max_seqs)
    assert packed > 50


def test_pack_rejects_impossible_layouts():
    assert fs.pack([5000], [4096], 4) is None  # too long
    assert fs.pack([10, 10, 10], [100], 2) is None  # more than max_seqs
    assert fs.pack([10], [100, 100], 4) is None  # fewer sequences than micro-batches
    assert sorted(fs.pack([10, 10], [100, 100], 4)) == [[0], [1]]  # no micro-batch of padding only


def test_plan_layout_agrees_across_ranks_and_is_cheapest():
    rng = random.Random(1)
    rows = tuple(COSTS)
    for _ in range(25):
        rank_lens = [_random_lens(rng, 64) for _ in range(rng.choice([2, 4, 8]))]
        max_seqs = rng.choice([32, 64])
        plans = _lockstep_plans(rank_lens, rows, max_seqs, COSTS)
        assert len({plan.caps for plan in plans}) == 1
        for lens, plan in zip(rank_lens, plans, strict=True):
            _check_plan(lens, plan, max_seqs)
        # No strictly cheaper candidate is packable on every rank.
        total_max, item_max = max(map(sum, rank_lens)), max(max(lens) for lens in rank_lens)
        chosen = sum(COSTS[cap] for cap in plans[0].caps)
        for caps in fs.enumerate_layouts(total_max, 64, 64, item_max, rows, max_seqs, COSTS):
            if sum(COSTS[cap] for cap in caps) >= chosen - 1e-12:
                break
            assert not all(fs.pack(lens, caps, max_seqs) is not None for lens in rank_lens), caps


def test_plan_layout_uses_the_cost_table():
    lens = [4000] * 4  # one 16384-token row or two 8192-token rows
    rows = (8192, 16384)
    assert fs.plan_layout(lens, rows, 8, {row: fs.default_cost(row) for row in rows}, list).caps == (16384,)
    assert fs.plan_layout(lens, rows, 8, {8192: 1.0, 16384: 3.0}, list).caps == (8192, 8192)
    assert fs.plan_layout(lens, rows, 8, {8192: 1.0, 16384: 1.5}, list).caps == (16384,)
    assert fs.plan_layout(lens, rows, 2, {8192: 1.0, 16384: 1.5}, list).caps == (8192, 8192)  # max_seqs


def test_plan_layout_errors():
    with pytest.raises(ValueError, match="does not fit"):
        fs.plan_layout([20000], (8192, 16384), 8, {8192: 1.0, 16384: 2.0}, list)
    # One rank needs three micro-batches, the other has two sequences for them.
    with pytest.raises(RuntimeError, match="no fixed-shape layout fits"):
        _lockstep_plans([[200, 200, 200], [10, 10]], (256,), 8, {256: 1.0})


def _gloo_plan_worker(rank, world, init_file, out_dir):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        lens = _random_lens(random.Random(100 + rank), 64)
        if rank == world - 1:
            lens[:10] = [4352] * 10  # a heavier rank
        rows = (8192, 12288, 16384)
        plan = fs.plan_layout(lens, rows, 32, {row: COSTS[row] for row in rows}, fs.allreduce_max)
        _check_plan(lens, plan, 32)
        result = {"caps": list(plan.caps), "floats": fs.allreduce_max([float(rank), 2.5])}
        with open(f"{out_dir}/{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_plan_layout_agrees_across_gloo_processes(tmp_path):
    world = 4
    torch.multiprocessing.start_processes(
        _gloo_plan_worker,
        args=(world, str(tmp_path / "pg"), str(tmp_path)),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    results = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(world)]
    assert len({tuple(r["caps"]) for r in results}) == 1
    assert all(r["floats"] == [3.0, 2.5] for r in results)


# ---------------------------------------------------------------------------------------------------------------------
# PPO loss inputs with fixed rows (verl's unmodified ppo_loss)
# ---------------------------------------------------------------------------------------------------------------------


def _nested(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _vanilla_config():
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
    )


def _bypass_config():
    """Off-policy correction in bypass mode with token-level IS weights, as common for TPU rollouts."""
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


def _ppo_micro_batch(lengths, seed=0, zero_mask_row=None):
    gen = torch.Generator().manual_seed(seed)
    masks = [torch.ones(r, dtype=torch.int64) for _, r in lengths]
    if zero_mask_row is not None:
        masks[zero_mask_row] = torch.zeros_like(masks[zero_mask_row])
    micro_batch = TensorDict(
        {
            "prompts": _nested([torch.randint(0, 100, (p,), generator=gen) for p, _ in lengths]),
            "responses": _nested([torch.randint(0, 100, (r,), generator=gen) for _, r in lengths]),
            "input_ids": _nested([torch.randint(0, 100, (p + r,), generator=gen) for p, r in lengths]),
            "response_mask": _nested(masks),
            "old_log_probs": _nested([-torch.rand(r, generator=gen) for _, r in lengths]),
            "advantages": _nested([torch.randn(r, generator=gen) for _, r in lengths]),
            "ref_log_prob": _nested([-torch.rand(r, generator=gen) for _, r in lengths]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(micro_batch, dp_size=2, batch_num_tokens=sum(r for _, r in lengths) * 2, global_batch_size=64)
    return micro_batch


def _ppo_model_output(lengths, padded_len, old_log_probs, seed=0):
    """Bucket-padded log-probs and entropy (the device leaves) with their trimmed CPU views.

    The log-probs of response tokens stay within 0.15 of the old ones, so ratios and IS weights vary around 1.
    """
    gen = torch.Generator().manual_seed(seed + 1)
    total = sum(p + r for p, r in lengths)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    log_probs = -torch.rand(padded_len, generator=gen)
    offset = 0
    for (p, r), old in zip(lengths, old_log_probs, strict=True):
        log_probs[offset + p - 1 : offset + p - 1 + r] = old + 0.3 * (torch.rand(r, generator=gen) - 0.5)
        offset += p + r
    leaves = {"log_probs": log_probs, "entropy": torch.rand(padded_len, generator=gen)}
    model_output = {}
    for key, leaf in leaves.items():
        leaf.requires_grad_(True)
        nested = torch.nested.nested_tensor_from_jagged(leaf.detach()[:total], cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, leaf)
        model_output[key] = nested
    return model_output, leaves, cu_seqlens


def _plain_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if hasattr(value, "aggregate"):
            value = value.aggregate()
        out[key] = float(value.item() if hasattr(value, "item") else value)
    return out


def _run_loss(config, lengths, padded_len, *, num_rows=0, width=None, seed=0, zero_mask_row=None, exact=False):
    """Loss, gradients of the device leaves, metrics and loss inputs of one micro-batch."""
    micro_batch = _ppo_micro_batch(lengths, seed=seed, zero_mask_row=zero_mask_row)
    model_output, leaves, cu_seqlens = _ppo_model_output(
        lengths, padded_len, micro_batch["old_log_probs"].unbind(), seed
    )
    if exact:  # verl's own path on exact (unbucketed) packed outputs, as on GPU
        total = int(cu_seqlens[-1])
        loss_output = {k: torch.nested.nested_tensor_from_jagged(v[:total], cu_seqlens) for k, v in leaves.items()}
        loss_data = micro_batch
    else:
        tu.assign_non_tensor_data(micro_batch, "max_response_len", bucket_length(max(r for _, r in lengths), 16))
        if num_rows:
            tu.assign_non_tensor_data(micro_batch, fs.TPU_PADDED_NUM_SEQS_KEY, num_rows)
            tu.assign_non_tensor_data(micro_batch, "tpu_max_response_len", width)
        loss_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = ppo_loss(config, loss_output, loss_data)
    loss.backward()
    grads = {key: leaf.grad.detach().clone() for key, leaf in leaves.items()}
    return loss.detach(), grads, _plain_metrics(metrics), loss_output, loss_data


LENGTHS = [(5, 7), (3, 2), (6, 9), (4, 1), (7, 13)]


def _assert_same_loss(a, b, metrics_to_compare=None):
    torch.testing.assert_close(a[0], b[0], rtol=1e-6, atol=1e-6)
    assert set(a[1]) == set(b[1])
    for key in a[1]:
        torch.testing.assert_close(a[1][key], b[1][key], rtol=0, atol=1e-7)
    assert set(a[2]) == set(b[2])
    for key in metrics_to_compare if metrics_to_compare is not None else a[2]:
        assert math.isclose(a[2][key], b[2][key], rel_tol=1e-5, abs_tol=1e-6), (key, a[2][key], b[2][key])


def test_fixed_rows_give_the_same_loss_gradients_and_metrics():
    reference = _run_loss(_vanilla_config(), LENGTHS, 96)
    fixed = _run_loss(_vanilla_config(), LENGTHS, 96, num_rows=12, width=32)  # 7 padding rows, wider responses
    _assert_same_loss(reference, fixed)
    exact = _run_loss(_vanilla_config(), LENGTHS, 96, exact=True)
    _assert_same_loss(exact, fixed)


def test_fixed_rows_with_rollout_correction_give_the_same_loss_and_gradients():
    reference = _run_loss(_bypass_config(), LENGTHS, 96)
    fixed = _run_loss(_bypass_config(), LENGTHS, 96, num_rows=12, width=32)
    assert any(key.startswith("rollout_corr/") for key in fixed[2])
    # Per-sequence rollout-correction statistics average over every loss row unless verl skips rows without valid
    # tokens; every other metric, the loss and the gradients do not depend on the padding rows.
    _assert_same_loss(reference, fixed, [key for key in reference[2] if not key.startswith("rollout_corr/")])


def test_fixed_rows_shapes_do_not_depend_on_the_micro_batch():
    a = _run_loss(_vanilla_config(), [(5, 7), (9, 11)], 64, num_rows=8, width=32, seed=1)
    b = _run_loss(_vanilla_config(), [(12, 16), (3, 2), (4, 4), (2, 30), (6, 1)], 96, num_rows=8, width=32, seed=2)
    for key in ("log_probs", "entropy"):
        assert a[3][key].values().shape == b[3][key].values().shape == torch.Size([8 * (32 + 1)])
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert tuple(a[4][key].shape) == tuple(b[4][key].shape) == (8, 32)
    for key in ("prompts", "responses"):
        assert a[4][key].offsets().tolist() == b[4][key].offsets().tolist()
    assert tu.get_non_tensor_data(a[4], "max_response_len", None) == 32
    assert tu.get_non_tensor_data(a[4], "dp_size", None) == 2  # scalar metadata kept
    assert "input_ids" not in a[4].keys()  # per-sequence entries left out


def test_fixed_rows_keep_the_exact_layout_for_a_sequence_without_valid_tokens():
    result = _run_loss(_vanilla_config(), LENGTHS, 96, num_rows=12, width=32, zero_mask_row=1)
    assert result[4]["response_mask"].shape[0] == len(LENGTHS)


def test_fixed_rows_smaller_than_the_micro_batch_are_rejected():
    with pytest.raises(ValueError, match="smaller than the micro-batch"):
        _run_loss(_vanilla_config(), LENGTHS, 96, num_rows=3, width=32)


# ---------------------------------------------------------------------------------------------------------------------
# Partition, padding and restore
# ---------------------------------------------------------------------------------------------------------------------


def _mini_batch(n=24, seed=0, max_prompt=20, max_response=40, dtype=torch.float32):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    lengths = [
        (rng.randint(2, max_prompt), max_response if rng.random() < 0.15 else rng.randint(1, max_response))
        for _ in range(n)
    ]
    data = TensorDict(
        {
            "input_ids": _nested([torch.randint(1, 50, (p + r,), generator=gen) for p, r in lengths]),
            "position_ids": _nested([torch.arange(p + r) for p, r in lengths]),
            "prompts": _nested([torch.randint(1, 50, (p,), generator=gen) for p, _ in lengths]),
            "responses": _nested([torch.randint(1, 50, (r,), generator=gen) for _, r in lengths]),
            "response_mask": _nested([torch.ones(r, dtype=torch.int64) for _, r in lengths]),
            "loss_mask": _nested(
                [torch.cat([torch.zeros(p, dtype=torch.int64), torch.ones(r, dtype=torch.int64)]) for p, r in lengths]
            ),
            "old_log_probs": _nested([(-0.5 - torch.rand(r, generator=gen)).to(dtype) for _, r in lengths]),
            "advantages": _nested([torch.randn(r, generator=gen).to(dtype) for _, r in lengths]),
        },
        batch_size=[n],
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=True,
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        max_token_len_per_gpu=256,
        dp_size=1,
        global_batch_size=n,
        batch_num_tokens=sum(r for _, r in lengths),
        temperature=1.0,
    )
    return data, lengths


def _config(prewarm=False):
    config = fs.FixedShapeConfig.from_engine_config(_engine_config(fixed_shape_prewarm=prewarm))
    assert config is not None
    return config


def test_build_micro_batches_tags_pads_and_restores():
    config = _config()
    data, lengths = _mini_batch()
    assert fs.active_config(config, data, SimpleNamespace(cp_enabled=False, pp_enabled=False)) is config
    micro_batches, indices = fs.build_micro_batches(data, config)
    assert sorted(i for bins in indices for i in bins) == list(range(len(lengths)))
    assert fs.row_lengths(micro_batches) == [
        tu.get_non_tensor_data(mb, "tpu_padded_seq_len", 0) for mb in micro_batches
    ]
    outputs = []
    for micro_batch, bins in zip(micro_batches, indices, strict=True):
        row = tu.get_non_tensor_data(micro_batch, "tpu_padded_seq_len", None)
        assert row in config.rows
        assert tu.get_non_tensor_data(micro_batch, "tpu_max_response_len", None) == 64
        assert tu.get_non_tensor_data(micro_batch, fs.TPU_PADDED_NUM_SEQS_KEY, None) == 8
        assert 1 <= micro_batch.batch_size[0] <= 8
        for vocab_size in (None, 248320):
            ids, positions, labels, orig = pad_packed_inputs_for_tpu(
                micro_batch["input_ids"], micro_batch["position_ids"], micro_batch, "cpu", 32, vocab_size=vocab_size
            )
            assert ids.shape == positions.shape == labels.shape == (row,)
            assert orig == sum(p + r for p, r in (lengths[i] for i in bins))
            assert not positions[orig:].any() and not labels[orig:].any()
        assert tu.get_non_tensor_data(micro_batch, "max_response_len", None) == 64
        # A per-token output equal to the token id, to check the restored order.
        log_probs = torch.nested.nested_tensor_from_jagged(
            micro_batch["input_ids"].values().float(), micro_batch["input_ids"].offsets()
        )
        outputs.append({"model_output": {"log_probs": log_probs}, "loss": 0.5, "metrics": {}})
    output = postprocess_batch_func(outputs, indices, data)
    for got, want in zip(output["model_output"]["log_probs"].unbind(), data["input_ids"].unbind(), strict=True):
        assert torch.equal(got, want.float())
    fs.sum_losses_on_host(output)
    assert output["loss"] == [0.5 * len(micro_batches)]


def test_sum_losses_on_host():
    output = {"loss": [0.1] * 10}
    fs.sum_losses_on_host(output)
    assert output["loss"] == [math.fsum([0.1] * 10)]
    single = {"loss": [2.0]}
    fs.sum_losses_on_host(single)
    assert single["loss"] == [2.0]


def test_inapplicable_batches_keep_verls_partition(caplog):
    config = _config()
    parallel = SimpleNamespace(cp_enabled=False, pp_enabled=False)
    data, _ = _mini_batch()
    tu.assign_non_tensor(data, use_dynamic_bsz=False)
    with caplog.at_level(logging.WARNING, logger=fs.logger.name):
        assert fs.active_config(config, data, parallel) is None
        assert fs.active_config(config, data, parallel) is None
    assert sum("do not apply" in record.getMessage() for record in caplog.records) == 1
    data, _ = _mini_batch()
    assert fs.active_config(config, data, SimpleNamespace(cp_enabled=True, pp_enabled=False)) is None
    assert fs.active_config(None, data, parallel) is None
    assert fs.inapplicable_reasons(data, parallel) == []


# ---------------------------------------------------------------------------------------------------------------------
# One training step of a toy model: verl's partition vs fixed shapes
# ---------------------------------------------------------------------------------------------------------------------


class _Toy:
    """Log-prob of the label at packed position t: -softplus(theta[label_t] + w * position_t / 64).

    It depends only on the token and its position in its document, so every partition gives the same per-token values.
    """

    def __init__(self, dtype):
        gen = torch.Generator().manual_seed(7)
        self.theta = (0.3 * torch.randn(64, generator=gen)).to(dtype).requires_grad_(True)
        self.w = torch.tensor(0.2, dtype=dtype, requires_grad=True)

    def log_probs(self, labels, positions):
        return -torch.nn.functional.softplus(self.theta[labels] + self.w * positions.to(self.theta.dtype) / 64)

    def forward_step(self, micro_batch, loss_function, calls):
        """The engine's per-micro-batch path on CPU: pad, "model", trimmed view with the padded leaf, loss inputs."""
        ids, positions, labels, orig = pad_packed_inputs_for_tpu(
            micro_batch["input_ids"], micro_batch["position_ids"], micro_batch, "cpu", 32
        )
        padded = self.log_probs(labels, positions)
        offsets = micro_batch["input_ids"].offsets()
        log_probs = torch.nested.nested_tensor_from_jagged(padded.detach()[:orig], offsets)
        setattr(log_probs, TPU_PADDED_VALUES_ATTR, padded)
        loss_output, loss_data = prepare_tpu_loss_inputs({"log_probs": log_probs}, micro_batch)
        calls.append((int(ids.shape[0]), tuple(loss_data["advantages"].shape), int(micro_batch.batch_size[0])))
        loss, metrics = loss_function(model_output=loss_output, data=loss_data)
        model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(padded.detach()[:orig].clone(), offsets)}
        return loss, {"model_output": model_output, "loss": float(loss.detach()), "metrics": metrics}

    def zero_grad(self):
        self.theta.grad = self.w.grad = None


def _loss_function(model_output, data, dp_group=None):
    return ppo_loss(_bypass_config(), model_output, data)


@pytest.mark.parametrize("dtype,rtol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_training_step_matches_verls_partition(dtype, rtol):
    data, _ = _mini_batch(n=24, seed=3, dtype=dtype)
    results = {}
    for fixed in (False, True):
        toy, calls = _Toy(dtype), []
        if fixed:
            micro_batches, _ = fs.build_micro_batches(data, _config())
        else:
            micro_batches, _ = prepare_micro_batches(data=data, dp_group=None, same_micro_num_in_dp=True)
        total = 0.0
        for micro_batch in micro_batches:
            loss, _ = toy.forward_step(micro_batch, _loss_function, calls)
            loss.backward()
            total += float(loss.detach())
        results[fixed] = (total, toy.theta.grad.clone(), toy.w.grad.clone(), calls)
    (loss_v, theta_v, w_v, _), (loss_f, theta_f, w_f, calls_f) = results[False], results[True]
    assert math.isclose(loss_v, loss_f, rel_tol=rtol, abs_tol=rtol)
    torch.testing.assert_close(theta_f, theta_v, rtol=rtol, atol=rtol * float(theta_v.abs().max()))
    torch.testing.assert_close(w_f, w_v, rtol=rtol, atol=rtol * float(w_v.abs()))
    assert {call[0] for call in calls_f} <= {96, 160, 256}  # rows from the fixed set
    assert {call[1] for call in calls_f} == {(8, 64)}  # loss rows and width fixed


# ---------------------------------------------------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------------------------------------------------


class _FakeClock:
    """A clock that only moves when the test advances it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_prewarm_compiles_times_every_row_and_discards_gradients():
    toy, calls, clock = _Toy(torch.float32), [], _FakeClock()
    config = _config(prewarm=True)
    data, _ = _mini_batch(n=24, seed=5)

    def run_micro_batch(micro_batch):
        loss, _ = toy.forward_step(micro_batch, _loss_function, calls)
        loss.backward()
        # A micro-batch takes (row length / 1000) s; the first run of each row compiles and takes 1 s more.
        compiling = (len(calls) - 1) % (1 + fs.PREWARM_TIMED_RUNS) == 0
        clock.now += calls[-1][0] / 1000 + (1.0 if compiling else 0.0)

    fs.prewarm(data, config, run_micro_batch, toy.zero_grad, clock=clock)
    assert len(calls) == (1 + fs.PREWARM_TIMED_RUNS) * len(config.rows)
    assert [call[0] for call in calls[:: 1 + fs.PREWARM_TIMED_RUNS]] == sorted(config.rows, reverse=True)
    assert all(call[1] == (config.max_seqs, config.response_len) and call[2] <= config.max_seqs for call in calls)
    assert config.cost_source == "measured"
    assert config.costs == pytest.approx({row: row / 1000 for row in config.rows})
    assert toy.theta.grad is None and toy.w.grad is None
    fs.prewarm(data, config, run_micro_batch, toy.zero_grad, clock=clock)  # once per engine
    assert len(calls) == (1 + fs.PREWARM_TIMED_RUNS) * len(config.rows)


def test_prewarm_disabled_or_unfit_rows_keep_the_default_costs():
    calls = []
    config = _config(prewarm=False)
    data, _ = _mini_batch(n=24, seed=5)
    fs.prewarm(data, config, calls.append, lambda: calls.append("zero_grad"))
    assert calls == [] and config.prewarmed and config.cost_source == "default"

    config = _config(prewarm=True)
    tiny = min(fs.sequence_lengths(data)) - 1
    config.rows = (tiny, *config.rows)  # no sequence fits this row: it is skipped
    config.costs = {row: fs.default_cost(row) for row in config.rows}
    fs.prewarm(data, config, calls.append, lambda: calls.append("zero_grad"))
    assert {fs.row_lengths([mb])[0] for mb in calls[:-1]} == {96, 160, 256}
    assert calls[-1] == "zero_grad"
    assert config.cost_source == "default"


# ---------------------------------------------------------------------------------------------------------------------
# TorchTitanTPUEngine.forward_backward_batch on CPU
# ---------------------------------------------------------------------------------------------------------------------


class TokamaxGatedDeltaKernel(torch.nn.Module):
    """Stand-in for torchtitan's Tokamax GatedDeltaNet kernel module (matched by class name)."""

    def __init__(self):
        super().__init__()
        self.max_num_segments = 1


@pytest.fixture
def torchtitan_tpu(monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    # Nothing to synchronize on a CPU host (and torch_tpu, when installed, needs a TPU for it).
    monkeypatch.setattr(torchtitan_tpu, "tpu_synchronize", lambda *args, **kwargs: None)
    return torchtitan_tpu


def _engine(torchtitan_tpu, fixed, toy, calls, kernel):
    engine = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        engine_kwargs=_engine_config(rows=ROWS if fixed else None).engine_kwargs,
        tensor_parallel_size=1,
        context_parallel_size=1,
        data_parallel_shard_size=1,
        data_parallel_replicate_size=1,
    )
    engine._fixed_shapes = fs.FixedShapeConfig.from_engine_config(engine.engine_config)
    engine.parallel_dims = SimpleNamespace(cp_enabled=False, pp_enabled=False)
    engine._use_simple_fsdp, engine._tpu_eager_mode, engine._simple_fsdp_loss_scale = True, None, None
    engine._tpu_kernels = frozenset({"kda"})
    engine.module = [torch.nn.ModuleList([kernel])]
    engine.trainer = SimpleNamespace(train_context=contextlib.nullcontext)
    engine.get_data_parallel_group = lambda: None

    def forward_step(micro_batch, loss_function, forward_only):
        loss, output = toy.forward_step(micro_batch, loss_function, calls)
        calls[-1] = (*calls[-1], kernel.max_num_segments)
        return loss, output

    engine.forward_step = forward_step
    engine.optimizer_zero_grad = toy.zero_grad
    return engine


def test_engine_forward_backward_batch_fixed_shapes_vs_verls_partition(torchtitan_tpu):
    results = {}
    for fixed in (False, True):
        toy, calls, kernel = _Toy(torch.float64), [], TokamaxGatedDeltaKernel()
        data, _ = _mini_batch(n=24, seed=11, dtype=torch.float64)
        engine = _engine(torchtitan_tpu, fixed, toy, calls, kernel)
        output = engine.forward_backward_batch(data, _loss_function, forward_only=False)
        results[fixed] = (output, toy.theta.grad.clone(), toy.w.grad.clone(), calls)
    (out_v, theta_v, w_v, calls_v), (out_f, theta_f, w_f, calls_f) = results[False], results[True]

    assert len(out_v["loss"]) > 1  # verl's partition: one loss per micro-batch
    assert len(out_f["loss"]) == 1  # fixed shapes: summed on the host
    assert math.isclose(sum(out_v["loss"]), out_f["loss"][0], rel_tol=1e-12, abs_tol=1e-12)
    torch.testing.assert_close(theta_f, theta_v, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(w_f, w_v, rtol=1e-12, atol=1e-14)
    for a, b in zip(
        out_v["model_output"]["log_probs"].unbind(), out_f["model_output"]["log_probs"].unbind(), strict=True
    ):
        # Same input order and values; a sequence's last position predicts the next packed document's first token,
        # which the loss never reads.
        assert a.shape == b.shape
        torch.testing.assert_close(a[:-1], b[:-1], rtol=0, atol=0)

    prewarm_calls = (1 + fs.PREWARM_TIMED_RUNS) * 3
    assert len(calls_f) > prewarm_calls
    assert {call[0] for call in calls_f} <= {96, 160, 256}
    assert {call[1] for call in calls_f} == {(8, 64)}
    # Tokamax document budget: constant max_seqs + 1 from the pre-warm on; verl's partition: largest micro-batch + 1.
    assert {call[3] for call in calls_f} == {8 + 1}
    assert {call[3] for call in calls_v} == {max(call[2] for call in calls_v) + 1}


def test_engine_forward_only_uses_fixed_shapes_without_prewarm(torchtitan_tpu):
    toy, calls, kernel = _Toy(torch.float64), [], TokamaxGatedDeltaKernel()
    data, _ = _mini_batch(n=24, seed=12, dtype=torch.float64)
    engine = _engine(torchtitan_tpu, True, toy, calls, kernel)
    with torch.no_grad():
        output = engine.forward_backward_batch(data, _loss_function, forward_only=True)
    assert not engine._fixed_shapes.prewarmed
    assert len(output["loss"]) == 1
    assert len(calls) > 1
    assert {call[0] for call in calls} <= {96, 160, 256}
    assert {call[1] for call in calls} == {(8, 64)}
