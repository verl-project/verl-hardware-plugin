# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for global bin packing across DP ranks (``VERL_TPU_GLOBAL_PACK=1``).

Covers the planner (``verl.utils.global_pack``), its agreement with the engine's per-rank packing
(``pack_micro_batches_into_bins``), the fill logging, and loss/gradient equivalence of multi-segment static rows
(laid out by ``bin_slot_permutation`` with position-derived document ids) versus unpacked sequences on a tiny model.
"""

from __future__ import annotations

import math
import random
import sys

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    bin_slot_permutation,
    log_micro_batch_fill,
    pack_micro_batches_into_bins,
)

try:  # The planner ships with verl (verl.utils.global_pack); older verl releases do not have it.
    from verl.utils.global_pack import (
        bin_lower_bound,
        ffd_pack_bins,
        get_global_pack_partitions,
        global_pack_partitions_from_env,
        segment_slot_len,
    )

    HAS_GLOBAL_PACK = True
except ImportError:
    HAS_GLOBAL_PACK = False

requires_global_pack = pytest.mark.skipif(not HAS_GLOBAL_PACK, reason="verl without verl.utils.global_pack")

BIN = 65536


def _bench397_lengths(seed: int, num_trajs: int = 1024, max_response_len: int = 57344) -> list[int]:
    """``verl.trainer.bench_35b_trainer`` lengths for ``BENCH_DIST=397b`` (prompt + response, 64K model length)."""
    g = torch.Generator().manual_seed(seed)
    prompt_lens = torch.randint(2160, 2400 + 1, (num_trajs,), generator=g).tolist()
    out = []
    for i, p_len in enumerate(prompt_lens):
        cap = min(max_response_len, BIN - p_len)
        rem = i % 20
        if rem < 8:
            lo, hi = min(50000, cap), min(62000, cap)
        elif rem < 14:
            lo, hi = min(40000, cap), min(50000, cap)
        elif rem < 18:
            lo, hi = min(28000, cap), min(40000, cap)
        else:
            lo, hi = min(12000, cap), min(28000, cap)
        r_len = int(torch.randint(lo, max(lo, hi) + 1, (1,), generator=g).item())
        out.append(p_len + min(r_len, cap))
    return out


def _gamma_lengths(seed: int, n: int = 1024, mean: float = 33000.0) -> list[int]:
    torch.manual_seed(seed)
    dist = torch.distributions.Gamma(torch.tensor(2.0), torch.tensor(2.0 / mean))
    return [int(x) for x in dist.sample((n,)).clamp(1500, BIN).tolist()]


def _nested_batch(lengths: list[int]) -> TensorDict:
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    tokens = torch.arange(int(offsets[-1]), dtype=torch.int64)
    return TensorDict(
        {"input_ids": torch.nested.nested_tensor_from_jagged(tokens, offsets)},
        batch_size=[len(lengths)],
    )


def _per_rank_ffd_max(lengths: list[int], dp: int, bin_len: int) -> int:
    k = len(lengths) // dp
    return max(len(ffd_pack_bins(lengths[r * k : (r + 1) * k], bin_len)) for r in range(dp))


def _check_plan(lengths, partitions, dp, bin_len, align, max_segments=0):
    n = len(lengths)
    assert len(partitions) == dp
    flat = [i for p in partitions for i in p]
    assert sorted(flat) == list(range(n)), "every sequence must be assigned exactly once"
    assert {len(p) for p in partitions} == {n // dp}, "equal-size dispatch chunks"
    costs = []
    for p in partitions:
        bins = ffd_pack_bins([lengths[i] for i in p], bin_len, align=align, max_segments=max_segments)
        for b in bins:
            assert sum(segment_slot_len(lengths[p[i]], bin_len, align) for i in b) <= bin_len
            if max_segments:
                assert len(b) <= max_segments
        costs.append(len(bins))
    return costs


# --------------------------------------------------------------------------------------------- planner


@requires_global_pack
def test_ffd_pack_bins_matches_legacy_engine_ffd():
    """align=1 / no segment cap is exactly the engine's historical per-rank FFD."""
    rng = random.Random(0)
    for _ in range(50):
        lengths = [rng.randint(1, 900) for _ in range(rng.randint(1, 30))]
        _, mb_indices = pack_micro_batches_into_bins(_nested_batch(lengths), dp_group=None, bin_len=1024, num_bins=1)
        assert len(mb_indices) == len(ffd_pack_bins(lengths, 1024))


@requires_global_pack
def test_ffd_alignment_and_segment_cap():
    assert segment_slot_len(1, 64, 64) == 64
    assert segment_slot_len(65, 256, 64) == 128
    assert segment_slot_len(250, 256, 64) == 256
    with pytest.raises(ValueError):
        segment_slot_len(257, 256, 64)
    # 4 x 60 fits one 256 bin unaligned, but 64-aligned (4 x 64) still fits; 5 x 50 needs 2 bins aligned.
    assert len(ffd_pack_bins([60] * 4, 256, align=64)) == 1
    assert len(ffd_pack_bins([50] * 5, 256, align=1)) == 1
    assert len(ffd_pack_bins([50] * 5, 256, align=64)) == 2
    assert len(ffd_pack_bins([10] * 8, 256, max_segments=3)) == 3


@requires_global_pack
@pytest.mark.parametrize("seed", [1001, 1002, 1003])
def test_bench397_distribution_reaches_lower_bound(seed):
    """BENCH_DIST=397b, 1024 trajectories, CP=2 on 256 devices -> 128 DP ranks, 65536-token bins."""
    dp = 128
    lengths = _bench397_lengths(seed)
    partitions, stats = get_global_pack_partitions(lengths, dp, BIN, align=64)
    costs = _check_plan(lengths, partitions, dp, BIN, align=64)
    per_rank = _per_rank_ffd_max(lengths, dp, BIN)
    lb = sum(stats.lower_bound_per_rank)
    token_bound = math.ceil(sum(lengths) / (dp * BIN))
    print(
        f"seed={seed} tokens={sum(lengths)} per-rank FFD={per_rank} global={max(costs)} lower_bound={lb} "
        f"token_bound={token_bound} fill={stats.fill_ratio:.3f}"
    )
    assert max(costs) == stats.micro_batches
    assert stats.micro_batches == lb  # optimal: > half of the trajectories exceed 32K, so 7 bins/rank is a hard floor
    assert stats.micro_batches < per_rank
    # The engine reproduces the plan: per-rank FFD of each reordered dispatch chunk.
    reordered = [lengths[j] for p in partitions for j in p]
    assert _per_rank_ffd_max(reordered, dp, BIN) == max(_check_plan(lengths, partitions, dp, BIN, align=1))


@requires_global_pack
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_33k_mean_distribution_near_token_bound(seed):
    """A ~33K-mean mix (~33M tokens/step): global packing reaches ceil(tokens / (dp * 64K))."""
    dp = 128
    lengths = _gamma_lengths(seed)
    partitions, stats = get_global_pack_partitions(lengths, dp, BIN, align=64)
    costs = _check_plan(lengths, partitions, dp, BIN, align=64)
    token_bound = math.ceil(sum(lengths) / (dp * BIN))
    print(
        f"seed={seed} tokens={sum(lengths)} per-rank FFD={_per_rank_ffd_max(lengths, dp, BIN)} global={max(costs)} "
        f"token_bound={token_bound} fill={stats.fill_ratio:.3f}"
    )
    assert max(costs) <= token_bound + 1
    assert max(costs) <= _per_rank_ffd_max(lengths, dp, BIN)


@requires_global_pack
@pytest.mark.parametrize("dp,n,bin_len,max_segments", [(4, 32, 1024, 0), (8, 64, 2048, 3), (3, 27, 512, 0)])
def test_random_small_plans_are_valid(dp, n, bin_len, max_segments):
    rng = random.Random(dp * 1000 + n)
    for _ in range(20):
        lengths = [rng.randint(1, bin_len) for _ in range(n)]
        partitions, stats = get_global_pack_partitions(lengths, dp, bin_len, align=8, max_segments=max_segments)
        costs = _check_plan(lengths, partitions, dp, bin_len, align=8, max_segments=max_segments)
        slots = [segment_slot_len(x, bin_len, 8) for x in lengths]
        assert max(costs) >= -(-bin_lower_bound(slots, bin_len, max_segments) // dp)
        assert max(costs) == stats.micro_batches


@requires_global_pack
def test_mini_batches_are_planned_separately():
    """Each rank's m-th mini-batch (contiguous slice of its chunk, no shuffle) is its share of mini-batch m."""
    dp, m = 4, 2
    rng = random.Random(5)
    lengths = [rng.randint(100, 4000) for _ in range(64)]
    partitions, stats = get_global_pack_partitions(lengths, dp, 4096, align=64, num_mini_batches=m)
    per_rank = len(lengths) // dp
    per_mb = per_rank // m
    for p in partitions:
        assert len(p) == per_rank
        for mb in range(m):
            chunk = p[mb * per_mb : (mb + 1) * per_mb]
            assert all(mb * 32 <= i < (mb + 1) * 32 for i in chunk)
    assert len(stats.max_bins_per_rank) == m


@requires_global_pack
def test_plan_rejects_uneven_batches_and_oversized_sequences():
    with pytest.raises(ValueError):
        get_global_pack_partitions([10] * 10, 4, 64)
    with pytest.raises(ValueError):
        get_global_pack_partitions([100] * 4, 4, 64)


@requires_global_pack
def test_env_settings(monkeypatch):
    monkeypatch.delenv("VERL_TPU_GLOBAL_PACK_BIN_LEN", raising=False)
    monkeypatch.delenv("VERL_TPU_BIN_SLOT_LEN", raising=False)
    assert global_pack_partitions_from_env([10] * 8, 4) is None
    monkeypatch.setenv("VERL_TPU_BIN_SLOT_LEN", "256")
    monkeypatch.setenv("VERL_TPU_GLOBAL_PACK_ALIGN", "64")
    partitions, stats = global_pack_partitions_from_env([100] * 8, 4)
    assert stats.bin_len == 256 and all(len(p) == 2 for p in partitions)
    assert stats.micro_batches == 1  # two 128-slot segments per 256 bin


# --------------------------------------------------------------------------------------------- engine side


@requires_global_pack
def test_engine_packing_with_global_plan_balances_micro_batches():
    """Simulate DP ranks: after the controller's reorder, every rank's engine FFD needs the planned bin count."""
    dp, bin_len = 16, 4096
    rng = random.Random(11)
    lengths = [rng.choice([rng.randint(2200, 4000), rng.randint(300, 1800)]) for _ in range(dp * 8)]
    partitions, stats = get_global_pack_partitions(lengths, dp, bin_len, align=64)
    counts_global, counts_contig = [], []
    for r in range(dp):
        chunk = [lengths[j] for j in partitions[r]]
        mbs, idx = pack_micro_batches_into_bins(
            _nested_batch(chunk), dp_group=None, bin_len=bin_len, num_bins=1, align=64
        )
        assert sorted(i for g in idx for i in g) == list(range(len(chunk)))
        for mb in mbs:
            assert int(mb["input_ids"].offsets()[-1]) <= bin_len
        counts_global.append(len(mbs))
        contig = lengths[r * 8 : (r + 1) * 8]
        counts_contig.append(len(pack_micro_batches_into_bins(_nested_batch(contig), None, bin_len, 1)[0]))
    assert max(counts_global) == stats.micro_batches
    assert max(counts_global) <= max(counts_contig)


def test_engine_packing_without_verl_global_pack(monkeypatch):
    """verl without ``verl.utils.global_pack``: the engine falls back to its own first-fit-decreasing packing."""
    monkeypatch.setitem(sys.modules, "verl.utils.global_pack", None)  # makes the import raise ImportError
    rng = random.Random(7)
    for _ in range(20):
        lengths = [rng.randint(1, 900) for _ in range(rng.randint(1, 30))]
        mbs, idx = pack_micro_batches_into_bins(_nested_batch(lengths), dp_group=None, bin_len=1024, num_bins=1)
        assert sorted(i for g in idx for i in g) == list(range(len(lengths)))
        for mb in mbs:
            assert int(mb["input_ids"].offsets()[-1]) <= 1024
        # FFD never needs more than twice the token lower bound (plus one bin).
        assert len(mbs) <= 2 * math.ceil(sum(lengths) / 1024) + 1


def test_log_micro_batch_fill(capsys):
    data = _nested_batch([100, 50, 30])
    fill = log_micro_batch_fill(data, num_micro_batches=2, static_len=128, dp_group=None, tag="t", global_pack=True)
    assert fill == pytest.approx(180 / 256)
    out = capsys.readouterr().out
    assert "micro_batches/rank=2" in out and "fill=0.703" in out
    data2 = _nested_batch([100, 50, 30])
    # As the actor worker stores it (engine_workers.train_mini_batch): NonTensorData of all DP ranks' lengths.
    tu.assign_non_tensor(data2, global_token_num=NonTensorData([100, 50, 30, 200]))
    assert log_micro_batch_fill(data2, 2, 128, None) == pytest.approx(380 / 256)


# --------------------------------------------------------------------------------------------- loss / grad semantics


class _TinyLM(torch.nn.Module):
    """Embedding -> one causal attention layer with document masks -> LM head."""

    def __init__(self, vocab: int = 37, dim: int = 16):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, dim)
        self.pos = torch.nn.Embedding(512, dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.out = torch.nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Document ids from positions, as torchtitan's segment_ids_from_positions (Splash / GDN segment ids).
        seg = torch.cumsum((positions == 0).to(torch.int64), dim=-1)
        h = self.emb(ids) + self.pos(positions)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        t = ids.shape[0]
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        mask = causal & (seg[:, None] == seg[None, :])
        att = (q @ k.T) / math.sqrt(q.shape[-1])
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.out(h + att @ v)


def _token_logprobs(model, ids, positions, labels):
    return torch.gather(F.log_softmax(model(ids, positions), -1), -1, labels[:, None]).squeeze(-1)


@requires_global_pack
def test_packed_rows_match_unpacked_loss_and_grads():
    """Token-mean PPO-style loss with the global normalizer: packed multi-segment rows == per-sequence forward."""
    vocab, bin_len, dp = 37, 96, 3
    rng = random.Random(3)
    lengths = [rng.randint(5, 60) for _ in range(12)]
    seqs = [torch.randint(1, vocab, (n,), generator=torch.Generator().manual_seed(i)) for i, n in enumerate(lengths)]
    masks = [(torch.arange(n) >= n // 3).float() for n in lengths]  # prompt tokens masked out
    advs = [torch.full((n,), (i % 5 - 2) * 0.5) for i, n in enumerate(lengths)]
    global_tokens = sum(float(m[:-1].sum()) for m in masks)

    def seq_terms(lp, i):
        n = lengths[i]
        return -(lp[: n - 1] * advs[i][: n - 1] * masks[i][: n - 1]).sum()

    # Reference: each sequence alone.
    ref = _TinyLM(vocab)
    loss_ref = 0.0
    for i, s in enumerate(seqs):
        lp = _token_logprobs(ref, s[:-1], torch.arange(lengths[i] - 1), s[1:])
        loss_ref = loss_ref + seq_terms(torch.cat([lp, lp.new_zeros(1)]), i)
    loss_ref = loss_ref / global_tokens
    loss_ref.backward()

    # Global pack -> per-rank engine bins -> static [bin_len] rows laid out by bin_slot_permutation.
    model = _TinyLM(vocab)
    partitions, _ = get_global_pack_partitions(lengths, dp, bin_len, align=8)
    loss_packed = 0.0
    for part in partitions:
        chunk = [lengths[j] for j in part]
        _, mb_idx = pack_micro_batches_into_bins(_nested_batch(chunk), None, bin_len, 1, align=8)
        for mb in mb_idx:
            items = [part[i] for i in mb]
            offsets = torch.tensor([0, *torch.tensor([lengths[i] for i in items]).cumsum(0).tolist()])
            used = int(offsets[-1])
            ids = torch.cat([seqs[i] for i in items] + [torch.zeros(bin_len - used, dtype=torch.int64)])
            pos = torch.cat(
                [torch.arange(lengths[i]) for i in items] + [torch.zeros(bin_len - used, dtype=torch.int64)]
            )
            labels = torch.roll(ids, -1)
            slot_src, contig_src, tails = bin_slot_permutation(offsets, bin_len, bin_len)
            ids_s, labels_s, pos_s = ids[slot_src], labels[slot_src], pos[slot_src].clone()
            for a, b in tails:
                pos_s[a:b] = torch.arange(b - a)
            lp = _token_logprobs(model, ids_s, pos_s, labels_s)[contig_src]
            for k, i in enumerate(items):
                loss_packed = loss_packed + seq_terms(lp[int(offsets[k]) : int(offsets[k + 1])], i)
    loss_packed = loss_packed / global_tokens
    loss_packed.backward()

    torch.testing.assert_close(loss_packed, loss_ref, rtol=1e-5, atol=1e-6)
    for (name, p_ref), (_, p) in zip(ref.named_parameters(), model.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, p_ref.grad, rtol=1e-4, atol=1e-6, msg=name)
