# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Fixed-shape micro-batches for the TorchTitan TPU engine (opt-in: ``engine_kwargs.tpu.fixed_shape_rows``).

With ``use_dynamic_bsz=True``, verl's ``rearrange_micro_batches`` picks a different number of micro-batches, and of
sequences per micro-batch, on every update, and the engine pads each packed row up to the sequence bucket. On TPU,
every new packed-row length compiles new XLA programs for the transformer blocks, and every new sequence count new
programs for the loss and its metrics, whose batch dimension is that count. In RL fine-tuning this keeps recompiling
long after the first update.

With ``fixed_shape_rows`` set, the engine replaces verl's partition by one whose shapes come from small fixed sets:

1. :func:`build_micro_batches`: every rank packs its sequences into the same layout, a list of packed-row lengths taken
   from ``fixed_shape_rows``. The layout is the cheapest one (by a per-row-length cost table) that every rank can pack
   with at most ``fixed_shape_max_seqs`` sequences and at least one sequence per micro-batch. The ranks agree through
   CPU all-reduces of feasibility flags, so all of them take the same decision. Micro-batch ``i`` is padded to its
   row length (``tpu_padded_seq_len``) and its response width to ``fixed_shape_response_len``
   (``tpu_max_response_len``).
2. ``losses/loss_inputs.py`` lays the PPO loss inputs out as ``[fixed_shape_max_seqs, fixed_shape_response_len]``.
   The extra rows are empty sequences (all-zero response mask, log-probs and advantages), so the loss, its gradients
   and the global token-mean normalization (``batch_num_tokens``) are unchanged.
3. :func:`prewarm`: on the first training call, every row length runs once (compile) and :data:`PREWARM_TIMED_RUNS`
   more times (timed) through the engine's forward / loss / backward path, on sequences of the real mini-batch. The
   gradients are then discarded (no optimizer step). The timings, maximum over ranks, become the cost table.
4. :func:`sum_losses_on_host`: the per-micro-batch losses are summed on the host, so verl reduces a one-element list
   instead of one whose length is the number of micro-batches.

The transformer blocks then see one shape per row length, and the loss one shape per row length (the response
gather) plus one for everything after it; with the pre-warm, all of them compile before the first real micro-batch.
With Tokamax's GatedDeltaNet kernels, the engine also keeps their packed-document budget constant
(``fixed_shape_max_seqs + 1``).

Per-sequence statistics over the loss rows must skip rows without valid tokens to be unchanged by the padding rows
(verl's rollout-correction sequence-level metrics, for example); the loss and its gradients never depend on them.

Fixed-shape micro-batches need ``use_dynamic_bsz=True`` and packed inputs (``use_remove_padding=True``), without
context or pipeline parallelism. Otherwise the engine keeps verl's partition and logs a warning once.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
import torch.distributed
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    bucket_length,
    get_tpu_seq_bucket_size,
    unwrap_metadata,
)
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options, parse_fixed_shape_rows
from verl_hardware_plugin.accelerators.tpu.profilers import timeline

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Non-tensor key of a fixed-shape micro-batch: the number of rows of its PPO loss inputs (``losses/loss_inputs.py``).
TPU_PADDED_NUM_SEQS_KEY = "tpu_padded_num_seqs"
DEFAULT_MAX_SEQS = 64
# Timed pre-warm runs per row length, after the run that compiles it.
PREWARM_TIMED_RUNS = 2
# Per-micro-batch overhead of the default cost model, in tokens.
DEFAULT_OVERHEAD_TOKENS = 2048

AllReduceMax = Callable[[list], list]


# ---------------------------------------------------------------------------------------------------------------------
# Layout planner
# ---------------------------------------------------------------------------------------------------------------------


def default_cost(row: int) -> float:
    """Cost of a micro-batch of packed-row length ``row`` before any is measured: its padded tokens plus a fixed
    per-micro-batch overhead, in tokens."""
    return float(row + DEFAULT_OVERHEAD_TOKENS)


def enumerate_layouts(
    total_max: int,
    n_max: int,
    n_min: int,
    item_max: int,
    rows: Sequence[int],
    max_seqs: int,
    costs: dict[int, float],
    extra_k: int = 2,
    limit: int = 200_000,
) -> list[tuple[int, ...]]:
    """Every multiset of row lengths (a non-increasing tuple) that could hold the largest rank's sequences, cheapest
    first. Deterministic, so every rank builds the same list.

    Args:
        total_max: Largest number of tokens on any rank.
        n_max: Largest number of sequences on any rank.
        n_min: Smallest number of sequences on any rank (every micro-batch needs one).
        item_max: Longest sequence on any rank.
        rows: Allowed packed-row lengths.
        max_seqs: Maximum number of sequences per micro-batch.
        costs: Cost of one micro-batch per row length.
        extra_k: Micro-batches allowed beyond the fewest that hold ``total_max`` tokens in the shortest rows.
        limit: Maximum number of candidates.

    Raises:
        ValueError: when there are more than ``limit`` candidates.
    """
    rows = sorted({int(r) for r in rows})
    k_lo = max(1, -(-total_max // rows[-1]), -(-n_max // max_seqs))
    k_hi = min(n_min, -(-total_max // rows[0]) + extra_k)
    candidates = []
    for k in range(k_lo, k_hi + 1):
        for combo in itertools.combinations_with_replacement(rows[::-1], k):
            if combo[0] < item_max or sum(combo) < total_max:
                continue
            candidates.append((sum(costs[r] for r in combo), k, sum(combo), combo))
            if len(candidates) > limit:
                raise ValueError(f"more than {limit} fixed-shape layouts; use fewer fixed_shape_rows")
    candidates.sort()
    return [c[3] for c in candidates]


def pack(lens: Sequence[int], caps: Sequence[int], max_seqs: int) -> Optional[list[list[int]]]:
    """Best-fit-decreasing packing of sequences of ``lens`` tokens into micro-batches of row lengths ``caps``.

    Every micro-batch gets at least one and at most ``max_seqs`` sequences. Returns the (sorted) sequence indices of
    each micro-batch, or ``None`` when this heuristic finds no packing.
    """
    n, k = len(lens), len(caps)
    if n < k:
        return None
    remaining = list(caps)
    bins: list[list[int]] = [[] for _ in range(k)]
    for i in sorted(range(n), key=lambda j: (-lens[j], j)):
        best = -1
        for b in range(k):
            if len(bins[b]) < max_seqs and remaining[b] >= lens[i] and (best < 0 or remaining[b] < remaining[best]):
                best = b
        if best < 0:
            return None
        bins[best].append(i)
        remaining[best] -= lens[i]
    # A micro-batch of padding only is not allowed (the loss needs a valid token): move the shortest sequence that
    # fits from a micro-batch that has two or more.
    for b in range(k):
        if bins[b]:
            continue
        best_move = None
        for d in range(k):
            if len(bins[d]) < 2:
                continue
            for pos, i in enumerate(bins[d]):
                if lens[i] <= remaining[b] and (best_move is None or (lens[i], d, pos) < best_move):
                    best_move = (lens[i], d, pos)
        if best_move is None:
            return None
        _, d, pos = best_move
        i = bins[d].pop(pos)
        remaining[d] += lens[i]
        bins[b].append(i)
        remaining[b] -= lens[i]
    return [sorted(b) for b in bins]


@dataclasses.dataclass(frozen=True)
class Plan:
    """A fixed-shape layout and this rank's packing of it."""

    caps: tuple[int, ...]  # packed-row length of micro-batch i, the same on every rank
    bins: list[list[int]]  # this rank's sequence indices of micro-batch i
    cost: float  # sum of the costs of ``caps``
    candidate: int  # index of the layout in the cheapest-first list, -1 for the fallback
    collectives: int  # all-reduces issued


def plan_layout(
    lens: Sequence[int],
    rows: Sequence[int],
    max_seqs: int,
    costs: dict[int, float],
    allreduce_max: AllReduceMax,
    *,
    per_round: int = 64,
    max_rounds: int = 4,
    extra_k: int = 2,
) -> Plan:
    """Chooses the cheapest layout every rank can pack, and this rank's packing of it.

    ``allreduce_max(values)`` must return the element-wise maximum over all ranks. Every rank must pass the same
    ``rows``, ``max_seqs``, ``costs`` and options: the control flow only depends on reduced values, so all ranks issue
    the same collectives and return the same ``caps``. Candidates are tried cheapest first, ``per_round`` per
    all-reduce; if none of the first ``max_rounds * per_round`` fits on every rank, the layout is the fewest
    largest rows that every rank can pack.

    Raises:
        ValueError: when a sequence is longer than the largest row, or a rank has no sequences.
        RuntimeError: when no layout fits (e.g. a rank has fewer sequences than the micro-batches another needs).
    """
    rows = sorted({int(r) for r in rows})
    lens = [int(x) for x in lens]
    n = len(lens)
    stats = allreduce_max([sum(lens), n, max(lens) if lens else 0, -n])
    total_max, n_max, item_max, n_min = int(stats[0]), int(stats[1]), int(stats[2]), -int(stats[3])
    collectives = 1
    if item_max > rows[-1]:
        raise ValueError(f"a sequence of {item_max} tokens does not fit the largest fixed_shape_rows ({rows[-1]})")
    if n_min < 1:
        raise ValueError("a rank has no sequences in this mini-batch")
    candidates = enumerate_layouts(total_max, n_max, n_min, item_max, rows, max_seqs, costs, extra_k=extra_k)
    for rnd in range(max_rounds):
        chunk = candidates[rnd * per_round : (rnd + 1) * per_round]
        if not chunk:
            break
        packs = [pack(lens, caps, max_seqs) for caps in chunk]
        flags = allreduce_max([0 if p is not None else 1 for p in packs] + [1] * (per_round - len(chunk)))
        collectives += 1
        for j, caps in enumerate(chunk):
            if int(flags[j]) == 0:
                bins = packs[j]
                assert bins is not None
                return Plan(tuple(caps), bins, sum(costs[r] for r in caps), rnd * per_round + j, collectives)
    k = max(1, -(-total_max // rows[-1]), -(-n_max // max_seqs))
    while k <= n_min:
        caps = (rows[-1],) * k
        bins = pack(lens, caps, max_seqs)
        flags = allreduce_max([0 if bins is not None else 1])
        collectives += 1
        if int(flags[0]) == 0:
            assert bins is not None
            return Plan(caps, bins, sum(costs[r] for r in caps), -1, collectives)
        k += 1
    raise RuntimeError(
        f"no fixed-shape layout fits: up to {total_max} tokens and {n_min} to {n_max} sequences per rank, "
        f"fixed_shape_rows {rows}, fixed_shape_max_seqs {max_seqs}"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class FixedShapeConfig:
    """Fixed-shape settings of one engine (see ``TPUEngineOptions``) and its cost table.

    Attributes:
        rows: Allowed packed-row lengths, ascending.
        max_seqs: Maximum number of sequences per micro-batch and row count of the PPO loss inputs.
        response_len: Response width of the PPO loss inputs (a multiple of the sequence bucket).
        prewarm: Whether :func:`prewarm` compiles and times the rows on the first training call.
        costs: Cost of one micro-batch per row length: :func:`default_cost` until the pre-warm measured them (seconds).
        cost_source: ``"default"`` or ``"measured"``.
        prewarmed: Whether the pre-warm already ran (or was skipped).
        warned_inapplicable: Whether :func:`active_config` already warned that a batch cannot use fixed shapes.
    """

    rows: tuple[int, ...]
    max_seqs: int
    response_len: int
    prewarm: bool = True
    costs: dict[int, float] = dataclasses.field(default_factory=dict)
    cost_source: str = "default"
    prewarmed: bool = False
    warned_inapplicable: bool = False

    def __post_init__(self) -> None:
        if not self.costs:
            self.costs = {row: default_cost(row) for row in self.rows}

    @classmethod
    def from_engine_config(cls, engine_config: Any) -> Optional[FixedShapeConfig]:
        """The engine's fixed-shape settings from ``engine_kwargs.tpu``, or ``None`` when ``fixed_shape_rows`` is unset.

        Raises:
            ValueError: when a row length is not a multiple of the sequence bucket.
        """
        options = get_tpu_engine_options(engine_config)
        if options.fixed_shape_rows is None:
            return None
        rows = parse_fixed_shape_rows(options.fixed_shape_rows)
        # pad_packed_inputs_for_tpu rounds the real lengths up to the bucket and pads to the larger of that and the
        # requested length, so the rows must be bucket multiples and the response width is rounded up to one.
        step = get_tpu_seq_bucket_size(engine_config)
        misaligned = [row for row in rows if row % step]
        if misaligned:
            raise ValueError(
                f"engine_kwargs.tpu.fixed_shape_rows {misaligned} are not multiples of the sequence bucket ({step}, "
                "engine_kwargs.tpu.seq_bucket_size)"
            )
        assert options.fixed_shape_response_len is not None  # get_tpu_engine_options requires it with the rows
        config = cls(
            rows=rows,
            max_seqs=options.fixed_shape_max_seqs or DEFAULT_MAX_SEQS,
            response_len=bucket_length(options.fixed_shape_response_len, step),
            prewarm=True if options.fixed_shape_prewarm is None else options.fixed_shape_prewarm,
        )
        logger.info(
            "Fixed-shape micro-batches: rows=%s max_seqs=%d response_len=%d prewarm=%s",
            list(config.rows),
            config.max_seqs,
            config.response_len,
            config.prewarm,
        )
        return config


def inapplicable_reasons(data: TensorDict, parallel_dims: Any) -> list[str]:
    """Why fixed-shape micro-batches cannot partition ``data`` (empty when they can).

    Depends only on the config and the batch layout, never on the token counts, so every rank decides the same.
    """

    def meta(key: str, default: Any) -> Any:
        return unwrap_metadata(tu.get_non_tensor_data(data=data, key=key, default=default))

    reasons = []
    if not meta("use_dynamic_bsz", True):
        reasons.append("use_dynamic_bsz=False")
    if not meta("use_remove_padding", True) or meta("pad_mode", DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING:
        reasons.append("padded inputs")
    if int(meta("force_group_size", 1) or 1) != 1:
        reasons.append("force_group_size != 1")
    if not getattr(data["input_ids"], "is_nested", False):
        reasons.append("input_ids not nested")
    if getattr(parallel_dims, "cp_enabled", False) or getattr(parallel_dims, "pp_enabled", False):
        reasons.append("context or pipeline parallelism")
    return reasons


def active_config(
    config: Optional[FixedShapeConfig], data: TensorDict, parallel_dims: Any
) -> Optional[FixedShapeConfig]:
    """``config`` if fixed-shape micro-batches apply to ``data``, else ``None`` (verl's partition is used)."""
    if config is None:
        return None
    reasons = inapplicable_reasons(data, parallel_dims)
    if reasons:
        if not config.warned_inapplicable:
            logger.warning("Fixed-shape micro-batches do not apply (%s); using verl's partition", ", ".join(reasons))
            config.warned_inapplicable = True
        return None
    return config


def allreduce_max(values: Sequence[float]) -> list:
    """Element-wise maximum of ``values`` over all ranks, on the host.

    CPU tensors on the default process group, as :func:`align_micro_batch_shapes_across_ranks` does; integers stay
    integers unless a value is a float. The identity without a process group.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return list(values)
    dtype = torch.float64 if any(isinstance(v, float) for v in values) else torch.int64
    tensor = torch.tensor(list(values), dtype=dtype)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.tolist()


def _is_rank0() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def sequence_lengths(data: TensorDict) -> list[int]:
    """Token count of every (packed, nested) sequence of ``data``."""
    return [int(x) for x in data["input_ids"].offsets().diff().tolist()]


def _select(data: TensorDict, indices: list[int], config: FixedShapeConfig, row: int) -> TensorDict:
    micro_batch = tu.index_select_tensor_dict(data, indices)
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_seq_len", int(row))
    tu.assign_non_tensor_data(micro_batch, "tpu_max_response_len", int(config.response_len))
    tu.assign_non_tensor_data(micro_batch, TPU_PADDED_NUM_SEQS_KEY, int(config.max_seqs))
    return micro_batch


def build_micro_batches(
    data: TensorDict, config: FixedShapeConfig, allreduce_max: AllReduceMax = allreduce_max
) -> tuple[list[TensorDict], list[list[int]]]:
    """Partitions ``data`` into fixed-shape micro-batches.

    Returns ``(micro_batches, indices)`` like verl's ``prepare_micro_batches`` (``indices`` restores the input order
    in ``postprocess_batch_func``). Collective: every rank must call it, with the same ``config``.
    """
    lens = sequence_lengths(data)
    start = time.perf_counter()
    plan = plan_layout(lens, config.rows, config.max_seqs, config.costs, allreduce_max)
    micro_batches = [_select(data, bins, config, cap) for cap, bins in zip(plan.caps, plan.bins, strict=True)]
    if "responses" in data.keys() and getattr(data["responses"], "is_nested", False):
        longest = int(data["responses"].offsets().diff().max().item())
        if longest > config.response_len:
            logger.warning(
                "A response of %d tokens exceeds fixed_shape_response_len (%d): its loss inputs get a new shape",
                longest,
                config.response_len,
            )
    if _is_rank0():
        used = [sum(lens[i] for i in bins) for bins in plan.bins]
        logger.info(
            "Fixed-shape layout: rows=%s fill=%.3f sequences=%s cost=%.3f (%s) candidate=%d planning=%.1f ms",
            list(plan.caps),
            sum(used) / max(1, sum(plan.caps)),
            [len(bins) for bins in plan.bins],
            plan.cost,
            config.cost_source,
            plan.candidate,
            1e3 * (time.perf_counter() - start),
        )
    return micro_batches, plan.bins


def row_lengths(micro_batches: list[TensorDict]) -> list[int]:
    """Packed-row length of every micro-batch (for the ``fwd_bwd`` timeline span)."""
    return [int(tu.get_non_tensor_data(mb, "tpu_padded_seq_len", 0) or 0) for mb in micro_batches]


def sum_losses_on_host(output: dict[str, Any]) -> None:
    """Replaces the per-micro-batch ``loss`` list of ``output`` by its exact (``math.fsum``) one-element sum.

    verl sums the list on the device; a one-element list keeps that reduction's shape fixed.
    """
    losses = output.get("loss")
    if isinstance(losses, list) and len(losses) > 1:
        output["loss"] = [math.fsum(float(x) for x in losses)]


def _prewarm_indices(lens: Sequence[int], row: int, max_seqs: int) -> list[int]:
    """The shortest sequences whose tokens fit ``row`` (at most ``max_seqs``, in batch order); empty if none fits.

    Only the shapes matter for compiling and timing a row: its length and the ``max_seqs`` loss rows.
    """
    picked: list[int] = []
    used = 0
    for i in sorted(range(len(lens)), key=lambda j: (lens[j], j)):
        if len(picked) >= max_seqs or used + lens[i] > row:
            break
        picked.append(i)
        used += lens[i]
    return sorted(picked)


def prewarm(
    data: TensorDict,
    config: FixedShapeConfig,
    run_micro_batch: Callable[[TensorDict], None],
    zero_grad: Callable[[], None],
    allreduce_max: AllReduceMax = allreduce_max,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Compiles every row length before the first real micro-batch and measures its cost (once per engine).

    For each row length, largest first, a micro-batch of the shortest sequences of ``data`` that fit it runs once
    (compile) and :data:`PREWARM_TIMED_RUNS` more times (timed) through ``run_micro_batch``, which must run the
    engine's forward, loss and backward of one micro-batch and wait for the device. ``zero_grad`` then discards the
    accumulated gradients; no optimizer step runs. The fastest timed run of each row, maximum over ranks, replaces the
    default costs. A row length that no sequence fits on some rank is skipped on every rank (it compiles on first use),
    and the default costs are kept. Collective: every rank must call it, with the same ``config``.

    Module state other than gradients that the forward pass may update (e.g. MoE expert-load counters) is not reset.
    """
    if config.prewarmed:
        return
    config.prewarmed = True
    if not config.prewarm:
        return
    lens = sequence_lengths(data)
    rows = sorted(config.rows, reverse=True)
    picks = {row: _prewarm_indices(lens, row, config.max_seqs) for row in rows}
    skips = allreduce_max([0 if picks[row] else 1 for row in rows])
    measured: dict[int, float] = {}
    start_all = clock()
    with timeline.span(
        "fixed_shape_prewarm", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane(), rows=rows
    ) as details:
        for row, skip in zip(rows, skips, strict=True):
            if int(skip):
                logger.warning(
                    "Fixed-shape pre-warm: no sequences fit row %d on some rank; it compiles on first use", row
                )
                continue
            times = []
            for _ in range(1 + PREWARM_TIMED_RUNS):
                micro_batch = _select(data, picks[row], config, row)
                start = clock()
                run_micro_batch(micro_batch)
                times.append(clock() - start)
            measured[row] = min(times[1:])
            if _is_rank0():
                logger.info(
                    "Fixed-shape pre-warm: row %d first run %.2f s, timed %s s",
                    row,
                    times[0],
                    [round(t, 4) for t in times[1:]],
                )
        zero_grad()
        if measured:
            keys = sorted(measured)
            agreed = dict(zip(keys, allreduce_max([float(measured[k]) for k in keys]), strict=True))
            if all(row in agreed and math.isfinite(agreed[row]) and agreed[row] > 0 for row in config.rows):
                config.costs = {row: float(agreed[row]) for row in config.rows}
                config.cost_source = "measured"
            details["measured_s"] = {str(k): round(v, 4) for k, v in agreed.items()}
        details["cost_source"] = config.cost_source
    if _is_rank0():
        logger.info(
            "Fixed-shape pre-warm done in %.1f s; costs %s (%s)", clock() - start_all, config.costs, config.cost_source
        )
