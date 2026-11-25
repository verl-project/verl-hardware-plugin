# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``gdn_slot_schedule`` and its ``TorchTitanTPUEngineWithLMHead`` input/output wiring."""

from __future__ import annotations

import contextlib
import logging
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl_hardware_plugin.accelerators.tpu.engines import gdn_slot_schedule, torchtitan_tpu, tpu_model_kernels


def _make_packed_mb(
    seqs_pos: list[list[int]],
    *,
    vocab: int = 32,
    padded_seq_len: int | None = None,
    calculate_entropy: bool = False,
    seed: int = 0,
) -> TensorDict:
    gen = torch.Generator().manual_seed(seed)
    id_seqs = [torch.randint(1, vocab, (len(p),), generator=gen, dtype=torch.int64) for p in seqs_pos]
    pos_seqs = [torch.tensor(p, dtype=torch.int64) for p in seqs_pos]
    mb = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(id_seqs, layout=torch.jagged),
            "position_ids": torch.nested.as_nested_tensor(pos_seqs, layout=torch.jagged),
        },
        batch_size=[len(seqs_pos)],
    )
    tu.assign_non_tensor(
        mb,
        use_remove_padding=True,
        calculate_entropy=calculate_entropy,
        temperature=1.0,
    )
    if padded_seq_len is not None:
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", padded_seq_len)
    return mb


@pytest.fixture(autouse=True)
def _no_spmd_input_annotation(monkeypatch):
    """The stub engines here have no real ``ParallelDims`` / SPMD mesh for TorchTitan's input type annotation."""
    monkeypatch.setattr(torchtitan_tpu, "_annotate_input_spmd_types", lambda *args, **kwargs: None)


def test_has_internal_document_resets_checks_exact_sequence_starts():
    # Two standard single-document sequences starting at 0.
    pos = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    cu = torch.tensor([0, 3, 7])
    assert gdn_slot_schedule.has_internal_document_resets(pos, cu) is False

    # Sub-document reset inside the first sequence.
    pos_multi = torch.tensor([0, 1, 0, 1, 0, 1, 2])
    assert gdn_slot_schedule.has_internal_document_resets(pos_multi, cu) is True

    # Continuation sequence starting at position > 0 that later resets to 0 inside the same sequence:
    # (pos == 0).sum() == 1 <= n_seqs == 1, yet index 2 is an interior document reset.
    pos_cont_reset = torch.tensor([5, 6, 0, 1])
    cu_single = torch.tensor([0, 4])
    assert gdn_slot_schedule.has_internal_document_resets(pos_cont_reset, cu_single) is True

    # Continuation sequence without interior reset.
    pos_cont_clean = torch.tensor([5, 6, 7, 8])
    assert gdn_slot_schedule.has_internal_document_resets(pos_cont_clean, cu_single) is False


def test_classify_and_sort_micro_batches_agree_on_categories():
    # padded_len=16, auto_slot_len=8 -> 2 row slots of length 8.
    mb_seg = _make_packed_mb([[0, 1, 0, 1], [0, 1, 2]], padded_seq_len=16)  # internal reset -> SEGMENTED (4+3=7)
    mb_single = _make_packed_mb([list(range(12))], padded_seq_len=16)  # 1 seq > 8, <= 16 -> SINGLE_DOC (12)
    mb_row_small = _make_packed_mb([list(range(5)), list(range(4))], padded_seq_len=16)  # 2 seqs <= 8 -> ROW_SLOT (9)
    mb_row_large = _make_packed_mb([list(range(8)), list(range(6))], padded_seq_len=16)  # 2 seqs <= 8 -> ROW_SLOT (14)

    for mb, expected_cat in (
        (mb_seg, gdn_slot_schedule.SLOT_CAT_SEGMENTED),
        (mb_single, gdn_slot_schedule.SLOT_CAT_SINGLE_DOC),
        (mb_row_small, gdn_slot_schedule.SLOT_CAT_ROW_SLOT),
        (mb_row_large, gdn_slot_schedule.SLOT_CAT_ROW_SLOT),
    ):
        _, cat, _ = gdn_slot_schedule.classify_gdn_micro_batch(
            mb,
            auto_slot_len=8,
            padded_len=16,
            forward_only=False,
            allow_row_slot=True,
        )
        assert cat == expected_cat

    sorted_mbs, sorted_idx = gdn_slot_schedule.sort_micro_batches_for_slots(
        [mb_seg, mb_single, mb_row_small, mb_row_large],
        ["seg", "single", "row_small", "row_large"],
        auto_slot_len=8,
        static_len=16,
        bucket_size=8,
        forward_only=False,
        allow_row_slot=True,
    )
    assert sorted_idx == ["row_large", "row_small", "single", "seg"]
    assert len(sorted_mbs) == 4


def test_classify_excludes_differentiated_entropy_and_3d_mrope_from_row_slots():
    mb_ent = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]], padded_seq_len=16, calculate_entropy=True)
    # Training pass (forward_only=False) with entropy requested -> row-slot disallowed.
    mask_train, cat_train, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_ent, auto_slot_len=8, padded_len=16, forward_only=False, allow_row_slot=True
    )
    assert mask_train == 0
    assert cat_train == gdn_slot_schedule.SLOT_CAT_SEGMENTED

    # Forward-only pass (old log-probs) computes chunked entropy without logits -> row-slot allowed.
    mask_fwd, cat_fwd, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_ent, auto_slot_len=8, padded_len=16, forward_only=True, allow_row_slot=True
    )
    assert mask_fwd == 1
    assert cat_fwd == gdn_slot_schedule.SLOT_CAT_ROW_SLOT

    # 3-D MRoPE positions must not be flattened into 1-D slots.
    mb_mrope = _make_packed_mb([[0, 1, 2]], padded_seq_len=8)
    mb_mrope["position_ids"] = torch.nested.as_nested_tensor(
        [torch.zeros((3, 3), dtype=torch.int64)], layout=torch.jagged
    )
    mask_mrope, cat_mrope, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_mrope, auto_slot_len=4, padded_len=8, forward_only=True, allow_row_slot=True
    )
    assert mask_mrope == 0
    assert cat_mrope == gdn_slot_schedule.SLOT_CAT_SEGMENTED


def test_classify_rejects_continuation_sequences_starting_above_zero():
    """Sequences starting at position > 0 must run segmented so Splash/Varlen attention never merges segments."""
    mb_two = _make_packed_mb([[0, 1, 2], [5, 6, 7, 8]], padded_seq_len=16)
    mask_two, cat_two, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_two, auto_slot_len=8, padded_len=16, forward_only=True, allow_row_slot=True
    )
    assert mask_two == 0
    assert cat_two == gdn_slot_schedule.SLOT_CAT_SEGMENTED

    mb_single_cont = _make_packed_mb([[5, 6, 7, 8]], padded_seq_len=8)
    mask_one, cat_one, _ = gdn_slot_schedule.classify_gdn_micro_batch(
        mb_single_cont, auto_slot_len=4, padded_len=8, forward_only=True, allow_row_slot=True
    )
    assert mask_one == 0
    assert cat_one == gdn_slot_schedule.SLOT_CAT_SEGMENTED


def test_plan_micro_batch_gdn_slots_band_all_reduce_and_refreshes_stale_key():
    mb_row = _make_packed_mb([[0, 1, 2, 3]], padded_seq_len=16, calculate_entropy=True)
    # Forward-only: eligible for both row-slot (8) and single-doc (16) -> picks row-slot 8.
    slots_fwd = gdn_slot_schedule.plan_micro_batch_gdn_slots(
        [mb_row], auto_slot_len=8, forward_only=True, allow_row_slot=True
    )
    assert slots_fwd == [8]
    assert tu.get_non_tensor_data(data=mb_row, key=gdn_slot_schedule.MB_SLOT_LEN_KEY, default=-1) == 8

    # Reusing the same TensorDict on a training pass where a peer rank is segmented (BAND -> 0) must overwrite 8 -> 0.
    def _peer_is_segmented(tensor: torch.Tensor, op: object = None) -> None:
        assert op == dist.ReduceOp.BAND
        tensor.zero_()

    with (
        mock.patch.object(dist, "is_initialized", return_value=True),
        mock.patch.object(dist, "get_world_size", return_value=2),
        mock.patch.object(dist, "all_reduce", side_effect=_peer_is_segmented) as ar,
    ):
        slots_train = gdn_slot_schedule.plan_micro_batch_gdn_slots(
            [mb_row], auto_slot_len=8, forward_only=False, allow_row_slot=True
        )
    ar.assert_called_once()
    assert slots_train == [0]
    assert tu.get_non_tensor_data(data=mb_row, key=gdn_slot_schedule.MB_SLOT_LEN_KEY, default=-1) == 0


def test_build_cp_full_positions_modes_and_validation():
    mb = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]], padded_seq_len=12)

    # Segmented multi-doc: zero-padded tail matching pad_packed_inputs_for_tpu and TpuQwen35Model.preprocess_inputs.
    seg_pos = gdn_slot_schedule.build_cp_full_positions(mb, 12, unsegmented=False)
    assert seg_pos.tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 0, 0, 0, 0]

    # Single-doc unsegmented: zero-padded tail.
    mb_one = _make_packed_mb([[0, 1, 2, 3, 4]], padded_seq_len=8)
    unseg_pos = gdn_slot_schedule.build_cp_full_positions(mb_one, 8, unsegmented=True)
    assert unseg_pos.tolist() == [0, 1, 2, 3, 4, 0, 0, 0]

    # Row-slot permuted: each 6-token row starts with its sequence and pads with zeros.
    cu = mb["input_ids"].offsets()
    ids_pad = torch.zeros((1, 12), dtype=torch.int64)
    pos_pad = torch.zeros((1, 12), dtype=torch.int64)
    lbl_pad = torch.zeros((1, 12), dtype=torch.int64)
    _, _, _, slot_src_cpu, _ = gdn_slot_schedule.permute_packed_for_row_slots(cu, 12, 6, ids_pad, pos_pad, lbl_pad)
    slot_pos = gdn_slot_schedule.build_cp_full_positions(mb, 12, unsegmented=True, slot_src_cpu=slot_src_cpu)
    assert slot_pos.tolist() == [0, 1, 2, 0, 0, 0, 0, 1, 2, 3, 0, 0]

    # Multimodal and 3-D MRoPE raise ValueError instead of silently corrupting positions.
    mb_mm = _make_packed_mb([[0, 1, 2]], padded_seq_len=4)
    mb_mm["multi_modal_inputs"] = [{"pixel_values": torch.zeros(1)}]
    with pytest.raises(ValueError, match="cp_full_positions"):
        gdn_slot_schedule.build_cp_full_positions(mb_mm, 4, unsegmented=False)

    mb_mrope = _make_packed_mb([[0, 1, 2]], padded_seq_len=4)
    mb_mrope["position_ids"] = torch.nested.as_nested_tensor(
        [torch.zeros((3, 3), dtype=torch.int64)], layout=torch.jagged
    )
    with pytest.raises(ValueError, match="cp_full_positions"):
        gdn_slot_schedule.build_cp_full_positions(mb_mrope, 4, unsegmented=False)

    # use_remove_padding=False validation in prepare_model_inputs: 1-D text works; multimodal / MRoPE fail loudly.
    eng = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    eng.parallel_dims = SimpleNamespace(tp_enabled=False, cp_enabled=True, get_mesh=lambda _n: "cp_mesh")
    eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=32))
    eng.engine_config = SimpleNamespace(engine_kwargs={"tpu": {"seq_bucket_size": 4}})
    eng.trainer = SimpleNamespace(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(context_parallel_load_balancer=None, context_parallel_ptrr_mask_key=None)
        )
    )
    eng._gdn_row_slot_len = 0
    eng._gdn_auto_row_slot_len = 0
    eng._use_fused_linear_cross_entropy = True
    eng._cp_full_positions = True
    eng._cp_attention_owner = lambda: SimpleNamespace(
        cp_shard=lambda batch, *_a: {
            "input": batch["input"][:4],
            "positions": batch["positions"][..., :4],
        }
    )

    mb_no_rmpad = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]])
    tu.assign_non_tensor(mb_no_rmpad, use_remove_padding=False)
    with mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")):
        _, extra_in, _, _ = eng.prepare_model_inputs(mb_no_rmpad)
    assert extra_in["cp_full_positions"].tolist() == [0, 1, 2, 0, 0, 1, 2, 3]

    mb_no_rmpad_mm = _make_packed_mb([[0, 1, 2]])
    tu.assign_non_tensor(mb_no_rmpad_mm, use_remove_padding=False)
    mb_no_rmpad_mm["multi_modal_inputs"] = [{"pixel_values": torch.zeros(1)}]
    with (
        mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
        pytest.raises(ValueError, match="cp_full_positions"),
    ):
        eng.prepare_model_inputs(mb_no_rmpad_mm)

    mb_no_rmpad_mrope = _make_packed_mb([[0, 1, 2]])
    tu.assign_non_tensor(mb_no_rmpad_mrope, use_remove_padding=False)
    mb_no_rmpad_mrope["position_ids"] = torch.nested.as_nested_tensor(
        [torch.zeros((4, 3), dtype=torch.int64)], layout=torch.jagged
    )
    with (
        mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
        pytest.raises(ValueError, match="cp_full_positions"),
    ):
        eng.prepare_model_inputs(mb_no_rmpad_mrope)


def test_enable_fused_gdn_projections_and_model_kwarg_validation():
    @dataclass
    class _GDNCfg:
        fuse_input_projections: bool = False

    class _SlottedLayer:
        __slots__ = ("attention",)

        def __init__(self, attn: object) -> None:
            self.attention = attn

    gdn_a, gdn_b = _GDNCfg(), _GDNCfg()
    root_cfg = SimpleNamespace(layers=[_SlottedLayer(gdn_a), {"linear_attn": gdn_b}, SimpleNamespace(other=1)])
    assert gdn_slot_schedule.enable_fused_gdn_projections(root_cfg) == 2
    assert gdn_a.fuse_input_projections is True
    assert gdn_b.fuse_input_projections is True

    with pytest.raises(RuntimeError, match="torchtitan#216"):
        gdn_slot_schedule.enable_fused_gdn_projections(SimpleNamespace(layers=[SimpleNamespace(other=1)]))

    class _LegacyModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, *, positions: torch.Tensor | None = None) -> torch.Tensor:
            return tokens

    class _GenericKwargsModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return tokens

    class _UpdatedModel(torch.nn.Module):
        def forward(
            self,
            tokens: torch.Tensor,
            *,
            positions: torch.Tensor | None = None,
            cp_full_positions: torch.Tensor | None = None,
            gated_delta_document_resets: bool | None = None,
            **kwargs: object,
        ) -> torch.Tensor:
            return tokens

    with pytest.raises(RuntimeError, match="cp_full_positions"):
        gdn_slot_schedule.validate_model_cp_slot_support(
            [_LegacyModel()], need_cp_full_positions=True, need_document_resets_override=False
        )
    # Generic **kwargs (or Module.parallelize's forward_with_redistribution on a legacy model) must NOT falsely pass.
    with pytest.raises(RuntimeError, match="cp_full_positions"):
        gdn_slot_schedule.validate_model_cp_slot_support(
            [_GenericKwargsModel()], need_cp_full_positions=True, need_document_resets_override=False
        )
    parallelized_legacy = _LegacyModel()

    def forward_with_redistribution(*args: object, **kwargs: object) -> object:
        return args[0]

    parallelized_legacy.forward = forward_with_redistribution  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="gated_delta_document_resets"):
        gdn_slot_schedule.validate_model_cp_slot_support(
            [parallelized_legacy], need_cp_full_positions=False, need_document_resets_override=True
        )

    # Parallelized + _orig_mod wrapped TpuQwen35Model with explicit #216 kwargs succeeds.
    parallelized_updated = _UpdatedModel()
    parallelized_updated.forward = forward_with_redistribution  # type: ignore[method-assign]
    compiled_wrapper = SimpleNamespace(_orig_mod=parallelized_updated)
    assert gdn_slot_schedule.validate_model_cp_slot_support(
        [compiled_wrapper], need_cp_full_positions=True, need_document_resets_override=True
    ) == (True, True)


@pytest.mark.parametrize("cp_enabled", [False, True])
@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_engine_prepare_inputs_and_outputs_row_slot_roundtrip_matches_baseline(
    cp_enabled: bool, calculate_entropy: bool
):
    """Row-slot permutation + optional 2-way CP sharding produces identical jagged log_probs and entropy."""
    torch.manual_seed(7)
    vocab, hidden_dim, bucket_size, slot_len = 24, 8, 16, 8
    seqs_pos = [[0, 1, 2, 3, 4], [0, 1, 2, 3]]  # 9 tokens -> padded_seq_len=16, 2 slots of 8

    # Token-wise deterministic embedding table so a position-independent token model gives the exact same hidden
    # representation per token regardless of where the token sits in the padded/permuted/CP-sharded buffer.
    token_embed = torch.randn(vocab, hidden_dim)
    lm_head_weight_t = torch.randn(hidden_dim, vocab)

    def _make_stub_engine(
        *, cp: bool, cp_full_pos: bool, auto_slot: int = 0
    ) -> torchtitan_tpu.TorchTitanTPUEngineWithLMHead:
        eng = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
        eng.parallel_dims = SimpleNamespace(
            tp_enabled=False,
            cp_enabled=cp,
            get_mesh=lambda name: "cp_mesh",
        )
        eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab))
        eng.engine_config = SimpleNamespace(
            max_token_len_per_gpu=bucket_size,
            entropy_checkpointing=False,
            entropy_from_logits_with_chunking=False,
            engine_kwargs={"tpu": {"seq_bucket_size": bucket_size}},
        )
        eng.trainer = SimpleNamespace(
            config=SimpleNamespace(
                parallelism=SimpleNamespace(
                    context_parallel_load_balancer=None,
                    context_parallel_ptrr_mask_key=None,
                )
            )
        )
        eng._gdn_row_slot_len = 0
        eng._gdn_auto_row_slot_len = auto_slot
        eng._use_fused_linear_cross_entropy = True
        eng._cp_full_positions = cp_full_pos
        eng._tpu_kernels = frozenset()
        eng.compute_entropy_from_logits = verl_F.entropy_from_logits
        return eng

    def _exact_logprobs(hidden_2d: torch.Tensor, labels_1d: torch.Tensor, weight_t: torch.Tensor, **kwargs):
        temperature = float(kwargs.get("temperature", 1.0))
        logits = (hidden_2d @ weight_t).float() / temperature
        log_probs_all = torch.log_softmax(logits, dim=-1)
        safe_labels = labels_1d.clamp(0, weight_t.shape[1] - 1)
        return log_probs_all.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # 1. Baseline: no row slots, no CP.
    mb_base = _make_packed_mb(seqs_pos, vocab=vocab, padded_seq_len=16, calculate_entropy=calculate_entropy, seed=11)
    base_engine = _make_stub_engine(cp=False, cp_full_pos=False)
    with (
        mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
        mock.patch.object(torchtitan_tpu, "fused_linear_logprobs_tpu", side_effect=_exact_logprobs),
        torch.no_grad(),
    ):
        ids_b, extra_in_b, extra_kw_b, out_args_b = base_engine.prepare_model_inputs(mb_base)
        assert "cp_full_positions" not in extra_in_b
        assert extra_kw_b == {}
        hidden_b = token_embed[ids_b.clamp(0, vocab - 1)]
        out_base = base_engine.prepare_model_outputs((hidden_b, lm_head_weight_t), out_args_b, mb_base)

    # 2. Row-slot scheduled path (with optional 2-way contiguous CP sharding + cp_full_positions).
    mb_slot = _make_packed_mb(seqs_pos, vocab=vocab, padded_seq_len=16, calculate_entropy=calculate_entropy, seed=11)
    slots = gdn_slot_schedule.plan_micro_batch_gdn_slots(
        [mb_slot], auto_slot_len=slot_len, forward_only=True, allow_row_slot=True
    )
    assert slots == [slot_len]

    slot_engine = _make_stub_engine(cp=cp_enabled, cp_full_pos=cp_enabled, auto_slot=slot_len)
    rank_outputs: list[dict[str, torch.Tensor]] = []
    cp_world = 2 if cp_enabled else 1
    peer_Hidden_logprobs: dict[str, list[torch.Tensor]] = {"lp": [], "ent": []}

    # First run prepare_model_inputs + local forward on each simulated CP rank.
    per_rank_state = []
    for cp_rank in range(cp_world):
        if cp_enabled:

            def _cp_shard(batch, *_args, _r=cp_rank):
                half = batch["input"].shape[-1] // 2
                sl = slice(_r * half, (_r + 1) * half)
                return {
                    "input": batch["input"][..., sl],
                    "positions": batch["positions"][..., sl],
                    "labels": batch["labels"][..., sl],
                    "attention_masks": f"mask_rank_{_r}",
                }

            slot_engine._cp_attention_owner = lambda _fn=_cp_shard: SimpleNamespace(cp_shard=_fn)

        with (
            mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
            torch.no_grad(),
        ):
            ids_r, extra_in_r, extra_kw_r, out_args_r = slot_engine.prepare_model_inputs(mb_slot)

        if calculate_entropy and cp_rank == 0:
            # With grad enabled, differentiated entropy must safely bypass row-slot permutation even if MB_SLOT_LEN_KEY
            # was left on the micro-batch.
            with mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")):
                _, _, extra_kw_grad, out_args_grad = slot_engine.prepare_model_inputs(mb_slot)
            assert "gated_delta_document_resets" not in extra_kw_grad
            assert gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY not in out_args_grad

        assert extra_kw_r.get("gated_delta_document_resets") is False
        if cp_enabled:
            assert extra_kw_r.get("attention_masks") == f"mask_rank_{cp_rank}"
            assert extra_in_r["cp_full_positions"].tolist() == [0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]
        else:
            assert "cp_full_positions" not in extra_in_r

        hidden_r = token_embed[ids_r.clamp(0, vocab - 1)]
        lbl_1d = out_args_r["labels"].squeeze(0)
        peer_Hidden_logprobs["lp"].append(_exact_logprobs(hidden_r.squeeze(0), lbl_1d, lm_head_weight_t))
        if calculate_entropy:
            peer_Hidden_logprobs["ent"].append(
                torchtitan_tpu.chunked_entropy_from_hidden_tpu(hidden_r.squeeze(0), lm_head_weight_t, 1.0)
            )
        per_rank_state.append((hidden_r, out_args_r))

    for _cp_rank, (hidden_r, out_args_r) in enumerate(per_rank_state):
        if cp_enabled:
            call_idx = {"n": 0}

            def _gather_cp(local_tensor: torch.Tensor, padded_len: int, _state=call_idx) -> torch.Tensor:
                key = "ent" if (calculate_entropy and _state["n"] == 0) else "lp"
                _state["n"] += 1
                full = torch.cat(peer_Hidden_logprobs[key], dim=0)
                assert full.numel() == padded_len
                return full

            slot_engine._gather_context_parallel = _gather_cp

        with (
            mock.patch.object(torchtitan_tpu, "fused_linear_logprobs_tpu", side_effect=_exact_logprobs),
            torch.no_grad(),
        ):
            out_r = slot_engine.prepare_model_outputs((hidden_r, lm_head_weight_t), out_args_r, mb_slot)
        rank_outputs.append(out_r)

    for out_r in rank_outputs:
        torch.testing.assert_close(out_r["log_probs"].values(), out_base["log_probs"].values(), atol=1e-6, rtol=1e-6)
        if calculate_entropy:
            torch.testing.assert_close(out_r["entropy"].values(), out_base["entropy"].values(), atol=1e-6, rtol=1e-6)


def test_unflatten_seq_len_context_dual_torchtitan_and_missing_error():
    """Supports ``components.unflatten_shim``, legacy ``unflatten_shim``, and fails loudly when absent."""
    events: list[tuple[str, int | None]] = []

    @contextlib.contextmanager
    def _new_set_unflatten(n: int | None):
        events.append(("components", n))
        yield

    @contextlib.contextmanager
    def _old_set_unflatten(n: int | None):
        events.append(("legacy", n))
        yield

    # 1. google-pytorch/torchtitan HEAD (c7ba92e27): components.unflatten_shim
    tpu_pkg = types.ModuleType("torchtitan.experiments.tpu")
    tpu_pkg.__path__ = []  # type: ignore[attr-defined]
    comp_pkg = types.ModuleType("torchtitan.experiments.tpu.components")
    comp_pkg.__path__ = []  # type: ignore[attr-defined]
    new_shim = types.ModuleType("torchtitan.experiments.tpu.components.unflatten_shim")
    new_shim.set_unflatten_seq_len = _new_set_unflatten  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {
            "torchtitan": types.ModuleType("torchtitan"),
            "torchtitan.experiments": types.ModuleType("torchtitan.experiments"),
            "torchtitan.experiments.tpu": tpu_pkg,
            "torchtitan.experiments.tpu.components": comp_pkg,
            "torchtitan.experiments.tpu.components.unflatten_shim": new_shim,
            "torchtitan.experiments.tpu.unflatten_shim": None,
        },
    ):
        with gdn_slot_schedule.unflatten_seq_len_context(32768):
            pass
    assert events == [("components", 32768)]

    # 2. Private branch (0590025cb): torchtitan.experiments.tpu.unflatten_shim
    events.clear()
    old_shim = types.ModuleType("torchtitan.experiments.tpu.unflatten_shim")
    old_shim.set_unflatten_seq_len = _old_set_unflatten  # type: ignore[attr-defined]
    with mock.patch.dict(
        sys.modules,
        {
            "torchtitan": types.ModuleType("torchtitan"),
            "torchtitan.experiments": types.ModuleType("torchtitan.experiments"),
            "torchtitan.experiments.tpu": SimpleNamespace(unflatten_shim=old_shim),
            "torchtitan.experiments.tpu.components": None,
            "torchtitan.experiments.tpu.components.unflatten_shim": None,
            "torchtitan.experiments.tpu.unflatten_shim": old_shim,
        },
    ):
        with gdn_slot_schedule.unflatten_seq_len_context(16384):
            pass
    assert events == [("legacy", 16384)]

    # 3. Upstream pytorch/torchtitan (no experiments.tpu): clear RuntimeError
    with (
        mock.patch.dict(
            sys.modules,
            {
                "torchtitan": types.ModuleType("torchtitan"),
                "torchtitan.experiments": None,
                "torchtitan.experiments.tpu": None,
                "torchtitan.experiments.tpu.components": None,
                "torchtitan.experiments.tpu.components.unflatten_shim": None,
                "torchtitan.experiments.tpu.unflatten_shim": None,
            },
        ),
        pytest.raises(RuntimeError, match="set_unflatten_seq_len"),
    ):
        gdn_slot_schedule.unflatten_seq_len_context(8192)


def test_static_gdn_row_slot_len_rejected_under_cp_in_prepare_model_inputs():
    """Static ``_gdn_row_slot_len`` is rejected when ``cp_enabled`` so unpermuted ``cp_full_positions`` never occurs."""
    eng = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    eng.parallel_dims = SimpleNamespace(tp_enabled=False, cp_enabled=True, get_mesh=lambda _n: "cp_mesh")
    eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=32))
    eng.engine_config = SimpleNamespace(engine_kwargs={"tpu": {"seq_bucket_size": 8}})
    eng.trainer = SimpleNamespace(
        config=SimpleNamespace(
            parallelism=SimpleNamespace(context_parallel_load_balancer=None, context_parallel_ptrr_mask_key=None)
        )
    )
    eng._gdn_row_slot_len = 8
    eng._gdn_auto_row_slot_len = 0
    eng._use_fused_linear_cross_entropy = True
    eng._cp_full_positions = True

    mb = _make_packed_mb([[0, 1, 2], [0, 1, 2, 3]], padded_seq_len=16)
    with (
        mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")),
        pytest.raises(ValueError, match="gdn_row_slot_len is not supported with context parallelism"),
    ):
        eng.prepare_model_inputs(mb)


def test_reused_tensordict_stale_mb_slot_len_gated_and_entropy_excluded_before_unflatten():
    """HIGH-3 + MEDIUM-3: stale ``MB_SLOT_LEN_KEY`` never desyncs ``set_unflatten_seq_len`` / ``prepare_model_inputs``,
    and ``segmented_document_count`` + cross-rank ``MAX`` sync set identical ``max_num_segments`` budgets."""
    # 1. Sub-document resets inside 1 packed sequence count every pos == 0 reset.
    subdoc_pos = [0, 1, 2] * 7  # 7 sub-documents inside 1 sequence (batch_size == 1)
    mb_subdocs = _make_packed_mb([subdoc_pos], padded_seq_len=32)
    assert int(mb_subdocs.batch_size[0]) == 1
    assert gdn_slot_schedule.segmented_document_count(mb_subdocs) == 7

    # 2. Training pass with calculate_entropy=True on a reused TensorDict carrying stale MB_SLOT_LEN_KEY=8:
    #    neither unflatten_seq_len_context(8) nor prepare_model_inputs row-slot permutation may activate, AND
    #    segmented_mbs must count the micro-batch and sync global_max_docs across SPMD ranks.
    class TokamaxGatedDeltaKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_num_segments = 16

    kernel = TokamaxGatedDeltaKernel()
    root_mod = torch.nn.Sequential(kernel)

    mb_reused = _make_packed_mb([[0, 1, 2, 3], [0, 1, 2]], padded_seq_len=16, calculate_entropy=True)
    tu.assign_non_tensor_data(mb_reused, gdn_slot_schedule.MB_SLOT_LEN_KEY, 8)
    mb_reused["loss_mask"] = torch.ones((2, 4), dtype=torch.int64)

    def _build_fwb_stub(*, auto_slot_len: int) -> torchtitan_tpu.TorchTitanTPUEngineWithLMHead:
        eng = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
        eng.parallel_dims = SimpleNamespace(tp_enabled=False, cp_enabled=False, pp_enabled=False)
        eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=32))
        tpu_kw: dict[str, object] = {"seq_bucket_size": 8, "static_micro_batch_len": 16}
        if auto_slot_len > 0:
            tpu_kw["gdn_auto_row_slot_len"] = auto_slot_len
        eng.engine_config = SimpleNamespace(
            tensor_parallel_size=1,
            context_parallel_size=1,
            data_parallel_shard_size=1,
            data_parallel_replicate_size=1,
            entropy_checkpointing=False,
            entropy_from_logits_with_chunking=False,
            engine_kwargs={"tpu": tpu_kw},
        )
        eng.trainer = SimpleNamespace(
            config=SimpleNamespace(parallelism=SimpleNamespace(context_parallel_load_balancer=None)),
            train_context=contextlib.nullcontext,
        )
        eng.module = [root_mod]
        eng._graph_trainer_mode = False
        eng._use_simple_fsdp = False
        eng._simple_fsdp_loss_scale = None
        eng._tpu_eager_mode = None
        eng._gdn_row_slot_len = 0
        eng._gdn_auto_row_slot_len = auto_slot_len
        eng._sort_micro_batches_by_slot = False
        eng._use_fused_linear_cross_entropy = True
        eng._cp_full_positions = False
        eng._tpu_kernels = frozenset({"kda"})
        eng.get_data_parallel_group = lambda: None
        return eng

    unflatten_calls: list[int | None] = []
    seen_forward_kwargs: list[dict[str, object]] = []
    seen_post_gather_keys: list[bool] = []

    @contextlib.contextmanager
    def _spy_unflatten(seq_len: int | None):
        unflatten_calls.append(seq_len)
        yield

    def _peer_has_20_docs(tensor: torch.Tensor, op: object = None) -> None:
        if op == dist.ReduceOp.MAX and tensor.numel() == 1:
            tensor.copy_(torch.maximum(tensor, torch.tensor([20], dtype=tensor.dtype)))

    for auto_slot_len in (8, 0):
        kernel.max_num_segments = 16
        tu.assign_non_tensor_data(mb_reused, gdn_slot_schedule.MB_SLOT_LEN_KEY, 8)
        eng = _build_fwb_stub(auto_slot_len=auto_slot_len)

        def _fake_forward_step(mb: TensorDict, loss_function: object, forward_only: bool, _eng=eng, **_kw: object):
            with mock.patch.object(torchtitan_tpu, "get_device_id", return_value=torch.device("cpu")):
                _, _, extra_kw, out_args = _eng.prepare_model_inputs(mb)
            seen_forward_kwargs.append(dict(extra_kw))
            seen_post_gather_keys.append(gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY in out_args)
            loss = torch.ones((), requires_grad=True)
            return loss, {"model_output": {}, "loss": 1.0, "metrics": {}}

        eng.forward_step = _fake_forward_step  # type: ignore[method-assign]

        # Even if plan_micro_batch_gdn_slots were bypassed (stale key left at 8), forward_backward_batch must check
        # allow_auto_row_slot and differentiated-entropy exclusion BEFORE unflatten_seq_len_context.
        with (
            mock.patch.object(torchtitan_tpu, "prepare_micro_batches", return_value=([mb_reused], [[0, 1]])),
            mock.patch.object(torchtitan_tpu, "postprocess_batch_func", return_value={"ok": True}),
            mock.patch.object(torchtitan_tpu, "tpu_synchronize"),
            mock.patch.object(gdn_slot_schedule, "plan_micro_batch_gdn_slots", return_value=[8]),
            mock.patch.object(gdn_slot_schedule, "unflatten_seq_len_context", side_effect=_spy_unflatten),
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(dist, "all_reduce", side_effect=_peer_has_20_docs),
        ):
            res = eng.forward_backward_batch(mb_reused, loss_function=lambda **_kw: None, forward_only=False)

        assert res == {"ok": True}
        assert unflatten_calls == []
        assert seen_forward_kwargs[-1] == {}
        assert seen_post_gather_keys[-1] is False
        # Peer rank had 20 docs -> packed_document_budget(20) == 32 on this rank too!
        assert kernel.max_num_segments == tpu_model_kernels.packed_document_budget(20) == 32


def _make_fake_torchtitan_env(*, layout: str, has_216_features: bool) -> dict[str, object]:
    """Minimal ``sys.modules`` TorchTitan environment for ``c7ba92e27`` (``head``) or ``0590025cb`` (``priv``)."""
    assert layout in ("head", "priv")

    class _InnerAttnCfg:
        class _Owner:
            @classmethod
            def cp_shard(cls, batch, *_args):
                return batch

        _owner = _Owner

    class _GDNCfg:
        def __init__(self) -> None:
            if has_216_features:
                self.fuse_input_projections = False

    class _FakeModelCfg:
        dim = 16
        n_layers = 2
        vocab_size = 32
        first_full_attention_backend = _InnerAttnCfg()

        def __init__(self) -> None:
            self.layers = [SimpleNamespace(delta_net=_GDNCfg())]

        def traverse(self, _cls):
            return []

    class _FakeQwen35Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner_attention = torch.nn.Identity()
            self.w = torch.nn.Parameter(torch.ones(2, 2))

        if has_216_features:

            def forward(
                self,
                tokens: torch.Tensor,
                *,
                attention_masks: object = None,
                positions: torch.Tensor | None = None,
                cp_full_positions: torch.Tensor | None = None,
                gated_delta_document_resets: bool | None = None,
                **kwargs: object,
            ) -> torch.Tensor:
                return tokens

        else:

            def forward(
                self,
                tokens: torch.Tensor,
                *,
                attention_masks: object = None,
                positions: torch.Tensor | None = None,
                special_tokens: dict[str, int] | None = None,
            ) -> torch.Tensor:
                return tokens

    @dataclass
    class _ParallelismConfig:
        data_parallel_replicate_degree: int = 1
        data_parallel_shard_degree: int = 1
        fsdp_reshard_after_forward: object = "default"
        tensor_parallel_degree: int = 1
        pipeline_parallel_degree: int = 1
        context_parallel_degree: int = 1
        expert_parallel_degree: int = 1
        spmd_backend: str = "default"
        use_simple_fsdp: bool = False
        enable_tpu_topo_mesh: bool = False
        context_parallel_load_balancer: str | None = None
        context_parallel_ptrr_mask_key: str | None = None

    @dataclass
    class _TrainingConfig:
        enable_cpu_offload: bool = False
        disable_cuda_graphs: bool = True
        max_norm: float = 1.0
        mixed_precision_param: str = "bfloat16"
        mixed_precision_reduce: str = "float32"
        dtype: str = "float32"
        max_context_length: int = 4096

    @dataclass
    class _CompileConfig:
        enable: bool = False
        backend: str = "tpu"

    class _SelectiveAC:
        @dataclass
        class Config:
            pass

    class _FullAC:
        @dataclass
        class Config:
            pass

    class _Trainer:
        @dataclass
        class Config:
            model_spec: Any
            hf_assets_path: str
            optimizer: Any
            lr_scheduler: Any
            parallelism: _ParallelismConfig
            checkpoint: Any
            compile: _CompileConfig
            training: _TrainingConfig
            activation_checkpoint: Any
            dataloader: Any
            loss: Any
            override: Any = None

            def __post_init__(self) -> None:
                if self.override is None:
                    self.override = SimpleNamespace(imports=[])

        def __init__(self, cfg: Config) -> None:
            self.config = cfg
            p = cfg.parallelism
            self.parallel_dims = SimpleNamespace(
                tp=p.tensor_parallel_degree,
                pp=p.pipeline_parallel_degree,
                cp=p.context_parallel_degree,
                ep=p.expert_parallel_degree,
                tp_enabled=p.tensor_parallel_degree > 1,
                pp_enabled=p.pipeline_parallel_degree > 1,
                cp_enabled=p.context_parallel_degree > 1,
                ep_enabled=p.expert_parallel_degree > 1,
                get_optional_mesh=lambda _n: None,
                get_mesh=lambda _n: "cp_mesh",
            )
            self.model_parts = [_FakeQwen35Model()]

    tt_pkg = types.ModuleType("torchtitan")
    tt_pkg.__path__ = []  # type: ignore[attr-defined]
    exp_pkg = types.ModuleType("torchtitan.experiments")
    exp_pkg.__path__ = []  # type: ignore[attr-defined]
    tpu_pkg = types.ModuleType("torchtitan.experiments.tpu")
    tpu_pkg.__path__ = []  # type: ignore[attr-defined]

    cp_attn_mod = types.ModuleType("cp_attention")
    cp_attn_mod.install_tpu_cp_patches = lambda: None  # type: ignore[attr-defined]
    cp_attn_mod.tpu_cp_inner_attention = lambda base, _strat: base  # type: ignore[attr-defined]
    cp_attn_mod.tpu_cp_load_balancer = lambda strat: "headtail" if strat == "all_gather" else None  # type: ignore[attr-defined]
    cp_attn_mod.validate_tpu_context_parallel = lambda _m, _p: None  # type: ignore[attr-defined]
    cp_attn_mod.KVAllGatherCPInnerAttention = type("KVAllGatherCPInnerAttention", (torch.nn.Module,), {})  # type: ignore[attr-defined]

    ac_mod = types.ModuleType("activation_checkpoint")
    ac_mod.TPUSelectiveAC = _SelectiveAC  # type: ignore[attr-defined]

    qwen35_mod = types.ModuleType("torchtitan.experiments.tpu.qwen3_5")
    qwen35_mod.qwen3_5_configs = {"debug": _FakeModelCfg()}  # type: ignore[attr-defined]
    qwen35_mod.model_registry = lambda _flavor, **_kw: SimpleNamespace(  # type: ignore[attr-defined]
        model=_FakeModelCfg(), parallelize_fn=lambda m, **_k: m
    )
    qwen3_reg = types.ModuleType("torchtitan.experiments.tpu.qwen3.config_registry")
    qwen3_reg._FUSED_LOSS_OVERRIDE = "fused_loss_override"  # type: ignore[attr-defined]

    splash_mod = types.ModuleType("torchtitan.experiments.tpu.kernels.splash_attention")

    class _SplashAttention(torch.nn.Module):
        def __init__(self, *, original_module=None, **_kw) -> None:
            super().__init__()
            self.original_module = original_module

    splash_mod.SplashAttention = _SplashAttention  # type: ignore[attr-defined]
    splash_mod.splash_block_size_for = lambda s, max_block_size=512: min(s, max_block_size)  # type: ignore[attr-defined]

    mods: dict[str, object] = {
        "torchtitan": tt_pkg,
        "torchtitan.components": SimpleNamespace(),
        "torchtitan.components.checkpointer": SimpleNamespace(
            CheckpointManager=SimpleNamespace(Config=lambda **kw: SimpleNamespace(**kw))
        ),
        "torchtitan.components.loss": SimpleNamespace(
            CrossEntropyLoss=SimpleNamespace(Config=lambda: SimpleNamespace())
        ),
        "torchtitan.components.optimizer": SimpleNamespace(
            LRSchedulersContainer=SimpleNamespace(Config=lambda **kw: SimpleNamespace(**kw)),
            OptimizersContainer=SimpleNamespace(Config=lambda **kw: SimpleNamespace(**kw)),
            ParamGroupConfig=lambda **kw: SimpleNamespace(**kw),
        ),
        "torchtitan.components.data": SimpleNamespace(),
        "torchtitan.components.data.loader": SimpleNamespace(
            BaseDataLoader=type("BaseDataLoader", (), {"Config": type("Config", (), {})})
        ),
        "torchtitan.config": SimpleNamespace(
            CompileConfig=_CompileConfig,
            ParallelismConfig=_ParallelismConfig,
            TrainingConfig=_TrainingConfig,
        ),
        "torchtitan.config.transform": SimpleNamespace(
            ContextParallelTransform=lambda inner_attention: inner_attention,
            transform_model_config_=lambda m, _t: m,
        ),
        "torchtitan.distributed": SimpleNamespace(),
        "torchtitan.distributed.activation_checkpoint": SimpleNamespace(FullAC=_FullAC, SelectiveAC=_SelectiveAC),
        "torchtitan.trainer": SimpleNamespace(Trainer=_Trainer),
        "torchtitan.models": SimpleNamespace(),
        "torchtitan.models.common": SimpleNamespace(),
        "torchtitan.models.common.feed_forward": SimpleNamespace(
            FeedForward=type("FeedForward", (torch.nn.Module,), {})
        ),
        "torchtitan.experiments": exp_pkg,
        "torchtitan.experiments.tpu": tpu_pkg,
        "torchtitan.experiments.tpu.qwen3_5": qwen35_mod,
        "torchtitan.experiments.tpu.qwen3_5.config_registry": SimpleNamespace(),
        "torchtitan.experiments.tpu.qwen3": SimpleNamespace(config_registry=qwen3_reg),
        "torchtitan.experiments.tpu.qwen3.config_registry": qwen3_reg,
        "torchtitan.experiments.tpu.kernels": SimpleNamespace(splash_attention=splash_mod),
        "torchtitan.experiments.tpu.kernels.splash_attention": splash_mod,
    }
    if layout == "head":
        comp_pkg = types.ModuleType("torchtitan.experiments.tpu.components")
        comp_pkg.__path__ = []  # type: ignore[attr-defined]
        dist_pkg = types.ModuleType("torchtitan.experiments.tpu.distributed")
        dist_pkg.__path__ = []  # type: ignore[attr-defined]
        mods.update(
            {
                "torchtitan.experiments.tpu.components": comp_pkg,
                "torchtitan.experiments.tpu.components.activation_checkpoint": ac_mod,
                "torchtitan.experiments.tpu.distributed": dist_pkg,
                "torchtitan.experiments.tpu.distributed.cp_attention": cp_attn_mod,
                "torchtitan.experiments.tpu.activation_checkpoint": None,
                "torchtitan.experiments.tpu.cp_attention": None,
            }
        )
    else:
        mods.update(
            {
                "torchtitan.experiments.tpu.components": None,
                "torchtitan.experiments.tpu.components.activation_checkpoint": None,
                "torchtitan.experiments.tpu.distributed": None,
                "torchtitan.experiments.tpu.distributed.cp_attention": None,
                "torchtitan.experiments.tpu.activation_checkpoint": ac_mod,
                "torchtitan.experiments.tpu.cp_attention": cp_attn_mod,
            }
        )
    return mods


def test_engine_init_dual_torchtitan_compatibility_and_validation_guards(caplog):
    """Tests dual ``c7ba92e27`` / ``0590025cb`` layouts, loud ``RuntimeError``s on missing ``#216`` features,
    CRITICAL-1 (``gdn_row_slot_len`` + CP), HIGH-2 (``fuse_gdn_projections`` + TP), and MEDIUM-2 bucket warning."""
    from verl_hardware_plugin.accelerators.tpu.engines import fused_logprob

    def _build_engine(
        tpu_kw: dict[str, object],
        *,
        layout: str,
        has_216: bool,
        tp: int = 1,
        cp: int = 1,
    ) -> torchtitan_tpu.TorchTitanTPUEngineWithLMHead:
        hf_cfg = SimpleNamespace(model_type="qwen3_5_text", hidden_size=16, num_hidden_layers=2, vocab_size=32)
        model_cfg = SimpleNamespace(hf_config=hf_cfg, path="/tmp/fake_hf")
        eng_cfg = SimpleNamespace(
            use_torch_compile=False,
            attn_type="sdpa",
            max_seq_len=4096,
            data_parallel_replicate_size=1,
            data_parallel_shard_size=1,
            reshard_after_forward="default",
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            context_parallel_size=cp,
            expert_parallel_size=1,
            spmd_backend="default",
            forward_only=False,
            pad_to_length=False,
            pad_to_length_bucket=256,
            activation_checkpoint="selective",
            full_determinism=False,
            param_offload=False,
            optimizer_offload=False,
            entropy_from_logits_with_chunking=False,
            engine_kwargs={"tpu": {"use_fused_linear_cross_entropy": tp == 1, **tpu_kw}},
        )
        opt_cfg = SimpleNamespace(
            name="AdamW",
            lr=1e-4,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            total_training_steps=10,
            lr_warmup_steps=1,
            lr_warmup_steps_ratio=0.0,
            decay_type="cosine",
            min_lr_factor=0.1,
            clip_grad=1.0,
        )
        ckpt_cfg = SimpleNamespace(async_save=False)
        fake_env = _make_fake_torchtitan_env(layout=layout, has_216_features=has_216)
        with (
            mock.patch.dict(sys.modules, fake_env),
            mock.patch.object(fused_logprob, "require_tokamax", return_value=None),
        ):
            return torchtitan_tpu.TorchTitanTPUEngineWithLMHead(model_cfg, eng_cfg, opt_cfg, ckpt_cfg)  # type: ignore[arg-type]

    # 1. On torchtitan HEAD (c7ba92e27: no #216 features), baseline + CP initialize cleanly with zero errors.
    eng_head = _build_engine({}, layout="head", has_216=False)
    assert eng_head._cp_full_positions is False and eng_head._gdn_auto_row_slot_len == 0
    eng_head_cp = _build_engine({"context_parallel_strategy": "ulysses"}, layout="head", has_216=False, cp=2)
    assert eng_head_cp.parallel_dims.cp_enabled is True

    # 2. Enabling #216-specific options against c7ba92e27 fails loudly with clear RuntimeError messages.
    with pytest.raises(RuntimeError, match="cp_full_positions"):
        _build_engine(
            {"cp_full_positions": True, "context_parallel_strategy": "ulysses"},
            layout="head",
            has_216=False,
            cp=2,
        )
    with pytest.raises(RuntimeError, match="gated_delta_document_resets"):
        _build_engine({"gdn_auto_row_slot_len": 256}, layout="head", has_216=False)
    with pytest.raises(RuntimeError, match="fuse_input_projections"):
        _build_engine({"fuse_gdn_projections": True}, layout="head", has_216=False)

    # 3. On private branch (0590025cb: #216 features present), all #216 options initialize cleanly, AND MEDIUM-2 logs
    #    a warning when gdn_auto_row_slot_len is set without static_micro_batch_len and seq_bucket_size % slot != 0.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=torchtitan_tpu.__name__):
        eng_priv = _build_engine(
            {
                "cp_full_positions": True,
                "gdn_auto_row_slot_len": 1024,
                "fuse_gdn_projections": True,
                "sort_micro_batches_by_slot": True,
                "context_parallel_strategy": "ulysses",
                "seq_bucket_size": 256,
            },
            layout="priv",
            has_216=True,
            cp=2,
        )
    assert eng_priv._cp_full_positions is True and eng_priv._gdn_auto_row_slot_len == 1024
    assert eng_priv.config.model_spec.model.layers[0].delta_net.fuse_input_projections is True
    assert "gdn_auto_row_slot_len=1024 is set without static_micro_batch_len" in caplog.text

    # 4. CRITICAL-1: static gdn_row_slot_len + CP > 1 raises ValueError in __init__.
    with pytest.raises(ValueError, match="gdn_row_slot_len is not supported with context"):
        _build_engine(
            {"gdn_row_slot_len": 512, "context_parallel_strategy": "ulysses"},
            layout="priv",
            has_216=True,
            cp=2,
        )

    # 5. HIGH-2: fuse_gdn_projections + tensor_parallel_size > 1 raises ValueError in __init__.
    with pytest.raises(ValueError, match="fuse_gdn_projections does not support tensor parallelism"):
        _build_engine({"fuse_gdn_projections": True}, layout="priv", has_216=True, tp=2)
