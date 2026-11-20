# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Micro-batch slot planning, sorting and CP full-position helpers for linear-attention / hybrid models.

Pairs with ``google-pytorch/torchtitan#216``, which adds three general capabilities upstream in torchtitan so no
plugin-side monkeypatches are needed:

1. ``cp_full_positions`` on ``model.forward``: passes the pre-``cp_shard`` unsharded 1-D positions to every CP
   attention and head-parallel GatedDeltaNet layer, eliminating per-layer ``all_gather(positions)`` collectives across
   both Ulysses and KV all-gather CP.
2. ``gated_delta_document_resets`` per-batch override on ``model.forward``: runs unsegmented GatedDeltaNet kernels when
   every ``[B, S]`` row (``set_unflatten_seq_len``) holds at most one document.
3. ``fuse_input_projections`` on ``TpuGatedDeltaNet.Config``: fuses the six bias-free GatedDeltaNet input projections
   into a single matmul.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.model import extract_multi_modal_inputs
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    bucket_length,
    row_slot_permutation,
    unwrap_metadata,
)

MB_SLOT_LEN_KEY = "tpu_gdn_mb_slot_len"
POST_GATHER_CONTIG_SRC_KEY = "gdn_contig_src_post_gather"

SLOT_CAT_ROW_SLOT = 0
SLOT_CAT_SINGLE_DOC = 1
SLOT_CAT_SEGMENTED = 2

_MASK_ROW_SLOT = 1
_MASK_SINGLE_DOC = 2


def unflatten_seq_len_context(seq_len: Optional[int]) -> Any:
    """Sets ``torchtitan``'s TPU unflatten row length across ``components.unflatten_shim`` and legacy layouts."""
    try:
        from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
    except ImportError:
        try:
            from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
        except ImportError as exc:
            raise RuntimeError(
                "GDN row-slot scheduling requires 'set_unflatten_seq_len' from google-pytorch/torchtitan "
                "(torchtitan.experiments.tpu.components.unflatten_shim or "
                "torchtitan.experiments.tpu.unflatten_shim)"
            ) from exc
    return set_unflatten_seq_len(seq_len)


def forward_accepts_kwarg(module: Any, name: str) -> bool:
    """True when ``module.forward`` explicitly declares keyword argument ``name`` (unwrapping compiled/parallelized)."""
    seen: set[int] = set()
    while getattr(module, "_orig_mod", None) is not None and id(module) not in seen:
        seen.add(id(module))
        module = module._orig_mod
    fwd = getattr(module, "_orig_forward", None) or getattr(module, "forward", None)
    if callable(fwd) and getattr(fwd, "__name__", "") == "forward_with_redistribution":
        cls_fwd = getattr(type(module), "forward", None)
        if callable(cls_fwd):
            fwd = cls_fwd
    if not callable(fwd):
        return False
    try:
        params = inspect.signature(fwd).parameters
    except (TypeError, ValueError):
        return False
    return name in params


def validate_model_cp_slot_support(
    module_parts: Any,
    *,
    need_cp_full_positions: bool,
    need_document_resets_override: bool,
) -> tuple[bool, bool]:
    """Checks that ``module_parts[0].forward`` supports the requested ``google-pytorch/torchtitan#216`` kwargs."""
    parts = module_parts if isinstance(module_parts, list | tuple | torch.nn.ModuleList) else [module_parts]
    root = parts[0] if parts else None
    has_full_pos = forward_accepts_kwarg(root, "cp_full_positions") if root is not None else False
    has_doc_resets = forward_accepts_kwarg(root, "gated_delta_document_resets") if root is not None else False
    if need_cp_full_positions and not has_full_pos:
        raise RuntimeError(
            "engine_kwargs.tpu.cp_full_positions=True requires a torchtitan model whose forward() accepts "
            "'cp_full_positions' (google-pytorch/torchtitan#216)"
        )
    if need_document_resets_override and not has_doc_resets:
        raise RuntimeError(
            "engine_kwargs.tpu.gdn_auto_row_slot_len requires a torchtitan model whose forward() accepts "
            "'gated_delta_document_resets' (google-pytorch/torchtitan#216)"
        )
    return has_full_pos, has_doc_resets


def enable_fused_gdn_projections(model_cfg: Any) -> int:
    """Sets ``fuse_input_projections = True`` on every sub-config of ``model_cfg`` that declares it.

    Raises ``RuntimeError`` if no sub-config supports ``fuse_input_projections``, so an older torchtitan or a model
    without GatedDeltaNet layers fails loudly at init instead of silently ignoring the option.
    """
    visited: set[int] = set()
    matched = 0

    def _walk(obj: Any) -> None:
        nonlocal matched
        if obj is None or id(obj) in visited:
            return
        if isinstance(obj, int | float | str | bool | bytes):
            return
        visited.add(id(obj))
        if hasattr(obj, "fuse_input_projections"):
            obj.fuse_input_projections = True
            matched += 1
        if isinstance(obj, list | tuple):
            for item in obj:
                _walk(item)
            return
        if isinstance(obj, dict):
            for item in obj.values():
                _walk(item)
            return
        if hasattr(obj, "__dict__") or hasattr(type(obj), "__slots__"):
            names: set[str] = set()
            if hasattr(obj, "__dict__"):
                names.update(obj.__dict__.keys())
            for cls in type(obj).__mro__:
                slots = getattr(cls, "__slots__", ())
                if isinstance(slots, str):
                    names.add(slots)
                else:
                    names.update(slots)
            for attr in names:
                if not attr.startswith("_"):
                    _walk(getattr(obj, attr, None))

    _walk(model_cfg)
    if matched == 0:
        raise RuntimeError(
            "engine_kwargs.tpu.fuse_gdn_projections=True matched 0 modules with 'fuse_input_projections' "
            "(requires Qwen3.5 GatedDeltaNet with google-pytorch/torchtitan#216)"
        )
    return matched


def has_internal_document_resets(position_values_1d: torch.Tensor, cu_seqlens: torch.Tensor) -> bool:
    """True when any sequence in ``cu_seqlens`` resets ``position == 0`` at an interior token (packed sub-documents).

    Unlike a coarse ``(positions == 0).sum() <= n_seqs`` check, this inspects the exact reset indices against sequence
    starts ``cu_seqlens[:-1]``: a continuation sequence that starts at ``position > 0`` and later resets to ``0`` at an
    interior offset is correctly detected as multi-document.
    """
    offs = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if len(offs) < 2:
        return False
    total = offs[-1]
    pos = position_values_1d.detach().cpu().reshape(-1)[:total]
    if pos.numel() != total:
        return True
    seq_starts = set(offs[:-1])
    zero_indices = torch.nonzero(pos == 0, as_tuple=False).reshape(-1).tolist()
    return any(int(idx) not in seq_starts for idx in zero_indices if int(idx) > 0)


def _extract_1d_packed_positions(mb: TensorDict, expected_tokens: int) -> Optional[torch.Tensor]:
    """Returns 1-D CPU packed positions when ``mb['position_ids']`` is 1-D text positions; ``None`` for 3-D MRoPE."""
    if "position_ids" not in mb.keys():
        return None
    if "multi_modal_inputs" in mb.keys() and extract_multi_modal_inputs(mb.get("multi_modal_inputs", [])):
        return None
    pos = mb["position_ids"]
    vals = pos.values() if getattr(pos, "is_nested", False) else pos
    if vals.ndim > 1:
        # Nested 3-D MRoPE has values shape (total_tokens, 3) or (3, total_tokens); refuse rather than flatten.
        if vals.numel() != expected_tokens:
            return None
        vals = vals.reshape(-1)
    if int(vals.numel()) != expected_tokens:
        return None
    return vals.detach().cpu()


def segmented_document_count(mb: TensorDict) -> int:
    """Document count of a segmented micro-batch, accounting for interior ``position == 0`` resets."""
    batch_seqs = int(mb.batch_size[0]) if len(mb.batch_size) > 0 else 0
    if "input_ids" not in mb.keys() or "position_ids" not in mb.keys():
        return batch_seqs
    ids = mb["input_ids"]
    use_rmpad = bool(unwrap_metadata(tu.get_non_tensor_data(data=mb, key="use_remove_padding", default=True)))
    if not use_rmpad:
        return batch_seqs
    total_tokens = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
    pos_1d = _extract_1d_packed_positions(mb, total_tokens)
    zero_resets = int((pos_1d == 0).sum().item()) if pos_1d is not None and pos_1d.numel() > 0 else 0
    return max(batch_seqs, zero_resets)


def sync_max_segmented_documents(local_max_docs: int) -> int:
    """Synchronizes the maximum segmented document count across all SPMD ranks via CPU ``MAX`` all-reduce."""
    max_docs = int(local_max_docs)
    if dist.is_initialized() and dist.get_world_size() > 1:
        doc_t = torch.tensor([max_docs], dtype=torch.int64)
        dist.all_reduce(doc_t, op=dist.ReduceOp.MAX)
        max_docs = int(doc_t.item())
    return max_docs


def classify_gdn_micro_batch(
    mb: TensorDict,
    *,
    auto_slot_len: int,
    padded_len: int,
    forward_only: bool,
    allow_row_slot: bool,
    allow_single_doc: bool = True,
) -> tuple[int, int, int]:
    """Returns ``(eligibility_mask, sort_category, total_tokens)`` for one micro-batch.

    Shared by both ``sort_micro_batches_for_slots`` and ``plan_micro_batch_gdn_slots`` so sort order and resolved slot
    assignments always agree on category boundaries.
    """
    ids = mb["input_ids"]
    use_rmpad = bool(unwrap_metadata(tu.get_non_tensor_data(data=mb, key="use_remove_padding", default=True)))
    if not use_rmpad or not getattr(ids, "is_nested", False):
        n_tok = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
        return 0, SLOT_CAT_SEGMENTED, n_tok

    offs_t = ids.offsets().detach().cpu()
    offs = [int(x) for x in offs_t.tolist()]
    seq_lens = [b - a for a, b in zip(offs[:-1], offs[1:], strict=True)]
    total_tokens = offs[-1] if offs else 0
    n_seqs = len(seq_lens)
    max_seq = max(seq_lens, default=0)
    if padded_len <= 0 or n_seqs == 0 or min(seq_lens) <= 0:
        return 0, SLOT_CAT_SEGMENTED, total_tokens

    pos_1d = _extract_1d_packed_positions(mb, total_tokens)
    if pos_1d is None or has_internal_document_resets(pos_1d, offs_t):
        return 0, SLOT_CAT_SEGMENTED, total_tokens

    starts_at_zero = all(int(pos_1d[s].item()) == 0 for s in offs[:-1])
    calc_ent = bool(unwrap_metadata(tu.get_non_tensor_data(data=mb, key="calculate_entropy", default=False)))
    mask = 0
    if (
        allow_row_slot
        and starts_at_zero
        and auto_slot_len > 0
        and (not calc_ent or forward_only)
        and padded_len >= auto_slot_len
        and padded_len % auto_slot_len == 0
        and n_seqs <= padded_len // auto_slot_len
        and max_seq <= auto_slot_len
    ):
        mask |= _MASK_ROW_SLOT
    if allow_single_doc and n_seqs == 1 and int(pos_1d[0].item()) == 0 and max_seq <= padded_len:
        mask |= _MASK_SINGLE_DOC

    if mask & _MASK_ROW_SLOT:
        cat = SLOT_CAT_ROW_SLOT
    elif mask & _MASK_SINGLE_DOC:
        cat = SLOT_CAT_SINGLE_DOC
    else:
        cat = SLOT_CAT_SEGMENTED
    return mask, cat, total_tokens


def sort_micro_batches_for_slots(
    micro_batches: list[TensorDict],
    indices: list[Any],
    *,
    auto_slot_len: int,
    static_len: int,
    bucket_size: int,
    forward_only: bool,
    allow_row_slot: bool,
    allow_single_doc: bool = True,
) -> tuple[list[TensorDict], list[Any]]:
    """Orders ``(micro_batches, indices)`` by ``(slot_category, -total_tokens, original_idx)``."""
    if len(micro_batches) <= 1:
        return micro_batches, indices
    step = max(1, int(bucket_size))

    def _key(idx: int) -> tuple[int, int, int]:
        mb = micro_batches[idx]
        ids = mb["input_ids"]
        n_tok = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
        est_padded = bucket_length(max(n_tok, int(static_len or 0)), step)
        _, cat, total = classify_gdn_micro_batch(
            mb,
            auto_slot_len=auto_slot_len,
            padded_len=est_padded,
            forward_only=forward_only,
            allow_row_slot=allow_row_slot,
            allow_single_doc=allow_single_doc,
        )
        return (cat, -total, idx)

    order = sorted(range(len(micro_batches)), key=_key)
    return [micro_batches[k] for k in order], [indices[k] for k in order]


def plan_micro_batch_gdn_slots(
    micro_batches: list[TensorDict],
    *,
    auto_slot_len: int,
    forward_only: bool,
    allow_row_slot: bool,
    allow_single_doc: bool = True,
) -> list[int]:
    """Resolves per-micro-batch GDN unflatten slot lengths across all ranks via ``BAND`` all-reduce.

    Stores ``MB_SLOT_LEN_KEY`` on each micro-batch: ``auto_slot_len`` when every rank's micro-batch fits row slots,
    ``padded_len`` when every rank's micro-batch is a single unsegmented document, or ``0`` (segmented).
    """
    if not micro_batches:
        return []
    masks: list[int] = []
    padded_lens: list[int] = []
    for mb in micro_batches:
        ids = mb["input_ids"]
        fallback_len = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
        padded_len = int(tu.get_non_tensor_data(data=mb, key="tpu_padded_seq_len", default=fallback_len) or 0)
        padded_lens.append(padded_len)
        mask, _, _ = classify_gdn_micro_batch(
            mb,
            auto_slot_len=auto_slot_len,
            padded_len=padded_len,
            forward_only=forward_only,
            allow_row_slot=allow_row_slot,
            allow_single_doc=allow_single_doc,
        )
        masks.append(mask)

    if dist.is_initialized() and dist.get_world_size() > 1:
        mask_t = torch.tensor(masks, dtype=torch.int64)
        dist.all_reduce(mask_t, op=dist.ReduceOp.BAND)
        masks = [int(x) for x in mask_t.tolist()]

    resolved: list[int] = []
    for mb, m, padded_len in zip(micro_batches, masks, padded_lens, strict=True):
        slot = auto_slot_len if (m & _MASK_ROW_SLOT) else (padded_len if (m & _MASK_SINGLE_DOC) else 0)
        tu.assign_non_tensor_data(mb, MB_SLOT_LEN_KEY, slot)
        resolved.append(slot)
    return resolved


def build_cp_full_positions(
    micro_batch: TensorDict,
    padded_seq_len: int,
    *,
    unsegmented: bool = False,
    slot_src_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Builds the 1-D unsharded ``cp_full_positions`` tensor (length ``padded_seq_len``) on CPU.

    Zero-pads trailing padding tokens (``value=0``) across both segmented and unsegmented micro-batches (and permutes
    by ``slot_src_cpu`` when row-slotted), matching ``pad_packed_inputs_for_tpu`` and
    ``TpuQwen35Model.preprocess_inputs`` in ``google-pytorch/torchtitan#216``. Each padding token has position ``0``,
    so ``segment_ids_from_positions`` isolates every sequence's real tokens from trailing padding, preserves Splash
    Attention block skipping on trailing padding tiles under CP, and keeps ``cp_full_positions=True`` numerically
    identical to ``cp_full_positions=False``.

    Raises ``ValueError`` if ``micro_batch`` has multimodal inputs or non-1D MRoPE positions.
    """
    del unsegmented
    ids = micro_batch["input_ids"]
    total_tokens = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
    orig_pos_cpu = _extract_1d_packed_positions(micro_batch, total_tokens)
    if orig_pos_cpu is None:
        raise ValueError(
            "cp_full_positions supports 1-D text position_ids only (got multimodal inputs or multi-dimensional MRoPE "
            "position_ids)"
        )
    pad_len = padded_seq_len - int(orig_pos_cpu.numel())
    if pad_len < 0:
        raise ValueError(
            f"cp_full_positions: position_ids has {orig_pos_cpu.numel()} tokens > padded_seq_len={padded_seq_len}"
        )
    padded_zero = torch.nn.functional.pad(orig_pos_cpu, (0, pad_len), value=0)
    if slot_src_cpu is not None:
        return padded_zero[slot_src_cpu].contiguous()
    return padded_zero.contiguous()


def permute_packed_for_row_slots(
    cu_seqlens: torch.Tensor,
    padded_seq_len: int,
    slot_len: int,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permutes ``(input_ids, position_ids, labels)`` into ``slot_len`` row slots and returns index maps."""
    slot_src_cpu, contig_src_cpu = row_slot_permutation(cu_seqlens, padded_seq_len, slot_len)
    slot_src_dev = slot_src_cpu.to(input_ids.device)
    contig_src_dev = contig_src_cpu.to(input_ids.device)
    return (
        input_ids[..., slot_src_dev],
        position_ids[..., slot_src_dev],
        labels[..., slot_src_dev],
        slot_src_cpu,
        contig_src_dev,
    )
