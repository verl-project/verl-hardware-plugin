# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct ``tpu-sync`` install of Qwen3.5 (hybrid GatedDeltaNet, dense or MoE) into vLLM's parameters.

Without this module, hybrid / MoE models install the received weights through vLLM's own checkpoint loaders: every
rollout rank all-gathers every tensor over its TP group and streams the full HF tensors into
``model_runner.reload_weights``, whose loaders fuse, TP-slice and re-layout them (``_install_with_vllm_loaders``).
For Qwen3.5-35B-A3B on 8 TensorCores that takes ~55 s of every ~60 s ``update_weights``: the full 70 GB model is
gathered on every rank, and vLLM's MoE loaders loop over the experts.

Here, as in the dense Qwen3 path, every rollout rank receives only its own TP slice of each tensor and writes it
straight into vLLM's (fused, TP-sharded, TPU-laid-out) parameters, with no gather and no loader. tpu-sync can only
deliver one contiguous block per tensor and rank, so the trainer exports the tensors that vLLM shards in several
pieces as those pieces (``make_tpu_sync_local_layout``), and each rank concatenates its pieces locally:

==========================================  ============================  =========================================
Trainer tensor (tpu-sync name)              Received by rank ``r``        Written into (vLLM, TP rank ``r``)
==========================================  ============================  =========================================
``linear_attn.in_proj_{q,k,v}`` (split      dim-0 slice                   ``in_proj_qkvz`` = [q_r; k_r; v_r; z_r]
from HF's ``in_proj_qkv``), ``in_proj_z``                                 (transposed: vllm-torchtpu stores the
``linear_attn.in_proj_b/a`` [Hv, H]         dim-0 slice                   dense weights as [n_in, n_out])
                                                                          ``in_proj_ba`` = [b_r; a_r]
``linear_attn.conv1d_{q,k,v}`` (split       dim-0 slice                   ``conv1d`` = [q_r; k_r; v_r]
from HF's ``conv1d``)
``linear_attn.A_log`` / ``dt_bias``         whole (tiny)                  their rank-``r`` slices
``linear_attn.out_proj`` [H, V]             dim-1 slice (row parallel)    ``out_proj``
``mlp.experts.gate_proj/up_proj`` [E, I,    dim-1 slice (moe_intermed.)   ``experts.w13_weight`` [E, H, 2A]:
H] (split from HF's ``gate_up_proj``)                                     gate^T in [0, I/tp), up^T in [A, A+I/tp)
                                                                          (A: I/tp padded to 128 by vllm-torchtpu)
``mlp.experts.down_proj`` [E, H, I]         dim-2 slice                   ``experts.w2_weight`` [E, A, H]: down^T
``mlp.gate``, ``shared_expert_gate``        whole (replicated in vLLM)    as is
==========================================  ============================  =========================================

Everything else (attention q/k/v/o with the replicated KV heads, the shared expert, norms, embeddings) goes through
the dense path's rules in ``tpu_sync_worker_extension``.

The layouts above are those of vLLM 0.29's Qwen3.5 (non-interleaved ``in_proj_qkvz`` / ``in_proj_ba``) and of
vllm-torchtpu's unquantized MoE method. ``plan_direct_install`` checks them on the live vLLM model and falls back to
vLLM's loaders for anything else (other models, quantized or expert-parallel MoE, a replicated ``in_proj_ba``, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, NamedTuple, Optional

import torch

# HF model types of the Qwen3.5 text models this module installs.
QWEN3_5_MODEL_TYPES = ("qwen3_5_text", "qwen3_5_moe_text")

# Trainer tensors that every rank receives whole: vLLM keeps them unsharded (the MoE router and shared-expert
# gate), or they are tiny and sliced on the receiver (A_log / dt_bias).
REPLICATED_SUFFIXES = (
    ".linear_attn.A_log",
    ".linear_attn.dt_bias",
    ".mlp.gate.weight",
    ".mlp.shared_expert_gate.weight",
)
# Row-parallel (input dimension, dim 1, sharded) dense weights in addition to the dense path's o_proj / down_proj.
ROW_PARALLEL_SUFFIXES = (".linear_attn.out_proj.weight",)


class SplitTensor(NamedTuple):
    """An HF tensor that the trainer exports as its pieces (see ``make_tpu_sync_local_layout``).

    vLLM TP-shards each piece separately, so rank ``r``'s slice of the fused tensor is several blocks, while its
    slice of each piece is one.
    """

    fused: str  # suffix of the HF tensor
    pieces: tuple[str, ...]  # suffixes of its pieces, in concat order
    dim: int  # concat dimension
    qkv: bool  # pieces are GatedDeltaNet q / k / v rows (sizes from the config); else two equal halves


SPLIT_TENSORS = (
    SplitTensor(
        ".linear_attn.in_proj_qkv.weight",
        (".linear_attn.in_proj_q.weight", ".linear_attn.in_proj_k.weight", ".linear_attn.in_proj_v.weight"),
        dim=0,
        qkv=True,
    ),
    SplitTensor(
        ".linear_attn.conv1d.weight",
        (".linear_attn.conv1d_q.weight", ".linear_attn.conv1d_k.weight", ".linear_attn.conv1d_v.weight"),
        dim=0,
        qkv=True,
    ),
    SplitTensor(".mlp.experts.gate_up_proj", (".mlp.experts.gate_proj", ".mlp.experts.up_proj"), dim=-2, qkv=False),
)
# The dimension that vLLM's TP shards in each piece, and in the experts' down_proj (the expert intermediate size).
SPLIT_TP_DIMS = {
    ".linear_attn.in_proj_q.weight": 0,
    ".linear_attn.in_proj_k.weight": 0,
    ".linear_attn.in_proj_v.weight": 0,
    ".linear_attn.conv1d_q.weight": 0,
    ".linear_attn.conv1d_k.weight": 0,
    ".linear_attn.conv1d_v.weight": 0,
    ".mlp.experts.gate_proj": 1,
    ".mlp.experts.up_proj": 1,
    ".mlp.experts.down_proj": 2,
}
# Every tensor of one GatedDeltaNet layer / MoE block that ``install`` reads, as the trainer sends them.
GDN_LAYER_SUFFIXES = (
    *SPLIT_TENSORS[0].pieces,
    *SPLIT_TENSORS[1].pieces,
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.A_log",
    ".linear_attn.dt_bias",
)
EXPERT_BLOCK_SUFFIXES = (*SPLIT_TENSORS[2].pieces, ".mlp.experts.down_proj")
# vLLM fuses these into one parameter: (fused suffix, source suffixes in concat order).
FUSED_TARGETS = (
    (
        ".mlp.shared_expert.gate_up_proj.weight",
        (".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight"),
    ),
    (".linear_attn.in_proj_ba.weight", (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight")),
)

# vLLM MoE methods whose expert weight layout ``_write_w13`` / ``_write_w2`` know: vllm-torchtpu's unquantized method
# (transposed, each w13 half padded to a multiple of 128) and vLLM's own unquantized method ([E, 2I, H] / [E, H, I]).
MOE_LAYOUT_TPU = "tpu_transposed_padded"
MOE_LAYOUT_VLLM = "vllm"
_MOE_LAYOUTS = {"VllmUnquantizedFusedMoEMethod": MOE_LAYOUT_TPU, "UnquantizedFusedMoEMethod": MOE_LAYOUT_VLLM}
_MOE_PAD_MULTIPLE = 128  # vllm-torchtpu's VllmUnquantizedFusedMoEMethod.process_weights_after_loading

_ERROR_PREFIX = "tpu-sync direct install"
_GDN_RE = re.compile(r"^(.*\.linear_attn)\.in_proj_q\.weight$")
_EXPERTS_RE = re.compile(r"^(.*\.mlp\.experts)\.gate_proj$")


class Qwen3_5Dims(NamedTuple):
    """GatedDeltaNet and MoE sizes of a Qwen3.5 text config."""

    key_dim: int  # linear_num_key_heads * linear_key_head_dim (each of q and k)
    value_dim: int  # linear_num_value_heads * linear_value_head_dim
    num_k_heads: int
    num_v_heads: int
    moe_intermediate_size: Optional[int]  # None for the dense models

    @classmethod
    def from_config(cls, hf_text_config: Any) -> Qwen3_5Dims:
        num_k_heads = int(hf_text_config.linear_num_key_heads)
        num_v_heads = int(hf_text_config.linear_num_value_heads)
        moe_inter = getattr(hf_text_config, "moe_intermediate_size", None)
        return cls(
            key_dim=num_k_heads * int(hf_text_config.linear_key_head_dim),
            value_dim=num_v_heads * int(hf_text_config.linear_value_head_dim),
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            moe_intermediate_size=int(moe_inter) if moe_inter else None,
        )


@dataclass
class DirectInstallPlan:
    """What ``install`` needs, checked once against the live vLLM model by ``plan_direct_install``."""

    dims: Qwen3_5Dims
    tp_rank: int
    tp_size: int
    moe_layout: Optional[str]  # MOE_LAYOUT_TPU / MOE_LAYOUT_VLLM, None without experts
    # Updated by ``install``: whether vllm-torchtpu's zero MoE padding was verified (first sync only), and vLLM's
    # resolved expert parameter keys (``(prefix, leaf) -> key``), which do not change between syncs.
    padding_checked: bool = False
    expert_keys: dict[tuple[str, str], str] = field(default_factory=dict)


def make_tpu_sync_local_layout(hf_text_config: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Trainer side: the ``TrainerWeights.local_transform`` that exports the ``SPLIT_TENSORS`` as their pieces.

    The pieces are views of the fused tensors along a dimension that FSDP does not shard for the experts (dim 0, the
    expert, stays sharded, so they are still exported as local shards). ``in_proj_qkv`` / ``conv1d`` come out of the
    state-dict adapter's concatenation of torchtitan's separate q / k / v, so their rows are split back at the
    config's q / k / v boundaries.
    """
    # TODO(tpu): torchtitan keeps the GatedDeltaNet q / k / v (and conv q / k / v) FSDP-sharded, but its state-dict
    # adapter concatenates them into HF's in_proj_qkv / conv1d, which all-gathers them; the pieces below are views of
    # that replicated tensor. Small for Qwen3.5-35B-A3B (~32 MiB per layer, ~1 GB per sync, one layer at a time). To
    # export them as local shards with no gather, take torchtitan's unfused q / k / v before the adapter fuses them
    # (for local_shards() only: plain iteration needs HF's fused tensors).
    dims = Qwen3_5Dims.from_config(hf_text_config)
    qkv_rows = (dims.key_dim, dims.key_dim, dims.value_dim)

    def local_layout(chunk: dict[str, Any]) -> dict[str, Any]:
        for split in SPLIT_TENSORS:
            for name in [n for n in chunk if n.endswith(split.fused)]:
                tensor = chunk.pop(name)
                total = tensor.shape[split.dim]
                sizes = qkv_rows if split.qkv else (total // 2, total // 2)
                if sum(sizes) != total:
                    raise RuntimeError(f"tpu-sync: {name} {tuple(tensor.shape)} does not split into {sizes}")
                prefix, start = name[: -len(split.fused)], 0
                for piece, size in zip(split.pieces, sizes, strict=True):
                    chunk[prefix + piece] = tensor.narrow(split.dim, start, size)  # a view
                    start += size
        return chunk

    return local_layout


def refuse_split_tensors(weights: Iterator[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    """Undoes ``make_tpu_sync_local_layout`` for vLLM's checkpoint loaders: yields HF's fused tensors.

    Used when the rollout falls back to vLLM's loaders (``plan_direct_install`` returned no plan), which expect HF's
    ``in_proj_qkv`` / ``conv1d`` / stacked ``mlp.experts.gate_up_proj``. A fused tensor is yielded once all of its
    pieces arrived (sorted names keep a layer's pieces together); other tensors pass through.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        for split in SPLIT_TENSORS:
            piece = next((p for p in split.pieces if name.endswith(p)), None)
            if piece is None:
                continue
            fused_name = name[: -len(piece)] + split.fused
            parts = pending.setdefault(fused_name, {})
            parts[piece] = tensor
            if len(parts) == len(split.pieces):
                del pending[fused_name]
                yield fused_name, torch.cat([parts[p] for p in split.pieces], dim=split.dim)
            break
        else:
            yield name, tensor
    if pending:
        raise RuntimeError(f"tpu-sync: split tensors with missing pieces: {sorted(pending)}")


def plan_direct_install(
    vllm_model: Any, global_shapes: dict[str, list[int]], hf_text_config: Any, tp_size: int, worker_rank: int
) -> tuple[Optional[DirectInstallPlan], str]:
    """``(plan, "")`` if this module can install into ``vllm_model`` directly, else ``(None, reason)``.

    Checks, on the live vLLM model rather than by name: a Qwen3.5 text config; the trainer's split tensor layout
    with the config's sizes; GatedDeltaNet heads and expert intermediate size that split evenly over the TP ranks;
    one fused, non-interleaved, TP-sharded ``in_proj_qkvz`` / ``in_proj_ba`` per received GatedDeltaNet layer; a
    known unquantized MoE method without expert parallelism; and that this worker's rank is its TP rank (tpu-sync
    registers it at that position of the ``[1, tp]`` mesh).
    """
    model_type = getattr(hf_text_config, "model_type", None)
    if model_type not in QWEN3_5_MODEL_TYPES:
        return None, f"model_type {model_type!r} is not a Qwen3.5 text model"
    fused_names = [n for n in global_shapes if any(n.endswith(split.fused) for split in SPLIT_TENSORS)]
    if fused_names:
        return None, f"the trainer sent fused tensors (no tpu-sync split layout), e.g. {fused_names[0]}"
    try:
        dims = Qwen3_5Dims.from_config(hf_text_config)
    except (AttributeError, TypeError, ValueError) as e:
        return None, f"incomplete Qwen3.5 config: {e}"
    if dims.num_k_heads % tp_size or dims.num_v_heads % tp_size:
        return None, f"GatedDeltaNet heads ({dims.num_k_heads} k, {dims.num_v_heads} v) do not split over TP={tp_size}"
    if dims.moe_intermediate_size and dims.moe_intermediate_size % tp_size:
        return None, f"moe_intermediate_size {dims.moe_intermediate_size} does not split over TP={tp_size}"

    expected_rows = {
        ".in_proj_q.weight": dims.key_dim,
        ".in_proj_k.weight": dims.key_dim,
        ".in_proj_v.weight": dims.value_dim,
        ".in_proj_z.weight": dims.value_dim,
    }
    for name, shape in global_shapes.items():
        for suffix, rows in expected_rows.items():
            if name.endswith(".linear_attn" + suffix) and shape[0] != rows:
                return None, f"{name} has {shape[0]} rows, the config gives {rows}"

    # Every layer must be complete: a missing piece would otherwise fail mid-install, after earlier layers of the
    # live vLLM model were already overwritten.
    gdn_layers = {n.split(".linear_attn.", 1)[0] for n in global_shapes if ".linear_attn." in n}
    if not gdn_layers:
        return None, "the trainer sent no GatedDeltaNet (linear_attn) tensors"
    expert_blocks = {n.split(".mlp.experts.", 1)[0] for n in global_shapes if ".mlp.experts." in n}
    for layers, suffixes in ((gdn_layers, GDN_LAYER_SUFFIXES), (expert_blocks, EXPERT_BLOCK_SUFFIXES)):
        for layer in sorted(layers):
            missing = [s for s in suffixes if layer + s not in global_shapes]
            if missing:
                return None, f"{layer} lacks {missing}"

    modules = list(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else []
    gdn_modules = [m for _, m in modules if hasattr(m, "in_proj_qkvz") and hasattr(m, "in_proj_ba")]
    if len(gdn_modules) != len(gdn_layers):
        return None, f"vLLM has {len(gdn_modules)} fused in_proj_qkvz layers, the trainer sent {len(gdn_layers)}"
    tp_ranks = set()
    for m in gdn_modules:
        if getattr(m, "gqa_interleaved_layout", False):
            return None, "vLLM's GatedDeltaNet uses the interleaved (Qwen3-Next) in_proj_qkvz layout"
        if getattr(m, "disable_tp_for_ba_proj", False):
            return None, "vLLM replicates in_proj_ba (disable_tp_for_ba_proj)"
        if getattr(m, "tp_size", tp_size) != tp_size:
            return None, f"vLLM's GatedDeltaNet tp_size {m.tp_size} != {tp_size}"
        tp_ranks.add(int(getattr(m, "tp_rank", worker_rank)))
    if len(tp_ranks) > 1:
        return None, f"vLLM's GatedDeltaNet layers disagree on the TP rank: {sorted(tp_ranks)}"
    tp_rank = tp_ranks.pop()
    if tp_rank != worker_rank:
        return None, f"worker rank {worker_rank} is not its TP rank {tp_rank}"

    moe_layout = None
    if expert_blocks:
        layouts = set()
        for _, m in modules:
            if not (hasattr(m, "w13_weight") and hasattr(m, "quant_method")):
                continue
            if getattr(m, "use_ep", False):
                return None, "vLLM uses expert parallelism"
            method = type(m.quant_method).__name__
            if method not in _MOE_LAYOUTS:
                return None, f"unsupported vLLM MoE method {method} (quantized?)"
            layouts.add(_MOE_LAYOUTS[method])
        if len(layouts) != 1:
            return None, f"vLLM MoE layouts {sorted(layouts) or 'not found'}"
        moe_layout = layouts.pop()
    return DirectInstallPlan(dims, tp_rank, tp_size, moe_layout), ""


def sampler_sharding_rule(name: str, global_shape: list[int], tp_size: int) -> Optional[tuple[list[str], list[int]]]:
    """``(sharding_spec, local_shape)`` of a Qwen3.5-specific tensor, or ``None`` to use the dense path's rules."""
    if any(name.endswith(s) for s in REPLICATED_SUFFIXES):
        return [""] * len(global_shape), list(global_shape)
    tp_dim = None
    if any(name.endswith(s) for s in ROW_PARALLEL_SUFFIXES):
        tp_dim = 1
    for suffix, dim in SPLIT_TP_DIMS.items():
        if name.endswith(suffix):
            tp_dim = dim
    if tp_dim is None:
        return None
    if global_shape[tp_dim] % tp_size:
        raise RuntimeError(f"{_ERROR_PREFIX}: {name} {global_shape} dim {tp_dim} does not split over {tp_size}")
    spec = [""] * len(global_shape)
    local = list(global_shape)
    spec[tp_dim] = "tp"
    local[tp_dim] = global_shape[tp_dim] // tp_size
    return spec, local


def install(
    staging: dict[str, torch.Tensor],
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    copy_into: Callable[[str, torch.Tensor], None],
    plan: DirectInstallPlan,
) -> set[str]:
    """Writes the Qwen3.5-specific tensors of ``staging`` into vLLM's parameters; returns the staging names used.

    ``copy_into(vllm_key, tensor)`` copies a tensor in vLLM's checkpoint ([n_out, n_in]) layout, transposing it
    for the weights that vllm-torchtpu keeps as [n_in, n_out]. Updates ``plan.padding_checked`` and
    ``plan.expert_keys`` (see ``DirectInstallPlan``).
    """
    consumed: set[str] = set()
    dims, tp_rank, tp_size = plan.dims, plan.tp_rank, plan.tp_size

    def write(name: str, src: torch.Tensor) -> None:
        key = find_model_key(name)
        if key is None:
            raise RuntimeError(f"{_ERROR_PREFIX}: vLLM has no parameter {name}")
        if _local(model_sd[key]).numel() != src.numel():  # copy_ would broadcast a too-small source silently
            raise RuntimeError(
                f"{_ERROR_PREFIX}: {name} has {src.numel()} elements, vLLM's {key} {tuple(_local(model_sd[key]).shape)}"
            )
        copy_into(key, src)

    def rank_rows(name: str, total: int) -> torch.Tensor:
        """Rank ``tp_rank``'s rows of a tensor received whole."""
        full = staging[name]
        if full.shape[0] != total:
            raise RuntimeError(f"{_ERROR_PREFIX}: {name} has {full.shape[0]} rows, expected {total}")
        per_rank = total // tp_size
        return full[tp_rank * per_rank : (tp_rank + 1) * per_rank]

    for name in list(staging):
        m = _GDN_RE.match(name)
        if m is None:
            continue
        prefix = m.group(1)
        qkvz = [f"{prefix}.in_proj_{p}.weight" for p in "qkvz"]
        conv = [f"{prefix}.conv1d_{p}.weight" for p in "qkv"]
        rows = (dims.key_dim, dims.key_dim, dims.value_dim, dims.value_dim)
        for piece, total in zip(qkvz + conv, rows + rows[:3], strict=True):
            if staging[piece].shape[0] * tp_size != total:
                raise RuntimeError(f"{_ERROR_PREFIX}: {piece} has {staging[piece].shape[0]} rows on this rank")
        write(f"{prefix}.in_proj_qkvz.weight", torch.cat([staging[p] for p in qkvz], dim=0))
        write(f"{prefix}.conv1d.weight", torch.cat([staging[p] for p in conv], dim=0))
        for vec in ("A_log", "dt_bias"):
            write(f"{prefix}.{vec}", rank_rows(f"{prefix}.{vec}", dims.num_v_heads))
        consumed.update(qkvz + conv + [f"{prefix}.A_log", f"{prefix}.dt_bias"])

    for name in list(staging):
        m = _EXPERTS_RE.match(name)
        if m is None:
            continue
        if plan.moe_layout is None:
            raise RuntimeError(f"{_ERROR_PREFIX}: received experts but planned no MoE layout")
        prefix = m.group(1)
        gate, up, down = staging[name], staging[f"{prefix}.up_proj"], staging[f"{prefix}.down_proj"]
        w13, w2 = (
            model_sd[_expert_param(model_sd, find_model_key, prefix, leaf, plan.expert_keys)]
            for leaf in ("w13_weight", "w2_weight")
        )
        _write_w13(w13, gate, up, plan.moe_layout, check_padding=not plan.padding_checked)
        _write_w2(w2, down, plan.moe_layout, check_padding=not plan.padding_checked)
        consumed.update({name, f"{prefix}.up_proj", f"{prefix}.down_proj"})
    plan.padding_checked = True
    return consumed


def _expert_param(
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    prefix: str,
    leaf: str,
    cache: dict[tuple[str, str], str],
) -> str:
    """vLLM's key of expert weight ``leaf`` (``w13_weight`` / ``w2_weight``) of the MoE block ``prefix``.

    vLLM's MoE layer keeps the expert weights in a submodule of ``mlp.experts`` (e.g. its ``MoERunner``'s
    ``routed_experts``), so the key is ``<prefix>.<submodule...>.<leaf>``; the submodule path depends on the vLLM
    version and platform, so it is looked up (once; ``cache`` keeps the result).
    """
    cached = cache.get((prefix, leaf))
    if cached is not None and cached in model_sd:
        return cached
    key = find_model_key(f"{prefix}.{leaf}")
    if key is not None:
        cache[(prefix, leaf)] = key
        return key
    # ``model.layers.3.mlp.experts`` -> any ``...(^|.)layers.3.mlp.experts.<submodules>.<leaf>`` key of vLLM.
    stem = re.escape(prefix.split("model.", 1)[-1])
    pattern = re.compile(rf"(?:^|\.){stem}\.(?:.+\.)?{re.escape(leaf)}$")
    matches = [k for k in model_sd if pattern.search(k)]
    if len(matches) != 1:
        raise RuntimeError(f"{_ERROR_PREFIX}: expected one vLLM parameter {prefix}.*.{leaf}, found {matches}")
    cache[(prefix, leaf)] = matches[0]
    return matches[0]


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if hasattr(param, "to_local") else param


def _aligned(inter: int) -> int:
    return (inter + _MOE_PAD_MULTIPLE - 1) // _MOE_PAD_MULTIPLE * _MOE_PAD_MULTIPLE


def _check_zero(region: torch.Tensor, what: str) -> None:
    if region.numel() and bool(region.ne(0).any()):
        raise RuntimeError(f"{_ERROR_PREFIX}: vllm-torchtpu's {what} padding is not zero")


def _write_w13(param: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, layout: str, check_padding: bool) -> None:
    """Writes rank-local ``gate`` / ``up`` [E, I_r, H] into vLLM's ``w13_weight``.

    ``MOE_LAYOUT_TPU``: vllm-torchtpu's unquantized MoE keeps it as [E, H, 2A] (transposed, each half padded with
    zeros to ``A``, the next multiple of 128). ``MOE_LAYOUT_VLLM``: vLLM's own [E, 2 I_r, H].
    """
    w13 = _local(param)
    num_experts, inter, hidden = gate.shape
    if layout == MOE_LAYOUT_VLLM:
        expected: tuple[int, ...] = (num_experts, 2 * inter, hidden)
    elif layout == MOE_LAYOUT_TPU:
        expected = (num_experts, hidden, 2 * _aligned(inter))
    else:
        raise RuntimeError(f"{_ERROR_PREFIX}: unknown MoE layout {layout}")
    if tuple(w13.shape) != expected:
        raise RuntimeError(f"{_ERROR_PREFIX}: w13_weight {tuple(w13.shape)}, expected {expected} ({layout})")
    if layout == MOE_LAYOUT_VLLM:
        w13.copy_(torch.cat([gate, up], dim=1))
        return
    aligned = _aligned(inter)
    if check_padding:
        _check_zero(w13[..., inter:aligned], "w13 gate")
        _check_zero(w13[..., aligned + inter :], "w13 up")
    w13[..., :inter].copy_(gate.transpose(1, 2))
    w13[..., aligned : aligned + inter].copy_(up.transpose(1, 2))


def _write_w2(param: torch.Tensor, down: torch.Tensor, layout: str, check_padding: bool) -> None:
    """Writes rank-local ``down`` [E, H, I_r] into ``w2_weight``: [E, A, H] (``MOE_LAYOUT_TPU``) or [E, H, I_r]."""
    w2 = _local(param)
    num_experts, hidden, inter = down.shape
    if layout == MOE_LAYOUT_VLLM:
        expected: tuple[int, ...] = (num_experts, hidden, inter)
    elif layout == MOE_LAYOUT_TPU:
        expected = (num_experts, _aligned(inter), hidden)
    else:
        raise RuntimeError(f"{_ERROR_PREFIX}: unknown MoE layout {layout}")
    if tuple(w2.shape) != expected:
        raise RuntimeError(f"{_ERROR_PREFIX}: w2_weight {tuple(w2.shape)}, expected {expected} ({layout})")
    if layout == MOE_LAYOUT_VLLM:
        w2.copy_(down)
        return
    if check_padding:
        _check_zero(w2[:, inter:, :], "w2")
    w2[:, :inter, :].copy_(down.transpose(1, 2))
