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
vllm-torchtpu's MoE methods (``moe_weights.MOE_LAYOUTS``: unquantized, or FP8 experts requantized on the rollout from
the received bf16 slices). Under vLLM expert parallelism over the TP group each rank receives whole experts (dim 0)
instead. ``plan_direct_install`` checks all of this on the live vLLM model and falls back to vLLM's loaders for
anything else (other models, unknown MoE methods or placements, a replicated ``in_proj_ba``, ...).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, NamedTuple, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.direct_install import (
    DirectInstallLayout,
    non_moe_quantized_reason,
)
from verl_hardware_plugin.accelerators.tpu.rollout.moe_weights import (
    EP_EXPERT_BLOCK_SUFFIXES,
    EP_EXPERT_TENSORS,
    EP_FP8_SCALE_SUFFIXES,
    FP8_MOE_DTYPE,
    MOE_LAYOUT_TPU_EP,
    MOE_LAYOUT_TPU_EP_QUANT,
    MOE_QUANTIZED_LAYOUTS,
    ExpertWireQuant,
    MoeRequant,
    ParityCheckFn,
    ep_quant_loader_error,
    ep_zero_copy_targets,
    format_ep_expert_weight,
    plan_ep_moe_layers,
    plan_moe_layers,
    write_ep_scales,
    write_experts,
)

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


SPLIT_TP_DIMS.update({sfx: 0 for sfx in EP_EXPERT_BLOCK_SUFFIXES + EP_FP8_SCALE_SUFFIXES})
# vLLM fuses these into one parameter: (fused suffix, source suffixes in concat order).
FUSED_TARGETS = (
    (
        ".mlp.shared_expert.gate_up_proj.weight",
        (".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight"),
    ),
    (".linear_attn.in_proj_ba.weight", (".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight")),
)

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
    num_experts: Optional[int] = None  # None for the dense models (or inferred from wire shapes)

    @classmethod
    def from_config(cls, hf_text_config: Any) -> Qwen3_5Dims:
        num_k_heads = int(hf_text_config.linear_num_key_heads)
        num_v_heads = int(hf_text_config.linear_num_value_heads)
        moe_inter = getattr(hf_text_config, "moe_intermediate_size", None)
        num_exp = getattr(hf_text_config, "num_experts", None)
        return cls(
            key_dim=num_k_heads * int(hf_text_config.linear_key_head_dim),
            value_dim=num_v_heads * int(hf_text_config.linear_value_head_dim),
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            moe_intermediate_size=int(moe_inter) if moe_inter else None,
            num_experts=int(num_exp) if num_exp else None,
        )


@dataclass
class DirectInstallPlan:
    """What ``install`` needs, checked once against the live vLLM model by ``plan_direct_install``."""

    dims: Qwen3_5Dims
    tp_rank: int
    tp_size: int
    # One of ``moe_weights.MOE_LAYOUTS``' values (e.g. MOE_LAYOUT_TPU, MOE_LAYOUT_TPU_FP8) or MOE_LAYOUT_TPU_EP; None
    # without experts.
    moe_layout: Optional[str]
    # True when vLLM runs expert parallelism over the whole TP group (whole experts per rank, dim-0 sharded); False
    # when vLLM TP-slices every expert's intermediate dimension.
    expert_parallel: bool = False
    # Updated by ``install``: whether vllm-torchtpu's zero MoE padding was verified (first sync only), and vLLM's
    # resolved expert parameter keys (``(prefix, leaf) -> key``), which do not change between syncs.
    padding_checked: bool = False
    expert_keys: dict[tuple[str, str], str] = field(default_factory=dict)
    # The rollout's quantized expert format for ``moe_weights.MOE_QUANTIZED_LAYOUTS`` (resolved once by the plan).
    moe_requant: Optional[MoeRequant] = None


def make_tpu_sync_local_layout(
    hf_text_config: Any,
    expert_parallel: bool = False,
    fp8_moe: bool | ExpertWireQuant = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Trainer side: the ``TrainerWeights.local_transform`` that exports the ``SPLIT_TENSORS`` as their pieces.

    With ``expert_parallel`` (``engine_kwargs.tpu.rollout_expert_parallel``), the expert stacks are sent in vLLM's
    expert-parallel layout instead (``EP_EXPERT_TENSORS``): a transposed copy of each, still sharded on the expert
    dimension, so the rollout can bind its expert weights directly with no HBM staging copy.

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
    if fp8_moe and not expert_parallel:
        raise ValueError("tpu-sync: rollout_fp8_moe needs rollout_expert_parallel")
    wire_quant: Optional[ExpertWireQuant] = (
        fp8_moe if isinstance(fp8_moe, ExpertWireQuant) else (ExpertWireQuant(FP8_MOE_DTYPE, None) if fp8_moe else None)
    )
    dims = Qwen3_5Dims.from_config(hf_text_config)
    qkv_rows = (dims.key_dim, dims.key_dim, dims.value_dim)

    def local_layout(chunk: dict[str, Any]) -> dict[str, Any]:
        if expert_parallel:
            for ep in EP_EXPERT_TENSORS:
                for name in [n for n in chunk if n.endswith(ep.hf)]:
                    prefix = name[: -len(ep.hf)]
                    hf_tensor = chunk.pop(name)
                    ep_tensor, scale = format_ep_expert_weight(hf_tensor, ep.kind, wire_quant)
                    del hf_tensor
                    ep_tensor._tpu_sync_owned = True
                    if scale is not None:
                        ep_tensor._tpu_sync_keep_dtype = True
                        scale._tpu_sync_owned = True
                        scale._tpu_sync_keep_dtype = True
                        chunk[prefix + ep.scale] = scale
                    chunk[prefix + ep.ep] = ep_tensor
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


def refuse_split_tensors(
    weights: Iterator[tuple[str, torch.Tensor]], *, rewrite_ep: bool = True
) -> Iterator[tuple[str, torch.Tensor]]:
    """Undoes ``make_tpu_sync_local_layout`` for vLLM's checkpoint loaders: yields HF's fused tensors.

    Used when the rollout falls back to vLLM's loaders (``plan_direct_install`` returned no plan), which expect HF's
    ``in_proj_qkv`` / ``conv1d`` / stacked ``mlp.experts.gate_up_proj`` / ``down_proj`` (also undoes the
    expert-parallel layout when ``rewrite_ep`` is True). A fused tensor is yielded once all of its
    pieces arrived (sorted names keep a layer's pieces together); other tensors pass through.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        if name.endswith(EP_FP8_SCALE_SUFFIXES) or (
            name.endswith(EP_EXPERT_BLOCK_SUFFIXES)
            and tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise RuntimeError(
                f"tpu-sync: cannot install trainer-quantized expert tensor {name} ({tensor.dtype}) via vLLM's "
                "checkpoint loaders; inspect the rejected direct-install plan log above"
            )
        if rewrite_ep:
            ep = next((t for t in EP_EXPERT_TENSORS if name.endswith(t.ep)), None)
            if ep is not None:
                yield name[: -len(ep.ep)] + ep.hf, tensor.transpose(-2, -1)
                continue
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
    vllm_model: Any,
    global_shapes: dict[str, list[int]],
    hf_text_config: Any,
    tp_size: int,
    worker_rank: int,
    allow_quantized_moe: bool = True,
    global_dtypes: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[DirectInstallPlan], str]:
    """``(plan, "")`` if this module can install into ``vllm_model`` directly, else ``(None, reason)``.

    Checks, on the live vLLM model rather than by name: a Qwen3.5 text config; the trainer's split tensor layout
    with the config's sizes; GatedDeltaNet heads and expert intermediate size that split evenly over the TP ranks;
    one fused, non-interleaved, TP-sharded ``in_proj_qkvz`` / ``in_proj_ba`` per received GatedDeltaNet layer; MoE
    layers of one known method and parallel mode (``moe_weights.plan_moe_layers``: TP-sliced experts or expert
    parallelism over the whole TP group; for a quantized method, also its requantizer and every live weight and scale;
    ``allow_quantized_moe=False`` refuses quantized experts); and that this worker's rank is its TP rank (tpu-sync
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
    wire_ep = any(n.endswith(EP_EXPERT_BLOCK_SUFFIXES) for n in global_shapes)
    wire_quant = any(n.endswith(EP_FP8_SCALE_SUFFIXES) for n in global_shapes)
    if wire_quant and not wire_ep:
        return None, "the trainer sent quantized expert scales without the expert-parallel layout"

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
    expert_suffixes = EP_EXPERT_BLOCK_SUFFIXES if wire_ep else EXPERT_BLOCK_SUFFIXES
    required_expert_suffixes = expert_suffixes + (EP_FP8_SCALE_SUFFIXES if wire_quant else ())
    for layers, suffixes in ((gdn_layers, GDN_LAYER_SUFFIXES), (expert_blocks, required_expert_suffixes)):
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

    moe_modules = [(n, m) for n, m in modules if hasattr(m, "w13_weight") and hasattr(m, "quant_method")]
    quant_why = non_moe_quantized_reason(vllm_model, moe_modules if expert_blocks else ())
    if quant_why:
        return None, quant_why

    moe_layout = None
    expert_parallel = False
    moe_requant = None
    if expert_blocks:
        non_3d = sorted(
            name for name, shape in global_shapes.items() if name.endswith(expert_suffixes) and len(shape) != 3
        )
        if non_3d:
            return None, f"expected 3-D expert tensors, got {non_3d[0]} {list(global_shapes[non_3d[0]])}"
        expert_counts = {int(shape[0]) for name, shape in global_shapes.items() if name.endswith(expert_suffixes)}
        if len(expert_counts) != 1:
            return None, f"inconsistent expert counts in wire shapes: {sorted(expert_counts)}"
        wire_experts = expert_counts.pop()
        if dims.num_experts is not None and dims.num_experts != wire_experts:
            return None, f"the trainer sent {wire_experts} experts, the config gives {dims.num_experts}"
        num_experts = dims.num_experts or wire_experts
        if not dims.moe_intermediate_size:
            return None, "incomplete Qwen3.5 MoE config: missing moe_intermediate_size"
        if not moe_modules:
            return None, "vLLM MoE layouts not found"
        if len(moe_modules) != len(expert_blocks):
            return (
                None,
                f"vLLM has {len(moe_modules)} fused-MoE layers, the trainer sent {len(expert_blocks)} MoE blocks",
            )

        if wire_ep:
            ep_plan, why = plan_ep_moe_layers(
                moe_modules,
                global_shapes,
                expert_blocks,
                num_experts,
                dims.moe_intermediate_size,
                tp_size,
                tp_rank,
                allow_quantized=allow_quantized_moe,
                global_dtypes=global_dtypes,
            )
            if ep_plan is None:
                return None, why
            moe_layout, expert_parallel, moe_requant = ep_plan
        else:
            cfg_hidden = getattr(hf_text_config, "hidden_size", None)
            hidden = (
                int(cfg_hidden)
                if cfg_hidden
                else next(int(s[-1]) for n, s in sorted(global_shapes.items()) if n.endswith(".mlp.experts.gate_proj"))
            )
            moe_plan, why = plan_moe_layers(
                moe_modules,
                num_experts,
                dims.moe_intermediate_size,
                hidden,
                tp_size,
                tp_rank,
                allow_quantized=allow_quantized_moe,
                use_ep_at_tp1=True,
            )
            if moe_plan is None:
                return None, why
            expected_expert_shapes = {
                ".mlp.experts.gate_proj": [num_experts, dims.moe_intermediate_size, hidden],
                ".mlp.experts.up_proj": [num_experts, dims.moe_intermediate_size, hidden],
                ".mlp.experts.down_proj": [num_experts, hidden, dims.moe_intermediate_size],
            }
            for layer in sorted(expert_blocks):
                for suffix, exp_shape in expected_expert_shapes.items():
                    t_name = layer + suffix
                    if list(global_shapes[t_name]) != exp_shape:
                        return None, (
                            f"{t_name} is {list(global_shapes[t_name])}, the config gives {exp_shape} "
                            f"(moe_intermediate_size={dims.moe_intermediate_size})"
                        )
            moe_layout, expert_parallel, moe_requant = moe_plan
    elif dims.moe_intermediate_size and dims.moe_intermediate_size % tp_size:
        return None, f"moe_intermediate_size {dims.moe_intermediate_size} does not split over TP={tp_size}"
    return (
        DirectInstallPlan(dims, tp_rank, tp_size, moe_layout, expert_parallel=expert_parallel, moe_requant=moe_requant),
        "",
    )


def zero_copy_targets(
    plan: DirectInstallPlan,
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    global_shapes: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    """Rollout side: ``{tpu-sync tensor: vLLM weight}`` for the tensors received straight into vLLM's weights.

    In the zero-copy expert-parallel layout (``EP_EXPERT_TENSORS``, ``rollout_expert_parallel=True``), rank r's slice
    of each expert tensor is its vLLM ``w13_weight`` / ``w2_weight``, so those weights are bound to tpu-sync as the
    receive buffers: no HBM staging copy of the experts, and nothing to install after the transfer.
    """
    return ep_zero_copy_targets(
        plan.moe_layout, plan.tp_size, plan.expert_keys, model_sd, find_model_key, global_shapes
    )


def sampler_sharding_rule(
    name: str, global_shape: list[int], tp_size: int, plan: Optional[DirectInstallPlan] = None
) -> Optional[tuple[list[str], list[int]]]:
    """``(sharding_spec, local_shape)`` of a Qwen3.5-specific tensor, or ``None`` to use the dense path's rules."""
    if any(name.endswith(s) for s in REPLICATED_SUFFIXES):
        return [""] * len(global_shape), list(global_shape)
    tp_dim = None
    if any(name.endswith(s) for s in ROW_PARALLEL_SUFFIXES):
        tp_dim = 1
    for suffix, dim in SPLIT_TP_DIMS.items():
        if name.endswith(suffix):
            tp_dim = 0 if (plan is not None and plan.expert_parallel and suffix in EXPERT_BLOCK_SUFFIXES) else dim
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
    check: Optional[ParityCheckFn] = None,
) -> set[str]:
    """Writes the Qwen3.5-specific tensors of ``staging`` into vLLM's parameters; returns the staging names used.

    ``copy_into(vllm_key, tensor)`` copies a tensor in vLLM's checkpoint ([n_out, n_in]) layout, transposing it
    for the weights that vllm-torchtpu keeps as [n_in, n_out]. ``check``, if given, is called on every expert
    weight region before it is overwritten (the exact parity check; ``copy_into`` runs it for the rest; quantized
    experts compare the requantized weights and scales, see ``moe_weights.write_requantized``). Updates
    ``plan.padding_checked`` and ``plan.expert_keys`` (see ``DirectInstallPlan``).
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

    if plan.moe_layout in (MOE_LAYOUT_TPU_EP, MOE_LAYOUT_TPU_EP_QUANT):
        # Received straight into vLLM's expert weights (zero_copy_targets); quantized runs also install the FP32 scales.
        consumed.update(n for n in staging if n.endswith(EP_EXPERT_BLOCK_SUFFIXES))
        if plan.moe_layout == MOE_LAYOUT_TPU_EP_QUANT:
            consumed.update(write_ep_scales(staging, model_sd, find_model_key, plan.expert_keys, check=check))
    for name in list(staging):
        m = _EXPERTS_RE.match(name)
        if m is None:
            continue
        if plan.moe_layout is None:
            raise RuntimeError(f"{_ERROR_PREFIX}: received experts but planned no MoE layout")
        prefix = m.group(1)
        pieces = (name, f"{prefix}.up_proj", f"{prefix}.down_proj")
        missing = [p for p in pieces if p not in staging]
        if missing:
            raise RuntimeError(f"{_ERROR_PREFIX}: {prefix} lacks {missing}")
        gate, up, down = (staging[p] for p in pieces)
        if gate.ndim != 3:
            raise RuntimeError(f"{_ERROR_PREFIX}: {name} is {tuple(gate.shape)}, expected a 3-D expert stack")
        local_experts = (
            (dims.num_experts // tp_size if plan.expert_parallel else dims.num_experts)
            if dims.num_experts is not None
            else gate.shape[0]
        )
        local_inter = (
            (dims.moe_intermediate_size if plan.expert_parallel else dims.moe_intermediate_size // tp_size)
            if dims.moe_intermediate_size is not None
            else gate.shape[1]
        )
        hidden = gate.shape[2]
        expected_shapes = (
            (local_experts, local_inter, hidden),
            (local_experts, local_inter, hidden),
            (local_experts, hidden, local_inter),
        )
        for piece, t, exp_shape in zip(pieces, (gate, up, down), expected_shapes, strict=True):
            if tuple(t.shape) != exp_shape:
                raise RuntimeError(f"{_ERROR_PREFIX}: {piece} is {tuple(t.shape)} on this rank, expected {exp_shape}")
        write_experts(
            model_sd,
            find_model_key,
            prefix,
            gate,
            up,
            down,
            layout=plan.moe_layout,
            expert_keys=plan.expert_keys,
            check_padding=not plan.padding_checked,
            requant=plan.moe_requant,
            check=check,
        )
        consumed.update(pieces)
    # Requantized experts are written whole (fresh zero padding), so no padding is checked for them.
    plan.padding_checked = plan.moe_layout not in (MOE_QUANTIZED_LAYOUTS | {MOE_LAYOUT_TPU_EP_QUANT})
    return consumed


def _local(param: torch.Tensor) -> torch.Tensor:
    return param.to_local() if hasattr(param, "to_local") else param


def _loader_weights(weights: Iterator[tuple[str, torch.Tensor]], hf_text_config: Any) -> Iterator:
    rewrite_ep = hf_text_config is None or getattr(hf_text_config, "model_type", None) in QWEN3_5_MODEL_TYPES
    return refuse_split_tensors(weights, rewrite_ep=rewrite_ep)


def _loader_error(
    global_shapes: dict[str, list[int]],
    hf_text_config: Any = None,
    *,
    global_dtypes: Optional[Mapping[str, str]] = None,
) -> str:
    del hf_text_config
    return ep_quant_loader_error(global_shapes, global_dtypes)


def _sharding_rule(name: str, global_shape: list[int], tp_size: int, plan: DirectInstallPlan) -> Optional[Any]:
    return sampler_sharding_rule(name, global_shape, tp_size, plan)


LAYOUT = DirectInstallLayout(
    name="qwen3_5",
    plan=plan_direct_install,
    sharding_rule=_sharding_rule,
    install=install,
    fused_targets=FUSED_TARGETS,
    zero_copy_targets=zero_copy_targets,
    loader_weights=_loader_weights,
    loader_error=_loader_error,
)
