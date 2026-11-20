# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM fused-MoE expert weights: the layouts of ``w13_weight`` / ``w2_weight``, how to plan them and writers for them.

Shared by the direct ``tpu-sync`` installs that write experts straight into vLLM (``qwen3_5_layout``,
``qwen3_moe_layout``). Nothing here is model-specific: callers pass each rank's expert tensors in HF's per-expert
orientation, ``gate`` / ``up`` [E, I, H] and ``down`` [E, H, I], plan the live vLLM MoE layers with
``plan_moe_layers`` and write every MoE block with ``write_experts``.

Quantized experts (``MOE_QUANTIZED_LAYOUTS``): the trainer sends bf16, and each rollout rank requantizes its own
slice with the rollout's own quantizer (``MoeRequant``) into the live quantized weights and scales. vllm-torchtpu
quantizes the TP-local shard on load too, so the result is what its ``process_weights_after_loading`` produces.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.device_api import tpu_synchronize
from verl_hardware_plugin.accelerators.tpu.rollout.direct_install import (
    UNQUANTIZED_PARAM_DTYPES,
    non_moe_quantized_reason,
)

# vLLM MoE methods (by class name) and their expert weight layout:
# * ``MOE_LAYOUT_TPU``: vllm-torchtpu's unquantized method, w13 [E, H, 2A] / w2 [E, A, H] (transposed; each w13 half
#   and w2's rows padded with zeros to ``A``, the next multiple of 128);
# * ``MOE_LAYOUT_VLLM``: vLLM's own unquantized method, [E, 2I, H] / [E, H, I];
# * ``MOE_LAYOUT_TPU_FP8``: vllm-torchtpu's FP8 method (``QUANTIZE_ON_LOAD_PREFIXES=experts``, or an FP8 checkpoint
#   whose non-expert layers stay unquantized): ``MOE_LAYOUT_TPU``'s layout in ``MOE_REQUANTIZE_WEIGHT_DTYPE`` (FP8
#   e4m3 by default) plus ``w*_weight_scale_inv`` scales, per output channel or per ``MOE_REQUANTIZE_BLOCK_SIZE`` block
#   of the contracting dim (``requant_shapes``). Written by requantizing the received bf16 (``write_requantized``).
MOE_LAYOUT_TPU = "tpu_transposed_padded"
MOE_LAYOUT_VLLM = "vllm"
MOE_LAYOUT_TPU_EP = "tpu_expert_parallel"  # vllm-torchtpu zero-copy EP layout (EP_EXPERT_TENSORS)
MOE_LAYOUT_TPU_EP_QUANT = "tpu_expert_parallel_quantized"  # trainer-quantized zero-copy EP + wire scales
MOE_LAYOUT_TPU_EP_FP8 = MOE_LAYOUT_TPU_EP_QUANT
MOE_LAYOUT_TPU_FP8 = "tpu_transposed_padded_fp8"
FP8_MOE_DTYPE = torch.float8_e4m3fn
MOE_LAYOUTS = {
    "VllmUnquantizedFusedMoEMethod": MOE_LAYOUT_TPU,
    "UnquantizedFusedMoEMethod": MOE_LAYOUT_VLLM,
    "VllmFp8MoEMethodTPU": MOE_LAYOUT_TPU_FP8,
}
_MOE_PAD_MULTIPLE = 128  # vllm-torchtpu's VllmUnquantizedFusedMoEMethod.process_weights_after_loading
# ``ExactParityCheck.check(name, target, received)``: called on a target region before it is overwritten.
ParityCheckFn = Callable[[str, torch.Tensor, torch.Tensor], None]

_ERROR_PREFIX = "tpu-sync direct install"
_WEIGHT_LEAVES = ("w13_weight", "w2_weight")


def moe_attr(module: Any, name: str, default: Any) -> Any:
    """``module.<name>``, else ``module.moe_config.<name>`` (vLLM keeps the EP geometry on both, version-dependent)."""
    value = getattr(module, name, None)
    if value is None:
        value = getattr(getattr(module, "moe_config", None), name, None)
    return default if value is None else value


def _module_activation(module: Any) -> str:
    """The activation string vllm-torchtpu quantized (and runs) ``module``'s experts with."""
    activation = getattr(getattr(module, "quant_method", None), "_tpu_activation_str", None)
    if activation is not None:
        return str(activation)
    activation = moe_attr(module, "activation", "silu")
    activation = getattr(activation, "value", activation)  # vLLM's MoEActivation enum
    return str(activation).lower()


def moe_parallel_mode(
    module: Any, num_experts: int, moe_intermediate_size: int, tp_size: int, tp_rank: int
) -> tuple[Optional[bool], str]:
    """``(expert_parallel, "")`` for a vLLM MoE layer that can be written directly on TP rank ``tp_rank``, else
    ``(None, reason)``.

    ``expert_parallel`` is True if the layer holds whole experts ``[r * E / tp, (r + 1) * E / tp)`` (EP over the whole
    TP group, linear placement), False if it holds slice ``r`` of every expert's intermediate dimension (no EP).
    """
    global_experts = int(moe_attr(module, "global_num_experts", 0) or 0)
    local_experts = int(moe_attr(module, "local_num_experts", 0) or 0)
    if global_experts and global_experts != num_experts:
        return None, f"{global_experts} experts, the config gives {num_experts}"
    activation = _module_activation(module)
    if (
        "swigluoai" in activation.lower()
        or getattr(module, "w13_bias", None) is not None
        or getattr(module, "w2_bias", None) is not None
    ):
        return None, f"activation {activation} / expert bias not supported"
    if tp_size == 1:
        return True, ""
    if not bool(moe_attr(module, "use_ep", False)):
        # vLLM slices every expert's intermediate dimension over TP: one dim-1 (gate / up) or dim-2 (down) block of
        # each stack per rank.
        if moe_intermediate_size % tp_size:
            return None, f"moe_intermediate_size {moe_intermediate_size} does not split over TP={tp_size}"
        if local_experts and local_experts != num_experts:
            return None, f"{local_experts} local experts without expert parallelism, the config gives {num_experts}"
        moe_tp_size, moe_tp_rank = (
            int(moe_attr(module, "tp_size", tp_size)),
            int(moe_attr(module, "tp_rank", tp_rank)),
        )
        if (moe_tp_size, moe_tp_rank) != (tp_size, tp_rank):
            return None, f"MoE TP size {moe_tp_size} / rank {moe_tp_rank}, needs the TP group ({tp_size} / {tp_rank})"
        return False, ""
    if num_experts % tp_size:
        return None, f"{num_experts} experts do not split over TP={tp_size} (expert parallelism)"
    ep_size, ep_rank = int(moe_attr(module, "ep_size", 0) or 0), int(moe_attr(module, "ep_rank", -1))
    if ep_size != tp_size or ep_rank != tp_rank:
        return (
            None,
            f"EP size {ep_size} / rank {ep_rank}, needs expert parallelism over the whole TP group "
            f"({tp_size} / {tp_rank})",
        )
    placement = str(moe_attr(module, "expert_placement_strategy", "linear"))
    if placement != "linear":
        return None, f"expert placement {placement!r}, needs 'linear'"
    if local_experts and local_experts * tp_size != num_experts:
        return (
            None,
            f"{local_experts} local experts on each of {tp_size} ranks, the config gives {num_experts} in total",
        )
    return True, ""


def expert_param(
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


def write_w13(
    param: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    layout: str,
    check_padding: bool,
    check: Optional[ParityCheckFn] = None,
    name: str = "w13_weight",
) -> None:
    """Writes rank-local ``gate`` / ``up`` [E, I_r, H] into vLLM's ``w13_weight``.

    ``MOE_LAYOUT_TPU``: vllm-torchtpu's unquantized MoE keeps it as [E, H, 2A] (transposed, each half padded with
    zeros to ``A``, the next multiple of 128). ``MOE_LAYOUT_VLLM``: vLLM's own [E, 2 I_r, H]. ``check(name,
    target, received)``, if given, sees every region before it is overwritten (``ExactParityCheck.check``).
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
        fused = torch.cat([gate, up], dim=1)
        if check is not None:
            check(name, w13, fused)
        w13.copy_(fused)
        return
    aligned = _aligned(inter)
    if check_padding:
        _check_zero(w13[..., inter:aligned], "w13 gate")
        _check_zero(w13[..., aligned + inter :], "w13 up")
    if check is not None:
        check(f"{name}[gate]", w13[..., :inter], gate.transpose(1, 2))
        check(f"{name}[up]", w13[..., aligned : aligned + inter], up.transpose(1, 2))
    w13[..., :inter].copy_(gate.transpose(1, 2))
    w13[..., aligned : aligned + inter].copy_(up.transpose(1, 2))


def write_w2(
    param: torch.Tensor,
    down: torch.Tensor,
    layout: str,
    check_padding: bool,
    check: Optional[ParityCheckFn] = None,
    name: str = "w2_weight",
) -> None:
    """Writes rank-local ``down`` [E, H, I_r] into ``w2_weight``: [E, A, H] (``MOE_LAYOUT_TPU``) or [E, H, I_r].

    ``check`` as in ``write_w13``.
    """
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
        if check is not None:
            check(name, w2, down)
        w2.copy_(down)
        return
    if check_padding:
        _check_zero(w2[:, inter:, :], "w2")
    if check is not None:
        check(name, w2[:, :inter, :], down.transpose(1, 2))
    w2[:, :inter, :].copy_(down.transpose(1, 2))


# ---------------------------------------------------------------------------------------------------------------
# Quantized experts: requantized on the rollout from the received bf16 slices.
# ---------------------------------------------------------------------------------------------------------------

# Bytes of the float32 copy of one requantization chunk (vllm-torchtpu's ``_REQUANT_CHUNK_BYTES`` when it has one).
_DEFAULT_REQUANT_CHUNK_BYTES = 1 << 30
VLLM_TORCHTPU_FP8_MODULE = "vllm_torchtpu.layers.adapter.quantization.fp8"


@dataclasses.dataclass(frozen=True)
class MoeRequant:
    """The rollout's quantized expert format, resolved once by ``plan_moe_layers`` (not re-read on every sync).

    ``quantize_and_format(w_fp32, *, name, quant_dtype, block_size, activation) -> (weight, scale)`` is the
    rollout's own quantizer of one chunk of an expert stack: ``name="w13"`` [e, 2 I_r, H] or ``"w2"`` [e, H, I_r] in
    float32 to vLLM's live layout, weight [e, H, 2A] / [e, A, H] and scale [e, K / block_size or 1, 1, N]
    (vllm-torchtpu's ``_quantize_and_format_single_moe_weight``: padding, quantization, transpose, GMM scale layout).
    """

    quant_dtype: torch.dtype
    block_size: Optional[int]
    activation: str
    quantize_and_format: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    chunk_bytes: int = _DEFAULT_REQUANT_CHUNK_BYTES
    scale_leaves: tuple[str, str] = ("w13_weight_scale_inv", "w2_weight_scale_inv")  # vLLM's w13 / w2 scales

    def describe(self) -> str:
        scales = "per output channel" if self.block_size is None else f"per block of {self.block_size}"
        return f"{self.quant_dtype}, scales {scales}, activation {self.activation}"


def _load_vllm_torchtpu_fp8() -> tuple[Optional[tuple[Callable, Callable, int]], str]:
    """``((quantize_and_format, get_moe_quant_config, chunk_bytes), "")`` from vllm-torchtpu, else ``(None, why)``.

    TODO(tpu): these are vllm-torchtpu's private helpers (the ones its ``process_weights_after_loading`` calls).
    Switch to a public requantization entry point once vllm-torchtpu has one. Until then the plan probes them here
    (import and signature), so a renamed helper makes the rollout fall back to vLLM's loaders instead of failing
    half-way through an install.
    """
    try:
        fp8 = importlib.import_module(VLLM_TORCHTPU_FP8_MODULE)
    except Exception as e:  # not installed, or not importable in this process
        return None, f"cannot import vllm-torchtpu's FP8 MoE requantizer ({type(e).__name__}: {e})"
    quantize_and_format = getattr(fp8, "_quantize_and_format_single_moe_weight", None)
    get_config = getattr(fp8, "_get_moe_quant_config", None)
    if not callable(quantize_and_format) or not callable(get_config):
        return None, f"{VLLM_TORCHTPU_FP8_MODULE} lacks _quantize_and_format_single_moe_weight / _get_moe_quant_config"
    try:
        inspect.signature(quantize_and_format).bind(
            torch.empty(0), name="w13", quant_dtype=torch.float8_e4m3fn, block_size=None, activation="silu"
        )
        inspect.signature(get_config).bind("")
    except (TypeError, ValueError) as e:
        return None, f"vllm-torchtpu's FP8 MoE requantizer has an unexpected signature ({e})"
    raw_chunk_bytes: Any = None
    try:
        raw_chunk_bytes = getattr(getattr(fp8, "envs", None), "TPU_MOE_REQUANT_CHUNK_BYTES", None)
    except Exception as e:
        return None, f"vllm-torchtpu's TPU_MOE_REQUANT_CHUNK_BYTES is invalid ({e})"
    if raw_chunk_bytes is None:
        raw_chunk_bytes = getattr(fp8, "_REQUANT_CHUNK_BYTES", None)
    if raw_chunk_bytes is None:
        raw_chunk_bytes = _DEFAULT_REQUANT_CHUNK_BYTES
    if isinstance(raw_chunk_bytes, bool):
        return None, f"vllm-torchtpu's MoE requant chunk size {raw_chunk_bytes!r} is not a positive int"
    try:
        chunk_bytes = int(raw_chunk_bytes)
    except (TypeError, ValueError) as e:
        return None, f"vllm-torchtpu's MoE requant chunk size {raw_chunk_bytes!r} is invalid ({e})"
    if chunk_bytes <= 0:
        return None, f"vllm-torchtpu's MoE requant chunk size must be positive, got {chunk_bytes}"
    return (quantize_and_format, get_config, chunk_bytes), ""


def requant_shapes(
    requant: MoeRequant, local_experts: int, local_inter: int, hidden: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``{leaf: (shape, dtype)}`` of one MoE layer's live quantized weights and scales on this rank."""
    aligned = _aligned(local_inter)
    bs = requant.block_size
    s13, s2 = requant.scale_leaves
    return {
        "w13_weight": ((local_experts, hidden, 2 * aligned), requant.quant_dtype),
        s13: ((local_experts, hidden // bs if bs else 1, 1, 2 * aligned), torch.float32),
        "w2_weight": ((local_experts, aligned, hidden), requant.quant_dtype),
        s2: ((local_experts, aligned // bs if bs else 1, 1, hidden), torch.float32),
    }


def _plan_vllm_torchtpu_fp8(
    moe_modules: list[tuple[str, Any]], local_experts: int, local_inter: int, hidden: int
) -> tuple[Optional[MoeRequant], str]:
    """``MOE_LAYOUT_TPU_FP8``: probes vllm-torchtpu's requantizer and checks every live weight and scale."""
    loaded, why = _load_vllm_torchtpu_fp8()
    if loaded is None:
        return None, why
    quantize_and_format, get_config, chunk_bytes = loaded
    try:
        _, quant_dtype, block_size = get_config("[tpu-sync direct install]: requantizing received MoE weights")
    except Exception as e:  # e.g. an unsupported MOE_REQUANTIZE_WEIGHT_DTYPE
        return None, f"vllm-torchtpu's MoE quantization config: {e}"
    if not isinstance(quant_dtype, torch.dtype):
        return None, f"vllm-torchtpu's MoE quantization dtype {quant_dtype!r} is not a torch.dtype"
    if block_size is not None and (isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0):
        return None, f"vllm-torchtpu's MOE_REQUANTIZE_BLOCK_SIZE={block_size!r} is not a positive int"
    activations = {_module_activation(m) for _, m in moe_modules}
    if len(activations) != 1:
        return None, f"vLLM's MoE layers use different activations {sorted(activations)}"
    requant = MoeRequant(quant_dtype, block_size, activations.pop(), quantize_and_format, chunk_bytes)
    aligned = _aligned(local_inter)
    if requant.block_size and (hidden % requant.block_size or aligned % requant.block_size):
        return None, (
            f"MOE_REQUANTIZE_BLOCK_SIZE={requant.block_size} does not divide the contracting dimensions "
            f"(hidden {hidden}, padded intermediate {aligned})"
        )
    for name, m in moe_modules:
        m_sd = m.state_dict() if isinstance(m, torch.nn.Module) else None
        for leaf, (shape, dtype) in requant_shapes(requant, local_experts, local_inter, hidden).items():
            t = m_sd.get(leaf) if m_sd is not None else getattr(m, leaf, None)
            if not isinstance(t, torch.Tensor):
                return None, f"vLLM's {name} has no {leaf}"
            t = _local(t)
            if tuple(t.shape) != shape or t.dtype != dtype:
                return None, (
                    f"vLLM's {name}.{leaf} is {tuple(t.shape)} {t.dtype}, requantizing gives {shape} {dtype} "
                    f"({requant.describe()})"
                )
    return requant, ""


# Quantized MoE layouts: how ``plan_moe_layers`` resolves and validates each one's ``MoeRequant``. Another quantized
# MoE method (dtype, MXFP4, ...) is one ``MOE_LAYOUTS`` entry plus a planner here; ``write_experts`` is shared.
_QUANTIZED_PLANNERS: dict[str, Callable[..., tuple[Optional[MoeRequant], str]]] = {
    MOE_LAYOUT_TPU_FP8: _plan_vllm_torchtpu_fp8,
}
MOE_QUANTIZED_LAYOUTS = frozenset(_QUANTIZED_PLANNERS)


class MoeLayerPlan(NamedTuple):
    """What ``plan_moe_layers`` found for the vLLM MoE layers."""

    layout: str  # one of ``MOE_LAYOUTS``' values
    expert_parallel: bool  # whole experts per rank (dim 0), else every expert's intermediate slice
    requant: Optional[MoeRequant]  # the quantized format, for ``MOE_QUANTIZED_LAYOUTS``


def plan_moe_layers(
    moe_modules: list[tuple[str, Any]],
    num_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
    *,
    allow_quantized: bool = True,
    use_ep_at_tp1: bool = False,
) -> tuple[Optional[MoeLayerPlan], str]:
    """``(MoeLayerPlan, "")`` if the vLLM MoE layers ``moe_modules`` (``(name, module)``) can be written directly,
    else ``(None, reason)``.

    Every layer must use one known MoE method (``MOE_LAYOUTS``) and one parallel mode (``moe_parallel_mode``). For a
    quantized layout the rollout's quantizer is probed and every live weight and scale is checked against what it will
    produce, so ``write_experts`` does not fail half-way through the live model. ``allow_quantized=False``
    (``engine_kwargs.tpu-sync.direct_install_quantized_moe=False``) refuses quantized layouts: vLLM's loaders then
    install them. ``use_ep_at_tp1``: at TP=1, where both modes hold every expert, report vLLM's ``use_ep``.
    """
    layouts: set[str] = set()
    modes: set[bool] = set()
    for name, m in moe_modules:
        qm = m.quant_method
        method = type(qm).__name__
        layout = MOE_LAYOUTS.get(method)
        if layout is None:
            return None, f"unsupported vLLM MoE method {method} (quantized?)"
        if layout in MOE_QUANTIZED_LAYOUTS and not allow_quantized:
            return None, (
                f"vLLM MoE method {method}: the direct install of quantized experts is off "
                "(engine_kwargs.tpu-sync.direct_install_quantized_moe=False)"
            )
        if getattr(qm, "_fused_moe_decode_op", None) is not None or getattr(qm, "_moe_tp_decode_op", None) is not None:
            return None, f"vLLM's {name} armed a fused shared+routed decode kernel (stages shared-expert buffers)"
        ep_mode, why = moe_parallel_mode(m, num_experts, moe_intermediate_size, tp_size, tp_rank)
        if ep_mode is None:
            return None, f"vLLM's {name}: {why}"
        layouts.add(layout)
        modes.add(bool(moe_attr(m, "use_ep", False)) if use_ep_at_tp1 and tp_size == 1 else ep_mode)
    if len(layouts) != 1:
        return None, f"vLLM MoE layouts {sorted(layouts) or 'not found'}"
    if len(modes) != 1:
        return None, "vLLM's MoE layers mix expert parallelism and TP-sliced experts"
    layout, expert_parallel = layouts.pop(), modes.pop()
    requant = None
    planner = _QUANTIZED_PLANNERS.get(layout)
    if planner is not None:
        local_experts = num_experts // tp_size if expert_parallel else num_experts
        local_inter = moe_intermediate_size if expert_parallel else moe_intermediate_size // tp_size
        requant, why = planner(moe_modules, local_experts, local_inter, hidden_size)
        if requant is None:
            return None, why
    return MoeLayerPlan(layout, expert_parallel, requant), ""


def _requantize(parts: tuple[torch.Tensor, ...], name: str, requant: MoeRequant) -> tuple[torch.Tensor, torch.Tensor]:
    """``requant.quantize_and_format`` of ``torch.cat(parts, dim=1)`` in float32, a few experts at a time.

    As vllm-torchtpu's own ``_requantize_expert_chunks``: each chunk's float32 copy is at most ``requant.chunk_bytes``
    and is materialized before the next one starts (pending lazy TPU tensors would otherwise pile up), and a single
    concatenation joins the quantized chunks (on TPU an in-place slice write rewrites the whole tensor, once per
    chunk). Experts are quantized independently, so the result is bit-identical to quantizing the whole stack.
    """
    num_experts = parts[0].shape[0]
    per_expert = max(1, sum(p[0].numel() for p in parts) * 4)
    per_chunk = max(1, requant.chunk_bytes // per_expert)
    weights, scales = [], []
    for start in range(0, num_experts, per_chunk):
        end = min(start + per_chunk, num_experts)
        chunk = parts[0][start:end] if len(parts) == 1 else torch.cat([p[start:end] for p in parts], dim=1)
        fp32 = chunk.to(torch.float32)
        del chunk
        w, s = requant.quantize_and_format(
            fp32,
            name=name,
            quant_dtype=requant.quant_dtype,
            block_size=requant.block_size,
            activation=requant.activation,
        )
        del fp32
        if per_chunk < num_experts:
            tpu_synchronize([w, s])
        weights.append(w)
        scales.append(s)
    if len(weights) == 1:
        return weights[0], scales[0]
    return torch.cat(weights, dim=0), torch.cat(scales, dim=0)


def write_requantized(
    targets: dict[str, torch.Tensor],
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    requant: MoeRequant,
    check: Optional[ParityCheckFn] = None,
    names: Optional[dict[str, str]] = None,
) -> None:
    """Requantizes rank-local ``gate`` / ``up`` [E, I_r, H] and ``down`` [E, H, I_r] with the rollout's quantizer and
    copies the results into vLLM's live ``w13_weight`` / ``w2_weight`` and scales (``targets``, by leaf).

    ``check`` (the step-0 exact parity check) compares the requantized weights and scales bit for bit with the live
    ones before overwriting them: vLLM quantized the same bf16 checkpoint with the same function on load
    (``QUANTIZE_ON_LOAD_PREFIXES``), so they must match exactly. ``names`` (by leaf) names the targets in its report.
    Each weight and its scale are materialized right after they are written, so no requantization temporaries pile up
    across layers.
    """
    names = names or {}
    s13, s2 = requant.scale_leaves
    for weight_leaf, scale_leaf, parts, name in (
        ("w13_weight", s13, (gate, up), "w13"),
        ("w2_weight", s2, (down,), "w2"),
    ):
        written = []
        for leaf, src in zip((weight_leaf, scale_leaf), _requantize(parts, name, requant), strict=True):
            target = _local(targets[leaf])
            what = names.get(leaf, leaf)
            if tuple(target.shape) != tuple(src.shape) or target.dtype != src.dtype:
                raise RuntimeError(
                    f"{_ERROR_PREFIX}: {what} is {tuple(target.shape)} {target.dtype}, the requantizer gives "
                    f"{tuple(src.shape)} {src.dtype}"
                )
            if check is not None:
                check(f"{what}[requantized]", target, src)
            target.copy_(src)
            written.append(target)
        tpu_synchronize(written)


def write_experts(
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    prefix: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    layout: str,
    expert_keys: dict[tuple[str, str], str],
    check_padding: bool,
    requant: Optional[MoeRequant] = None,
    check: Optional[ParityCheckFn] = None,
) -> None:
    """Writes one MoE block's rank-local ``gate`` / ``up`` [E, I_r, H] and ``down`` [E, H, I_r] into vLLM's expert
    weights of the block ``prefix`` in the planned ``layout`` (``plan_moe_layers``): copied (``write_w13`` /
    ``write_w2``), or requantized with ``requant`` for ``MOE_QUANTIZED_LAYOUTS`` (``write_requantized``).

    ``expert_keys`` caches vLLM's resolved keys (``expert_param``); ``check`` is the exact parity check.
    """
    w13_key, w2_key = (expert_param(model_sd, find_model_key, prefix, leaf, expert_keys) for leaf in _WEIGHT_LEAVES)
    if layout in MOE_QUANTIZED_LAYOUTS:
        if requant is None:
            raise RuntimeError(f"{_ERROR_PREFIX}: quantized MoE layout {layout} planned without its format")
        keys = {"w13_weight": w13_key, "w2_weight": w2_key}
        for leaf in requant.scale_leaves:
            keys[leaf] = expert_param(model_sd, find_model_key, prefix, leaf, expert_keys)
        targets = {leaf: model_sd[key] for leaf, key in keys.items()}
        write_requantized(targets, gate, up, down, requant, check=check, names=keys)
        return
    write_w13(model_sd[w13_key], gate, up, layout, check_padding, check=check, name=w13_key)
    write_w2(model_sd[w2_key], down, layout, check_padding, check=check, name=w2_key)


# ---------------------------------------------------------------------------------------------------------------
# Zero-copy expert-parallel wire layout (bf16 or trainer-quantized, shared across Qwen3.5 and Qwen3 MoE).
# ---------------------------------------------------------------------------------------------------------------


# TODO(tpu): Remove ``rollout_expert_parallel`` / ``EP_EXPERT_TENSORS`` / ``ep_zero_copy_targets`` once tpu-sync stages
# received weights in host DRAM, linearizes/transposes on host, and H2Ds directly into vLLM weights. Standard expert
# stacks already install directly under vLLM expert parallelism (``plan.expert_parallel`` shards them on dim 0); this
# transposed wire layout exists only to avoid the HBM staging copy on large MoEs such as Qwen3.5-397B-A17B.
class EPExpertTensor(NamedTuple):
    """An expert stack sent in vLLM's expert-parallel runtime layout (``engine_kwargs.tpu.rollout_expert_parallel``)."""

    hf: str  # suffix of Qwen3.5's fused HF tensor
    ep: str  # suffix of the tpu-sync weight tensor (received zero-copy into ``leaf``)
    scale: str  # suffix of the tpu-sync FP32 scale tensor [E, K_blocks, N] when trainer-quantized
    leaf: str  # vLLM's expert weight parameter
    scale_leaf: str  # vLLM's expert scale parameter [E_r, K_blocks, 1, N]
    kind: str  # "w13" or "w2"


EP_W13 = EPExpertTensor(
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.w13_ep",
    ".mlp.experts.w13_scale_ep",
    "w13_weight",
    "w13_weight_scale_inv",
    "w13",
)
EP_W2 = EPExpertTensor(
    ".mlp.experts.down_proj",
    ".mlp.experts.w2_ep",
    ".mlp.experts.w2_scale_ep",
    "w2_weight",
    "w2_weight_scale_inv",
    "w2",
)
EP_EXPERT_TENSORS: tuple[EPExpertTensor, ...] = (EP_W13, EP_W2)
EP_EXPERT_BLOCK_SUFFIXES: tuple[str, ...] = tuple(t.ep for t in EP_EXPERT_TENSORS)
EP_FP8_SCALE_SUFFIXES: tuple[str, ...] = tuple(t.scale for t in EP_EXPERT_TENSORS)

_SUPPORTED_QUANT_DTYPES: frozenset[torch.dtype] = frozenset(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
        torch.int8,
    )
    if isinstance(dt, torch.dtype)
)


def resolve_wire_dtype(raw: Any) -> Optional[torch.dtype]:
    """Resolves a ``torch.dtype`` or ``"torch.float8_e4m3fn"`` / ``"float32"`` wire string, else ``None``."""
    if isinstance(raw, torch.dtype):
        return raw
    if not isinstance(raw, str):
        return None
    dt = getattr(torch, raw.removeprefix("torch."), None)
    return dt if isinstance(dt, torch.dtype) else None


def _is_tpu_tensor(t: torch.Tensor) -> bool:
    return t.device.type == "tpu"


def _natural_block_order(blocks: set[str] | list[str]) -> list[str]:
    return sorted(blocks, key=lambda b: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", b)])


@dataclasses.dataclass(frozen=True)
class ExpertWireQuant:
    """Trainer-side MoE wire quantization format, discovered from the rollout's ``describe_vllm_params``."""

    quant_dtype: torch.dtype = FP8_MOE_DTYPE
    block_size: Optional[int] = None  # None: one scale per output channel
    activation: str = "silu"

    def __post_init__(self) -> None:
        if not isinstance(self.quant_dtype, torch.dtype) or self.quant_dtype not in _SUPPORTED_QUANT_DTYPES:
            raise ValueError(f"unsupported quantized expert dtype {self.quant_dtype!r}")
        if self.block_size is not None and (
            isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size <= 0
        ):
            raise ValueError(f"quantized expert block_size {self.block_size!r} is not a positive int")
        if "swigluoai" in str(self.activation).lower():
            raise ValueError(f"quantized expert activation {self.activation!r} is not supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_dtype": str(self.quant_dtype),
            "block_size": self.block_size,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExpertWireQuant:
        raw_dtype = d.get("quant_dtype")
        dtype = resolve_wire_dtype(str(raw_dtype)) if raw_dtype is not None else None
        if dtype is None or dtype not in _SUPPORTED_QUANT_DTYPES:
            raise ValueError(f"unknown or unsupported quantized expert dtype {raw_dtype!r}")
        raw_bs = d.get("block_size")
        if raw_bs is not None and (isinstance(raw_bs, bool) or not isinstance(raw_bs, int) or raw_bs <= 0):
            raise ValueError(f"quantized expert block_size {raw_bs!r} is not a positive int")
        raw_act = d.get("activation")
        activation = str("silu" if raw_act is None or raw_act == "" else raw_act).lower()
        if "swigluoai" in activation:
            raise ValueError(f"quantized expert activation {activation!r} is not supported")
        return cls(dtype, raw_bs, activation)

    @classmethod
    def from_description(cls, description: Optional[dict[str, Any]]) -> Optional[ExpertWireQuant]:
        """The format a rollout's ``describe_vllm_params`` reported, or ``None`` if its experts are not quantized."""
        moe_quant = (description or {}).get("moe_quant")
        return cls.from_dict(moe_quant) if isinstance(moe_quant, dict) and "quant_dtype" in moe_quant else None


def _probe_moe_quant(
    vllm_model: Any, tp_size: int = 1, tp_rank: Optional[int] = None
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """``(ExpertWireQuant.to_dict(), None)`` when ``vllm_model`` can receive trainer-quantized zero-copy EP experts,
    ``(None, None)`` when ``vllm_model`` has no quantized MoE layers, or ``(None, reason)`` when quantized MoE layers
    exist on the rollout but are ineligible for EP zero-copy direct install."""
    modules = list(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else []
    moe_modules = [(n, m) for n, m in modules if hasattr(m, "w13_weight") and hasattr(m, "quant_method")]
    if not moe_modules:
        return None, None
    methods = [type(m.quant_method).__name__ for _, m in moe_modules]
    layouts = {MOE_LAYOUTS.get(method) for method in methods}
    has_quant_moe = any(
        MOE_LAYOUTS.get(type(m.quant_method).__name__) in MOE_QUANTIZED_LAYOUTS
        or (
            isinstance(getattr(m, "w13_weight", None), torch.Tensor)
            and _local(m.w13_weight).dtype not in UNQUANTIZED_PARAM_DTYPES
        )
        for _, m in moe_modules
    )
    if not has_quant_moe:
        return None, None
    if None in layouts or len(layouts) != 1 or not (layouts <= MOE_QUANTIZED_LAYOUTS):
        return None, f"vLLM MoE layers use unsupported or mixed methods {sorted(set(methods))}"
    dense_why = non_moe_quantized_reason(vllm_model, moe_modules)
    if dense_why:
        return None, dense_why
    activations: set[str] = set()
    for mod_name, m in moe_modules:
        qm = m.quant_method
        if getattr(qm, "_fused_moe_decode_op", None) is not None or getattr(qm, "_moe_tp_decode_op", None) is not None:
            return None, f"vLLM's {mod_name} armed a fused shared+routed decode kernel"
        if not bool(moe_attr(m, "use_ep", False)):
            return None, f"vLLM's {mod_name} does not use expert parallelism (use_ep=False)"
        ep_size_raw = moe_attr(m, "ep_size", None)
        if tp_size > 1:
            ep_size = int(ep_size_raw or 0)
            if ep_size != tp_size:
                return None, f"vLLM's {mod_name} has ep_size={ep_size}, expected TP={tp_size}"
            if tp_rank is not None:
                ep_rank = int(moe_attr(m, "ep_rank", -1))
                if ep_rank != tp_rank:
                    return None, f"vLLM's {mod_name} has ep_rank={ep_rank}, expected TP rank {tp_rank}"
        elif ep_size_raw is not None and int(ep_size_raw) not in (0, 1):
            return None, f"vLLM's {mod_name} has ep_size={int(ep_size_raw)}, expected 1"
        placement = str(moe_attr(m, "expert_placement_strategy", "linear"))
        if placement != "linear":
            return None, f"vLLM's {mod_name} uses expert placement {placement!r}, needs 'linear'"
        global_e = int(moe_attr(m, "global_num_experts", 0) or 0)
        local_e = int(moe_attr(m, "local_num_experts", 0) or 0)
        if global_e and global_e % tp_size:
            return None, f"vLLM's {mod_name} has {global_e} experts, not divisible by TP={tp_size}"
        if global_e and local_e and local_e * tp_size != global_e:
            return None, f"vLLM's {mod_name} has {local_e} local experts * TP={tp_size} != {global_e}"
        if getattr(m, "w13_bias", None) is not None or getattr(m, "w2_bias", None) is not None:
            return None, f"vLLM's {mod_name} has expert bias (not supported)"
        act = _module_activation(m)
        if "swigluoai" in act.lower():
            return None, f"vLLM's {mod_name} uses unsupported activation {act!r}"
        activations.add(act)
    if len(activations) != 1:
        return None, f"vLLM's MoE layers use different activations {sorted(activations)}"
    activation = activations.pop()

    loaded, _ = _load_vllm_torchtpu_fp8()
    if loaded is not None:
        try:
            _, quant_dtype, raw_bs = loaded[1]("[tpu-sync]: describing the rollout's quantized MoE")
        except Exception as e:
            return None, f"vllm-torchtpu's MoE quantization config failed: {e}"
        if not isinstance(quant_dtype, torch.dtype) or quant_dtype not in _SUPPORTED_QUANT_DTYPES:
            return None, f"unsupported MoE quantization dtype {quant_dtype!r}"
        if raw_bs is not None and (isinstance(raw_bs, bool) or not isinstance(raw_bs, int) or raw_bs <= 0):
            return None, f"vllm-torchtpu's MOE_REQUANTIZE_BLOCK_SIZE={raw_bs!r} is not a positive int"
        block_size: Optional[int] = raw_bs
    else:
        # Fallback when vllm-torchtpu is not importable (e.g. CPU tests): infer and verify from live weights & scales.
        first_name, first_m = moe_modules[0]
        first_sd = first_m.state_dict() if isinstance(first_m, torch.nn.Module) else None
        w13_0 = first_sd.get("w13_weight") if first_sd is not None else getattr(first_m, "w13_weight", None)
        w2_0 = first_sd.get("w2_weight") if first_sd is not None else getattr(first_m, "w2_weight", None)
        s13_0 = (
            first_sd.get("w13_weight_scale_inv")
            if first_sd is not None
            else getattr(first_m, "w13_weight_scale_inv", None)
        )
        s2_0 = (
            first_sd.get("w2_weight_scale_inv")
            if first_sd is not None
            else getattr(first_m, "w2_weight_scale_inv", None)
        )
        if not all(isinstance(t, torch.Tensor) for t in (w13_0, w2_0, s13_0, s2_0)):
            return None, f"vLLM's {first_name} lacks quantized expert weights or scales in state_dict()"
        w13_l, w2_l = _local(w13_0), _local(w2_0)  # type: ignore[arg-type]
        s13_l, s2_l = _local(s13_0), _local(s2_0)  # type: ignore[arg-type]
        if w13_l.dtype not in _SUPPORTED_QUANT_DTYPES:
            return None, f"vLLM's {first_name}.w13_weight has unsupported dtype {w13_l.dtype}"
        if w13_l.ndim != 3 or w2_l.ndim != 3 or s13_l.ndim != 4 or s2_l.ndim != 4:
            return None, (
                f"vLLM's {first_name} has unexpected expert/scale ranks "
                f"(w13={w13_l.ndim}D, w2={w2_l.ndim}D, s13={s13_l.ndim}D, s2={s2_l.ndim}D)"
            )
        n_blocks_13 = int(s13_l.shape[1])
        n_blocks_2 = int(s2_l.shape[1])
        if n_blocks_13 < 1 or n_blocks_2 < 1:
            return None, f"vLLM's {first_name} has empty scale block dimension ({n_blocks_13}, {n_blocks_2})"
        if n_blocks_13 > 1:
            if int(w13_l.shape[1]) % n_blocks_13 != 0:
                return None, f"vLLM's {first_name} hidden {int(w13_l.shape[1])} not divisible by {n_blocks_13} blocks"
            block_size = int(w13_l.shape[1]) // n_blocks_13
        elif n_blocks_2 > 1:
            if int(w2_l.shape[1]) % n_blocks_2 != 0:
                return None, f"vLLM's {first_name} inter {int(w2_l.shape[1])} not divisible by {n_blocks_2} blocks"
            block_size = int(w2_l.shape[1]) // n_blocks_2
        else:
            block_size = None
        if block_size is not None and block_size <= 0:
            return None, f"vLLM's {first_name} inferred non-positive block_size={block_size}"
        quant_dtype = w13_l.dtype

    # Validate every MoE layer's live weights and scales against (quant_dtype, block_size) before advertising support.
    for mod_name, m in moe_modules:
        m_sd = m.state_dict() if isinstance(m, torch.nn.Module) else None
        tensors: dict[str, torch.Tensor] = {}
        for leaf in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            t = m_sd.get(leaf) if m_sd is not None else getattr(m, leaf, None)
            if not isinstance(t, torch.Tensor):
                return None, f"vLLM's {mod_name} has no {leaf} in state_dict()"
            lt = _local(t)
            if not lt.is_contiguous():
                return None, f"vLLM's {mod_name}.{leaf} is not contiguous"
            tensors[leaf] = lt
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        s13, s2 = tensors["w13_weight_scale_inv"], tensors["w2_weight_scale_inv"]
        if w13.ndim != 3 or w2.ndim != 3 or s13.ndim != 4 or s2.ndim != 4:
            return None, f"vLLM's {mod_name} has invalid expert/scale tensor ranks"
        if w13.dtype != quant_dtype or w2.dtype != quant_dtype:
            return None, f"vLLM's {mod_name} expert dtype ({w13.dtype}, {w2.dtype}) != {quant_dtype}"
        if s13.dtype != torch.float32 or s2.dtype != torch.float32:
            return None, f"vLLM's {mod_name} scale dtype ({s13.dtype}, {s2.dtype}) != torch.float32"
        e_local, hidden, two_aligned = (int(d) for d in w13.shape)
        if e_local < 1 or hidden < 1 or two_aligned < 2 * _MOE_PAD_MULTIPLE or two_aligned % (2 * _MOE_PAD_MULTIPLE):
            return None, f"vLLM's {mod_name}.w13_weight has unaligned GMM shape {tuple(w13.shape)}"
        aligned = two_aligned // 2
        if tuple(w2.shape) != (e_local, aligned, hidden):
            return None, f"vLLM's {mod_name}.w2_weight is {tuple(w2.shape)}, expected {(e_local, aligned, hidden)}"
        if block_size is not None and (hidden % block_size or aligned % block_size):
            return None, (
                f"vLLM's {mod_name} dimensions (hidden={hidden}, aligned={aligned}) are not divisible by "
                f"block_size={block_size}"
            )
        exp_s13 = (e_local, hidden // block_size if block_size else 1, 1, 2 * aligned)
        exp_s2 = (e_local, aligned // block_size if block_size else 1, 1, hidden)
        if tuple(s13.shape) != exp_s13 or tuple(s2.shape) != exp_s2:
            return None, (
                f"vLLM's {mod_name} scales are {tuple(s13.shape)} / {tuple(s2.shape)}, expected {exp_s13} / {exp_s2}"
            )
    try:
        wire_quant = ExpertWireQuant(quant_dtype, block_size, activation)
    except ValueError as e:
        return None, str(e)
    return wire_quant.to_dict(), None


def describe_moe_quant(vllm_model: Any, tp_size: int = 1, tp_rank: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Rollout side: ``ExpertWireQuant.to_dict()`` when all MoE layers on ``vllm_model`` are eligible for zero-copy
    trainer-quantized EP install, else ``None``."""
    moe_quant, _ = _probe_moe_quant(vllm_model, tp_size=tp_size, tp_rank=tp_rank)
    return moe_quant


def moe_quant_ineligible_reason(vllm_model: Any, tp_size: int = 1, tp_rank: Optional[int] = None) -> Optional[str]:
    """Why ``vllm_model``'s quantized MoE layers cannot receive trainer-quantized zero-copy EP weights, or ``None``."""
    _, reason = _probe_moe_quant(vllm_model, tp_size=tp_size, tp_rank=tp_rank)
    return reason


def quantize_expert_stack(weight: torch.Tensor, quant: ExpertWireQuant) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q, scale)`` of ``weight`` [..., N, K] quantized over its last (contracting) dimension in format ``quant``.

    Matches vllm-torchtpu's ``quantize_tensor(w, quant_dtype, axis=-1, block_size)`` bit for bit (explicit float32
    reciprocal multiply for tie-point stability, exact zeros for all-zero blocks, half-to-even rounding for integer
    dtypes), without importing vLLM on the trainer. ``q`` is [..., N, K] in ``quant.quant_dtype``; ``scale`` is
    float32 [..., K / block_size (1 per channel), N].
    """
    dtype = quant.quant_dtype
    is_float = torch.empty((), dtype=dtype).is_floating_point()
    info: Any = torch.finfo(dtype) if is_float else torch.iinfo(dtype)
    w = weight.to(torch.float32)
    k = w.shape[-1]
    block = quant.block_size or k
    if k % block:
        raise ValueError(f"tpu-sync: contracting dimension {k} is not divisible by block_size={block}")
    blocked = w.reshape(*w.shape[:-1], k // block, block)
    dtype_max_recip = torch.tensor(1.0 / float(info.max), dtype=torch.float32, device=w.device)
    scale = blocked.abs().amax(dim=-1, keepdim=True) * dtype_max_recip
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = torch.clamp(blocked * scale_inv, min=float(info.min), max=float(info.max))
    if not is_float:
        q = torch.round(q)
    q = q.reshape_as(w).to(dtype)
    return q, scale.squeeze(-1).transpose(-2, -1).contiguous()


def _pad_expert_for_gmm(w: torch.Tensor, kind: str, activation: str = "silu") -> torch.Tensor:
    """vllm-torchtpu's ``_quantize_and_format_single_moe_weight`` pre-quantization alignment to 128."""
    if "swigluoai" in activation.lower():
        raise ValueError(f"activation {activation!r} is not supported")
    if kind == "w13":
        half = w.shape[1] // 2
        aligned = _aligned(half)
        if aligned != half:
            pad = w.new_zeros(w.shape[0], aligned - half, w.shape[2])
            w = torch.cat([w[:, :half, :], pad, w[:, half:, :], pad], dim=1)
    elif kind == "w2":
        inter = w.shape[2]
        aligned = _aligned(inter)
        if aligned != inter:
            w = torch.cat([w, w.new_zeros(w.shape[0], w.shape[1], aligned - inter)], dim=2)
    else:
        raise ValueError(f"unknown expert tensor kind {kind!r}")
    return w


def format_ep_expert_weight(
    weight: torch.Tensor,
    kind: str,
    quant: Optional[ExpertWireQuant] = None,
    chunk_bytes: int = _DEFAULT_REQUANT_CHUNK_BYTES,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Trainer side: formats ``weight`` (``kind="w13"`` [E, 2I, H] or ``"w2"`` [E, H, I]) for zero-copy EP export.

    Without ``quant``, returns ``(weight.transpose(1, 2).contiguous(), None)``. With ``quant``, pads to vllm-torchtpu's
    128-multiple GMM layout and quantizes a few experts at a time (capping float32 temporaries at ``chunk_bytes``),
    returning ``(q_gmm, scale)`` where ``q_gmm`` is [E, H, 2A] / [E, A, H] in ``quant.quant_dtype`` and ``scale`` is
    float32 [E, K / block_size (1 per channel), N].
    """
    if quant is None:
        out_w = weight.transpose(-2, -1).contiguous()
        if _is_tpu_tensor(out_w):
            tpu_synchronize([out_w])
        return out_w, None
    num_experts = int(weight.shape[0])
    per_expert = max(1, int(weight[0].numel()) * 4)
    per_chunk = max(1, chunk_bytes // per_expert)
    weights, scales = [], []
    for start in range(0, num_experts, per_chunk):
        end = min(start + per_chunk, num_experts)
        padded = _pad_expert_for_gmm(weight[start:end].to(torch.float32), kind, quant.activation)
        q, scale = quantize_expert_stack(padded, quant)
        del padded
        q_gmm = q.transpose(1, 2).contiguous()
        if _is_tpu_tensor(q_gmm):
            tpu_synchronize([q_gmm, scale])
        weights.append(q_gmm)
        scales.append(scale)
    if len(weights) == 1:
        return weights[0], scales[0]
    cat_w, cat_s = torch.cat(weights, dim=0), torch.cat(scales, dim=0)
    if _is_tpu_tensor(cat_w):
        tpu_synchronize([cat_w, cat_s])
    return cat_w, cat_s


def ep_quant_loader_error(
    global_shapes: Mapping[str, list[int]],
    global_dtypes: Optional[Mapping[str, str]] = None,
) -> str:
    """Fast-fail message when trainer-quantized EP expert tensors or scales would hit vLLM's checkpoint loaders."""
    scale_names = sorted(n for n in global_shapes if n.endswith(EP_FP8_SCALE_SUFFIXES))
    if scale_names:
        return (
            f"cannot install trainer-quantized expert tensor {scale_names[0]} via vLLM's checkpoint loaders; "
            "direct install was refused"
        )
    if global_dtypes:
        for name in sorted(global_dtypes):
            if name not in global_shapes:
                continue
            dt = resolve_wire_dtype(global_dtypes[name])
            if dt is None or dt not in UNQUANTIZED_PARAM_DTYPES:
                return (
                    f"cannot install quantized wire tensor {name} ({global_dtypes[name]}) via vLLM's checkpoint "
                    "loaders; direct install was refused"
                )
    return ""


def plan_ep_moe_layers(
    moe_modules: list[tuple[str, Any]],
    global_shapes: dict[str, list[int]],
    expert_blocks: set[str] | list[str],
    num_experts: int,
    moe_intermediate_size: int,
    tp_size: int,
    tp_rank: int,
    *,
    allow_quantized: bool = True,
    global_dtypes: Optional[Mapping[str, str]] = None,
) -> tuple[Optional[MoeLayerPlan], str]:
    """Rollout side: validates the zero-copy expert-parallel wire layout against ``moe_modules`` up front."""
    if not moe_modules:
        return None, "vLLM MoE layouts not found"
    ordered_blocks = _natural_block_order(expert_blocks)
    if len(moe_modules) != len(ordered_blocks):
        return (
            None,
            f"vLLM has {len(moe_modules)} fused-MoE layers, the trainer sent {len(ordered_blocks)} MoE blocks",
        )
    wire_quant = any(n.endswith(EP_FP8_SCALE_SUFFIXES) for n in global_shapes)
    for block in ordered_blocks:
        req = EP_EXPERT_BLOCK_SUFFIXES + (EP_FP8_SCALE_SUFFIXES if wire_quant else ())
        missing = [sfx for sfx in req if block + sfx not in global_shapes]
        if missing:
            return None, f"{block} lacks {missing}"
    if wire_quant and not allow_quantized:
        return None, (
            "the direct install of quantized experts is off (engine_kwargs.tpu-sync.direct_install_quantized_moe=False)"
        )
    for block, (mod_name, m) in zip(ordered_blocks, moe_modules, strict=True):
        qm = m.quant_method
        method = type(qm).__name__
        layout = MOE_LAYOUTS.get(method)
        if getattr(qm, "_fused_moe_decode_op", None) is not None or getattr(qm, "_moe_tp_decode_op", None) is not None:
            return None, f"vLLM's {mod_name} armed a fused shared+routed decode kernel (stages shared-expert buffers)"
        if not bool(moe_attr(m, "use_ep", False)):
            return None, "the trainer sent the expert-parallel layout but vLLM does not use expert parallelism"
        if wire_quant and layout not in MOE_QUANTIZED_LAYOUTS:
            return None, (
                f"the trainer sent quantized experts (rollout_fp8_moe) but vLLM's MoE is {method}, not a supported "
                "quantized MoE method (set QUANTIZE_ON_LOAD_PREFIXES=experts on the rollout)"
            )
        if not wire_quant and method != "VllmUnquantizedFusedMoEMethod":
            hint = (
                " (set engine_kwargs.tpu.rollout_fp8_moe=True or enable describe_vllm_params auto-discovery)"
                if layout in MOE_QUANTIZED_LAYOUTS
                else ""
            )
            return None, f"expert-parallel direct install needs vllm-torchtpu's unquantized MoE, not {method}{hint}"
        ep_mode, why = moe_parallel_mode(m, num_experts, moe_intermediate_size, tp_size, tp_rank)
        if ep_mode is None:
            return None, f"vLLM's {mod_name}: {why}"
        m_sd = m.state_dict() if isinstance(m, torch.nn.Module) else None
        for ep in EP_EXPERT_TENSORS:
            w_name = f"{block}{ep.ep}"
            w_shape = global_shapes.get(w_name)
            if w_shape is None or len(w_shape) != 3:
                return None, f"missing or invalid 3-D {w_name} in global_shapes"
            if int(w_shape[0]) != num_experts:
                return None, f"{w_name} has {w_shape[0]} experts, the config gives {num_experts}"
            if w_shape[0] % tp_size:
                return None, f"{w_name} {list(w_shape)} dim 0 does not split over TP={tp_size}"
            expected_w = (w_shape[0] // tp_size, int(w_shape[1]), int(w_shape[2]))
            live_w_raw = m_sd.get(ep.leaf) if m_sd is not None else getattr(m, ep.leaf, None)
            if not isinstance(live_w_raw, torch.Tensor):
                return None, f"vLLM's {mod_name} has no {ep.leaf}"
            live_w = _local(live_w_raw)
            if not live_w.is_contiguous():
                return None, f"vLLM's {mod_name}.{ep.leaf} is not contiguous"
            if wire_quant:
                if live_w.dtype not in _SUPPORTED_QUANT_DTYPES:
                    return None, (
                        f"vLLM's {mod_name}.{ep.leaf} is {tuple(live_w.shape)} {live_w.dtype}, "
                        f"expected quantized {expected_w}"
                    )
                if global_dtypes is not None and w_name in global_dtypes:
                    w_wire_dtype = resolve_wire_dtype(global_dtypes[w_name])
                    if w_wire_dtype is None or w_wire_dtype not in _SUPPORTED_QUANT_DTYPES:
                        return None, f"{w_name} has unsupported wire dtype {global_dtypes[w_name]!r}"
                    if live_w.dtype != w_wire_dtype:
                        return None, (
                            f"vLLM's {mod_name}.{ep.leaf} has dtype {live_w.dtype}, "
                            f"trainer sent {w_wire_dtype} for {w_name}"
                        )
                if tuple(live_w.shape) != expected_w:
                    return None, (
                        f"vLLM's {mod_name}.{ep.leaf} is {tuple(live_w.shape)} {live_w.dtype}, "
                        f"expected quantized {expected_w}"
                    )
                s_name = f"{block}{ep.scale}"
                s_shape = global_shapes.get(s_name)
                if s_shape is None or len(s_shape) != 3:
                    return None, f"missing or invalid 3-D {s_name} in global_shapes"
                if int(s_shape[0]) != num_experts or s_shape[0] % tp_size:
                    return None, f"{s_name} {list(s_shape)} dim 0 does not match {num_experts} experts / TP={tp_size}"
                if global_dtypes is not None and s_name in global_dtypes:
                    s_wire_dtype = resolve_wire_dtype(global_dtypes[s_name])
                    if s_wire_dtype != torch.float32:
                        return None, f"{s_name} has wire dtype {global_dtypes[s_name]!r}, expected torch.float32"
                expected_s = (s_shape[0] // tp_size, int(s_shape[1]), 1, int(s_shape[2]))
                live_s_raw = m_sd.get(ep.scale_leaf) if m_sd is not None else getattr(m, ep.scale_leaf, None)
                if not isinstance(live_s_raw, torch.Tensor):
                    return None, f"vLLM's {mod_name} has no {ep.scale_leaf}"
                live_s = _local(live_s_raw)
                if tuple(live_s.shape) != expected_s or live_s.dtype != torch.float32 or not live_s.is_contiguous():
                    return None, (
                        f"vLLM's {mod_name}.{ep.scale_leaf} is {tuple(live_s.shape)} {live_s.dtype}, "
                        f"wire scale gives {expected_s} torch.float32"
                    )
            else:
                if global_dtypes is not None and w_name in global_dtypes:
                    w_wire_dtype = resolve_wire_dtype(global_dtypes[w_name])
                    if w_wire_dtype != torch.bfloat16:
                        return None, f"{w_name} has wire dtype {global_dtypes[w_name]!r} without expert scales"
                if tuple(live_w.shape) != expected_w or live_w.dtype != torch.bfloat16:
                    return None, (
                        f"vLLM's {mod_name}.{ep.leaf} is {tuple(live_w.shape)} {live_w.dtype}, "
                        f"the expert-parallel wire tensor {w_name} gives {expected_w} torch.bfloat16"
                    )
    return MoeLayerPlan(MOE_LAYOUT_TPU_EP_QUANT if wire_quant else MOE_LAYOUT_TPU_EP, True, None), ""


def ep_zero_copy_targets(
    moe_layout: Optional[str],
    tp_size: int,
    expert_keys: dict[tuple[str, str], str],
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    global_shapes: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    """Rollout side: ``{wire_name: vLLM weight}`` for zero-copy expert-parallel layouts."""
    if moe_layout not in (MOE_LAYOUT_TPU_EP, MOE_LAYOUT_TPU_EP_QUANT):
        return {}
    quantized = moe_layout == MOE_LAYOUT_TPU_EP_QUANT
    targets: dict[str, torch.Tensor] = {}
    for name, shape in global_shapes.items():
        ep = next((t for t in EP_EXPERT_TENSORS if name.endswith(t.ep)), None)
        if ep is None:
            continue
        prefix = name[: -len(ep.ep)] + ".mlp.experts"
        weight = _local(model_sd[expert_param(model_sd, find_model_key, prefix, ep.leaf, expert_keys)])
        expected = (shape[0] // tp_size, *shape[1:])
        bad_dtype = (weight.dtype not in _SUPPORTED_QUANT_DTYPES) if quantized else (weight.dtype != torch.bfloat16)
        if tuple(weight.shape) != expected or bad_dtype or not weight.is_contiguous():
            want_dtype = "quantized" if quantized else "bfloat16"
            raise RuntimeError(
                f"{_ERROR_PREFIX}: vLLM's {prefix}.{ep.leaf} is {tuple(weight.shape)} {weight.dtype} "
                f"(contiguous={weight.is_contiguous()}); the expert-parallel layout needs {expected}, {want_dtype}, "
                "contiguous"
            )
        targets[name] = weight
    return targets


def ep_sharding_rule(name: str, global_shape: list[int], tp_size: int) -> Optional[tuple[list[str], list[int]]]:
    """Dim-0 TP sharding rule for ``EP_EXPERT_TENSORS`` weights and scales, else ``None``."""
    if not name.endswith(EP_EXPERT_BLOCK_SUFFIXES + EP_FP8_SCALE_SUFFIXES):
        return None
    if global_shape[0] % tp_size:
        raise RuntimeError(f"{_ERROR_PREFIX}: {name} {global_shape} dim 0 does not split over {tp_size}")
    spec = ["tp"] + [""] * (len(global_shape) - 1)
    local = [global_shape[0] // tp_size, *global_shape[1:]]
    return spec, local


def write_wire_scale(target: torch.Tensor, src: torch.Tensor, name: str, check: Optional[ParityCheckFn] = None) -> None:
    """Copies a received 3-D scale [E_r, K / block, N] into vLLM's live 4-D ``w*_weight_scale_inv``."""
    target = _local(target)
    expected = (src.shape[0], src.shape[1], 1, src.shape[2]) if src.ndim == 3 else None
    if (
        expected is None
        or tuple(target.shape) != expected
        or target.dtype != torch.float32
        or src.dtype != torch.float32
    ):
        raise RuntimeError(
            f"{_ERROR_PREFIX}: vLLM's {name} is {tuple(target.shape)} {target.dtype}, received "
            f"{tuple(src.shape)} {src.dtype} (expected vLLM's {expected} torch.float32)"
        )
    formatted = src.unsqueeze(2)
    if check is not None:
        check(name, target, formatted)
    target.copy_(formatted)


def write_ep_scales(
    staging: dict[str, torch.Tensor],
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    expert_keys: dict[tuple[str, str], str],
    check: Optional[ParityCheckFn] = None,
) -> set[str]:
    """Installs every ``EP_EXPERT_TENSORS`` scale in ``staging`` and returns the staging names consumed."""
    consumed: set[str] = set()
    for name in list(staging):
        ep = next((t for t in EP_EXPERT_TENSORS if name.endswith(t.scale)), None)
        if ep is None:
            continue
        prefix = name[: -len(ep.scale)] + ".mlp.experts"
        key = expert_param(model_sd, find_model_key, prefix, ep.scale_leaf, expert_keys)
        write_wire_scale(model_sd[key], staging[name], key, check=check)
        consumed.add(name)
    return consumed
