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
from typing import Any, Callable, NamedTuple, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.device_api import tpu_synchronize

# vLLM MoE methods (by class name) and their expert weight layout:
# * ``MOE_LAYOUT_TPU``: vllm-torchtpu's unquantized method, w13 [E, H, 2A] / w2 [E, A, H] (transposed; each w13 half
#   and w2's rows padded with zeros to ``A``, the next multiple of 128);
# * ``MOE_LAYOUT_VLLM``: vLLM's own unquantized method, [E, 2I, H] / [E, H, I];
# * ``MOE_LAYOUT_TPU_FP8``: vllm-torchtpu's FP8 method (``QUANTIZE_ON_LOAD_PREFIXES=experts`` or an FP8 checkpoint):
#   ``MOE_LAYOUT_TPU``'s layout in its ``MOE_REQUANTIZE_WEIGHT_DTYPE`` (FP8 e4m3 by default) plus
#   ``w*_weight_scale_inv`` scales, per output channel or per ``MOE_REQUANTIZE_BLOCK_SIZE`` block of the contracting
#   dimension (``requant_shapes``). Written by requantizing the received bf16 (``write_requantized``).
MOE_LAYOUT_TPU = "tpu_transposed_padded"
MOE_LAYOUT_VLLM = "vllm"
MOE_LAYOUT_TPU_EP = "tpu_expert_parallel"  # vllm-torchtpu zero-copy EP layout (see qwen3_5_layout.EP_EXPERT_TENSORS)
MOE_LAYOUT_TPU_FP8 = "tpu_transposed_padded_fp8"
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
    activation = str(moe_attr(module, "activation", "silu")).lower()
    if "swigluoai" in activation or getattr(module, "w13_bias", None) is not None:
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
    chunk_bytes = getattr(fp8, "_REQUANT_CHUNK_BYTES", _DEFAULT_REQUANT_CHUNK_BYTES)
    return (quantize_and_format, get_config, int(chunk_bytes)), ""


def _module_activation(module: Any) -> str:
    """The activation string vllm-torchtpu quantized (and runs) ``module``'s experts with."""
    activation = getattr(getattr(module, "quant_method", None), "_tpu_activation_str", None)
    if activation is None:
        activation = moe_attr(module, "activation", "silu")
        activation = getattr(activation, "value", activation)  # vLLM's MoEActivation enum
    return str(activation).lower()


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
    activations = {_module_activation(m) for _, m in moe_modules}
    if len(activations) != 1:
        return None, f"vLLM's MoE layers use different activations {sorted(activations)}"
    requant = MoeRequant(
        quant_dtype, int(block_size) if block_size else None, activations.pop(), quantize_and_format, chunk_bytes
    )
    aligned = _aligned(local_inter)
    if requant.block_size and (hidden % requant.block_size or aligned % requant.block_size):
        return None, (
            f"MOE_REQUANTIZE_BLOCK_SIZE={requant.block_size} does not divide the contracting dimensions "
            f"(hidden {hidden}, padded intermediate {aligned})"
        )
    for name, m in moe_modules:
        for leaf, (shape, dtype) in requant_shapes(requant, local_experts, local_inter, hidden).items():
            t = getattr(m, leaf, None)
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
        method = type(m.quant_method).__name__
        layout = MOE_LAYOUTS.get(method)
        if layout is None:
            return None, f"unsupported vLLM MoE method {method} (quantized?)"
        if layout in MOE_QUANTIZED_LAYOUTS and not allow_quantized:
            return None, (
                f"vLLM MoE method {method}: the direct install of quantized experts is off "
                "(engine_kwargs.tpu-sync.direct_install_quantized_moe=False)"
            )
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
