# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM fused-MoE expert weights: the layouts of ``w13_weight`` / ``w2_weight`` and writers for them.

Shared by the direct ``tpu-sync`` installs that write experts straight into vLLM (``qwen3_5_layout``,
``qwen3_moe_layout``). Nothing here is model-specific: callers pass each rank's expert tensors in HF's per-expert
orientation, ``gate`` / ``up`` [E, I, H] and ``down`` [E, H, I].
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import torch

# vLLM MoE methods whose expert weight layout ``write_w13`` / ``write_w2`` know: vllm-torchtpu's unquantized method
# (transposed, each w13 half padded to a multiple of 128) and vLLM's own unquantized method ([E, 2I, H] / [E, H, I]).
MOE_LAYOUT_TPU = "tpu_transposed_padded"
MOE_LAYOUT_VLLM = "vllm"
MOE_LAYOUT_TPU_EP = "tpu_expert_parallel"  # vllm-torchtpu zero-copy EP layout (see qwen3_5_layout.EP_EXPERT_TENSORS)
MOE_LAYOUTS = {"VllmUnquantizedFusedMoEMethod": MOE_LAYOUT_TPU, "UnquantizedFusedMoEMethod": MOE_LAYOUT_VLLM}
_MOE_PAD_MULTIPLE = 128  # vllm-torchtpu's VllmUnquantizedFusedMoEMethod.process_weights_after_loading
# ``ExactParityCheck.check(name, target, received)``: called on a target region before it is overwritten.
ParityCheckFn = Callable[[str, torch.Tensor, torch.Tensor], None]

_ERROR_PREFIX = "tpu-sync direct install"


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
