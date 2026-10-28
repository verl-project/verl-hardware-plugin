# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The model-specific direct ``tpu-sync`` installs into vLLM, as one table the worker extension walks.

By default ``tpu_sync_worker_extension`` installs every received tensor with the dense path's rules (TP-sharded
copies, fused q/k/v and gate/up) and sends hybrid or MoE models through vLLM's checkpoint loaders (slow: every rank
all-gathers every tensor). A ``DirectInstallLayout`` lets one model family receive and write its special tensors
(GatedDeltaNet projections, expert stacks, ...) straight into vLLM's parameters instead. The extension asks the
layouts of ``direct_install_layouts()`` in order and uses the first that can install into the live vLLM model; it
names no model itself.

To add a model: write ``rollout/<model>_layout.py`` with a module-level ``LAYOUT = DirectInstallLayout(...)`` (thin
adapters to the signatures below) and add it to ``direct_install_layouts()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, NamedTuple, Optional

import torch

# ``(sharding_spec, local_shape)`` of a tensor on one TP rank, e.g. ``(["tp", ""], [rows // tp, cols])``.
ShardingRule = tuple[list[str], list[int]]
# ``(vllm_model, global_shapes, hf_text_config, tp_size, worker_rank) -> (plan, "")`` or ``(None, reason)``.
PlanFn = Callable[[Any, dict[str, list[int]], Any, int, int], tuple[Optional[Any], str]]
# ``(staging, model_sd, find_model_key, copy_into, plan, check) -> staging names used``.
# ``copy_into(vllm_key, tensor)`` copies a tensor in vLLM's checkpoint layout (and runs the parity check);
# ``check``, if not None, is the exact parity check for regions written without ``copy_into``.
InstallFn = Callable[..., set[str]]
WeightsIterator = Iterator[tuple[str, torch.Tensor]]


@dataclass(frozen=True)
class DirectInstallLayout:
    """The direct install of one model family (see the module docstring)."""

    name: str  # logged as ``install mode <name> direct``
    plan: PlanFn  # checks the live vLLM model and the trainer's tensors once, on the first sync
    # ``(name, global_shape, tp_size) -> ShardingRule`` of a model-specific tensor, or None for the dense path's rules
    sharding_rule: Callable[[str, list[int], int], Optional[ShardingRule]]
    install: InstallFn  # writes the model-specific tensors, every sync; the dense path installs the rest
    # vLLM parameters fused from several trainer tensors, in addition to the dense path's: (fused suffix, sources).
    fused_targets: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # vLLM-loader fallback (no layout planned): ``(weights, hf_text_config) -> weights`` back in HF's checkpoint
    # layout, undoing how the trainer sends this model's tensors over tpu-sync.
    loader_weights: Optional[Callable[[WeightsIterator, Any], WeightsIterator]] = None
    # vLLM-loader fallback: ``(global_shapes, hf_text_config) -> error`` ("" if ``loader_weights`` can undo them).
    loader_error: Optional[Callable[[dict[str, list[int]], Any], str]] = None


class LayoutSelection(NamedTuple):
    """What ``select_layout`` chose."""

    layout: Optional[DirectInstallLayout]  # None: no layout plans (dense path or vLLM's loaders)
    plan: Any  # ``layout.plan``'s plan, passed to ``layout.install``
    reasons: list[str]  # why the layouts tried before ``layout`` (all of them if None) refused


def direct_install_layouts() -> tuple[DirectInstallLayout, ...]:
    """Every direct-install layout, in priority order (the first that plans is used)."""
    # Imported here: every layout module imports this one for ``DirectInstallLayout``.
    from verl_hardware_plugin.accelerators.tpu.rollout import qwen3_5_layout, qwen3_moe_layout

    return (qwen3_5_layout.LAYOUT, qwen3_moe_layout.LAYOUT)


def select_layout(
    vllm_model: Any,
    global_shapes: dict[str, list[int]],
    hf_text_config: Any,
    tp_size: int,
    worker_rank: int,
    layouts: Optional[tuple[DirectInstallLayout, ...]] = None,
) -> LayoutSelection:
    """The first of ``layouts`` (default ``direct_install_layouts()``) that can install into ``vllm_model``."""
    reasons: list[str] = []
    for layout in direct_install_layouts() if layouts is None else layouts:
        plan, reason = layout.plan(vllm_model, global_shapes, hf_text_config, tp_size, worker_rank)
        if plan is not None:
            return LayoutSelection(layout, plan, reasons)
        reasons.append(reason)
    return LayoutSelection(None, None, reasons)


def loader_weights(
    weights: WeightsIterator, hf_text_config: Any, layouts: Optional[tuple[DirectInstallLayout, ...]] = None
) -> WeightsIterator:
    """The received tensors in HF's checkpoint layout for vLLM's loaders: every layout's ``loader_weights`` applied."""
    for layout in direct_install_layouts() if layouts is None else layouts:
        if layout.loader_weights is not None:
            weights = layout.loader_weights(weights, hf_text_config)
    return weights


def loader_error(
    global_shapes: dict[str, list[int]], hf_text_config: Any, layouts: Optional[tuple[DirectInstallLayout, ...]] = None
) -> str:
    """Why vLLM's loaders cannot take the trainer's tensors (the first layout's ``loader_error``), or ``""``."""
    for layout in direct_install_layouts() if layouts is None else layouts:
        error = layout.loader_error(global_shapes, hf_text_config) if layout.loader_error is not None else ""
        if error:
            return error
    return ""
