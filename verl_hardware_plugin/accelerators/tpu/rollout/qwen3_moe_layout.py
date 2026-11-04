# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct ``tpu-sync`` install of Qwen3 MoE (e.g. Qwen3-30B-A3B) experts into vLLM's fused-MoE weights.

Without this module, Qwen3 MoE models install the received weights through vLLM's checkpoint loaders: the trainer
all-gathers every expert stack and sends one HF tensor per expert, every rollout rank all-gathers every tensor over
its TP group, and vLLM's MoE loaders loop over the experts (``_install_with_vllm_loaders``).

Trainer side (``expert_stack_wire_name``, used by ``TrainerWeights.local_shards``): torchtitan keeps each MoE layer's
routed experts as one stack per projection, ``moe.experts.w1`` / ``w3`` [E, I, H] and ``w2`` [E, H, I], whose expert
``e`` is HF's ``model.layers.N.mlp.experts.<e>.{gate,up,down}_proj.weight``. Each stack is sent as is, ONE 3-D tensor
named after those HF tensors without the expert index (the same orientation as Qwen3.5's expert pieces, see
``qwen3_5_layout``):

==============================================  ============  ===========================================
Trainer tensor (tpu-sync name)                  Shape         Received by TP rank ``r``
==============================================  ============  ===========================================
``model.layers.N.mlp.experts.gate_proj.weight``  [E, I, H]     EP: experts ``[r E/tp, (r+1) E/tp)`` (dim 0);
``model.layers.N.mlp.experts.up_proj.weight``    [E, I, H]     TP: every expert's ``I`` slice ``r`` (dim 1)
``model.layers.N.mlp.experts.down_proj.weight``  [E, H, I]     EP: dim 0; TP: ``I`` slice ``r`` (dim 2)
``model.layers.N.mlp.gate.weight`` (router)      [E, H]        whole (vLLM keeps it replicated)
==============================================  ============  ===========================================

FSDP's ``Shard(0)`` of a stack holds whole experts, a dim-0 block of the wire tensor, so the trainer exports it
without a gather and tpu-sync reshards it to whatever the rollout holds. Each rank's piece is one block of the 3-D
tensor in both rollout modes:

* **EP** (vLLM ``enable_expert_parallel``, EP size = TP size, linear placement): rank ``r`` holds whole experts
  ``[r * E / tp, (r + 1) * E / tp)``;
* **TP** (no expert parallelism): rank ``r`` holds slice ``r`` of every expert's intermediate dimension, as for
  Qwen3.5's experts.

Each rank writes its gate / up / down into its ``w13_weight`` / ``w2_weight`` with ``moe_weights``' writers
(vllm-torchtpu's transposed, 128-padded layout or vLLM's own). Everything else (attention, norms, embeddings) goes
through the dense path's rules in ``tpu_sync_worker_extension``.

``plan_direct_install`` checks this on the live vLLM model. For anything else (quantized experts, a non-linear expert
placement, EP that does not cover the TP group, a sharded router, ...) the rollout falls back to vLLM's loaders,
after ``split_expert_stacks`` restores HF's per-expert tensors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, NamedTuple, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.direct_install import DirectInstallLayout
from verl_hardware_plugin.accelerators.tpu.rollout.moe_weights import (
    MOE_LAYOUTS,
    ParityCheckFn,
    expert_param,
    moe_parallel_mode,
    write_w2,
    write_w13,
)

# HF model types of the Qwen3 MoE text models this module installs.
QWEN3_MOE_MODEL_TYPES = ("qwen3_moe",)
# Projections of an expert stack, in the order of vLLM's w13 (gate, up) and w2 (down).
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
# The MoE router, which vLLM keeps whole (ReplicatedLinear) on every rank.
ROUTER_SUFFIX = ".mlp.gate.weight"

_ERROR_PREFIX = "tpu-sync direct install"
# An expert stack: ``<prefix>.mlp.experts.<proj>.weight`` (3-D).
_WIRE_RE = re.compile(r"^(.*\.mlp\.experts)\.(gate_proj|up_proj|down_proj)\.weight$")
# One HF expert tensor: ``<prefix>.mlp.experts.<e>.<proj>.weight``.
_SLOT_RE = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_PER_EXPERT_RE = re.compile(r"\.experts\.\d+\.")


class Qwen3MoeDims(NamedTuple):
    """MoE sizes of a Qwen3 MoE text config."""

    num_experts: int
    moe_intermediate_size: int
    hidden_size: int

    @classmethod
    def from_config(cls, hf_text_config: Any) -> Qwen3MoeDims:
        return cls(
            num_experts=int(hf_text_config.num_experts),
            moe_intermediate_size=int(hf_text_config.moe_intermediate_size),
            hidden_size=int(hf_text_config.hidden_size),
        )

    def wire_shapes(self) -> dict[str, list[int]]:
        """Full shape of each expert stack, by ``EXPERT_PROJS`` name."""
        e, i, h = self
        return {"gate_proj": [e, i, h], "up_proj": [e, i, h], "down_proj": [e, h, i]}


@dataclass
class DirectInstallPlan:
    """What ``install`` needs, checked once against the live vLLM model by ``plan_direct_install``."""

    dims: Qwen3MoeDims
    tp_rank: int
    tp_size: int
    moe_layout: str  # moe_weights.MOE_LAYOUT_TPU / MOE_LAYOUT_VLLM
    # vLLM's expert parallelism over the whole TP group (whole experts per rank), else every expert's intermediate
    # dimension sliced over TP. Either is the same at TP=1.
    expert_parallel: bool = True
    # Updated by ``install``: whether vllm-torchtpu's zero MoE padding was verified (first sync only), and vLLM's
    # resolved expert parameter keys (``(prefix, leaf) -> key``), which do not change between syncs.
    padding_checked: bool = False
    expert_keys: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def local_experts(self) -> int:
        """Experts on this rank."""
        return self.dims.num_experts // self.tp_size if self.expert_parallel else self.dims.num_experts

    @property
    def local_intermediate(self) -> int:
        """Intermediate size of each expert on this rank."""
        inter = self.dims.moe_intermediate_size
        return inter if self.expert_parallel else inter // self.tp_size

    def stack_tp_dim(self, proj: str) -> int:
        """The dimension of the ``proj`` stack that this rollout shards over TP."""
        if self.expert_parallel:
            return 0
        return 2 if proj == "down_proj" else 1


def trainer_expert_stack_wire(model_type: Optional[str], expert_parallel_size: int) -> Optional[Callable]:
    """The trainer's ``TrainerWeights.expert_stack_wire`` hook: ``expert_stack_wire_name`` for Qwen3 MoE, else None.

    Raises:
        ValueError: for Qwen3 MoE with trainer ``expert_parallel_size > 1``. Only FSDP-sharded expert stacks
            (trainer EP = 1) are supported for now.
    """
    if model_type not in QWEN3_MOE_MODEL_TYPES:
        return None
    if expert_parallel_size > 1:
        raise ValueError(
            f"tpu-sync: Qwen3 MoE with trainer expert_parallel_size={expert_parallel_size} is not supported yet; "
            "use expert_parallel_size=1 (experts sharded by FSDP)."
        )
    return expert_stack_wire_name


def expert_stack_wire_name(slots: list[tuple[str, tuple]], stack_shape: tuple[int, ...]) -> Optional[str]:
    """Trainer side: the tpu-sync name of an expert stack sent as one 3-D tensor, or ``None`` to send it per expert.

    ``slots`` are the stack's HF tensors (``TorchTitanTPUEngine._expert_stack_slots``), slot ``e`` being ``stack[e]``.
    The stack is sent whole only if it is 3-D and slot ``e`` is ``<prefix>.mlp.experts.<e>.<proj>.weight`` of shape
    ``stack_shape[1:]`` for one ``<prefix>`` and ``<proj>``; the name is then ``<prefix>.mlp.experts.<proj>.weight``.
    """
    if len(stack_shape) != 3 or len(slots) != stack_shape[0]:
        return None
    names = set()
    for e, (slot_name, slot_shape) in enumerate(slots):
        m = _SLOT_RE.match(slot_name)
        if m is None or int(m.group(2)) != e or tuple(int(d) for d in slot_shape) != tuple(stack_shape[1:]):
            return None
        names.add(f"{m.group(1)}.{m.group(3)}.weight")
    return names.pop() if len(names) == 1 else None


def split_expert_stacks(weights: Iterator[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
    """Undoes the trainer's stacking for vLLM's checkpoint loaders: yields one HF tensor per expert.

    Used when the rollout falls back to vLLM's loaders (``plan_direct_install`` returned no plan), which expect HF's
    ``<prefix>.mlp.experts.<e>.<proj>.weight``. Other tensors (including Qwen3.5's expert pieces, which carry no
    ``.weight`` suffix) pass through.
    """
    for name, tensor in weights:
        m = _WIRE_RE.match(name)
        if m is None or tensor.ndim != 3:
            yield name, tensor
            continue
        prefix, proj = m.groups()
        for e, block in enumerate(tensor.unbind(0)):
            yield f"{prefix}.{e}.{proj}.weight", block


def _moe_parallel_mode(module: Any, dims: Qwen3MoeDims, tp_size: int, tp_rank: int) -> tuple[Optional[bool], str]:
    return moe_parallel_mode(module, dims.num_experts, dims.moe_intermediate_size, tp_size, tp_rank)


def plan_direct_install(
    vllm_model: Any, global_shapes: dict[str, list[int]], hf_text_config: Any, tp_size: int, worker_rank: int
) -> tuple[Optional[DirectInstallPlan], str]:
    """``(plan, "")`` if this module can install the experts into ``vllm_model`` directly, else ``(None, reason)``.

    Checks, on the live vLLM model rather than by name: a Qwen3 MoE text config; complete 3-D expert stacks with the
    config's shapes (no per-expert tensors); one known unquantized vLLM MoE layer per received MoE block, all either
    with linear expert parallelism over the whole TP group or without expert parallelism (``_moe_parallel_mode``),
    whose EP / TP rank is this worker's rank (tpu-sync registers it at that position of the ``[1, tp]`` mesh); and a
    router that vLLM keeps whole.
    """
    model_type = getattr(hf_text_config, "model_type", None)
    if model_type not in QWEN3_MOE_MODEL_TYPES:
        return None, f"model_type {model_type!r} is not a Qwen3 MoE text model"
    try:
        dims = Qwen3MoeDims.from_config(hf_text_config)
    except (AttributeError, TypeError, ValueError) as e:
        return None, f"incomplete Qwen3 MoE config: {e}"
    per_expert = sorted(n for n in global_shapes if _PER_EXPERT_RE.search(n))
    if per_expert:
        return None, f"the trainer sent per-expert tensors (no expert stacks), e.g. {per_expert[0]}"
    blocks = sorted({m.group(1) for n in global_shapes if (m := _WIRE_RE.match(n)) is not None})
    if not blocks:
        return None, "the trainer sent no expert stacks"
    # Every block must be complete and match the config: a bad one would otherwise fail mid-install, after earlier
    # layers of the live vLLM model were already overwritten.
    for block in blocks:
        for proj, shape in dims.wire_shapes().items():
            name = f"{block}.{proj}.weight"
            if name not in global_shapes:
                return None, f"{block} lacks {proj}"
            if list(global_shapes[name]) != shape:
                return None, f"{name} is {list(global_shapes[name])}, the config gives {shape}"

    modules = list(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else []
    moe_modules = [(n, m) for n, m in modules if hasattr(m, "w13_weight") and hasattr(m, "quant_method")]
    if len(moe_modules) != len(blocks):
        return None, f"vLLM has {len(moe_modules)} fused-MoE layers, the trainer sent {len(blocks)} MoE blocks"
    layouts, modes = set(), set()
    for name, m in moe_modules:
        method = type(m.quant_method).__name__
        if method not in MOE_LAYOUTS:
            return None, f"unsupported vLLM MoE method {method} (quantized?)"
        layouts.add(MOE_LAYOUTS[method])
        expert_parallel, why = _moe_parallel_mode(m, dims, tp_size, worker_rank)
        if expert_parallel is None:
            return None, f"vLLM's {name}: {why}"
        modes.add(expert_parallel)
    if len(layouts) != 1:
        return None, f"vLLM MoE layouts {sorted(layouts)}"
    if len(modes) != 1:
        return None, "vLLM's MoE layers mix expert parallelism and TP-sliced experts"

    model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else {}
    for name in sorted(n for n in global_shapes if n.endswith(ROUTER_SUFFIX)):
        key = next((k for k in (name, name.removeprefix("model."), f"model.{name}") if k in model_sd), None)
        if key is None or model_sd[key].numel() != dims.num_experts * dims.hidden_size:
            return None, f"vLLM does not keep the router {name} whole on every rank"
    return DirectInstallPlan(dims, worker_rank, tp_size, layouts.pop(), expert_parallel=modes.pop()), ""


def sampler_sharding_rule(
    name: str, global_shape: list[int], tp_size: int, plan: DirectInstallPlan
) -> Optional[tuple[list[str], list[int]]]:
    """``(sharding_spec, local_shape)`` of a Qwen3-MoE-specific tensor, or ``None`` to use the dense path's rules."""
    if name.endswith(ROUTER_SUFFIX):
        return [""] * len(global_shape), list(global_shape)
    m = _WIRE_RE.match(name)
    if m is None:
        return None
    dim = plan.stack_tp_dim(m.group(2))
    if len(global_shape) != 3 or global_shape[dim] % tp_size:
        raise RuntimeError(f"{_ERROR_PREFIX}: {name} {global_shape} does not split over {tp_size} on dim {dim}")
    spec, local = [""] * 3, list(global_shape)
    spec[dim], local[dim] = "tp", global_shape[dim] // tp_size
    return spec, local


def install(
    staging: dict[str, torch.Tensor],
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    plan: DirectInstallPlan,
    check: Optional[ParityCheckFn] = None,
) -> set[str]:
    """Writes this rank's expert slices of ``staging`` into vLLM's ``w13_weight`` / ``w2_weight``.

    Returns the staging names used. ``check``, if given, is called on every expert weight
    region before it is overwritten (the exact parity check). Updates ``plan.padding_checked`` and
    ``plan.expert_keys``. The router and every other tensor go through the dense path.
    """
    consumed: set[str] = set()
    num_local, inter, hidden = plan.local_experts, plan.local_intermediate, plan.dims.hidden_size
    for name in list(staging):
        m = _WIRE_RE.match(name)
        if m is None or m.group(2) != "gate_proj":
            continue
        prefix = m.group(1)
        names = [f"{prefix}.{proj}.weight" for proj in EXPERT_PROJS]
        missing = [n for n in names if n not in staging]
        if missing:
            raise RuntimeError(f"{_ERROR_PREFIX}: {prefix} lacks {missing}")
        gate, up, down = (staging[n] for n in names)
        expected = ((num_local, inter, hidden),) * 2 + ((num_local, hidden, inter),)
        for n, t, shape in zip(names, (gate, up, down), expected, strict=True):
            if tuple(t.shape) != shape:
                raise RuntimeError(f"{_ERROR_PREFIX}: {n} is {tuple(t.shape)} on this rank, expected {shape}")
        w13_key, w2_key = (
            expert_param(model_sd, find_model_key, prefix, leaf, plan.expert_keys)
            for leaf in ("w13_weight", "w2_weight")
        )
        check_padding = not plan.padding_checked
        write_w13(model_sd[w13_key], gate, up, plan.moe_layout, check_padding, check=check, name=w13_key)
        write_w2(model_sd[w2_key], down, plan.moe_layout, check_padding, check=check, name=w2_key)
        consumed.update(names)
    plan.padding_checked = True
    return consumed


def _install(staging, model_sd, find_model_key, copy_into, plan, check) -> set[str]:
    return install(staging, model_sd, find_model_key, plan, check)  # writes the experts itself, without copy_into


def _loader_weights(weights: Iterator[tuple[str, torch.Tensor]], hf_text_config: Any) -> Iterator:
    return split_expert_stacks(weights)


LAYOUT = DirectInstallLayout(
    name="qwen3_moe",
    plan=plan_direct_install,
    sharding_rule=sampler_sharding_rule,
    install=_install,
    loader_weights=_loader_weights,
)
