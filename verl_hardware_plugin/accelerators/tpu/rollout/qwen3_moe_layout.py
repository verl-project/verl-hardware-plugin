# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct ``tpu-sync`` install of Qwen3 MoE (e.g. Qwen3-30B-A3B) experts into vLLM's expert-parallel MoE weights.

Without this module, Qwen3 MoE models install the received weights through vLLM's checkpoint loaders: the trainer
all-gathers every expert stack and sends one HF tensor per expert, every rollout rank all-gathers every tensor over
its TP group, and vLLM's MoE loaders loop over the experts (``_install_with_vllm_loaders``).

Trainer side (``expert_stack_wire_name``, used by ``TrainerWeights.local_shards``): torchtitan keeps each MoE layer's
routed experts as one stack per projection, ``moe.experts.w1`` / ``w3`` [E, I, H] and ``w2`` [E, H, I], whose expert
``e`` is HF's ``model.layers.N.mlp.experts.<e>.{gate,up,down}_proj.weight``. Each stack is sent as ONE 2-D tensor
with the experts folded into rows, named after those HF tensors without the expert index:

==============================================  ============  ================================================
Trainer tensor (tpu-sync name)                  Shape         Written into (vLLM, TP = EP rank ``r``)
==============================================  ============  ================================================
``model.layers.N.mlp.experts.gate_proj.weight``  [E * I, H]    ``experts.w13_weight`` [E/tp, H, 2A]: gate^T in
``model.layers.N.mlp.experts.up_proj.weight``    [E * I, H]    [0, I), up^T in [A, A + I) (A: I padded to 128)
``model.layers.N.mlp.experts.down_proj.weight``  [E * H, I]    ``experts.w2_weight`` [E/tp, A, H]: down^T
``model.layers.N.mlp.gate.weight`` (router)      [E, H]        received whole (vLLM keeps it replicated)
==============================================  ============  ================================================

FSDP's ``Shard(0)`` of a stack holds whole experts, so a trainer rank's block is a contiguous dim-0 block of the wire
tensor and tpu-sync reshards it like any dense weight: the trainer never gathers the experts.

Rollout side: with expert parallelism over the whole TP group (vLLM ``enable_expert_parallel``, EP size = TP size,
linear placement), rank ``r`` holds experts ``[r * E / tp, (r + 1) * E / tp)``, again a contiguous dim-0 block of
each wire tensor, which it receives and writes into its ``w13_weight`` / ``w2_weight`` with ``moe_weights``'
writers (vllm-torchtpu's transposed, 128-padded layout or vLLM's own). Everything else (attention, norms,
embeddings) goes through the dense path's rules in ``tpu_sync_worker_extension``.

``plan_direct_install`` checks this on the live vLLM model. For anything else (no expert parallelism, which slices
every expert's intermediate dimension over TP instead; quantized experts; a replicated router missing; ...) the
rollout falls back to vLLM's loaders, after ``unfold_expert_stacks`` restores HF's per-expert tensors.
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
    write_w2,
    write_w13,
)

# HF model types of the Qwen3 MoE text models this module installs.
QWEN3_MOE_MODEL_TYPES = ("qwen3_moe",)
# Projections of a folded expert stack, in the order of vLLM's w13 (gate, up) and w2 (down).
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
# The MoE router, which vLLM keeps whole (ReplicatedLinear) on every rank.
ROUTER_SUFFIX = ".mlp.gate.weight"

_ERROR_PREFIX = "tpu-sync direct install"
# A folded expert stack: ``<prefix>.mlp.experts.<proj>.weight`` (2-D).
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
        """Full shape of each folded projection, by ``EXPERT_PROJS`` name."""
        e, i, h = self
        return {"gate_proj": [e * i, h], "up_proj": [e * i, h], "down_proj": [e * h, i]}


@dataclass
class DirectInstallPlan:
    """What ``install`` needs, checked once against the live vLLM model by ``plan_direct_install``."""

    dims: Qwen3MoeDims
    tp_rank: int
    tp_size: int
    moe_layout: str  # moe_weights.MOE_LAYOUT_TPU / MOE_LAYOUT_VLLM
    # Updated by ``install``: whether vllm-torchtpu's zero MoE padding was verified (first sync only), and vLLM's
    # resolved expert parameter keys (``(prefix, leaf) -> key``), which do not change between syncs.
    padding_checked: bool = False
    expert_keys: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def local_experts(self) -> int:
        return self.dims.num_experts // self.tp_size


def is_expert_stack(name: str) -> bool:
    """Whether ``name`` is a folded expert stack (see the module docstring)."""
    return _WIRE_RE.match(name) is not None


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
    """Trainer side: the tpu-sync name of an expert stack sent folded to 2-D, or ``None`` to send it per expert.

    ``slots`` are the stack's HF tensors (``TorchTitanTPUEngine._expert_stack_slots``), slot ``e`` being ``stack[e]``.
    The stack is folded only if it is 3-D and slot ``e`` is ``<prefix>.mlp.experts.<e>.<proj>.weight`` of shape
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


def num_experts_of(hf_text_config: Any) -> Optional[int]:
    """``num_experts`` of an HF text config, or ``None``."""
    num = getattr(hf_text_config, "num_experts", None)
    return int(num) if num else None


def unfold_expert_stacks(
    weights: Iterator[tuple[str, torch.Tensor]], num_experts: Optional[int]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Undoes the trainer's folding for vLLM's checkpoint loaders: yields one HF tensor per expert.

    Used when the rollout falls back to vLLM's loaders (``plan_direct_install`` returned no plan), which expect HF's
    ``<prefix>.mlp.experts.<e>.<proj>.weight``. Other tensors pass through.
    """
    for name, tensor in weights:
        m = _WIRE_RE.match(name)
        if m is None or tensor.ndim != 2:
            yield name, tensor
            continue
        if not num_experts or tensor.shape[0] % num_experts:
            raise RuntimeError(f"tpu-sync: cannot unfold {name} {tuple(tensor.shape)} into {num_experts} experts")
        prefix, proj = m.groups()
        for e, block in enumerate(tensor.reshape(num_experts, -1, tensor.shape[1]).unbind(0)):
            yield f"{prefix}.{e}.{proj}.weight", block


def _moe_attr(module: Any, name: str, default: Any) -> Any:
    """``module.<name>``, else ``module.moe_config.<name>`` (vLLM keeps the EP geometry on both, version-dependent)."""
    value = getattr(module, name, None)
    if value is None:
        value = getattr(getattr(module, "moe_config", None), name, None)
    return default if value is None else value


def _expert_parallel_mismatch(module: Any, num_experts: int, tp_size: int, tp_rank: int) -> str:
    """Why this vLLM MoE layer does not hold experts ``[r * E / tp, (r + 1) * E / tp)`` on TP rank ``r``, or ``""``."""
    global_experts = int(_moe_attr(module, "global_num_experts", 0) or 0)
    local_experts = int(_moe_attr(module, "local_num_experts", 0) or 0)
    if global_experts and global_experts != num_experts:
        return f"{global_experts} experts, the config gives {num_experts}"
    activation = str(_moe_attr(module, "activation", "silu")).lower()
    if "swigluoai" in activation or getattr(module, "w13_bias", None) is not None:
        return f"activation {activation} / expert bias not supported"
    if tp_size > 1:
        if not bool(_moe_attr(module, "use_ep", False)):
            # vLLM then slices every expert's intermediate dimension over TP, which a folded stack cannot deliver as
            # one contiguous block per rank.
            return "no expert parallelism (set enable_expert_parallel so that EP covers the TP group)"
        ep_size, ep_rank = int(_moe_attr(module, "ep_size", 0) or 0), int(_moe_attr(module, "ep_rank", -1))
        if ep_size != tp_size or ep_rank != tp_rank:
            return f"EP size {ep_size} / rank {ep_rank}, needs EP over the whole TP group ({tp_size} / {tp_rank})"
        placement = str(_moe_attr(module, "expert_placement_strategy", "linear"))
        if placement != "linear":
            return f"expert placement {placement!r}, needs 'linear'"
    if local_experts and local_experts * tp_size != num_experts:
        return f"{local_experts} local experts on each of {tp_size} ranks, the config gives {num_experts} in total"
    return ""


def plan_direct_install(
    vllm_model: Any, global_shapes: dict[str, list[int]], hf_text_config: Any, tp_size: int, worker_rank: int
) -> tuple[Optional[DirectInstallPlan], str]:
    """``(plan, "")`` if this module can install the experts into ``vllm_model`` directly, else ``(None, reason)``.

    Checks, on the live vLLM model rather than by name: a Qwen3 MoE text config; complete folded expert stacks with
    the config's shapes (no per-expert tensors); experts that split evenly over the TP ranks; one known unquantized
    vLLM MoE layer per received MoE block, with linear expert parallelism over the whole TP group whose EP rank is
    this worker's rank (tpu-sync registers it at that position of the ``[1, tp]`` mesh); and a router that vLLM
    keeps whole.
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
        return None, f"the trainer sent per-expert tensors (expert stacks not folded), e.g. {per_expert[0]}"
    blocks = sorted({m.group(1) for n in global_shapes if (m := _WIRE_RE.match(n)) is not None})
    if not blocks:
        return None, "the trainer sent no folded expert stacks"
    # Every block must be complete and match the config: a bad one would otherwise fail mid-install, after earlier
    # layers of the live vLLM model were already overwritten.
    for block in blocks:
        for proj, shape in dims.wire_shapes().items():
            name = f"{block}.{proj}.weight"
            if name not in global_shapes:
                return None, f"{block} lacks {proj}"
            if list(global_shapes[name]) != shape:
                return None, f"{name} is {list(global_shapes[name])}, the config gives {shape}"
    if dims.num_experts % tp_size:
        return None, f"{dims.num_experts} experts do not split over TP={tp_size}"

    modules = list(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else []
    moe_modules = [(n, m) for n, m in modules if hasattr(m, "w13_weight") and hasattr(m, "quant_method")]
    if len(moe_modules) != len(blocks):
        return None, f"vLLM has {len(moe_modules)} fused-MoE layers, the trainer sent {len(blocks)} MoE blocks"
    layouts = set()
    for name, m in moe_modules:
        method = type(m.quant_method).__name__
        if method not in MOE_LAYOUTS:
            return None, f"unsupported vLLM MoE method {method} (quantized?)"
        layouts.add(MOE_LAYOUTS[method])
        mismatch = _expert_parallel_mismatch(m, dims.num_experts, tp_size, worker_rank)
        if mismatch:
            return None, f"vLLM's {name}: {mismatch}"
    if len(layouts) != 1:
        return None, f"vLLM MoE layouts {sorted(layouts)}"

    model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else {}
    for name in sorted(n for n in global_shapes if n.endswith(ROUTER_SUFFIX)):
        key = next((k for k in (name, name.removeprefix("model."), f"model.{name}") if k in model_sd), None)
        if key is None or model_sd[key].numel() != dims.num_experts * dims.hidden_size:
            return None, f"vLLM does not keep the router {name} whole on every rank"
    return DirectInstallPlan(dims, worker_rank, tp_size, layouts.pop()), ""


def sampler_sharding_rule(name: str, global_shape: list[int], tp_size: int) -> Optional[tuple[list[str], list[int]]]:
    """``(sharding_spec, local_shape)`` of a Qwen3-MoE-specific tensor, or ``None`` to use the dense path's rules."""
    if name.endswith(ROUTER_SUFFIX):
        return [""] * len(global_shape), list(global_shape)
    if is_expert_stack(name):
        if len(global_shape) != 2 or global_shape[0] % tp_size:
            raise RuntimeError(f"{_ERROR_PREFIX}: {name} {global_shape} does not split over {tp_size} on dim 0")
        return ["tp", ""], [global_shape[0] // tp_size, global_shape[1]]
    return None


def install(
    staging: dict[str, torch.Tensor],
    model_sd: dict[str, torch.Tensor],
    find_model_key: Callable[[str], Optional[str]],
    plan: DirectInstallPlan,
    check: Optional[ParityCheckFn] = None,
) -> set[str]:
    """Writes this rank's experts of ``staging`` into vLLM's ``w13_weight`` / ``w2_weight``.

    Returns the staging names used. ``check``, if given, is called on every expert weight
    region before it is overwritten (the exact parity check). Updates ``plan.padding_checked`` and
    ``plan.expert_keys``. The router and every other tensor go through the dense path.
    """
    consumed: set[str] = set()
    num_local, inter, hidden = plan.local_experts, plan.dims.moe_intermediate_size, plan.dims.hidden_size
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
        expected = ((num_local * inter, hidden),) * 2 + ((num_local * hidden, inter),)
        for n, t, shape in zip(names, (gate, up, down), expected, strict=True):
            if tuple(t.shape) != shape:
                raise RuntimeError(f"{_ERROR_PREFIX}: {n} is {tuple(t.shape)} on this rank, expected {shape}")
        w13_key, w2_key = (
            expert_param(model_sd, find_model_key, prefix, leaf, plan.expert_keys)
            for leaf in ("w13_weight", "w2_weight")
        )
        check_padding = not plan.padding_checked
        write_w13(
            model_sd[w13_key],
            gate.view(num_local, inter, hidden),
            up.view(num_local, inter, hidden),
            plan.moe_layout,
            check_padding,
            check=check,
            name=w13_key,
        )
        write_w2(
            model_sd[w2_key],
            down.view(num_local, hidden, inter),
            plan.moe_layout,
            check_padding,
            check=check,
            name=w2_key,
        )
        consumed.update(names)
    plan.padding_checked = True
    return consumed


def _install(staging, model_sd, find_model_key, copy_into, plan, check) -> set[str]:
    return install(staging, model_sd, find_model_key, plan, check)  # writes the experts itself, without copy_into


def _loader_weights(weights: Iterator[tuple[str, torch.Tensor]], hf_text_config: Any) -> Iterator:
    return unfold_expert_stacks(weights, num_experts_of(hf_text_config))


def _loader_error(global_shapes: dict[str, list[int]], hf_text_config: Any) -> str:
    folded = [n for n in global_shapes if is_expert_stack(n)]
    if folded and not num_experts_of(hf_text_config):
        return (
            f"received folded expert stacks (e.g. {folded[0]}) but the model config has no num_experts to unfold "
            "them for vLLM's loaders"
        )
    return ""


LAYOUT = DirectInstallLayout(
    name="qwen3_moe",
    plan=plan_direct_install,
    sharding_rule=sampler_sharding_rule,
    install=_install,
    loader_weights=_loader_weights,
    loader_error=_loader_error,
)
