# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Trainer-side export in vLLM's parameter layout, so tpu-sync can receive straight into vLLM's weights.

vLLM is not transparent to the checkpoint it loads: it fuses ``q_proj`` / ``k_proj`` / ``v_proj`` into ``qkv_proj``,
``gate_proj`` / ``up_proj`` into ``gate_up_proj``, and vllm-torchtpu stores dense linear weights transposed,
``(in, out)``. tpu-sync moves whole buffers: a variable is one tensor on both ends, sharded differently. So for the
rollout to bind its vLLM parameters as the receive buffers, the trainer must send each parameter as the rollout holds
it: one virtual global tensor per vLLM parameter whose TP slices are the rollout ranks' local weights.

``build_vllm_layout_export`` turns the trainer's HF-named tensors (full, or this rank's FSDP shards) into that form,
given the rollout's parameter description (``describe_vllm_params`` on the rollout worker). The description carries
vLLM's own fusion table: each fused parameter lists its HF sources (from the model class's ``packed_modules_mapping``,
which every vLLM model declares and its weight loader follows), the per-source output sizes and, for ``qkv_proj``,
how many ranks replicate each KV head. Nothing here is specific to a model family. Unfused tensors keep their FSDP
blocks (transposed when vLLM flipped them) and tpu-sync reshards them for the rollout's TP split as usual. A fused
parameter's rollout-rank slice is the concatenation of the sources' slices, so with FSDP degree F equal to the
rollout TP degree, trainer rank t's shards of q/k/v are exactly rollout rank t's slices and the fused block is a local
concatenation; no communication is needed.

TODO(tpu): F != TP, interleaved placements (MoE round-robin experts, Qwen3.5 head interleaving) and replicated GQA KV
heads need an exchange among trainer ranks before the export; they raise ``VllmLayoutError`` for now.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VllmParam:
    """One vLLM parameter as one rollout rank holds it (the same shape on every rank).

    ``sources`` names the HF tensors a fused parameter concatenates along its output dim, in order (from the vLLM
    model's ``packed_modules_mapping``); empty for an unfused parameter. ``output_sizes`` are the sources' global
    output sizes (the fused module's ``output_sizes``) and ``kv_replicas`` how many TP ranks share each KV head
    (``QKVParallelLinear.num_kv_head_replicas``; 1 when every rank has its own KV rows).
    """

    local_shape: tuple[int, ...]
    dtype: str  # e.g. "torch.bfloat16"
    flipped: bool  # vllm-torchtpu transposed it to (in, out)
    sources: tuple[str, ...] = ()
    output_sizes: tuple[int, ...] = ()
    kv_replicas: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VllmParam:
        return cls(
            tuple(d["local_shape"]),
            str(d["dtype"]),
            bool(d["flipped"]),
            tuple(d.get("sources") or ()),
            tuple(int(s) for s in d.get("output_sizes") or ()),
            int(d.get("kv_replicas") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_shape": list(self.local_shape),
            "dtype": self.dtype,
            "flipped": self.flipped,
            "sources": list(self.sources),
            "output_sizes": list(self.output_sizes),
            "kv_replicas": self.kv_replicas,
        }


@dataclass
class VllmLayoutExport:
    """The trainer's tensors in vLLM's layout: ``(vllm name, tensor)`` plus how each is sharded across trainer ranks."""

    weights: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    # vllm name -> (num_shards, shard_index, shard_dim): this rank's FSDP block of the virtual global tensor.
    shards: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    # vllm name -> the HF names it was built from.
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)


class VllmLayoutError(RuntimeError):
    """The trainer's tensors cannot be exported in vLLM's layout without communication (see the module TODOs)."""


def _candidates(name: str) -> tuple[str, str]:
    """``name`` and its twin with the ``model.`` prefix toggled (vLLM and HF names differ only there)."""
    return name, (name[6:] if name.startswith("model.") else f"model.{name}")


def _lookup(params: dict[str, VllmParam], name: str) -> Optional[tuple[str, VllmParam]]:
    """Finds ``name`` in vLLM's parameters, tolerating a ``model.`` prefix difference."""
    for candidate in _candidates(name):
        if candidate in params:
            return candidate, params[candidate]
    return None


def _find_weight(weights: dict[str, torch.Tensor], name: str) -> Optional[str]:
    """Finds the trainer tensor for a vLLM-side source name, tolerating a ``model.`` prefix difference."""
    for candidate in _candidates(name):
        if candidate in weights:
            return candidate
    return None


def _rank_slices(
    sources: list[torch.Tensor], tp_size: int, local_out: int, kv_replicas: int
) -> list[list[torch.Tensor]]:
    """Per rollout rank, each full source's slice (HF dim 0), for a fused parameter with ``local_out`` rows per rank.

    The first source (q) is split ``tp_size`` ways. With GQA (``tp_size`` > KV heads) vLLM replicates each KV head on
    ``kv_replicas`` ranks, so the other sources (k, v) are split ``tp_size // kv_replicas`` ways and rank r takes
    block ``r // kv_replicas``. The per-rank row count is checked against ``local_out``.
    """
    q_rows = sources[0].shape[0]
    m = kv_replicas
    if q_rows % tp_size:
        raise VllmLayoutError(f"first source has {q_rows} rows, not a multiple of TP={tp_size}")
    if m < 1 or tp_size % m or any(s.shape[0] % (tp_size // m) for s in sources[1:]):
        raise VllmLayoutError(f"KV replication factor {m} does not divide TP={tp_size} or the KV rows")
    rows = q_rows // tp_size + sum(s.shape[0] // (tp_size // m) for s in sources[1:])
    if rows != local_out:
        raise VllmLayoutError(
            f"sources {[s.shape[0] for s in sources]} give {rows} rows per rank at TP={tp_size}, "
            f"kv_replicas={m}; vLLM holds {local_out}"
        )
    out = []
    for r in range(tp_size):
        slices = [sources[0].chunk(tp_size, dim=0)[r]]
        slices += [s.chunk(tp_size // m, dim=0)[r // m] for s in sources[1:]]
        out.append(slices)
    return out


def _is_tp_extension(global_shape: list[int], local_shape: tuple[int, ...], tp_size: int) -> bool:
    """Whether ``global_shape`` is ``local_shape`` with exactly one dim multiplied by ``tp_size``, or equal."""
    if len(global_shape) != len(local_shape):
        return False
    scaled = [g != l_ for g, l_ in zip(global_shape, local_shape, strict=True)]
    if not any(scaled):
        return True
    if sum(scaled) != 1:
        return False
    d = scaled.index(True)
    return global_shape[d] == local_shape[d] * tp_size


def build_vllm_layout_export(
    named_weights: list[tuple[str, torch.Tensor]],
    shard_info: dict[str, tuple[int, int]],
    vllm_params: dict[str, VllmParam],
    tp_size: int,
) -> VllmLayoutExport:
    """Re-packs HF-named trainer tensors into vLLM's per-parameter layout.

    Args:
        named_weights: ``(hf_name, tensor)``: this rank's dim-0 FSDP block when ``hf_name`` is in ``shard_info``,
            else the full tensor (the same on every rank).
        shard_info: ``hf_name -> (num_shards, shard_index)`` of the dim-0 FSDP blocks.
        vllm_params: ``vllm name -> VllmParam`` from the rollout (``describe_vllm_params``).
        tp_size: the rollout's TP degree.

    Returns:
        The export. For every vLLM parameter, the virtual global tensor is the concatenation over rollout ranks r of
        rank r's local weight along its TP dim; a fused parameter's rank-r slice is ``[src0_r; src1_r; ...]`` in HF
        layout, transposed when vLLM flipped it. A sharded source needs F == TP, so this rank's block is rollout rank
        ``shard_index``'s slice and no exchange is needed. HF tensors vLLM has no parameter for (e.g. ``lm_head``
        tied to the embedding) are dropped.

    Raises:
        VllmLayoutError: a tensor cannot be exported without communication, or does not match vLLM's shape or dtype.
    """
    weights = dict(named_weights)
    export = VllmLayoutExport()
    consumed: set[str] = set()

    def emit(vllm_name: str, param: VllmParam, hf_names: tuple[str, ...]) -> None:
        shards = {shard_info.get(n) for n in hf_names}
        if len(shards) != 1:
            raise VllmLayoutError(f"{vllm_name}: sources {hf_names} must be sharded alike, got {shards}")
        shard = shards.pop()
        fused = len(hf_names) > 1
        # vLLM's rows per rank of a fused parameter, in HF layout (its output dim; dim 1 of a flipped weight).
        local_out = param.local_shape[1] if param.flipped else param.local_shape[0]
        if fused and shard is not None:
            # vLLM shards a fused parameter on its output dim, the dim the sources are concatenated on. With
            # F == TP, this rank's dim-0 block of every source is rollout rank `index`'s slice, so the fused block
            # is a local concatenation. Otherwise, or when vLLM replicates KV heads (TP > KV heads: rank r needs a
            # whole head, more than its FSDP block), the slices live on other ranks (exchange not implemented).
            if shard[0] != tp_size:
                raise VllmLayoutError(
                    f"{vllm_name}: FSDP degree {shard[0]} != rollout TP {tp_size}; exporting a fused parameter in "
                    "vLLM's layout needs an exchange among trainer ranks (not implemented)"
                )
            tensor = torch.cat([weights[n] for n in hf_names], dim=0)
            if tensor.shape[0] != local_out:
                raise VllmLayoutError(
                    f"{vllm_name}: this rank's fused block has {tensor.shape[0]} rows, vLLM's rank slice {local_out} "
                    "(replicated KV heads need an exchange among trainer ranks, not implemented)"
                )
        elif fused:
            # Full sources: rank r's slice is the r-th slice of each source, in source order.
            per_rank = _rank_slices([weights[n] for n in hf_names], tp_size, local_out, param.kv_replicas)
            tensor = torch.cat([s for slices in per_rank for s in slices], dim=0)
        else:
            # A single source: the same tensor (tpu-sync reshards the FSDP blocks for the rollout's TP split).
            tensor = weights[hf_names[0]]

        shard_dim = 0  # the FSDP blocks are dim-0 blocks of the HF tensor
        if param.flipped:
            if tensor.dim() != 2:
                raise VllmLayoutError(f"{vllm_name}: vLLM flipped it but the trainer tensor is {tensor.dim()}-D")
            tensor = tensor.t()
            shard_dim = 1
        tensor = tensor.contiguous()

        # The virtual global tensor must be vLLM's local shape with one dim (its TP dim) multiplied by TP, or equal.
        global_shape = list(tensor.shape)
        if shard is not None:
            global_shape[shard_dim] *= shard[0]
            export.shards[vllm_name] = (shard[0], shard[1], shard_dim)
        if not _is_tp_extension(global_shape, param.local_shape, tp_size):
            raise VllmLayoutError(
                f"{vllm_name}: virtual global shape {global_shape} (this rank's tensor {tuple(tensor.shape)}) is not "
                f"vLLM's local shape {param.local_shape} extended {tp_size}x on one dim (flipped={param.flipped})"
            )
        if str(tensor.dtype) != param.dtype:
            raise VllmLayoutError(f"{vllm_name}: trainer dtype {tensor.dtype} != vLLM {param.dtype}")
        export.weights.append((vllm_name, tensor))
        export.sources[vllm_name] = hf_names
        consumed.update(hf_names)

    # 1. Fused parameters: vLLM told us their sources (its packed_modules_mapping).
    for vllm_name in sorted(vllm_params):
        param = vllm_params[vllm_name]
        if not param.sources:
            continue
        resolved = [_find_weight(weights, s) for s in param.sources]
        if any(r is None for r in resolved):
            missing = [s for s, r in zip(param.sources, resolved, strict=True) if r is None]
            raise VllmLayoutError(f"{vllm_name}: the trainer has no tensor for its vLLM sources {missing}")
        hf_names = tuple(r for r in resolved if r is not None)
        if param.output_sizes and len(param.output_sizes) != len(hf_names):
            raise VllmLayoutError(
                f"{vllm_name}: vLLM lists {len(param.output_sizes)} output sizes for {len(hf_names)} sources"
            )
        if param.output_sizes:
            # Global rows per source: a dim-0 FSDP block holds 1/F of them.
            rows = [weights[n].shape[0] * (shard_info[n][0] if n in shard_info else 1) for n in hf_names]
            if rows != list(param.output_sizes):
                raise VllmLayoutError(
                    f"{vllm_name}: trainer source rows {rows} != vLLM's output sizes {list(param.output_sizes)}"
                )
        emit(vllm_name, param, hf_names)

    # 2. Everything else: the same tensor, transposed when vLLM flipped it.
    for hf_name in weights:
        if hf_name in consumed:
            continue
        found = _lookup(vllm_params, hf_name)
        if found is None:
            logger.debug(f"vllm layout export: vLLM has no parameter for {hf_name}; dropped")
            continue
        emit(found[0], found[1], (hf_name,))

    export.weights.sort(key=lambda x: x[0])
    return export
