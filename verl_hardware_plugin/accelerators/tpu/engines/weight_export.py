# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Exporting the trainer's weights for the weight sync, with or without gathering the FSDP shards.

``TorchTitanTPUEngine.get_per_tensor_param`` returns a ``TrainerWeights``. Iterating it yields full tensors (every
sharded DTensor all-gathered), which is what verl and the ``ray-plasma-store`` engine expect. The ``tpu-sync`` engine
calls ``TrainerWeights.local_shards()`` instead, which keeps each parameter that tpu-sync can place from its local
block (``fsdp_dim0_shard``) on this rank and returns every other parameter full. See ``fsdp_dim0_shard`` for which
layouts are exported locally and which are gathered.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Callable, Iterable, Iterator, Optional

import torch
from torch.distributed.tensor import DTensor, Shard

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def fsdp_dim0_shard(param: Any) -> Optional[tuple[int, int]]:
    """``(num_shards, shard_index)`` if this rank holds one of ``num_shards`` equal-size dim-0 blocks of ``param``.

    A receiver that knows the full shape places the local block from these two numbers alone (block
    ``shard_index`` of ``num_shards``), which is all tpu-sync needs to reshard it for the rollout. Layouts:

    * **FSDP** (``Shard(0)`` on a 1-D mesh) with dim 0 divisible by the mesh size: a block, exported locally.
    * **FSDP + CP**: torchtitan shards parameters over the flattened ``dp_shard`` x ``cp`` mesh dimension (CP splits
      activations, not parameters), so the parameters are again ``Shard(0)`` on a 1-D mesh: exported locally.
    * **Replicated** (``Replicate()`` placements, or a plain tensor): ``None``. Every rank already holds the full
      tensor, so it is exported full without further communication here. It may have been gathered upstream,
      though: torchtitan's ``FusedQKVLinear`` (``fuse_qkv=True``) keeps ``wqkv`` FSDP sharded (``Shard(0)``) in
      training, but its ``state_dict()`` hook (``_split_qkv_on_save``) redistributes it to ``Replicate()`` (an FSDP
      all-gather) before splitting it into q/k/v_proj. TODO: split ``wqkv`` locally so q/k/v export as local shards.
    * **Uneven dim-0 cuts**: ``None``, gathered. torch gives the remainder rows to the leading ranks, which is not
      the equal-block layout tpu-sync assumes.
    * **FSDP x TP, HSDP** (2-D meshes) and ``_StridedShard``: ``None``, gathered. TODO: register these as n-D
      blocks (``mesh_shape`` / ``global_shard_indices`` per dimension) once tpu-sync is confirmed to describe them.
    * **Expert stacks** (torchtitan's fused ``[E, ...]`` MoE weights): with an ``expert_stack_wire`` hook (Qwen3 MoE,
      ``rollout.qwen3_moe_layout``), ``local_shards()`` sends each stack folded to 2-D (``[E * rows, cols]``). An
      FSDP ``Shard(0)`` of whole experts, as this function recognizes it, is then a dim-0 block of the folded
      tensor and is exported locally. Any other layout of the stack (e.g. torchtitan's dim-1 expert sharding when
      the FSDP degree exceeds the expert count, or an uneven expert split) is gathered first. Without the hook,
      stacks are gathered and split into one HF tensor per expert.
    * **1-D tensors** (norm weights, biases, Qwen3.5's ``A_log`` / ``dt_bias``): ``None``, gathered. They are tiny,
      and tpu-sync mis-delivers some of their small blocks (e.g. a 128-element norm weight split 8 ways arrived with
      only its first block), so they are always sent whole.

    Every tensor that ends up gathered is logged by ``TrainerWeights.local_shards()``.
    """
    if not isinstance(param, DTensor) or param.ndim < 2:
        return None
    mesh = param.device_mesh
    if mesh.ndim != 1 or len(param.placements) != 1:
        return None
    placement = param.placements[0]
    # type() rather than isinstance(): _StridedShard subclasses Shard but permutes which block a rank holds.
    if type(placement) is not Shard or placement.dim != 0:
        return None
    num_shards = mesh.size(0)
    full_rows = int(param.shape[0])
    if num_shards <= 1 or full_rows % num_shards:
        return None
    if int(param.to_local().shape[0]) != full_rows // num_shards:
        return None
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        return None
    return num_shards, int(coordinate[0])


def _is_replicated(param: Any) -> bool:
    """True if every rank already holds the whole of ``param``, so exporting it full needs no further communication."""
    return not isinstance(param, DTensor) or all(p.is_replicate() for p in param.placements)


def _bf16_on(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    """A bf16 copy of ``tensor`` on ``device`` that never aliases the trainer's parameter storage."""
    out = tensor.to(device, dtype=torch.bfloat16, non_blocking=True)
    if out is tensor:
        # Already bf16 on the device: .to() returned the parameter itself. The receiver reads the exported tensor
        # after the export returns, so it must not change under a later optimizer step.
        out = out.clone()
    return out


class TrainerWeights:
    """HF-named weights returned by ``TorchTitanTPUEngine.get_per_tensor_param``, exported lazily and only once.

    Iterating yields ``(hf_name, full_tensor)`` pairs: every sharded DTensor is all-gathered, as verl expects from
    ``get_per_tensor_param``. ``local_shards()`` yields ``(hf_name, tensor, shard)`` instead, where ``tensor`` is this
    rank's local block and ``shard`` its ``(num_shards, shard_index)`` for parameters that ``fsdp_dim0_shard``
    recognizes, and the full tensor with ``shard=None`` for every other parameter. Both paths run the same
    collectives in the same order on every rank. A second export raises: the parameters are released as they go.

    ``dense`` is a dict of HF-named parameters, or an iterable of such dicts that is consumed lazily during the export
    (``TorchTitanTPUEngine`` converts one transformer layer at a time, so that the tensors its state-dict adapter
    fuses, e.g. Qwen3.5's MoE ``gate_up_proj``, exist for one layer at a time). A parameter reachable under several
    names (tied embeddings) must appear in one dict.

    ``local_transform``, if given, rewrites each dense dict before ``local_shards()`` exports it: the layout that the
    ``tpu-sync`` receiver installs from (e.g. Qwen3.5's fused ``in_proj_qkv`` / ``conv1d`` / expert ``gate_up_proj``
    split into the pieces that vLLM TP-shards separately, see ``rollout.qwen3_5_layout``). Plain iteration keeps
    verl's HF layout.

    ``expert_stacks`` holds ``(stack, slots)`` pairs: a fused ``[E, ...]`` expert weight and the HF name and shape of
    each expert's slice. ``expert_stack_wire(slots, stack_shape)``, if given, names the 2-D tensor that
    ``local_shards()`` sends a stack as, its experts folded into rows (``None`` keeps the per-expert tensors), e.g.
    ``rollout.qwen3_moe_layout.expert_stack_wire_name``. Plain iteration always yields one tensor per expert.
    """

    def __init__(
        self,
        dense: dict[str, Any] | Iterable[dict[str, Any]],
        expert_stacks: list[tuple[Any, list[tuple[str, tuple]]]],
        device: Any,
        local_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        expert_stack_wire: Optional[Callable[[list[tuple[str, tuple]], tuple[int, ...]], Optional[str]]] = None,
    ) -> None:
        self._dense_chunks = iter([dense]) if isinstance(dense, dict) else iter(dense)
        self._local_transform = local_transform
        self._expert_stacks = expert_stacks
        self._expert_stack_wire = expert_stack_wire
        self._device = device
        self._iter: Optional[Iterator[tuple[str, torch.Tensor]]] = None
        self._exported = False

    def __iter__(self) -> TrainerWeights:
        return self

    def __next__(self) -> tuple[str, torch.Tensor]:
        if self._iter is None:
            self._iter = ((name, tensor) for name, tensor, _ in self._export(local=False))
        return next(self._iter)

    def local_shards(self) -> Iterator[tuple[str, torch.Tensor, Optional[tuple[int, int]]]]:
        """Yields ``(hf_name, tensor, shard)``; see the class docstring."""
        return self._export(local=True)

    def _dense_items(self, local: bool = False) -> Iterator[tuple[str, Any, int]]:
        """Pops ``(hf_name, param, number of names of param in its dict)`` from the dense dicts, one dict at a time."""
        for chunk in self._dense_chunks:
            if local and self._local_transform is not None:
                chunk = self._local_transform(chunk)
            names_per_param = Counter(id(p) for p in chunk.values())
            for name in list(chunk.keys()):
                param = chunk.pop(name)
                yield name, param, names_per_param[id(param)]

    def _export(self, local: bool) -> Iterator[tuple[str, torch.Tensor, Optional[tuple[int, int]]]]:
        if self._exported:
            raise RuntimeError("TrainerWeights can only be exported once; call get_per_tensor_param() again")
        self._exported = True
        device = self._device
        # A parameter reachable under several names (tied input embedding and LM head) is exported once.
        # The entries keep their parameter referenced, so its id is not reused while it is a key.
        exported: dict[int, tuple[torch.Tensor, Optional[tuple[int, int]], Any]] = {}
        counts = {"local": 0, "replicated": 0, "gathered": 0}
        for name, param, num_names in self._dense_items(local):
            pid = id(param)
            if pid in exported:
                tensor, shard, _ = exported[pid]
            elif isinstance(param, DTensor):
                shard = fsdp_dim0_shard(param) if local else None
                if shard is not None:
                    tensor = _bf16_on(param.to_local(), device)
                    counts["local"] += 1
                else:
                    if local:
                        kind = "replicated" if _is_replicated(param) else "gathered"
                        counts[kind] += 1
                        if kind == "gathered":
                            logger.info(
                                f"{name}: all-gathered for the weight sync (mesh ndim={param.device_mesh.ndim}, "
                                f"placements={param.placements}, shape={tuple(param.shape)}, "
                                f"local={tuple(param.to_local().shape)})"
                            )
                    tensor = param.to(device, dtype=torch.bfloat16, non_blocking=True).full_tensor()
                if num_names > 1:
                    exported[pid] = (tensor, shard, param)
            else:
                tensor, shard = param.to(torch.bfloat16, non_blocking=True), None
                if local:
                    counts["replicated"] += 1
            yield name, tensor, shard
            del tensor, param
        exported.clear()
        # Fused expert stacks: folded to one 2-D tensor each when the receiver installs that layout (local shards of
        # whole experts stay local), else gathered and split into one HF tensor per expert.
        while self._expert_stacks:
            stack, slots = self._expert_stacks.pop(0)
            wire = None
            if local and self._expert_stack_wire is not None:
                wire = self._expert_stack_wire(slots, tuple(int(d) for d in stack.shape))
            if wire is not None:
                tensor, shard, kind = self._folded_expert_stack(wire, stack)
                counts[kind] += 1
                yield wire, tensor, shard
                del tensor, stack
                continue
            if local and isinstance(stack, DTensor) and not _is_replicated(stack):
                counts["gathered"] += 1
                logger.info(
                    f"Expert stack of {len(slots)} experts ({slots[0][0]}, ...): all-gathered for the weight sync"
                )
            full = stack.to(device, non_blocking=True)
            if isinstance(full, DTensor):
                full = full.full_tensor()
            full = full.to(torch.bfloat16, non_blocking=True)
            for e, (hf_name, _shape) in enumerate(slots):
                yield hf_name, full[e].clone(), None
            del full
        if local:
            logger.info(
                f"Weight export: {counts['local']} local shards, {counts['replicated']} replicated tensors (fused "
                f"q/k/v are all-gathered inside torchtitan's state_dict()), {counts['gathered']} all-gathered here"
            )

    def _folded_expert_stack(self, wire: str, stack: Any) -> tuple[torch.Tensor, Optional[tuple[int, int]], str]:
        """``(tensor, shard, kind)`` of an ``[E, rows, cols]`` expert stack sent as ``wire`` ``[E * rows, cols]``.

        Only a whole-expert FSDP block (``fsdp_dim0_shard``: ``Shard(0)`` on a 1-D mesh, experts divisible by the mesh
        size) is a dim-0 block of the folded tensor, exported locally (``kind`` ``"local"``). Every other layout is
        gathered (``"gathered"``) and sent folded and full; a replicated stack is sent as is (``"replicated"``).
        In particular, torchtitan's TPU SimpleFSDP shards expert weights on dim 1 once the FSDP degree exceeds the
        expert count (torchtitan 73ebc5f2, for Qwen3-235B); such a block holds a slice of every expert, not whole
        experts, so it is gathered here.
        """
        num_experts, rows, cols = (int(d) for d in stack.shape)
        if isinstance(stack, DTensor):
            shard = fsdp_dim0_shard(stack)
            if shard is not None:
                local = stack.to_local()
                return _bf16_on(local.reshape(int(local.shape[0]) * rows, cols), self._device), shard, "local"
            if not _is_replicated(stack):
                logger.info(
                    f"{wire}: expert stack all-gathered for the weight sync, not a dim-0 block of whole experts "
                    f"(mesh ndim={stack.device_mesh.ndim}, placements={stack.placements}, shape={tuple(stack.shape)}, "
                    f"local={tuple(stack.to_local().shape)})"
                )
                full = stack.to(self._device, dtype=torch.bfloat16, non_blocking=True).full_tensor()
                return full.reshape(num_experts * rows, cols), None, "gathered"
            stack = stack.to_local()
        return _bf16_on(stack.reshape(num_experts * rows, cols), self._device), None, "replicated"
