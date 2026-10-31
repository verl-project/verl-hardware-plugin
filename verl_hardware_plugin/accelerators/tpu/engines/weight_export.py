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

from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name

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
    * **EP**: routed experts sharded on the expert dimension over ``ep`` (mesh dimensions of size 1, e.g. ``efsdp``
      when EP spans every rank, are ignored) are blocks, exported locally. Fused expert stacks of other models
      never reach this function; ``TrainerWeights`` gathers them and splits them into one HF tensor per expert.
    * **1-D tensors** (norm weights, biases, Qwen3.5's ``A_log`` / ``dt_bias``): ``None``, gathered. They are tiny,
      and tpu-sync mis-delivers some of their small blocks (e.g. a 128-element norm weight split 8 ways arrived with
      only its first block), so they are always sent whole.

    Every tensor that ends up gathered is logged by ``TrainerWeights.local_shards()``.
    """
    if not isinstance(param, DTensor) or param.ndim < 2:
        return None
    mesh = param.device_mesh
    # Mesh dimensions of size 1 do not split anything (e.g. the ``efsdp`` dimension of the routed experts when EP
    # spans every rank), so only the others have to form a 1-D layout.
    active = [d for d in range(mesh.ndim) if mesh.size(d) > 1]
    if len(active) != 1 or len(param.placements) != mesh.ndim:
        return None
    dim = active[0]
    placement = param.placements[dim]
    # type() rather than isinstance(): _StridedShard subclasses Shard but permutes which block a rank holds.
    if type(placement) is not Shard or placement.dim != 0:
        return None
    # The other dimensions have size 1, so whatever their placement (torchtitan puts ``_StridedShard`` on the
    # routed experts' size-1 ``efsdp`` dimension under EP), they leave the local block unchanged.
    num_shards = mesh.size(dim)
    full_rows = int(param.shape[0])
    if full_rows % num_shards:
        return None
    if int(param.to_local().shape[0]) != full_rows // num_shards:
        return None
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        return None
    return num_shards, int(coordinate[dim])


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


def _owned(tensor: torch.Tensor, aliases_param: bool) -> torch.Tensor:
    """``tensor``, or a copy of it if it may alias the parameter or is a view into a larger buffer.

    The exported tensors are bound to tpu-sync as whole buffers, read after the export returns. With bf16
    parameters (``engine_kwargs.tpu.param_dtype``), the bf16 cast is a no-op: a replicated parameter's
    ``full_tensor()`` is then the parameter itself, or a view of it (e.g. Qwen3.5's q / k / v rows of the fused
    ``in_proj_qkv``), which tpu-sync rejects and a later optimizer step would change.
    """
    return tensor.clone() if aliases_param or tensor._base is not None else tensor


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
    """

    def __init__(
        self,
        dense: dict[str, Any] | Iterable[dict[str, Any]],
        expert_stacks: list[tuple[Any, list[tuple[str, tuple]]]],
        device: Any,
        local_transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        self._dense_chunks = iter([dense]) if isinstance(dense, dict) else iter(dense)
        self._local_transform = local_transform
        self._expert_stacks = expert_stacks
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
                    local_block = param.to_local()
                    if getattr(param, "_tpu_sync_owned", False) and local_block.dtype == torch.bfloat16:
                        tensor = local_block.to(device, non_blocking=True)  # already a fresh copy (local_transform)
                    else:
                        tensor = _bf16_on(local_block, device)
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
                    converted = param.to(device, dtype=torch.bfloat16, non_blocking=True)
                    tensor = _owned(converted.full_tensor(), aliases_param=converted is param)
                if num_names > 1:
                    exported[pid] = (tensor, shard, param)
            else:
                converted = param.to(torch.bfloat16, non_blocking=True)
                tensor, shard = _owned(converted, aliases_param=converted is param), None
                if local:
                    counts["replicated"] += 1
            yield name, tensor, shard
            del tensor, param
        exported.clear()
        # Fused expert stacks are always gathered, then split into one HF tensor per expert.
        while self._expert_stacks:
            stack, slots = self._expert_stacks.pop(0)
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
            if get_device_name() == "tpu":
                log_gpu_memory_usage("After the tpu-sync weight export", logger=logger)
            logger.info(
                f"Weight export: {counts['local']} local shards, {counts['replicated']} replicated tensors (fused "
                f"q/k/v are all-gathered inside torchtitan's state_dict()), {counts['gathered']} all-gathered here"
            )
