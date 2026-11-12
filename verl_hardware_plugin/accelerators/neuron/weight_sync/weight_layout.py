# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bounded BF16 HF buckets and dense native vLLM tensor-parallel layouts."""

import re
from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class BucketEntry:
    name: str
    shape: tuple
    numel: int
    offset: int
    param_offset: int


@dataclass(frozen=True)
class Bucket:
    buf_numel: int
    entries: tuple


@dataclass(frozen=True)
class BoxPiece:
    name: str
    dst_box: tuple
    box_offset: int
    numel: int
    entry_index: int
    src_box: tuple
    transposed: bool


@dataclass(frozen=True)
class BoxPlan:
    pieces: tuple
    numel: int
    skipped: tuple


def plan_buckets(schedule, itemsize=2, *, bucket_bytes):
    if itemsize != 2 or bucket_bytes < 2 or bucket_bytes % 2:
        raise ValueError("Weight synchronization requires even-sized BF16 buckets")
    capacity = bucket_bytes // itemsize
    buckets, entries, names = [], [], set()
    used = 0
    for name, shape, dtype in schedule:
        shape = tuple(shape)
        if name in names or dtype != "bfloat16" or len(shape) not in (1, 2) or min(shape) < 1:
            raise ValueError(f"Unsupported or duplicate BF16 weight schedule entry: {name}")
        names.add(name)
        row_size = prod(shape[1:])
        if row_size > capacity:
            raise ValueError(f"One row of {name} exceeds the weight bucket size")
        offset = 0
        while offset < prod(shape):
            length = min(capacity - used, prod(shape) - offset)
            length -= length % row_size
            if not length:
                buckets.append(Bucket(capacity, tuple(entries)))
                entries, used = [], 0
                continue
            entries.append(BucketEntry(name, shape, length, used, offset))
            used += length
            offset += length
    if entries:
        buckets.append(Bucket(capacity, tuple(entries)))
    if not buckets:
        raise ValueError("The weight schedule cannot be empty")
    return buckets


def _fused_region(name, shapes, inventory, rank, size):
    match = re.fullmatch(r"(.*\.self_attn\.)([qkv])_proj\.(weight|bias)", name)
    if match is None:
        return None
    prefix, projection, suffix = match.groups()
    target = prefix + "qkv_proj_" + suffix
    if target not in inventory:
        return None
    projections = [shapes.get(prefix + kind + "_proj." + suffix) for kind in "qkv"]
    if any(shape is None for shape in projections):
        raise ValueError(f"Incomplete QKV schedule for {target}")
    query, key, value = map(tuple, projections)
    resident = tuple(inventory[target])
    transposed = suffix == "weight"
    dimensions = 2 if transposed else 1
    if (
        any(len(shape) != dimensions for shape in (query, key, value, resident))
        or key != value
        or query[1:] != key[1:]
        or query[0] % size
        or (transposed and resident[0] != query[1])
    ):
        raise ValueError(f"Unsupported QKV shapes for {target}")
    query_width = query[0] // size
    remaining = resident[-1] - query_width
    if remaining <= 0 or remaining % 2:
        raise ValueError(f"Invalid fused QKV width for {target}")
    key_width = remaining // 2
    if key[0] % key_width or key_width * size % key[0]:
        raise ValueError(f"Invalid KV replication for {target}")
    replicas = key_width * size // key[0]
    if replicas < 1 or size % replicas:
        raise ValueError(f"Invalid KV replication for {target}")
    width = query_width if projection == "q" else key_width
    shard = rank if projection == "q" else rank // replicas
    column = {"q": 0, "k": query_width, "v": query_width + key_width}[projection]
    region = ((shard * width, (shard + 1) * width),) + tuple((0, extent) for extent in query[1:])
    origin = (0, column) if transposed else (column,)
    return target, region, origin, transposed


def _target_regions(name, shapes, inventory, rank, size):
    fused = _fused_region(name, shapes, inventory, rank, size)
    if fused is not None:
        return [fused]
    full = tuple(shapes[name])
    folded = re.sub(r"\.(weight|bias)$", r"_\1", name)
    targets = [name] if name in inventory else [folded] if folded in inventory else []
    if name == "model.embed_tokens.weight" and "lm_head.weight" not in shapes and "lm_head.weight" in inventory:
        targets.append("lm_head.weight")
    regions = []
    for target in targets:
        transpose = target == folded and len(full) == 2
        resident = tuple(inventory[target])
        expected = full[::-1] if transpose else full
        region = [(0, extent) for extent in full]
        if expected != resident:
            axes = [
                axis
                for axis in range(len(full))
                if len(resident) == len(full)
                and expected[axis] == resident[axis] * size
                and all(expected[other] == resident[other] for other in range(len(full)) if other != axis)
            ]
            if len(axes) != 1:
                raise ValueError(f"Unsupported TP layout for {target}: {full} -> {resident}")
            axis = axes[0]
            source_axis = 1 - axis if transpose else axis
            region[source_axis] = (rank * resident[axis], (rank + 1) * resident[axis])
        regions.append((target, tuple(region), (0,) * len(full), transpose))
    return regions


def plan_box(bucket, shapes, inventory, tp_rank, tp_size):
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid tensor-parallel coordinate")
    pieces, skipped = [], []
    offset = 0
    for entry_index, entry in enumerate(bucket.entries):
        if tuple(shapes[entry.name]) != entry.shape:
            raise ValueError(f"Bucket shape disagrees with schedule for {entry.name}")
        row_size = prod(entry.shape[1:])
        first, last = entry.param_offset // row_size, (entry.param_offset + entry.numel) // row_size
        targets = _target_regions(entry.name, shapes, inventory, tp_rank, tp_size)
        if not targets:
            skipped.append(entry.name)
        for target, region, origin, transpose in targets:
            lower, upper = max(first, region[0][0]), min(last, region[0][1])
            if lower >= upper:
                continue
            source = ((lower - first, upper - first),) + region[1:]
            destination = ((lower - region[0][0], upper - region[0][0]),) + tuple(
                (0, end - start) for start, end in region[1:]
            )
            if transpose:
                destination = destination[::-1]
            destination = tuple(
                (start + shift, end + shift) for (start, end), shift in zip(destination, origin, strict=True)
            )
            length = prod(end - start for start, end in destination)
            pieces.append(BoxPiece(target, destination, offset, length, entry_index, source, transpose))
            offset += length
    return BoxPlan(tuple(pieces), offset, tuple(skipped))


def slot_numel_for(plans, itemsize=2):
    alignment = 1024 * 1024 // itemsize
    maximum = max(plan.numel for plan in plans)
    return max(alignment, (maximum + alignment - 1) // alignment * alignment)


def _slices(box):
    return tuple(slice(start, end) for start, end in box)


def pack_box(slot, buffer, bucket, plan):
    for piece in plan.pieces:
        entry = bucket.entries[piece.entry_index]
        fragment = buffer[entry.offset : entry.offset + entry.numel].view(-1, *entry.shape[1:])
        source = fragment[_slices(piece.src_box)]
        if piece.transposed:
            source = source.transpose(0, 1)
        slot[piece.box_offset : piece.box_offset + piece.numel].copy_(source.reshape(-1))


def apply_box(host_buffer, plan, tensors):
    """Apply host-staged pieces without lite's unsafe offset/strided device copies."""
    if host_buffer.device.type != "cpu":
        raise ValueError("Native weight application requires a host-staged NIXL slot")
    with torch.no_grad():
        for piece in plan.pieces:
            target = tensors[piece.name]
            shape = tuple(end - start for start, end in piece.dst_box)
            source = host_buffer[piece.box_offset : piece.box_offset + piece.numel].view(shape)
            rows_only = all(
                start == 0 and end == extent
                for (start, end), extent in zip(piece.dst_box[1:], target.shape[1:], strict=True)
            )
            if target.device.type == "cpu" or rows_only:
                target[_slices(piece.dst_box)].copy_(source)
            else:
                staged = target.detach().to("cpu")
                staged[_slices(piece.dst_box)].copy_(source)
                target.copy_(staged)
    return len(plan.pieces)
