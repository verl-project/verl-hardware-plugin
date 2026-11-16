# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bounded streaming of verl's HF tensor exports and strict receiver coverage."""

import torch


def iter_weight_buckets(named_tensors, buckets, device):
    """Stream CPU or device tensors into bounded device buffers."""
    iterator = iter(named_tensors)
    current_name, current_tensor = None, None
    for bucket in buckets:
        buffer = torch.zeros(bucket.buf_numel, dtype=torch.bfloat16, device=device)
        for entry in bucket.entries:
            if entry.name != current_name:
                try:
                    current_name, current_tensor = next(iterator)
                except StopIteration as error:
                    raise ValueError(f"Missing exported parameter {entry.name}") from error
                if current_name != entry.name or tuple(current_tensor.shape) != tuple(entry.shape):
                    raise ValueError(f"Export does not match weight schedule at {entry.name}")
                if current_tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Export {entry.name} must use bfloat16")
                current_tensor = current_tensor.reshape(-1)
            buffer[entry.offset : entry.offset + entry.numel].copy_(
                current_tensor[entry.param_offset : entry.param_offset + entry.numel]
            )
        yield buffer
    if next(iterator, None) is not None:
        raise ValueError("Export contains parameters absent from the weight schedule")


def validate_box_coverage(plans, inventory):
    """Reject skipped exports and incomplete resident weight coverage."""
    from math import prod

    counts = {}
    boxes = {}
    for plan in plans:
        if plan.skipped:
            raise ValueError(f"Rollout cannot apply parameters: {plan.skipped}")
        for piece in plan.pieces:
            shape = inventory[piece.name]
            if len(piece.dst_box) != len(shape) or any(
                lower < 0 or upper > extent or lower >= upper
                for (lower, upper), extent in zip(piece.dst_box, shape, strict=True)
            ):
                raise ValueError(f"Invalid rollout weight box for {piece.name}")
            previous_boxes = boxes.setdefault(piece.name, [])
            for previous in previous_boxes:
                if all(
                    max(lower, other_lower) < min(upper, other_upper)
                    for (lower, upper), (other_lower, other_upper) in zip(piece.dst_box, previous, strict=True)
                ):
                    raise ValueError(f"Overlapping rollout weight coverage for {piece.name}")
            previous_boxes.append(piece.dst_box)
            counts[piece.name] = counts.get(piece.name, 0) + piece.numel
    for name, shape in inventory.items():
        if counts.get(name, 0) != prod(shape):
            raise ValueError(f"Incomplete or overlapping rollout weight coverage for {name}")
