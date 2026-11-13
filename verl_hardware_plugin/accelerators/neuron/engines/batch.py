# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Static model frames and differentiable host-side RL loss inputs."""

import torch


def padded_inputs(input_ids, width, pad_token_id):
    """Right-pad independent samples without introducing attention between rows."""
    if not input_ids.is_nested or input_ids.device.type != "cpu":
        raise ValueError("Trainium expects CPU jagged input_ids from verl's NO_PADDING dataset")
    offsets = input_ids.offsets()
    lengths = offsets.diff()
    if torch.any(lengths <= 0) or torch.any(lengths > width):
        raise ValueError("Every sample must contain between one and max_seq_len tokens")
    padded = input_ids.to_padded_tensor(pad_token_id, output_size=(len(lengths), width)).long()
    positions = torch.arange(width).expand(len(lengths), -1).contiguous()
    labels = torch.zeros_like(padded)
    labels[:, :-1] = padded[:, 1:]
    return padded, positions, labels, offsets


def host_jagged(values, offsets):
    """Move only token statistics to CPU; retain the autograd edge to Trainium."""
    host_values = values.to("cpu")
    lengths = offsets.diff()
    valid = torch.arange(host_values.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return torch.nested.nested_tensor_from_jagged(host_values[valid], offsets=offsets)
