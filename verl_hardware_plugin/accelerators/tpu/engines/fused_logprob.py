# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-token LM-head log-probs on TPU without materializing the ``[T, V]`` logits."""

import importlib
from typing import Any

import torch
import torch.nn.functional as F


def _loss_module() -> Any:
    """torchtitan's TPU loss-kernel module (it imports tokamax, working around tokamax#240)."""
    return importlib.import_module("torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss")


def require_tokamax() -> None:
    """Raises a ``RuntimeError`` if torchtitan's TPU loss kernels, tokamax or torch_tpu's Pallas bridge is missing."""
    try:
        _loss_module()
        importlib.import_module("torch_tpu._internal.pallas.pallas")
    except Exception as e:  # ImportError, or a jax / tokamax version mismatch at import time
        raise RuntimeError(
            f"the fused LM-head log-prob needs torchtitan's TPU kernels, tokamax and torch_tpu's Pallas bridge: {e!r}"
        ) from e


def fused_linear_logprobs_tpu(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    temperature: float = 1.0,
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> torch.Tensor:
    """Per-token ``log_softmax(hidden_states @ weight_t / temperature)[labels]`` in FP32, shape ``(T,)``.

    ``hidden_states`` ``(..., H)`` and ``labels`` ``(...)`` are flattened to ``T`` tokens; ``weight_t`` is the
    ``(H, V)`` transposed LM-head weight (BF16), passed as is. ``T`` is zero-padded to whole ``b_block_size`` row
    blocks for the kernel; padded rows get no gradient.
    """
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    if temperature != 1.0:
        x = x * (1.0 / float(temperature))  # logits / t == (x / t) @ W
    x, labels = x.to(weight_t.dtype), labels.reshape(-1).to(torch.int32)
    num_tokens, pad = x.shape[0], -x.shape[0] % b_block_size
    if pad:
        x, labels = F.pad(x, (0, 0, 0, pad)), F.pad(labels, (0, pad))
    loss = _loss_module().linear_softmax_cross_entropy_loss(
        x.contiguous(),
        labels.contiguous(),
        weight_t,
        reduction="none",
        implementation="mosaic_tpu",
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )
    return -loss[:num_tokens]
