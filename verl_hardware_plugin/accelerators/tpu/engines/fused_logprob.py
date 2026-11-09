# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-token LM-head log-probs on TPU without materializing the ``[T, V]`` logits.

Forward: Tokamax's Mosaic-TPU kernel (``reduction="none"``) through torchtitan's torch op. Backward: chunked PyTorch,
because torchtitan's op only backpropagates a reduced loss and tokamax 0.0.14's Mosaic backward returns a dW ~7% off.
"""

import importlib
from typing import Any

import torch
import torch.nn.functional as F

BACKWARD_CHUNK = 2048  # rows per backward chunk: FP32 logits take rows x vocab x 4 B (1.2 GiB for Qwen3)


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


def _forward_op(b_block: int, h_block: int, v_block: int) -> Any:
    """torchtitan's forward custom op ``(x, labels, weight_t) -> (loss, lse)`` for ``reduction="none"``."""
    fwd_name, _ = _loss_module()._get_loss_op_names("none", "mosaic_tpu", b_block, h_block, v_block)
    return getattr(torch.ops.pallas, fwd_name)


class _FusedLogprob(torch.autograd.Function):
    """``log p(label)`` per token: the Pallas forward kernel and a chunked PyTorch backward.

    TODO(torchtitan): upstream per-token (``reduction="none"``) backward support in torchtitan's
    ``linear_softmax_cross_entropy_loss`` (its ``_my_bwd`` currently reshapes ``g_loss`` to a scalar ``()``
    and tokamax 0.0.14's Mosaic-TPU backward has ~7% dW error), then call ``linear_softmax_cross_entropy_loss``
    directly and remove this custom ``autograd.Function``.
    """

    @staticmethod
    def forward(ctx: Any, x, labels, weight_t, blocks):  # type: ignore[override]
        loss, lse = _forward_op(*blocks)(x, labels, weight_t)
        ctx.save_for_backward(x, labels, weight_t, lse)
        return -loss

    @staticmethod
    def backward(ctx: Any, grad):  # type: ignore[override]
        # d log p(label) / d logits = onehot(label) - softmax(logits); one chunk of FP32 logits at a time.
        # On torch_tpu, FP32 matmuls already default to single-pass BF16 MXU with FP32 accumulation; honor the
        # process-wide float32_matmul_precision rather than mutating it here.
        x, labels, weight_t, lse = ctx.saved_tensors
        w32 = weight_t.float()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(w32)
        for start in range(0, x.shape[0], BACKWARD_CHUNK):
            rows = slice(start, start + BACKWARD_CHUNK)
            x32 = x[rows].float()
            g = -torch.exp(x32 @ w32 - lse[rows].unsqueeze(-1))
            g.scatter_add_(-1, labels[rows].long().unsqueeze(-1), torch.ones_like(g[:, :1]))
            g *= grad[rows].float().unsqueeze(-1)
            grad_x[rows] = (g @ w32.t()).to(x.dtype)
            grad_w += x32.t() @ g
        return grad_x, None, grad_w.to(weight_t.dtype), None


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
    blocks = (b_block_size, h_block_size, v_block_size)
    return _FusedLogprob.apply(x.contiguous(), labels.contiguous(), weight_t, blocks)[:num_tokens]
