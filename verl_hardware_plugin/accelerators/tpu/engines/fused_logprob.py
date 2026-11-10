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

BACKWARD_CHUNK_BYTES = 512 << 20  # FP32 logits bytes per backward chunk: rows x vocab-chunk x 4 B
BACKWARD_MIN_VOCAB_CHUNK = 1024


def backward_vocab_chunk(num_tokens: int, vocab: int, chunk_bytes: int = BACKWARD_CHUNK_BYTES) -> int:
    """Vocab columns per backward chunk, so that the chunk's FP32 logits take about ``chunk_bytes``.

    A multiple of 128 (TPU lane width) of at least ``BACKWARD_MIN_VOCAB_CHUNK``, or the whole vocab if it fits.
    """
    cols = chunk_bytes // (4 * max(num_tokens, 1))
    cols = max(BACKWARD_MIN_VOCAB_CHUNK, (cols // 128) * 128)
    return min(vocab, cols)


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
        # d log p(label) / d logits = onehot(label) - softmax(logits), one vocab chunk of FP32 logits at a time.
        # Chunking the vocab (not the tokens) keeps the peak at one [T, Vc] FP32 chunk plus the [H, Vc] FP32 weight
        # slice: never an FP32 copy of the whole LM head or of its gradient (2 x 2.3 GiB for Qwen3-8B), and each
        # grad_w column block is complete after one matmul, so it is written in the weight's dtype directly.
        # On torch_tpu, FP32 matmuls already default to single-pass BF16 MXU with FP32 accumulation; honor the
        # process-wide float32_matmul_precision rather than mutating it here.
        x, labels, weight_t, lse = ctx.saved_tensors
        num_tokens, vocab = x.shape[0], weight_t.shape[1]
        x32 = x.float()
        scale = grad.float().unsqueeze(-1)
        lse_col = lse.unsqueeze(-1)
        labels = labels.long()
        grad_x = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        grad_w = torch.empty_like(weight_t)
        chunk = backward_vocab_chunk(num_tokens, vocab)
        for start in range(0, vocab, chunk):
            cols = slice(start, min(start + chunk, vocab))
            w32 = weight_t[:, cols].float()
            g = -torch.exp(x32 @ w32 - lse_col)  # -softmax over this chunk's columns
            in_chunk = (labels >= cols.start) & (labels < cols.stop)
            local = (labels - cols.start).clamp_(0, cols.stop - cols.start - 1).unsqueeze(-1)
            g.scatter_add_(-1, local, in_chunk.to(g.dtype).unsqueeze(-1))
            g *= scale
            grad_x.addmm_(g, w32.t())
            grad_w[:, cols] = (x32.t() @ g).to(weight_t.dtype)
        return grad_x.to(x.dtype), None, grad_w, None


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
