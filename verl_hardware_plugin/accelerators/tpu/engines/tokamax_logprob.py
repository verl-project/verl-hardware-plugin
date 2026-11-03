# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-token LM-head log-probs on TPU with Tokamax's fused linear + softmax + cross-entropy kernel.

``tokamax_linear_logprob(x, labels, weight_t)`` returns ``log_softmax(x @ weight_t)[label]`` per token without writing
the ``[T, vocab]`` logits to HBM:

- forward: Tokamax's Mosaic-TPU Pallas kernel with ``reduction="none"`` (per-token loss ``-log p(label)`` and
  log-sum-exp), through torchtitan's torch custom op (``linear_softmax_cross_entropy_loss``'s
  ``torch.ops.pallas.linear_loss_fwd_*``);
- backward: in PyTorch, in chunks of ``BACKWARD_CHUNK`` tokens (see :class:`_FusedLogprob`), so only one chunk of
  logits is alive at a time.

Not Tokamax's backward kernels: with tokamax 0.0.14 (the verl TPU image) the Mosaic-TPU backward that torchtitan
uses returns a dW off by ~7% (relative L2 vs an FP32 reference on v5p; jialei777/verl-upstream#115 saw the same on
jax 0.11.1), and its chunked-XLA backward does not run through torch_tpu 0.2.0's JAX bridge. The PyTorch backward
matches the FP32 reference within BF16 rounding (dX, dW ~0.17%, like the kernel's own dX).

Numerics: BF16 operands, FP32 accumulation, FP32 log-sum-exp and softmax; gradients in the operands' dtypes.
"""

import importlib
from typing import Any

import torch

# Tokamax's default block sizes (torchtitan uses the same in its Qwen3-30B-A3B TPU recipe).
B_BLOCK_SIZE = 1024
H_BLOCK_SIZE = 512
V_BLOCK_SIZE = 2048
# Token rows per backward chunk: one chunk's FP32 logits take rows x vocab x 4 B (1.2 GiB for Qwen3's 151,936).
BACKWARD_CHUNK = 2048

_REQUIREMENT = (
    "the fused LM-head log-prob needs torchtitan's TPU kernels (torchtitan.experiments.tpu), tokamax and torch_tpu's "
    "Pallas bridge; all come with the verl TPU image (examples/tpu/grpo/Dockerfile)"
)


def _torchtitan_loss_module() -> Any:
    """torchtitan's ``linear_softmax_cross_entropy_loss`` module (it imports tokamax, working around tokamax#240)."""
    return importlib.import_module("torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss")


def require_tokamax() -> None:
    """Raises a ``RuntimeError`` if torchtitan's TPU loss kernels (and with them tokamax) cannot be imported."""
    try:
        _torchtitan_loss_module()
        importlib.import_module("torch_tpu._internal.pallas.pallas")
    except Exception as e:  # ImportError, or a jax / tokamax version mismatch at import time
        raise RuntimeError(f"{_REQUIREMENT}: {e!r}") from e


def _forward_op(b_block: int, h_block: int, v_block: int) -> Any:
    """torchtitan's forward custom op ``(x, labels, weight_t) -> (loss, lse)`` for ``reduction="none"``."""
    fwd_name, _ = _torchtitan_loss_module()._get_loss_op_names("none", "mosaic_tpu", b_block, h_block, v_block)
    return getattr(torch.ops.pallas, fwd_name)


class _FusedLogprob(torch.autograd.Function):
    """``log p(label)`` per token: the Pallas forward kernel, and a chunked PyTorch backward.

    Backward, for each chunk of ``BACKWARD_CHUNK`` rows: FP32 logits ``x @ W`` (BF16 operands, FP32 accumulation),
    ``p = exp(logits - lse) - onehot(label)`` (the gradient of ``-log p(label)``), scaled by the per-token cotangent
    of the loss, then ``dX = p @ W^T`` and ``dW += x^T @ p`` (FP32).
    """

    @staticmethod
    def forward(ctx: Any, x, labels, weight_t, b_block, h_block, v_block):  # type: ignore[override]
        loss, lse = _forward_op(b_block, h_block, v_block)(x, labels, weight_t)
        ctx.save_for_backward(x, labels, weight_t, lse)
        return -loss

    @staticmethod
    def backward(ctx: Any, grad_logp):  # type: ignore[override]
        x, labels, weight_t, lse = ctx.saved_tensors
        grad_loss = -grad_logp.float()
        w32 = weight_t.float()
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(w32)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium")  # BF16 passes with FP32 accumulation, as in the kernel
        try:
            for start in range(0, x.shape[0], BACKWARD_CHUNK):
                rows = slice(start, start + BACKWARD_CHUNK)
                x32 = x[rows].float()
                p = torch.exp(x32 @ w32 - lse[rows].unsqueeze(-1))
                p.scatter_add_(-1, labels[rows].long().unsqueeze(-1), -torch.ones_like(lse[rows]).unsqueeze(-1))
                p = p * grad_loss[rows].unsqueeze(-1)
                grad_x[rows] = (p @ w32.t()).to(x.dtype)
                grad_w += x32.t() @ p
        finally:
            torch.set_float32_matmul_precision(previous)
        return grad_x, None, grad_w.to(weight_t.dtype), None, None, None


def tokamax_linear_logprob(
    x: torch.Tensor,
    labels: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    b_block_size: int = B_BLOCK_SIZE,
    h_block_size: int = H_BLOCK_SIZE,
    v_block_size: int = V_BLOCK_SIZE,
) -> torch.Tensor:
    """Per-token ``log_softmax(x @ weight_t)[labels]`` in FP32, shape ``(T,)``.

    Args:
        x: ``(T, H)`` hidden states (cast to ``weight_t``'s dtype).
        labels: ``(T,)`` token ids in ``[0, V)``.
        weight_t: ``(H, V)`` transposed LM-head weight (BF16), e.g. ``lm_head.weight.t()`` (passed as is).

    Pads ``T`` with zero rows (label 0) up to a multiple of ``b_block_size``: the kernel needs whole row blocks.
    Padded rows get a zero cotangent.
    """
    x = x.to(weight_t.dtype)
    labels = labels.to(torch.int32)
    num_tokens = x.shape[0]
    pad = -num_tokens % b_block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        labels = torch.nn.functional.pad(labels, (0, pad))
    logp = _FusedLogprob.apply(x.contiguous(), labels.contiguous(), weight_t, b_block_size, h_block_size, v_block_size)
    return logp[:num_tokens] if pad else logp
