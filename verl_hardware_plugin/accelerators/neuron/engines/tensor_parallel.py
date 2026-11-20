# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Vocabulary-parallel token statistics without gathering full model logits."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint

from .compat import vocabulary_parallel_log_probs


class _VocabularyEntropy(torch.autograd.Function):
    """Differentiable entropy, which the pinned TorchTitan loss does not provide."""

    @staticmethod
    def forward(ctx, logits, group):
        maximum = logits.amax(dim=-1, keepdim=True)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        shifted = logits - maximum
        probabilities = shifted.exp()
        denominator = probabilities.sum(dim=-1, keepdim=True)
        dist.all_reduce(denominator, group=group)
        probabilities = probabilities / denominator
        expectation = (probabilities * shifted).sum(dim=-1)
        dist.all_reduce(expectation, group=group)
        centered = shifted - expectation.unsqueeze(-1)
        ctx.save_for_backward(probabilities, centered)
        return denominator.log().squeeze(-1) - expectation

    @staticmethod
    def backward(ctx, entropy_grad):
        probabilities, centered = ctx.saved_tensors
        return -probabilities * centered * entropy_grad.unsqueeze(-1), None


def token_statistics(logits, labels, temperature, calculate_entropy=False, chunk_size=0):
    """Return replicated token log probabilities and optional entropy, with local gradients.

    With chunk_size > 0, the FP32 log-softmax runs per sequence chunk and is recomputed in backward.
    """
    if isinstance(logits, DTensor):
        mesh = logits.device_mesh
        if (
            mesh.ndim != 1
            or not isinstance(logits.placements[0], Shard)
            or logits.placements[0].dim % logits.ndim != logits.ndim - 1
        ):
            raise ValueError(f"Expected vocabulary-sharded logits on a TP mesh, got {logits.placements}")
        if logits.shape[-1] % mesh.size():
            raise ValueError("Trainium requires vocabulary size divisible by tensor parallel size")
        local = logits.to_local()
        group = mesh.get_group()
        if local.ndim == 3 and 0 < chunk_size < local.shape[1]:
            pieces = [
                (local[:, start : start + chunk_size], labels[:, start : start + chunk_size])
                for start in range(0, local.shape[1], chunk_size)
            ]
        else:
            pieces = [(local, labels)]

        def chunk_statistics(piece, piece_labels):
            piece = piece.float() / temperature
            log_probs = vocabulary_parallel_log_probs(piece, piece_labels, group, logits.shape[-1])
            if calculate_entropy:
                return log_probs, _VocabularyEntropy.apply(piece, group)
            return (log_probs,)

        recompute = len(pieces) > 1 and torch.is_grad_enabled() and local.requires_grad
        log_probs, entropy = [], []
        for piece, piece_labels in pieces:
            if recompute:
                result = checkpoint(chunk_statistics, piece, piece_labels, use_reentrant=False)
            else:
                result = chunk_statistics(piece, piece_labels)
            log_probs.append(result[0])
            if calculate_entropy:
                entropy.append(result[1])
        log_probs = torch.cat(log_probs, dim=1) if len(pieces) > 1 else log_probs[0]
        if not calculate_entropy:
            return log_probs, None
        return log_probs, torch.cat(entropy, dim=1) if len(pieces) > 1 else entropy[0]
    normalized = torch.log_softmax(logits.float() / temperature, dim=-1)
    log_probs = normalized.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(normalized.exp() * normalized).sum(-1) if calculate_entropy else None
    return log_probs, entropy
