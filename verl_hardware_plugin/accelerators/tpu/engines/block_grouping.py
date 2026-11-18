# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compiling ``k`` consecutive transformer blocks as one XLA program (``VERL_TPU_BLOCKS_PER_COMPILE=k``).

TorchTitan's ``apply_compile`` compiles each transformer block on its own, and SimpleFSDP gathers a block's
parameters on attribute access inside the block. Every block's FSDP all-gathers therefore live in that block's
program: XLA's latency-hiding scheduler can overlap them only with the same block's compute, never with the
previous block's, and torch_tpu runs one program at a time. For Qwen3.5-397B (6.4 GB of bf16 expert weights per block
per EP2 rank) the gathers dominate the step (~54% of device time waiting in ``all-gather-done``).

With ``k > 1`` the first block of each group of ``k`` is given one compiled program that runs the whole group
(through each block's own ``_call_impl``, so activation checkpointing and hooks are unchanged) and the other blocks
of the group become pass-throughs. Inside the group program the scheduler can start block ``i+1``'s all-gathers
while block ``i`` computes (forward), and likewise for the recompute + backward. This is the in-program equivalent
of TorchTitan's graph-split ``layers_per_executable=2`` (24% shorter steps on Qwen3.5-35B) and of MaxText's
``prefetch_fsdp_weights``. It costs up to one extra block's gathered weights (and gradients in the backward) of HBM.

Parameter FQNs, the state dict and the weight sync are untouched: the group runner is not registered on the model.
Off by default; requires ``use_torch_compile=True``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from torch import nn

logger = logging.getLogger(__name__)

BLOCKS_PER_COMPILE_ENV = "VERL_TPU_BLOCKS_PER_COMPILE"


def blocks_per_compile() -> int:
    """``VERL_TPU_BLOCKS_PER_COMPILE`` (default ``1``: TorchTitan's per-block compile)."""
    try:
        return max(1, int(os.environ.get(BLOCKS_PER_COMPILE_ENV, "1") or 1))
    except ValueError:
        return 1


class _BlockGroup(nn.Module):
    """Runs ``blocks`` in sequence, each through its uncompiled ``_call_impl``.

    Not attached to the model, so its ``blocks`` registration does not change any parameter FQN.
    """

    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        for block in self.blocks:
            x = block._call_impl(x, *args, **kwargs)
        return x


def _passthrough(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    del args, kwargs
    return x


def group_compiled_blocks(
    model: nn.Module, k: int, *, compile_fn: Callable[[nn.Module], Callable] | None = None
) -> int:
    """Replaces the per-block compiled programs of ``model.layers`` by one program per ``k`` consecutive blocks.

    Args:
        model: The decoder whose ``layers`` (a ``ModuleDict`` iterated in order by its ``forward``) were compiled by
            ``apply_compile`` (``block._compiled_call_impl`` set on every block).
        k: Blocks per program; ``<= 1`` does nothing.
        compile_fn: Compiles a group runner; ``torch.compile(backend="tpu", fullgraph=True)`` by default.

    Returns:
        The number of group programs installed (0 when nothing changed).
    """
    layers = getattr(model, "layers", None)
    if k <= 1 or layers is None:
        return 0
    blocks = list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)
    if not blocks or any(getattr(b, "_compiled_call_impl", None) is None for b in blocks):
        logger.warning("%s=%d ignored: the transformer blocks are not compiled", BLOCKS_PER_COMPILE_ENV, k)
        return 0
    if compile_fn is None:

        def compile_fn(module: nn.Module) -> Callable:
            return torch.compile(module, backend="tpu", fullgraph=True)

    groups = 0
    for start in range(0, len(blocks), k):
        group = blocks[start : start + k]
        if len(group) == 1:
            continue  # a trailing single block keeps its own per-block program
        # object.__setattr__: assigning a Module through nn.Module.__setattr__ would register the runner (and the
        # group's later blocks) under block 0, changing parameter FQNs.
        object.__setattr__(group[0], "_compiled_call_impl", compile_fn(_BlockGroup(group)))
        for block in group[1:]:
            block._compiled_call_impl = _passthrough
        groups += 1
    logger.warning("Compiling %d transformer blocks as %d programs of up to %d blocks", len(blocks), groups, k)
    return groups
