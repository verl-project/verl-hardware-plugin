# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``VERL_TPU_BLOCKS_PER_COMPILE`` (``engines/block_grouping.py``)."""

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from verl_hardware_plugin.accelerators.tpu.engines import block_grouping


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, masks, positions):
        return x + torch.tanh(self.lin(x)) * masks + positions


class _Decoder(nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): checkpoint_wrapper(_Block(dim)) for i in range(n)})

    def forward(self, x, masks, positions):
        for layer in self.layers.values():
            x = layer(x, masks, positions)
        return x


def _compile_blocks(model: nn.Module) -> None:
    for block in model.layers.values():
        block.compile(backend="aot_eager", fullgraph=True)


def _run(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    out = model(x, 0.5, 0.1)
    out.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return out.detach(), grads


def test_group_compiled_blocks_matches_per_block_compile(monkeypatch):
    torch.manual_seed(0)
    ref = _Decoder(5, 8)
    grouped = _Decoder(5, 8)
    grouped.load_state_dict(ref.state_dict())
    _compile_blocks(ref)
    _compile_blocks(grouped)
    names_before = [n for n, _ in grouped.named_parameters()]

    compiled_groups = []

    def compile_fn(module):
        compiled_groups.append(module)
        return torch.compile(module, backend="aot_eager", fullgraph=True)

    assert block_grouping.group_compiled_blocks(grouped, 2, compile_fn=compile_fn) == 2
    assert len(compiled_groups) == 2 and all(len(g.blocks) == 2 for g in compiled_groups)
    # Blocks 1 and 3 are pass-throughs; block 4 keeps its own per-block program.
    assert grouped.layers["1"]._compiled_call_impl is block_grouping._passthrough
    assert grouped.layers["4"]._compiled_call_impl is not block_grouping._passthrough
    assert [n for n, _ in grouped.named_parameters()] == names_before

    x = torch.randn(4, 8, requires_grad=False)
    out_ref, grads_ref = _run(ref, x)
    out, grads = _run(grouped, x)
    torch.testing.assert_close(out, out_ref)
    for g, g_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(g, g_ref)


def test_group_compiled_blocks_noop_cases(monkeypatch):
    model = _Decoder(3, 4)
    assert block_grouping.group_compiled_blocks(model, 2) == 0  # not compiled
    _compile_blocks(model)
    assert block_grouping.group_compiled_blocks(model, 1) == 0
    monkeypatch.setenv(block_grouping.BLOCKS_PER_COMPILE_ENV, "3")
    assert block_grouping.blocks_per_compile() == 3
    monkeypatch.setenv(block_grouping.BLOCKS_PER_COMPILE_ENV, "x")
    assert block_grouping.blocks_per_compile() == 1
