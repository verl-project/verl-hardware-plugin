# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``engine_kwargs.tpu.blocks_per_compile`` (``engines/block_grouping.py``)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from verl_hardware_plugin.accelerators.tpu import options
from verl_hardware_plugin.accelerators.tpu.engines import block_grouping, tpu_ragged_ep, tpu_utils
from verl_hardware_plugin.accelerators.tpu.engines.block_grouping import BlockGroupingError

_BACKEND = "aot_eager"


@pytest.fixture(autouse=True)
def _reset_dynamo():
    # Every group program shares the code object of ``_BlockGroup.forward``; keep each test's cache entries (and its
    # recompile_limit budget) to itself.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x, masks, positions):
        return x + torch.tanh(self.lin(x)) * masks + positions


class _Decoder(nn.Module):
    """TorchTitan-style decoder: every block gets the previous block's output and the same arguments."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): checkpoint_wrapper(_Block(dim)) for i in range(n)})

    def forward(self, x, masks, positions):
        for layer in self.layers.values():
            x = layer(x, masks, positions)
        return x


class _PerLayerArgsDecoder(_Decoder):
    """Passes a different ``masks`` value to odd layers (e.g. masks / RoPE chosen by layer type)."""

    def forward(self, x, masks, positions):
        for i, layer in enumerate(self.layers.values()):
            x = layer(x, masks if i % 2 == 0 else masks * 2.0, positions)
        return x


class _WorkBetweenBlocksDecoder(_Decoder):
    """Collects hidden states and transforms them between blocks."""

    def forward(self, x, masks, positions):
        for layer in self.layers.values():
            x = layer(x, masks, positions) * 0.5
        return x


class _EarlyExitDecoder(_Decoder):
    """Runs only the first ``n_run`` blocks."""

    n_run = 3

    def forward(self, x, masks, positions):
        for layer in list(self.layers.values())[: self.n_run]:
            x = layer(x, masks, positions)
        return x


def _compile_fn(module: nn.Module):
    return torch.compile(module, backend=_BACKEND, fullgraph=True)


def _fresh(cls=_Decoder, n: int = 5, dim: int = 8, compile_blocks: bool = True) -> nn.Module:
    torch.manual_seed(0)
    model = cls(n, dim)
    if compile_blocks:
        for block in model.layers.values():
            block.compile(backend=_BACKEND, fullgraph=True)
    return model


def _run(model: nn.Module, x: torch.Tensor, masks=None, positions=0.1):
    masks = torch.full((1,), 0.5) if masks is None else masks
    out = model(x, masks, positions)
    out.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return out.detach(), grads


def _assert_same(model: nn.Module, ref: nn.Module, x: torch.Tensor) -> None:
    out_ref, grads_ref = _run(ref, x)
    out, grads = _run(model, x)
    torch.testing.assert_close(out, out_ref)
    for g, g_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(g, g_ref)


@pytest.mark.parametrize("k, expected_groups", [(2, 2), (3, 2), (5, 1), (8, 1)])
def test_group_compiled_blocks_matches_per_block_compile(k, expected_groups):
    ref = _fresh()
    grouped = _fresh()
    names_before = [n for n, _ in grouped.named_parameters()]
    state_before = {n: t.clone() for n, t in grouped.state_dict().items()}
    compiled_groups = []

    def compile_fn(module):
        compiled_groups.append(module)
        return _compile_fn(module)

    assert block_grouping.group_compiled_blocks(grouped, k, compile_fn=compile_fn) == expected_groups
    assert [len(g.blocks) for g in compiled_groups] == [min(k, 5 - i * k) for i in range(expected_groups)]
    assert [n for n, _ in grouped.named_parameters()] == names_before
    assert grouped.state_dict().keys() == state_before.keys()
    followers = [n for n, b in grouped.layers.items() if block_grouping.is_grouped_follower(b)]
    assert len(followers) == sum(len(g.blocks) - 1 for g in compiled_groups)

    x = torch.randn(4, 8)
    _assert_same(grouped, ref, x)
    _assert_same(grouped, ref, x)  # a second forward / backward (tracker reset per forward)
    with torch.no_grad():
        torch.testing.assert_close(grouped(x, torch.full((1,), 0.5), 0.1), ref(x, torch.full((1,), 0.5), 0.1))


def test_blocks_per_compile_one_leaves_model_untouched():
    model = _fresh()
    impls = {n: b.__dict__["_compiled_call_impl"] for n, b in model.layers.items()}
    assert block_grouping.group_compiled_blocks(model, 1, compile_fn=_compile_fn) == 0
    assert {n: b.__dict__["_compiled_call_impl"] for n, b in model.layers.items()} == impls
    assert all("compile" not in b.__dict__ for b in model.layers.values())
    assert not model._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize("k", [0, -3, True, 2.0, "2"])
def test_invalid_blocks_per_compile_raises(k):
    with pytest.raises(ValueError, match="blocks_per_compile"):
        block_grouping.group_compiled_blocks(_fresh(), k, compile_fn=_compile_fn)


def test_uncompiled_blocks_are_not_grouped():
    model = _fresh(compile_blocks=False)
    assert block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn) == 0
    assert not any(block_grouping._is_grouped(b) for b in model.layers.values())


def test_grouping_twice_raises():
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="already grouped"):
        block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)


@pytest.mark.parametrize("block", ["0", "1"])
def test_recompiling_a_grouped_block_raises(block):
    """Previously silent: a recompiled follower ran twice; a recompiled head dropped the rest of its group."""
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="compiled as part of a group"):
        model.layers[block].compile(backend=_BACKEND, fullgraph=True)


def test_redispatching_a_follower_behind_the_grouping_raises():
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    follower = model.layers["1"]
    follower._compiled_call_impl = torch.compile(follower._call_impl, backend=_BACKEND, fullgraph=True)
    with pytest.raises(BlockGroupingError, match="recompiled or re-dispatched"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_redispatching_a_head_behind_the_grouping_raises():
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    head = model.layers["0"]
    head._compiled_call_impl = torch.compile(head._call_impl, backend=_BACKEND, fullgraph=True)
    with pytest.raises(BlockGroupingError, match="out of order"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_per_layer_arguments_raise():
    """Previously silent: every block of a group ran with the group head's arguments."""
    model = _fresh(_PerLayerArgsDecoder)
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="per-layer arguments"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_equal_python_scalars_are_the_same_argument():
    ref = _fresh()
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    x = torch.randn(4, 8)
    masks = torch.full((1,), 0.5)
    # 1e-1 * 1 builds a new float object per call; equal scalars are accepted, tensors must be the same object.
    torch.testing.assert_close(model(x, masks, 1e-1 * 1), ref(x, masks, 0.1))


def test_work_between_blocks_raises():
    model = _fresh(_WorkBetweenBlocksDecoder)
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="works between blocks"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_calling_a_follower_directly_raises():
    model = _fresh()
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="out of order"):
        model.layers["1"](torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_skipping_part_of_a_group_raises():
    model = _fresh(_EarlyExitDecoder)  # runs blocks 0..2; block 3 is block 2's follower
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="was not called"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)
    # The failed forward leaves no state behind: a well-formed group run still works.
    model.n_run = 4
    model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


class _TupleBlock(_Block):
    def forward(self, x, masks, positions):
        return (super().forward(x, masks, positions),)


def test_tuple_returning_last_block_of_a_group_raises():
    model = _fresh()
    model.layers["1"] = _TupleBlock(8)
    model.layers["1"].compile(backend=_BACKEND, fullgraph=True)
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(BlockGroupingError, match="return one Tensor"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_tuple_returning_inner_block_of_a_group_fails_loudly():
    # The next block of the group gets the tuple inside the group program; Dynamo raises (not a silent result).
    model = _fresh()
    model.layers["0"] = _TupleBlock(8)
    model.layers["0"].compile(backend=_BACKEND, fullgraph=True)
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    with pytest.raises(Exception, match="must be Tensor, not tuple"):
        model(torch.randn(4, 8), torch.full((1,), 0.5), 0.1)


def test_relaxed_fullgraph_false_blocks_keep_their_own_program():
    ref = _fresh()
    model = _fresh()
    relaxed = model.layers["2"]
    relaxed.compile(backend=_BACKEND, fullgraph=False)
    setattr(relaxed, block_grouping.COMPILE_RELAXED_ATTR, True)
    relaxed_impl = relaxed.__dict__["_compiled_call_impl"]
    groups = []
    assert block_grouping.group_compiled_blocks(model, 2, compile_fn=lambda m: groups.append(m) or _compile_fn(m)) == 2
    assert [[id(b) for b in g.blocks] for g in groups] == [
        [id(model.layers["0"]), id(model.layers["1"])],
        [id(model.layers["3"]), id(model.layers["4"])],
    ]
    assert relaxed.__dict__["_compiled_call_impl"] is relaxed_impl
    _assert_same(model, ref, torch.randn(4, 8))


class _AllToAllTokenDispatcher:
    """A non-static EP dispatcher (host split syncs): ``_relax_non_static_ep_moe_compile`` relaxes its block."""

    num_experts = 8
    top_k = 2

    def dispatch(self, *args):
        return args

    def combine(self, *args):
        return args[0]


def _decoder_with_moe_block(moe_index: int) -> nn.Module:
    model = _fresh()
    moe = model.layers[str(moe_index)]
    moe.routed = nn.Module()
    moe.routed.token_dispatcher = _AllToAllTokenDispatcher()
    return model


@pytest.mark.parametrize("moe_index", [0, 1, 4])
def test_engine_order_groups_after_the_ep_dispatcher_recompiles(moe_index):
    """The engine's init sequence: parallelize relaxes non-static EP blocks, the ragged EP dispatcher recompiles them
    fullgraph=True, and only then are blocks grouped, so the recompiled block is grouped too."""
    ref = _fresh()
    model = _decoder_with_moe_block(moe_index)
    compile_config = SimpleNamespace(enable=True, backend=_BACKEND)
    tpu_utils._relax_non_static_ep_moe_compile(
        model, parallel_dims=SimpleNamespace(ep_enabled=True), compile_config=compile_config
    )
    assert getattr(model.layers[str(moe_index)], block_grouping.COMPILE_RELAXED_ATTR)
    _, n = tpu_ragged_ep.install_tpu_ep_token_dispatcher(
        [model], ep_dispatcher="ragged", capacity_factor=4.0, spmd_backend="default", compile_config=compile_config
    )
    assert n == 1 and not getattr(model.layers[str(moe_index)], block_grouping.COMPILE_RELAXED_ATTR)
    compile_fn = block_grouping.default_compile_fn(compile_config.backend)
    assert block_grouping.group_compiled_blocks(model, 2, compile_fn=compile_fn) == 2
    _assert_same(model, ref, torch.randn(4, 8))


@pytest.mark.parametrize("moe_index", [0, 1])
def test_ep_dispatcher_recompile_after_grouping_is_safe(moe_index):
    """The reverse (pre-fix) order: the relaxed block was left out of the groups, so its later recompile is safe."""
    ref = _fresh()
    model = _decoder_with_moe_block(moe_index)
    compile_config = SimpleNamespace(enable=True, backend=_BACKEND)
    tpu_utils._relax_non_static_ep_moe_compile(
        model, parallel_dims=SimpleNamespace(ep_enabled=True), compile_config=compile_config
    )
    block_grouping.group_compiled_blocks(model, 2, compile_fn=_compile_fn)
    assert not block_grouping._is_grouped(model.layers[str(moe_index)])
    tpu_ragged_ep.install_tpu_ep_token_dispatcher(
        [model], ep_dispatcher="ragged", capacity_factor=4.0, spmd_backend="default", compile_config=compile_config
    )
    _assert_same(model, ref, torch.randn(4, 8))


def test_blocks_per_compile_option_is_validated():
    def parse(value):
        return options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"blocks_per_compile": value}}))

    assert options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={})).blocks_per_compile is None
    assert parse(2).blocks_per_compile == 2
    assert parse("3").blocks_per_compile == 3
    for bad in ("x", 0, "-3", True):
        with pytest.raises(ValueError, match="blocks_per_compile"):
            parse(bad)
