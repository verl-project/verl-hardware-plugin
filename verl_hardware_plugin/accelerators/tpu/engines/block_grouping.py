# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compiling ``k`` consecutive transformer blocks as one XLA program (``engine_kwargs.tpu.blocks_per_compile=k``).

TorchTitan's ``apply_compile`` compiles each transformer block on its own, and SimpleFSDP gathers a block's weights
inside that block's program, so XLA can only overlap a block's all-gathers with the same block's compute. Compiling
``k`` consecutive blocks as one program lets XLA's latency-hiding scheduler start block ``i+1``'s gathers under block
``i``'s forward (and recompute + backward). Activation checkpointing stays per block: each block's ``_call_impl``, with
its checkpoint wrapper, runs inside the group program. Cost: up to one extra block of gathered weights and gradients
live in HBM.

How it works, and why it cannot silently compute something else. The decoder still loops over its blocks and calls
each one. The first block of a group (the head) runs the whole group's program with the arguments the decoder passed
to it, and returns the group's output. Every other block of the group (a follower) returns its input unchanged, after
checking that the per-block compute it skips is exactly what the group program already did:

* the follower is called right after its predecessor in the group, in order, within the same forward;
* its input is the very tensor the group program returned (the decoder did nothing between the blocks); and
* its other arguments are the head's (the same objects, or equal Python scalars / strings / ``None``).

Anything else raises :class:`BlockGroupingError`: a decoder that passes per-layer arguments, works between blocks, calls
a follower directly, or skips part of a group. Later changes to a grouped block are caught too: ``block.compile``
raises, and a head checks on every call that its followers still carry their guards, so a block recompiled or
re-dispatched behind the grouping's back fails at the next forward.

Only blocks compiled ``fullgraph=True`` are grouped. Blocks that a recipe relaxed to ``fullgraph=False`` (marked
``_tpu_compile_relaxed``, see ``tpu_utils._relax_non_static_ep_moe_compile``) keep their own program, and groups are
formed from the runs of consecutive eligible blocks between them. Call :func:`group_compiled_blocks` after every
module swap or recompile of the model (the engine calls it last in ``init_model``).

``copy.deepcopy`` / pickling drops ``_compiled_call_impl`` (``nn.Module.__getstate__``), so a copy runs every block
eagerly, which is still correct; calling ``.compile`` on a copied grouped block raises.

The clean general design lives upstream: a ``model.layers`` container that runs its own layers (TorchTitan's
``TpuLayerLoop``, ``experiments/tpu/components/layer_loop.py``) can compile groups of layers directly, because the
decoder then hands one argument list to all of them. This module is the plugin-side fallback for decoders that loop
over ``model.layers`` themselves.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Set (to True) by recipes on blocks they recompiled with ``fullgraph=False`` (``tpu_utils``).
COMPILE_RELAXED_ATTR = "_tpu_compile_relaxed"

_SCALAR_TYPES = (bool, int, float, complex, str, bytes, type(None))


class BlockGroupingError(RuntimeError):
    """A grouped block is used in a way the group program does not reproduce."""


class _BlockGroup(nn.Module):
    """Runs ``blocks`` in sequence, each through its uncompiled ``_call_impl`` (its own AC wrapper and hooks).

    Not attached to the model, so its ``blocks`` registration does not change any parameter FQN.
    """

    def __init__(self, blocks: Sequence[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        for block in self.blocks:
            x = block._call_impl(x, *args, **kwargs)
        return x


@dataclasses.dataclass
class _Pending:
    """The group whose program ran last, and the follower call it expects next."""

    group: _Group
    next_index: int
    output: torch.Tensor
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _Tracker:
    """State shared by the groups of one model; reset at the start of every forward of the model."""

    def __init__(self) -> None:
        self.pending: Optional[_Pending] = None

    def reset(self, *_: Any) -> None:
        self.pending = None

    def check_complete(self, *_: Any) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            raise BlockGroupingError(
                f"blocks_per_compile: {pending.group.name(pending.next_index)} was not called after "
                f"{pending.group.name(0)}, whose program computed it; the decoder must call every block of a group"
            )


def _same_arg(a: Any, b: Any) -> bool:
    if a is b:
        return True
    return type(a) is type(b) and isinstance(a, _SCALAR_TYPES) and a == b


class _Group:
    """``blocks`` (named ``names``) compiled as one ``program``; installs the head and follower dispatch."""

    def __init__(self, blocks: list[nn.Module], names: list[str], program: Callable, tracker: _Tracker):
        self.blocks = blocks
        self.names = names
        self.program = program
        self.tracker = tracker
        self.head = _HeadCall(self)
        self.followers = [_FollowerCall(self, i) for i in range(1, len(blocks))]

    def name(self, index: int) -> str:
        return f"block {self.names[index]!r} (group {self.names[0]}..{self.names[-1]})"

    def install(self) -> None:
        # object.__setattr__: assigning a Module through nn.Module.__setattr__ would register the program (an
        # OptimizedModule holding the group's blocks) under the head block, changing parameter FQNs.
        object.__setattr__(self.blocks[0], "_compiled_call_impl", self.head)
        for block, follower in zip(self.blocks[1:], self.followers, strict=True):
            object.__setattr__(block, "_compiled_call_impl", follower)
        for index, block in enumerate(self.blocks):
            object.__setattr__(block, "compile", _refuse_compile(self.name(index)))

    def check_installed(self) -> None:
        for index, (block, follower) in enumerate(zip(self.blocks[1:], self.followers, strict=True), start=1):
            if block.__dict__.get("_compiled_call_impl") is not follower:
                raise BlockGroupingError(
                    f"blocks_per_compile: {self.name(index)} was recompiled or re-dispatched after grouping, so the "
                    "group program would run it twice; group the blocks after every recompile"
                )


class _HeadCall:
    """``_compiled_call_impl`` of a group's first block: runs the group program."""

    def __init__(self, group: _Group):
        self.group = group

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        group = self.group
        group.check_installed()
        group.tracker.check_complete()  # the previous group must have been finished
        if not isinstance(x, torch.Tensor):
            raise BlockGroupingError(
                f"blocks_per_compile: {group.name(0)} got a {type(x).__name__} input; grouping needs a Tensor"
            )
        out = group.program(x, *args, **kwargs)
        if not isinstance(out, torch.Tensor):
            raise BlockGroupingError(
                f"blocks_per_compile: {group.name(0)} returned a {type(out).__name__}; grouping needs blocks that "
                "return one Tensor"
            )
        group.tracker.pending = _Pending(group, 1, out, args, dict(kwargs))
        return out


class _FollowerCall:
    """``_compiled_call_impl`` of a later block of a group: checks the group program computed it; returns its input."""

    def __init__(self, group: _Group, index: int):
        self.group = group
        self.index = index

    def _fail(self, why: str) -> BlockGroupingError:
        self.group.tracker.pending = None
        return BlockGroupingError(
            f"blocks_per_compile: {self.group.name(self.index)} {why}. Its compute already ran inside the program "
            f"of {self.group.name(0)} with that block's input and arguments, so this decoder would get wrong results; "
            "set engine_kwargs.tpu.blocks_per_compile=1 for this model"
        )

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        pending = self.group.tracker.pending
        if pending is None or pending.group is not self.group or pending.next_index != self.index:
            raise self._fail("was called out of order (not right after the previous block of its group)")
        if x is not pending.output:
            raise self._fail("got an input other than the previous block's output (the decoder works between blocks)")
        if len(args) != len(pending.args) or not all(map(_same_arg, args, pending.args)):
            raise self._fail("got positional arguments different from the group's first block (per-layer arguments)")
        if kwargs.keys() != pending.kwargs.keys() or not all(
            _same_arg(value, pending.kwargs[key]) for key, value in kwargs.items()
        ):
            raise self._fail("got keyword arguments different from the group's first block (per-layer arguments)")
        if self.index == len(self.group.blocks) - 1:
            self.group.tracker.pending = None
        else:
            pending.next_index += 1
        return x


def _refuse_compile(name: str) -> Callable[..., None]:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise BlockGroupingError(
            f"blocks_per_compile: {name} is compiled as part of a group; recompiling it alone would run it twice or "
            "drop the rest of its group. Recompile before grouping (group_compiled_blocks)"
        )

    return refuse


def is_grouped_follower(module: nn.Module) -> bool:
    """Whether ``module`` is a non-first block of a group, i.e. has no program of its own."""
    return isinstance(module.__dict__.get("_compiled_call_impl"), _FollowerCall)


def _is_grouped(module: nn.Module) -> bool:
    return isinstance(module.__dict__.get("_compiled_call_impl"), _HeadCall | _FollowerCall)


def default_compile_fn(backend: Any) -> Callable[[nn.Module], Callable]:
    """``torch.compile(group, backend=backend, fullgraph=True)``: the options ``apply_compile`` uses per block."""

    def compile_fn(module: nn.Module) -> Callable:
        return torch.compile(module, backend=backend, fullgraph=True)

    return compile_fn


def _eligible(block: nn.Module) -> bool:
    return getattr(block, "_compiled_call_impl", None) is not None and not getattr(block, COMPILE_RELAXED_ATTR, False)


def group_compiled_blocks(
    model: nn.Module, k: int, *, compile_fn: Callable[[nn.Module], Callable], layers_attr: str = "layers"
) -> int:
    """Compiles every ``k`` consecutive per-block-compiled blocks of ``model.<layers_attr>`` as one program.

    Call it after every module swap / recompile of ``model``; see the module docstring for what is checked at run time.

    Args:
        model: The decoder. Its ``forward`` calls the blocks of ``model.<layers_attr>`` (a ``ModuleDict``,
            ``ModuleList`` or ``Sequential``) in order, as ``block(h, *args, **kwargs)`` with the same ``args`` /
            ``kwargs`` for every block, and passes each block's output straight to the next one; otherwise the first
            forward raises :class:`BlockGroupingError`.
        k: Blocks per program; ``1`` does nothing.
        compile_fn: Compiles a group runner, e.g. :func:`default_compile_fn` of the engine's compile backend.
        layers_attr: Name of the block container on ``model``.

    Returns:
        The number of group programs installed (0 when nothing changed).

    Raises:
        ValueError: ``k`` is not an integer >= 1.
        BlockGroupingError: ``model`` is already grouped.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"blocks_per_compile must be an integer >= 1, got {k!r}")
    layers = getattr(model, layers_attr, None)
    if k == 1 or layers is None:
        return 0
    if isinstance(layers, nn.ModuleDict):
        named = list(layers.items())
    else:
        named = [(str(i), block) for i, block in enumerate(layers)]
    if any(_is_grouped(block) for _, block in named):
        raise BlockGroupingError(f"blocks_per_compile: {type(model).__name__}.{layers_attr} is already grouped")

    runs: list[list[tuple[str, nn.Module]]] = [[]]
    for name, block in named:
        if _eligible(block):
            runs[-1].append((name, block))
        elif runs[-1]:
            runs.append([])
    tracker = _Tracker()
    groups = 0
    for run in runs:
        for start in range(0, len(run), k):
            chunk = run[start : start + k]
            if len(chunk) < 2:
                continue  # a lone block keeps its own per-block program
            blocks = [block for _, block in chunk]
            _Group(blocks, [name for name, _ in chunk], compile_fn(_BlockGroup(blocks)), tracker).install()
            groups += 1
    if groups:
        model.register_forward_pre_hook(tracker.reset)
        model.register_forward_hook(tracker.check_complete)
    n_eligible = sum(len(run) for run in runs)
    logger.info(
        "blocks_per_compile=%d: %d of %d blocks eligible (compiled with fullgraph=True), %d group programs",
        k,
        n_eligible,
        len(named),
        groups,
    )
    if n_eligible < len(named):
        logger.warning(
            "blocks_per_compile=%d: %d of %d blocks keep their own program (not compiled, or compiled fullgraph=False)",
            k,
            len(named) - n_eligible,
            len(named),
        )
    return groups
