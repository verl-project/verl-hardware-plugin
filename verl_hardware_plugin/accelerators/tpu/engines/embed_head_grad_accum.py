# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Input embedding / LM head gradients under SimpleFSDP: one all-gather and one reduce-scatter per step.

torchtitan's SimpleFSDP (``experiments/graph_trainer/simple_fsdp.py``) all-gathers a parameter on every access and
reduce-scatters its gradient in every backward pass, so with ``M`` micro-batches per optimizer step every parameter
costs ``M`` all-gathers and ``M`` reduce-scatters. Inside the compiled TransformerBlocks those collectives overlap
with compute. The input embedding (``tok_embeddings``) and the LM head (``lm_head``) run eagerly outside the blocks,
so theirs run in series with compute, and with a large vocabulary these are among the largest parameters of a small
model (``2 x 248,320 x 1,024`` for an untied Qwen3.5-0.8B, about half of its parameters).

With ``engine_kwargs.tpu.embed_head_accum=True`` the engine runs the micro-batch loop of a training step inside
:func:`accumulate_embed_head_grads`, which replaces that traffic for these parameters:

1. Before the first micro-batch: one all-gather per parameter, through the parameter's own ``ReplicateComputation``
   (the op every access runs, so the weights used in the forward pass are bitwise the same), and one zero buffer of
   the full shape in the reduce dtype.
2. In the loop, every access returns a fresh leaf tensor aliasing the gathered weight. A post-accumulate-grad hook
   adds the leaf's gradient, upcast to the buffer's dtype, into the buffer and frees it.
3. After the last micro-batch: one reduce-scatter per parameter, in the reduce dtype, cast to the parameter dtype and
   assigned to (or added into) its sharded ``.grad``.

Forward outputs and the gradients of every other parameter are unchanged. The embedding and head gradients add the
same terms in a different order (over micro-batches first, then over ranks), so they match the default path to
rounding, not bit for bit. A tied embedding and head (one Parameter behind both modules) is gathered and
reduce-scattered once per step; every use still gets its own leaf, so both uses are summed in the buffer's dtype as
in the default path. Compiled TransformerBlocks are untouched.

Cost: per unique parameter, the gathered weight and the buffer stay allocated for the whole micro-batch loop (for
Qwen3.5-0.8B with bf16 compute and fp32 gradient reduction, 0.5 GB + 1.0 GB per parameter).

Supported: 1-D ``fully_shard`` SimpleFSDP (no TP / EP / HSDP on these weights, not the ``spmd_types`` backend), a
single model part (no pipeline parallelism), and an eager embedding and head. Anything else raises
:class:`UnsupportedLayoutError` before any collective runs, with the same outcome on every rank.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ContextManager, Optional

import torch
from torch import nn

# Backward order: the LM head's gradient arrives first, the embedding's last. The collectives of enter / exit run in
# this order on every rank.
EMBED_HEAD_MODULE_NAMES = ("lm_head", "tok_embeddings")


class UnsupportedLayoutError(RuntimeError):
    """The model is not in a layout this module handles (decided the same way on every rank, before collectives)."""


class _Use:
    """One SimpleFSDP-parametrized access site ``<module>.weight``."""

    __slots__ = ("name", "parametrizations", "index", "replicate_computation")

    def __init__(self, name: str, parametrizations: list, index: int, replicate_computation: nn.Module):
        self.name = name
        self.parametrizations = parametrizations
        self.index = index
        self.replicate_computation = replicate_computation


class _ParamGroup:
    """One unique sharded Parameter and its uses (two for a tied embedding and head)."""

    def __init__(self, param: nn.Parameter, uses: list[_Use]):
        self.param = param
        self.uses = uses
        rc = uses[0].replicate_computation
        self.mesh = rc.device_mesh
        self.grad_placements = list(rc.grad_placements)
        self.gather_dtype: torch.dtype = rc.param_dtype or param.dtype
        # The dtype of the reduce-scatter, as in the default path, and of the buffer.
        self.reduce_dtype: torch.dtype = rc.reduce_dtype or param.dtype
        self.buffer_dtype = torch.promote_types(self.reduce_dtype, param.dtype)
        self.full: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None
        self.num_grads = 0

    @property
    def name(self) -> str:
        return "=".join(use.name for use in self.uses)

    def accumulate(self, leaf: torch.Tensor) -> None:
        assert self.buffer is not None and leaf.grad is not None
        self.buffer.add_(leaf.grad.to(self.buffer_dtype))
        leaf.grad = None
        self.num_grads += 1


class _CachedReplicateComputation:
    """Stands in for a ``ReplicateComputation`` inside the loop: serves the weight gathered once."""

    def __init__(self, group: _ParamGroup, simple_fsdp: Any):
        self._group = group
        self._simple_fsdp = simple_fsdp

    def __call__(self, param: torch.Tensor) -> torch.Tensor:
        group = self._group
        if not self._simple_fsdp._active_parametrization:
            return param  # as ReplicateComputation: the sharded parameter itself
        if param is not group.param:
            raise RuntimeError(f"{group.name}: parametrization called with a different Parameter")
        assert group.full is not None
        if not (torch.is_grad_enabled() and param.requires_grad):
            return group.full
        leaf = group.full.detach().requires_grad_(True)
        leaf.register_post_accumulate_grad_hook(group.accumulate)
        return leaf


def _single_model(model_parts: Any) -> nn.Module:
    parts = list(model_parts) if isinstance(model_parts, list | tuple) else [model_parts]
    if len(parts) != 1:
        raise UnsupportedLayoutError(f"{len(parts)} model parts (pipeline parallelism is not supported)")
    model = parts[0]
    if hasattr(model, "_orig_mod"):
        raise UnsupportedLayoutError("the whole model is torch.compile'd; the embedding and LM head must run eagerly")
    return model


def plan_embed_head_grad_accumulation(
    model_parts: Any, module_names: Sequence[str] = EMBED_HEAD_MODULE_NAMES
) -> list[_ParamGroup]:
    """Returns the unique Parameters behind ``module_names`` with their uses.

    Raises:
        UnsupportedLayoutError: unless every module is a SimpleFSDP-parametrized, trainable, eager module sharded
            1-D ``fully_shard`` without TP / EP, and a shared Parameter has the same SimpleFSDP policy at every use.
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torchtitan.experiments.graph_trainer import simple_fsdp

    model = _single_model(model_parts)
    if not simple_fsdp._active_parametrization:
        raise UnsupportedLayoutError("SimpleFSDP's parametrization is disabled")
    groups: dict[int, _ParamGroup] = {}
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            raise UnsupportedLayoutError(f"the model has no {name}")
        if getattr(module, "_compiled_call_impl", None) is not None:
            raise UnsupportedLayoutError(f"{name} is torch.compile'd; it must run eagerly")
        parametrizations = getattr(module, "_simple_fsdp_parametrization_list", None)
        prop = type(module).__dict__.get("weight")
        # SimpleFSDP's property is ``lambda self, i=<index>, pn=<name>: parametrizations[i](self._parameters[pn])``.
        defaults = getattr(getattr(prop, "fget", None), "__defaults__", None)
        if parametrizations is None or not isinstance(prop, property) or not defaults or len(defaults) != 2:
            raise UnsupportedLayoutError(f"{name}.weight is not a SimpleFSDP-parametrized parameter")
        index, param_name = defaults
        if param_name != "weight" or not isinstance(index, int) or not 0 <= index < len(parametrizations):
            raise UnsupportedLayoutError(f"{name}: unexpected SimpleFSDP property layout {defaults!r}")
        rc = parametrizations[index]
        if not isinstance(rc, simple_fsdp.ReplicateComputation):
            raise UnsupportedLayoutError(f"{name}.weight is parametrized by {type(rc).__name__}")
        param = module._parameters.get("weight")
        if not isinstance(param, DTensor) or not param.requires_grad:
            raise UnsupportedLayoutError(f"{name}.weight is not a trainable DTensor")
        if type(param._local_tensor) is not torch.Tensor:
            raise UnsupportedLayoutError(
                f"{name}.weight has a {type(param._local_tensor).__name__} local tensor (quantized?)"
            )
        if rc.mode != "fully_shard" or rc.device_mesh.ndim != 1 or param.device_mesh.ndim != 1:
            raise UnsupportedLayoutError(
                f"{name}: SimpleFSDP mode {rc.mode!r} on a {param.device_mesh.ndim}-D mesh "
                "(only 1-D fully_shard without TP / EP / HSDP is supported)"
            )
        if rc.non_dp_mesh_types is not None:
            raise UnsupportedLayoutError(f"{name}: the spmd_types backend is not supported")
        if not all(isinstance(p, Shard | Replicate) for p in param.placements):
            raise UnsupportedLayoutError(f"{name}: unsupported placements {param.placements}")
        use = _Use(name, parametrizations, index, rc)
        group = groups.get(id(param))
        if group is None:
            groups[id(param)] = _ParamGroup(param, [use])
            continue
        first = group.uses[0].replicate_computation
        if (first.device_mesh, first.param_dtype, first.reduce_dtype) != (
            rc.device_mesh,
            rc.param_dtype,
            rc.reduce_dtype,
        ):
            raise UnsupportedLayoutError(
                f"{group.name} / {name}: a shared Parameter with different SimpleFSDP policies"
            )
        group.uses.append(use)
    return list(groups.values())


def validate_embed_head_accum(model_parts: Any, *, use_simple_fsdp: bool) -> list[str]:
    """Checks at engine start-up that ``engine_kwargs.tpu.embed_head_accum=True`` applies to the model.

    Returns:
        The names of the accumulated parameters (``lm_head=tok_embeddings`` for a tied pair).

    Raises:
        ValueError: without SimpleFSDP, or if the model's layout is not supported.
    """
    if not use_simple_fsdp:
        raise ValueError("engine_kwargs.tpu.embed_head_accum=True needs SimpleFSDP (use_simple_fsdp=True)")
    try:
        groups = plan_embed_head_grad_accumulation(model_parts)
    except UnsupportedLayoutError as e:
        raise ValueError(f"engine_kwargs.tpu.embed_head_accum=True is not supported for this model: {e}") from e
    return [group.name for group in groups]


class EmbedHeadGradAccumulator:
    """``enter()`` gathers once and swaps the parametrizations; ``exit()`` restores them and reduce-scatters once."""

    def __init__(self, model_parts: Any, module_names: Sequence[str] = EMBED_HEAD_MODULE_NAMES):
        self.groups = plan_embed_head_grad_accumulation(model_parts, module_names)
        self.active = False

    @property
    def buffer_bytes(self) -> int:
        """Device memory held during the loop: the gathered weight and the gradient buffer of every group."""
        return sum(g.param.numel() * (g.buffer_dtype.itemsize + g.gather_dtype.itemsize) for g in self.groups)

    def enter(self) -> None:
        from torch.distributed.tensor import DTensor
        from torchtitan.experiments.graph_trainer import simple_fsdp

        if self.active:
            raise RuntimeError("EmbedHeadGradAccumulator.enter() called twice")
        try:
            with torch.no_grad():
                for group in self.groups:
                    full = group.uses[0].replicate_computation.replicate_compute(group.param)
                    if isinstance(full, DTensor) or tuple(full.shape) != tuple(group.param.shape):
                        raise UnsupportedLayoutError(
                            f"{group.name}: the gathered weight is a {type(full).__name__} of shape "
                            f"{tuple(full.shape)}, expected a plain tensor of shape {tuple(group.param.shape)}"
                        )
                    group.full = full
                    group.buffer = torch.zeros(full.shape, dtype=group.buffer_dtype, device=full.device)
                    group.num_grads = 0
            for group in self.groups:
                cached = _CachedReplicateComputation(group, simple_fsdp)
                for use in group.uses:
                    use.parametrizations[use.index] = cached
            self.active = True
        except BaseException:
            self._restore()
            self._release()
            raise

    def _restore(self) -> None:
        for group in self.groups:
            for use in group.uses:
                use.parametrizations[use.index] = use.replicate_computation

    def _release(self) -> None:
        for group in self.groups:
            group.full = group.buffer = None

    def exit(self, reduce: bool, expected_micro_batches: Optional[int] = None) -> dict[str, int]:
        """Restores the parametrizations; with ``reduce``, reduce-scatters every buffer into its parameter's ``.grad``.

        Args:
            reduce: Whether to reduce the accumulated gradients (``False`` drops them, e.g. after an exception).
            expected_micro_batches: If given, every group must have received exactly one gradient per use and
                micro-batch; otherwise this raises before any collective instead of training on a partial sum.

        Returns:
            The number of gradients accumulated per group.
        """
        from torch.distributed.tensor import DTensor

        self._restore()
        self.active = False
        counts = {group.name: group.num_grads for group in self.groups}
        try:
            if not reduce:
                return counts
            for group in self.groups:
                expected = None if expected_micro_batches is None else len(group.uses) * int(expected_micro_batches)
                if group.num_grads == 0 or (expected is not None and group.num_grads != expected):
                    raise RuntimeError(
                        f"{group.name}: {group.num_grads} gradients accumulated, expected "
                        f"{expected if expected is not None else 'at least 1'} ({len(group.uses)} use(s) x "
                        f"{expected_micro_batches} micro-batch(es)); refusing to reduce a partial sum"
                    )
            with torch.no_grad():
                for group in self.groups:
                    assert group.buffer is not None
                    local = group.buffer.to(group.reduce_dtype)
                    partial = DTensor.from_local(
                        local,
                        group.mesh,
                        group.grad_placements,
                        run_check=False,
                        shape=local.shape,
                        stride=local.stride(),
                    )
                    grad = partial.redistribute(group.mesh, group.param.placements).to(group.param.dtype)
                    if group.param.grad is None:
                        group.param.grad = grad
                    else:
                        group.param.grad.add_(grad)
            return counts
        finally:
            self._release()


@contextlib.contextmanager
def accumulate_embed_head_grads(
    model_parts: Any,
    *,
    expected_micro_batches: Optional[int] = None,
    collective_context: Callable[[], ContextManager[Any]] = contextlib.nullcontext,
    synchronize: Optional[Callable[[], None]] = None,
    module_names: Sequence[str] = EMBED_HEAD_MODULE_NAMES,
) -> Iterator[EmbedHeadGradAccumulator]:
    """Runs the body's backward passes with the embedding and LM head gradients accumulated locally.

    On a normal exit the accumulated gradients are reduce-scattered into the parameters' ``.grad``; on an exception
    they are dropped and the parametrizations are restored.

    Args:
        model_parts: The engine's model parts (a single part).
        expected_micro_batches: Number of backward passes the body runs (see :meth:`EmbedHeadGradAccumulator.exit`).
        collective_context: Context in which the all-gathers and reduce-scatters run (e.g. the trainer's
            ``train_context``).
        synchronize: Called after the reduce-scatters are issued (e.g. to flush a deferred-execution queue).
        module_names: Attribute names of the modules whose ``weight`` is accumulated.
    """
    accumulator = EmbedHeadGradAccumulator(model_parts, module_names)
    with collective_context():
        accumulator.enter()
    ok = False
    try:
        yield accumulator
        ok = True
    finally:
        with collective_context():
            accumulator.exit(reduce=ok, expected_micro_batches=expected_micro_batches)
            if ok and synchronize is not None:
                synchronize()
