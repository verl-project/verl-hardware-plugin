# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``engines/embed_head_grad_accum.py`` (``engine_kwargs.tpu.embed_head_accum``).

A tiny decoder is sharded with a minimal stand-in of torchtitan's SimpleFSDP (this repo's CPU workflow does not
install torchtitan) and, when torchtitan is installed, with torchtitan's own SimpleFSDP; torchtitan's TPU Qwen3.5
debug model is checked too. Numerics and collective counts are checked on 2 ``gloo`` ranks with different data per
rank, for untied and tied embeddings.
"""

import ast
import collections
import dataclasses
import importlib.util
import sys
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

VOCAB, DIM, LAYERS = 64, 16, 2
HAS_TORCHTITAN = importlib.util.find_spec("torchtitan") is not None
BACKENDS = [
    "standin",
    pytest.param("torchtitan", marks=pytest.mark.skipif(not HAS_TORCHTITAN, reason="torchtitan is not installed")),
]


# ---------------------------------------------------------------------------------------------------------------
# SimpleFSDP: a minimal stand-in of torchtitan's experiments/graph_trainer/simple_fsdp.py, or torchtitan's own
# ---------------------------------------------------------------------------------------------------------------


def _standin_simple_fsdp_modules() -> dict[str, types.ModuleType]:
    """What ``embed_head_grad_accum`` relies on, as in torchtitan's SimpleFSDP, for ``sys.modules``.

    ``data_parallel`` shards every parameter on dim 0 and replaces ``module.<name>`` by a property calling the
    module's ``ReplicateComputation`` (all-gather in ``param_dtype``, reduce-scatter of the gradient in
    ``reduce_dtype``) on the sharded parameter.
    """
    from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

    simple_fsdp = types.ModuleType("torchtitan.experiments.graph_trainer.simple_fsdp")
    simple_fsdp._active_parametrization = True  # type: ignore[attr-defined]

    @contextmanager
    def disable_active_parametrization():
        try:
            simple_fsdp._active_parametrization = False  # type: ignore[attr-defined]
            yield
        finally:
            simple_fsdp._active_parametrization = True  # type: ignore[attr-defined]

    @dataclasses.dataclass(frozen=True)
    class MixedPrecisionPolicy:
        param_dtype: Optional[torch.dtype] = None
        reduce_dtype: Optional[torch.dtype] = None

    class ReplicateComputation(nn.Module):
        def __init__(self, device_mesh, param_sharding, mode, mp_policy, non_dp_mesh_types=None):
            super().__init__()
            self.device_mesh = device_mesh
            self.param_sharding = param_sharding
            self.mode = mode
            self.compute_placements = [Replicate()] * device_mesh.ndim
            self.grad_placements = [Partial(reduce_op="sum")] * device_mesh.ndim
            mp_policy = mp_policy or MixedPrecisionPolicy()
            self.param_dtype = mp_policy.param_dtype
            self.reduce_dtype = mp_policy.reduce_dtype
            self.non_dp_mesh_types = non_dp_mesh_types

        def replicate_compute(self, x):
            output = x.redistribute(
                placements=self.compute_placements, forward_dtype=self.param_dtype, backward_dtype=self.reduce_dtype
            )
            return output.to_local(grad_placements=self.grad_placements)

        def forward(self, x):
            if not simple_fsdp._active_parametrization:  # type: ignore[attr-defined]
                return x
            return self.replicate_compute(x)

    def data_parallel(model, device_mesh, mode="fully_shard", mp_policy=None):
        for module in list(model.modules()):
            names = [name for name, _ in module.named_parameters(recurse=False)]
            if not names or "SimpleFSDP" in type(module).__name__:
                continue
            for name in names:
                param = module._parameters[name]
                sharded = distribute_tensor(param.detach(), device_mesh, [Shard(0)])
                module.register_parameter(name, nn.Parameter(sharded, requires_grad=param.requires_grad))
            properties = {
                name: property(
                    lambda self, i=i, pn=name: self._simple_fsdp_parametrization_list[i](self._parameters[pn])
                )
                for i, name in enumerate(names)
            }
            module.__class__ = type(f"SimpleFSDP{type(module).__name__}", (type(module),), properties)
            module._simple_fsdp_parametrization_list = [
                ReplicateComputation(device_mesh, (Shard(0),), mode, mp_policy) for _ in names
            ]
        return model

    for name, value in (
        ("disable_active_parametrization", disable_active_parametrization),
        ("MixedPrecisionPolicy", MixedPrecisionPolicy),
        ("ReplicateComputation", ReplicateComputation),
        ("data_parallel", data_parallel),
    ):
        setattr(simple_fsdp, name, value)
    graph_trainer = types.ModuleType("torchtitan.experiments.graph_trainer")
    graph_trainer.simple_fsdp = simple_fsdp  # type: ignore[attr-defined]
    return {graph_trainer.__name__: graph_trainer, simple_fsdp.__name__: simple_fsdp}


def _simple_fsdp(backend: str):
    """The SimpleFSDP module that ``embed_head_grad_accum`` will import (installs the stand-in if asked to)."""
    if backend == "standin":
        sys.modules.update(_standin_simple_fsdp_modules())
    from torchtitan.experiments.graph_trainer import simple_fsdp

    return simple_fsdp


# ---------------------------------------------------------------------------------------------------------------
# Tiny decoder
# ---------------------------------------------------------------------------------------------------------------


class _Embedding(nn.Embedding):
    """Reads ``self.weight`` twice per forward, as torchtitan's ``Embedding`` does; the first read is discarded."""

    def forward(self, tokens):
        from torch.distributed.tensor import DTensor

        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return F.embedding(tokens, weight)


class _TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = _Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList(nn.Linear(DIM, DIM) for _ in range(LAYERS))
        self.lm_head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = h + torch.tanh(layer(h))
        return self.lm_head(h)


def _build(simple_fsdp, mesh, *, tied: bool, param_dtype=None, reduce_dtype=None, seed: int = 0) -> _TinyDecoder:
    torch.manual_seed(seed)
    model = _TinyDecoder()
    policy = simple_fsdp.MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    simple_fsdp.data_parallel(model, mesh, mode="fully_shard", mp_policy=policy)
    if tied:
        # As the engine's tied models: both modules hold one sharded Parameter, each through its own parametrization.
        model.tok_embeddings._parameters["weight"] = model.lm_head._parameters["weight"]
    return model


def _batches(rank: int, n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Micro-batches of different lengths, different on every rank."""
    out = []
    for k in range(n):
        g = torch.Generator().manual_seed(1000 * rank + k)
        tokens = torch.randint(0, VOCAB, (5 + 3 * k,), generator=g)
        out.append((tokens, torch.roll(tokens, -1)))
    return out


def _micro_batch_backward(model, batch, world: int) -> torch.Tensor:
    tokens, labels = batch
    logits = model(tokens)
    loss = F.cross_entropy(logits.float(), labels, reduction="sum") / 64.0 / world
    loss.backward()
    return logits.detach()


def _grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.to_local().clone() for n, p in model.named_parameters() if p.grad is not None}


EMBED_HEAD = ("tok_embeddings.weight", "lm_head.weight")


def _assert_grads_match(ref: dict, got: dict) -> None:
    """Embedding / head gradients equal to rounding; every other gradient bitwise."""
    assert set(ref) == set(got)
    for name, g_ref in ref.items():
        if name in EMBED_HEAD:
            assert g_ref.abs().max() > 0
            assert (got[name] - g_ref).abs().max() <= 1e-6 * g_ref.abs().max(), name
        else:
            assert torch.equal(got[name], g_ref), name


class _CollectiveCounter(TorchDispatchMode):
    """Counts the functional collectives DTensor issues (all-gathers and reduce-scatters)."""

    def __init__(self):
        super().__init__()
        self.counts: collections.Counter = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        for kind in ("all_gather", "reduce_scatter", "all_reduce"):
            if kind in name:
                self.counts[kind] += 1
        return func(*args, **(kwargs or {}))


def _parametrizations(model) -> list[tuple[list, list]]:
    return [
        (m._simple_fsdp_parametrization_list, list(m._simple_fsdp_parametrization_list))
        for m in (model.tok_embeddings, model.lm_head)
    ]


def _restored(snapshot) -> bool:
    return all(all(a is b for a, b in zip(plist, saved, strict=True)) for plist, saved in snapshot)


# ---------------------------------------------------------------------------------------------------------------
# 2-rank tests
# ---------------------------------------------------------------------------------------------------------------


def _spawn(worker, tmp_path, *args) -> None:
    torch.multiprocessing.start_processes(
        worker, args=(2, str(tmp_path / "pg"), *args), nprocs=2, join=True, start_method="spawn"
    )


def _accumulation_worker(rank: int, world: int, init_file: str, backend: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    simple_fsdp = _simple_fsdp(backend)
    from verl_hardware_plugin.accelerators.tpu.engines import embed_head_grad_accum as eh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        n_mb = 3
        for tied in (False, True):
            for param_dtype, reduce_dtype in ((torch.bfloat16, torch.float32), (None, None)):
                kw: dict[str, Any] = dict(tied=tied, param_dtype=param_dtype, reduce_dtype=reduce_dtype)
                ref, model = _build(simple_fsdp, mesh, **kw), _build(simple_fsdp, mesh, **kw)
                batches = _batches(rank, n_mb)
                groups = eh.plan_embed_head_grad_accumulation([model])
                assert [len(g.uses) for g in groups] == ([2] if tied else [1, 1])
                snapshot = _parametrizations(model)

                with _CollectiveCounter() as c_ref:
                    out_ref = [_micro_batch_backward(ref, b, world) for b in batches]
                acc = eh.EmbedHeadGradAccumulator([model])
                with _CollectiveCounter() as c_enter:
                    acc.enter()
                assert not _restored(snapshot)
                with _CollectiveCounter() as c_loop:
                    out = [_micro_batch_backward(model, b, world) for b in batches]
                with _CollectiveCounter() as c_exit:
                    counts = acc.exit(reduce=True, expected_micro_batches=n_mb)
                assert _restored(snapshot)

                # Forward outputs bitwise, gradients as the default path.
                assert all(torch.equal(a, b) for a, b in zip(out_ref, out, strict=True))
                _assert_grads_match(_grads(ref), _grads(model))
                assert counts == {g.name: len(g.uses) * n_mb for g in groups}

                # Per micro-batch, the default path gathers the embedding twice (two reads) and the head once,
                # and reduce-scatters each once; with accumulation, one all-gather and one reduce-scatter per
                # unique parameter per step.
                n_groups = len(groups)
                assert c_enter.counts == {"all_gather": n_groups}
                assert c_exit.counts == {"reduce_scatter": n_groups}
                assert c_ref.counts["all_gather"] - c_loop.counts["all_gather"] == 3 * n_mb
                assert c_ref.counts["reduce_scatter"] - c_loop.counts["reduce_scatter"] == 2 * n_mb

                # Restored: another default backward adds into the same .grad as on the reference.
                extra = _batches(rank + 7, 1)[0]
                _micro_batch_backward(ref, extra, world)
                _micro_batch_backward(model, extra, world)
                _assert_grads_match(_grads(ref), _grads(model))

                # The context manager adds into an existing .grad too.
                with eh.accumulate_embed_head_grads([model], expected_micro_batches=n_mb):
                    for b in batches:
                        _micro_batch_backward(model, b, world)
                for b in batches:
                    _micro_batch_backward(ref, b, world)
                _assert_grads_match(_grads(ref), _grads(model))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", BACKENDS)
def test_accumulation_matches_the_default_path_with_one_collective_per_step(tmp_path, backend):
    """Untied and tied, bf16 / fp32 and fp32 / fp32: same forward, same gradients, fewer collectives, restored."""
    _spawn(_accumulation_worker, tmp_path, backend)


def _failure_worker(rank: int, world: int, init_file: str, backend: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    simple_fsdp = _simple_fsdp(backend)
    from verl_hardware_plugin.accelerators.tpu.engines import embed_head_grad_accum as eh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        batches = _batches(rank, 3)
        for tied in (False, True):
            model = _build(simple_fsdp, mesh, tied=tied, param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            snapshot = _parametrizations(model)
            params = dict(model.named_parameters())

            # An exception in the loop: parametrizations restored, nothing reduced, the embedding / head .grad
            # untouched.
            with _CollectiveCounter() as counter, pytest.raises(KeyboardInterrupt):
                with eh.accumulate_embed_head_grads([model], expected_micro_batches=3):
                    _micro_batch_backward(model, batches[0], world)
                    raise KeyboardInterrupt
            assert _restored(snapshot)
            assert all(params[n].grad is None for n in EMBED_HEAD if n in params)
            assert counter.counts["reduce_scatter"] == 2 * LAYERS  # the layers' only

            # A wrong micro-batch count fails before any collective of the reduction.
            model.zero_grad(set_to_none=True)
            with _CollectiveCounter() as counter, pytest.raises(RuntimeError, match="refusing to reduce a partial sum"):
                with eh.accumulate_embed_head_grads([model], expected_micro_batches=4):
                    for b in batches:
                        _micro_batch_backward(model, b, world)
            assert _restored(snapshot)
            assert all(params[n].grad is None for n in EMBED_HEAD if n in params)
            assert counter.counts["reduce_scatter"] == 3 * 2 * LAYERS
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", BACKENDS)
def test_accumulation_fails_closed(tmp_path, backend):
    _spawn(_failure_worker, tmp_path, backend)


def _engine_worker(rank: int, world: int, init_file: str) -> None:
    import torch.distributed as dist
    from tensordict import TensorDict
    from torch.distributed.device_mesh import init_device_mesh

    simple_fsdp = _simple_fsdp("standin")
    from verl_hardware_plugin.accelerators.tpu.engines import embed_head_grad_accum as eh
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        batches = _batches(rank, 3)
        syncs: list = []
        seen: list = []

        def run(model, *, accumulate: bool, forward_only: bool):
            # A bare engine whose collaborators are replaced below.
            engine: Any = object.__new__(torchtitan_tpu.TorchTitanTPUEngine)
            engine.engine_config = SimpleNamespace(
                tensor_parallel_size=1, context_parallel_size=1, engine_kwargs={}, pad_to_length=False
            )
            engine.parallel_dims = SimpleNamespace(cp_enabled=False)
            engine.get_data_parallel_group = lambda: None
            engine.get_data_parallel_size = lambda: world
            engine.trainer = SimpleNamespace(train_context=nullcontext)
            engine.module = [model]
            engine._use_simple_fsdp, engine._tpu_kernels, engine._tpu_eager_mode = True, frozenset(), None
            engine._simple_fsdp_loss_scale = 1.0 / world
            engine._embed_head_accum = accumulate

            def forward_step(micro_batch, loss_function, forward_only):
                seen.append(type(model.lm_head._simple_fsdp_parametrization_list[0]).__name__)
                tokens, labels = batches[int(micro_batch["index"])]
                logits = model(tokens)
                loss = F.cross_entropy(logits.float(), labels, reduction="sum") / 64.0
                return loss, logits.detach()

            engine.forward_step = forward_step
            micro_batches = [TensorDict({"index": torch.tensor(i)}, batch_size=[]) for i in range(len(batches))]
            with (
                mock.patch.object(torchtitan_tpu, "compute_global_batch_num_tokens", return_value=64),
                mock.patch.object(torchtitan_tpu, "prepare_micro_batches", return_value=(micro_batches, None)),
                mock.patch.object(torchtitan_tpu, "align_micro_batch_shapes_across_ranks"),
                mock.patch.object(
                    torchtitan_tpu, "postprocess_batch_func", side_effect=lambda output_lst, **_: output_lst
                ),
                mock.patch.object(torchtitan_tpu, "tpu_synchronize", side_effect=lambda *a, **k: syncs.append(k)),
            ):
                return engine.forward_backward_batch(TensorDict({}, batch_size=[]), None, forward_only=forward_only)

        policy = dict(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for tied in (False, True):
            ref, model = _build(simple_fsdp, mesh, tied=tied, **policy), _build(simple_fsdp, mesh, tied=tied, **policy)
            snapshot = _parametrizations(model)

            seen.clear()
            syncs.clear()
            out_ref = run(ref, accumulate=False, forward_only=False)
            assert set(seen) == {"ReplicateComputation"}
            n_syncs_default = len(syncs)

            seen.clear()
            syncs.clear()
            out = run(model, accumulate=True, forward_only=False)
            assert set(seen) == {"_CachedReplicateComputation"}
            assert _restored(snapshot)
            # One drain after the reduce-scatters, on top of the per-micro-batch ones.
            assert len(syncs) == n_syncs_default + 1 and syncs[-1] == {"wait": True}
            assert all(torch.equal(a, b) for a, b in zip(out_ref, out, strict=True))
            _assert_grads_match(_grads(ref), _grads(model))

            # Forward-only passes never accumulate.
            seen.clear()
            model.zero_grad(set_to_none=True)
            out_fwd = run(model, accumulate=True, forward_only=True)
            assert set(seen) == {"ReplicateComputation"}
            assert all(p.grad is None for p in model.parameters())
            assert all(torch.equal(a, b) for a, b in zip(out_ref, out_fwd, strict=True))

        # The engine's start-up check.
        model = _build(simple_fsdp, mesh, tied=True)
        assert eh.validate_embed_head_accum([model], use_simple_fsdp=True) == ["lm_head=tok_embeddings"]
        with pytest.raises(ValueError, match="needs SimpleFSDP"):
            eh.validate_embed_head_accum([model], use_simple_fsdp=False)
        with pytest.raises(ValueError, match="pipeline parallelism"):
            eh.validate_embed_head_accum([model, model], use_simple_fsdp=True)
    finally:
        dist.destroy_process_group()


def test_engine_accumulates_only_in_training_passes_when_enabled(tmp_path):
    """``forward_backward_batch``: off is the default path; on accumulates in training passes only, then drains."""
    _spawn(_engine_worker, tmp_path)


def _qwen3_5_worker(rank: int, world: int, init_file: str) -> None:
    import torch.distributed as dist
    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig, derive
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.tpu.overrides.fused_loss import FusedLossLinearWrapper
    from torchtitan.experiments.tpu.qwen3_5 import model_registry

    from verl_hardware_plugin.accelerators.tpu.engines import embed_head_grad_accum as eh
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import get_tpu_simple_fsdp_parallelize_fn

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        parallel_dims = ParallelDims(dp_replicate=1, dp_shard=world, cp=1, tp=1, pp=1, ep=1, world_size=world)
        parallel_dims.build_mesh()
        parallelize_fn = get_tpu_simple_fsdp_parallelize_fn("qwen3_5")
        training = TrainingConfig(max_context_length=64)  # bf16 parameters in compute, fp32 gradient reduction
        parallelism = ParallelismConfig(data_parallel_shard_degree=world, use_simple_fsdp=True)

        def build(tied: bool):
            config = model_registry("debugmodel", seq_len=64).model
            config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = VOCAB
            config.lm_head = derive(config.lm_head, FusedLossLinearWrapper.Config)  # the engine's fused LM head
            config.enable_weight_tying = tied
            with torch.device("meta"):
                model = config.build()
            model = parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=training,
                parallelism=parallelism,
                compile_config=CompileConfig(enable=False),
                ac_config=None,
                dump_folder="",
            )
            model.to_empty(device="cpu")
            torch.manual_seed(0)
            model.init_states(buffer_device=None)
            return model

        def micro_batch_backward(model, k: int):
            g = torch.Generator().manual_seed(100 * rank + k)
            tokens = torch.randint(0, VOCAB, (32,), generator=g)  # one packed row, as the engine feeds it
            positions = torch.cat([torch.arange(12), torch.arange(20)])  # two documents
            h, weight_t = model(tokens, positions=positions)
            logp = torch.log_softmax(h.float() @ weight_t.float(), dim=-1)
            loss = -logp[:-1].gather(-1, tokens[1:, None]).sum() / 64.0 / world
            loss.backward()
            return logp.detach()

        n_mb = 3
        for tied in (False, True):
            ref, model = build(tied), build(tied)
            groups = eh.plan_embed_head_grad_accumulation([model])
            assert [len(g.uses) for g in groups] == ([2] if tied else [1, 1])
            out_ref = [micro_batch_backward(ref, k) for k in range(n_mb)]
            with eh.accumulate_embed_head_grads([model], expected_micro_batches=n_mb):
                out = [micro_batch_backward(model, k) for k in range(n_mb)]
            assert all(torch.equal(a, b) for a, b in zip(out_ref, out, strict=True))
            _assert_grads_match(_grads(ref), _grads(model))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not HAS_TORCHTITAN, reason="torchtitan is not installed")
def test_qwen3_5_with_torchtitan_simple_fsdp_matches_the_default_path(tmp_path):
    """torchtitan's TPU Qwen3.5 (debug flavor, fused LM head), sharded by the engine's SimpleFSDP recipe."""
    _spawn(_qwen3_5_worker, tmp_path)


# ---------------------------------------------------------------------------------------------------------------
# Single-process tests (1-rank gloo group)
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def mesh_1(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        with mock.patch.dict(sys.modules, _standin_simple_fsdp_modules()):
            yield init_device_mesh("cpu", (1,))
    finally:
        if owns_group:
            dist.destroy_process_group()


def test_unsupported_layouts_are_rejected_before_any_collective(mesh_1):
    from torchtitan.experiments.graph_trainer import simple_fsdp

    from verl_hardware_plugin.accelerators.tpu.engines.embed_head_grad_accum import (
        UnsupportedLayoutError,
        plan_embed_head_grad_accumulation,
    )

    def rejected(model_parts, match):
        with pytest.raises(UnsupportedLayoutError, match=match):
            plan_embed_head_grad_accumulation(model_parts)

    rejected([_TinyDecoder()], "not a SimpleFSDP-parametrized parameter")
    model = _build(simple_fsdp, mesh_1, tied=False)
    rejected([model, model], "pipeline parallelism")
    rejected([SimpleNamespace(_orig_mod=model)], "the whole model is torch.compile'd")
    with simple_fsdp.disable_active_parametrization():
        rejected([model], "parametrization is disabled")

    no_head = _build(simple_fsdp, mesh_1, tied=False)
    del no_head.lm_head
    rejected([no_head], "has no lm_head")

    compiled = _build(simple_fsdp, mesh_1, tied=False)
    compiled.lm_head._compiled_call_impl = object()
    rejected([compiled], "lm_head is torch.compile'd")

    frozen = _build(simple_fsdp, mesh_1, tied=False)
    frozen.tok_embeddings._parameters["weight"].requires_grad_(False)
    rejected([frozen], "tok_embeddings.weight is not a trainable DTensor")

    swapped = _build(simple_fsdp, mesh_1, tied=False)
    swapped.lm_head._simple_fsdp_parametrization_list[0] = lambda x: x
    rejected([swapped], "parametrized by function")

    spmd_types = _build(simple_fsdp, mesh_1, tied=False)
    spmd_types.lm_head._simple_fsdp_parametrization_list[0].non_dp_mesh_types = {}
    rejected([spmd_types], "spmd_types backend")

    replicated = _build(simple_fsdp, mesh_1, tied=False)
    replicated.lm_head._simple_fsdp_parametrization_list[0].mode = "replicate"
    rejected([replicated], "mode 'replicate'")

    mixed = _build(simple_fsdp, mesh_1, tied=True, param_dtype=torch.bfloat16)
    mixed.lm_head._simple_fsdp_parametrization_list[0].param_dtype = torch.float32
    rejected([mixed], "different SimpleFSDP policies")


def test_accesses_without_grad_or_parametrization_bypass_the_leaves(mesh_1):
    from torchtitan.experiments.graph_trainer import simple_fsdp

    from verl_hardware_plugin.accelerators.tpu.engines.embed_head_grad_accum import EmbedHeadGradAccumulator

    model = _build(simple_fsdp, mesh_1, tied=False, param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    acc = EmbedHeadGradAccumulator([model])
    assert acc.buffer_bytes == 2 * VOCAB * DIM * (4 + 2)
    acc.enter()
    with pytest.raises(RuntimeError, match="called twice"):
        acc.enter()
    try:
        with torch.no_grad():
            gathered = model.lm_head.weight
        assert gathered is acc.groups[0].full and gathered.dtype == torch.bfloat16 and not gathered.requires_grad
        leaf = model.lm_head.weight
        assert leaf.requires_grad and leaf.is_leaf and leaf.data_ptr() == gathered.data_ptr()
        with simple_fsdp.disable_active_parametrization():
            assert model.lm_head.weight is model.lm_head._parameters["weight"]
    finally:
        assert acc.exit(reduce=False) == {"lm_head": 0, "tok_embeddings": 0}
    assert all(g.full is None and g.buffer is None for g in acc.groups)


def test_no_top_level_torchtitan_import():
    """torchtitan is imported inside functions only, so the plugin loads on machines without it."""
    from verl_hardware_plugin.accelerators.tpu.engines import embed_head_grad_accum

    tree = ast.parse(open(embed_head_grad_accum.__file__).read())
    names = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert not any(name.startswith("torchtitan") for name in names)
