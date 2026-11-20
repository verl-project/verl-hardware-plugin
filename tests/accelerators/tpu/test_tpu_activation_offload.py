# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the saved-activation host offload (``engines/activation_offload.py``) and its engine wiring.

The real offload policy runs with ``device_types=("cpu",)`` so CPU tensors take the device path.
"""

import os
import tempfile
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from verl_hardware_plugin.accelerators.tpu.engines import activation_offload, tpu_utils
from verl_hardware_plugin.accelerators.tpu.engines.activation_offload import HostActivationOffload, _Packed


def _offload(model: nn.Module | None = None, **kwargs) -> HostActivationOffload:
    kwargs.setdefault("min_bytes", 0)
    kwargs.setdefault("device_types", ("cpu",))
    return HostActivationOffload(lambda: model.parameters() if model is not None else [], **kwargs)


class _Spy:
    """Records every (saved tensor, packed result) pair of ``offload.pack``."""

    def __init__(self, offload: HostActivationOffload):
        self.calls: list[tuple[torch.Tensor, object]] = []
        orig = offload.pack

        def pack(t):
            out = orig(t)
            self.calls.append((t, out))
            return out

        offload.pack = pack  # type: ignore[method-assign]

    def packed(self) -> list[_Packed]:
        return [o for _, o in self.calls if isinstance(o, _Packed)]


def _grads(model: nn.Module, fn, offload: HostActivationOffload | None = None) -> list[torch.Tensor]:
    with offload.hooks() if offload is not None else torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        out = fn()
    out.square().sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    if offload is not None:
        offload.end_micro_batch()
    return grads


def _mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 32))


def test_round_trip_matches_and_never_offloads_parameters():
    model = _mlp()
    x = torch.randn(16, 32)
    ref = _grads(model, lambda: torch.tanh(model(x)) * x)
    offload = _offload(model)
    spy = _Spy(offload)
    got = _grads(model, lambda: torch.tanh(model(x)) * x, offload)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r)
    assert spy.packed(), "activations should have been offloaded"
    params = list(model.parameters())

    def aliases_param(t):
        base = t
        while base is not None:
            if any(base is p for p in params):
                return True
            base = base._base
        return False

    param_saves = [(t, o) for t, o in spy.calls if aliases_param(t)]
    assert param_saves, "nn.Linear saves its weight (a view of it), so the param check must be exercised"
    assert all(o is t for t, o in param_saves)


def test_parameters_are_reread_every_micro_batch():
    model = _mlp()
    offload = _offload(model)
    with offload.hooks():
        pass
    old = model[0].weight
    model[0].weight = nn.Parameter(old.detach().clone())  # e.g. a param re-materialized by onload / resharding
    spy = _Spy(offload)
    _grads(model, lambda: model(torch.randn(4, 32)), offload)
    assert all(o is t for t, o in spy.calls if t._base is model[0].weight or t is model[0].weight)


def test_policy_skips_small_other_device_and_meta_tensors():
    offload = _offload(min_bytes=4096)
    small = torch.ones(16)
    assert offload.pack(small) is small
    big = torch.ones(4096)
    assert isinstance(offload.pack(big), _Packed)
    tpu_only = HostActivationOffload(list, min_bytes=0)  # engine default: device_types=("tpu",)
    assert tpu_only.pack(big) is big
    meta = torch.empty(4096, device="meta")
    assert offload.pack(meta) is meta
    assert offload.unpack(small) is small


def test_repeated_saves_are_deduplicated_and_share_one_device_copy():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(256))
    x = torch.randn(256)

    def fwd():
        h = x * w  # one tensor saved by several nodes
        return h * h + h.sin() + h.exp() + h.cos()

    ref = _grads(nn.ParameterList([w]), fwd)
    offload = _offload(nn.ParameterList([w]))
    spy = _Spy(offload)
    unpacked = []
    orig_unpack = offload.unpack

    def unpack(p):
        out = orig_unpack(p)
        if isinstance(p, _Packed):
            unpacked.append((p, out))
        return out

    offload.unpack = unpack  # type: ignore[method-assign]
    with offload.hooks():
        out = fwd()
    records = {id(p) for p in spy.packed()}
    assert offload.dedup_hits == 3  # h is saved 4 times (mul twice, sin, cos), exp saves its output
    assert offload.micro_batch_bytes == len(records) * 256 * 4  # one host copy per distinct tensor
    out.square().sum().backward()
    torch.testing.assert_close(w.grad, ref[0])
    by_record: dict[int, set[int]] = {}
    for p, t in unpacked:
        by_record.setdefault(id(p), set()).add(id(t))
    assert any(len(ts) == 1 and sum(1 for q, _ in unpacked if id(q) == rid) > 1 for rid, ts in by_record.items())
    offload.end_micro_batch()


def test_view_with_same_storage_is_deduplicated():
    offload = _offload()
    base = torch.randn(64, 64)
    a = offload.pack(base.view(64, 64))
    b = offload.pack(base.view(64, 64))
    assert a is b and a.refs == 2
    base.add_(1)  # a new version is a new save
    c = offload.pack(base.view(64, 64))
    assert c is not a


def test_unpack_prefetches_previous_records():
    offload = _offload(prefetch=2)
    ts = [torch.full((128,), float(i)) for i in range(4)]
    packed = [offload.pack(t) for t in ts]
    assert all(p.resident is None for p in packed)
    out = offload.unpack(packed[3])
    torch.testing.assert_close(out, ts[3])
    assert packed[3].resident is None  # its only save was unpacked
    assert packed[2].resident is not None and packed[1].resident is not None and packed[0].resident is None
    prefetched = packed[2].resident
    assert offload.unpack(packed[2]) is prefetched


def test_pool_reuses_tracks_shapes_caps_and_releases():
    allocs = []
    real = tpu_utils.alloc_cpu_staging

    def counting(shape, dtype):
        allocs.append(shape)
        return real(shape, dtype)

    with mock.patch.object(activation_offload, "alloc_cpu_staging", side_effect=counting):
        offload = _offload()
        for _ in range(2):
            offload.pack(torch.ones(1024))
            offload.pack(torch.ones(2048))
            offload.end_micro_batch()
        assert len(allocs) == 2  # the second micro-batch reused both buffers
        assert offload.stats()["pool_free_gib"] * (1 << 30) == 3072 * 4
        offload.pack(torch.ones(512))  # new shapes: unused buffers are dropped at the end of the micro-batch
        offload.end_micro_batch()
        assert offload.stats()["pool_free_gib"] * (1 << 30) == 512 * 4
        offload.release()
        assert offload.stats()["pool_free_gib"] == 0

        capped = _offload(host_cache_bytes=5000)
        capped.pack(torch.ones(1024))
        capped.pack(torch.ones(1024))
        capped.end_micro_batch()
        assert capped.stats()["pool_free_gib"] * (1 << 30) == 4096

        failed = _offload()
        failed.pack(torch.ones(1024))
        failed.end_micro_batch(reuse_buffers=False)
        assert failed.stats()["pool_free_gib"] == 0


def _sac_policy(ctx, func, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if func == torch.ops.aten.tanh.default else CheckpointPolicy.PREFER_RECOMPUTE


def _sac_block() -> nn.Module:
    torch.manual_seed(0)
    block = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64))
    return checkpoint_wrapper(block, context_fn=lambda: create_selective_checkpoint_contexts(_sac_policy))


def test_selective_ac_saves_reach_the_hooks_after_routing():
    x = torch.randn(32, 64, requires_grad=True)
    model = _sac_block()
    ref = _grads(model, lambda: model(x))

    offload = _offload(model)
    spy = _Spy(offload)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _grads(model, lambda: model(x), offload)
    assert len(spy.packed()) == 1  # without routing: only the checkpoint input

    assert activation_offload.route_sac_saves_through_hooks([model]) == 1
    assert activation_offload.route_sac_saves_through_hooks([model]) == 0  # idempotent
    spy = _Spy(offload)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        got = _grads(model, lambda: model(x), offload)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r)
    shapes = sorted(tuple(p.host.shape) for p in spy.packed())
    assert shapes == [(32, 64), (32, 64)]  # the checkpoint input and the SAC-saved tanh output


def test_full_ac_block_input_is_offloaded():
    torch.manual_seed(0)
    model = checkpoint_wrapper(nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64)))
    x = torch.randn(32, 64, requires_grad=True)
    ref = _grads(model, lambda: model(x))
    offload = _offload(model)
    spy = _Spy(offload)
    got = _grads(model, lambda: model(x), offload)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r)
    assert [p.host.shape for p in spy.packed()] == [torch.Size([32, 64])]


def test_compiled_selective_ac_block_offloads_without_recompiling():
    import torch._dynamo

    torch._dynamo.reset()
    model = _sac_block()
    compiled = torch.compile(model, backend="aot_eager")
    x = torch.randn(32, 64, requires_grad=True)
    ref = _grads(model, lambda: model(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _grads(model, lambda: compiled(x))  # compile with grad, no offload
        frames = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        offload = _offload(model)
        spy = _Spy(offload)
        got = _grads(model, lambda: compiled(x), offload)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == frames, "offload hooks caused a recompile"
    shapes = sorted(tuple(p.host.shape) for p in spy.packed())
    assert shapes.count((32, 64)) == 2  # input + tanh output of the compiled block; its weights are skipped
    torch._dynamo.reset()


@pytest.fixture
def _single_rank_group():
    import torch.distributed as dist

    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group("gloo", init_method=f"file://{os.path.join(tmp, 'pg')}", rank=0, world_size=1)
        try:
            yield
        finally:
            dist.destroy_process_group()


def test_dtensor_is_offloaded_and_rewrapped(_single_rank_group):
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor, init_device_mesh

    mesh = init_device_mesh("cpu", (1,))
    dt = distribute_tensor(torch.randn(64, 8), mesh, [Shard(0)])
    offload = _offload()
    packed = offload.pack(dt)
    assert isinstance(packed, _Packed)
    out = offload.unpack(packed)
    assert isinstance(out, DTensor) and out.placements == dt.placements
    torch.testing.assert_close(out.to_local(), dt.to_local())


# ----- engine wiring ---------------------------------------------------------------------------------------------


def _engine(model, offload, forward_only=False, enable=True, options=None):
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    engine = object.__new__(TorchTitanTPUEngine)
    engine.module = [model]
    engine.parallel_dims = SimpleNamespace(pp_enabled=False)
    engine.engine_config = SimpleNamespace(forward_only=forward_only)
    engine.model_config = SimpleNamespace(enable_activation_offload=enable)
    engine._tpu_options = options
    engine._activation_offload = offload
    return engine


def test_model_forward_step_offloads_only_with_grad_and_matches():
    model = _mlp()
    x = torch.randn(8, 32)
    off = _engine(model, None)
    ref = _grads(model, lambda: off.model_forward_step(inputs=x))
    offload = _offload(model)
    spy = _Spy(offload)
    on = _engine(model, offload)
    with torch.no_grad():
        on.model_forward_step(inputs=x)
    assert spy.calls == []  # forward-only passes do not enter the hooks
    out = on.model_forward_step(inputs=x)
    assert spy.packed()
    out.square().sum().backward()
    for p, r in zip(model.parameters(), ref, strict=True):
        torch.testing.assert_close(p.grad, r)
    offload.end_micro_batch()


def test_build_activation_offload_reads_verl_flag_and_tpu_options():
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu
    from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options

    model = _sac_block()
    assert _engine(model, None, enable=False)._build_activation_offload() is None
    assert _engine(model, None, forward_only=True)._build_activation_offload() is None
    with mock.patch.object(torchtitan_tpu, "get_device_name", return_value="cpu"):
        assert _engine(model, None)._build_activation_offload() is None

    options = get_tpu_engine_options(
        SimpleNamespace(
            engine_kwargs={
                "tpu": {
                    "activation_offload_min_mb": 2,
                    "activation_offload_host_cache_gb": 0.5,
                    "activation_offload_prefetch": 3,
                }
            }
        )
    )
    with mock.patch.object(torchtitan_tpu, "get_device_name", return_value="tpu"):
        offload = _engine(model, None, options=options)._build_activation_offload()
    assert offload is not None
    assert offload.min_bytes == 2 << 20 and offload.prefetch == 3 and offload.device_types == {"tpu"}
    assert offload._pool.cap_bytes == 1 << 29
    assert "respect_saved_tensors_hooks" in model.checkpoint_fn.keywords

    with pytest.raises(ValueError, match="activation_offload_prefetch"):
        get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"activation_offload_prefetch": 0}}))
