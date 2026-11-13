# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU unit tests of the in-process rollout's small decisions that the other suites rely on without asserting:
the trainer's eager-mode snapshot and its restore after the build, the engine scope's resolution of torch_tpu's API,
the collective counter's scope, the cross-rank agreement without a process group, the GC deferral around the pump,
and the logits-at-load hook (verl's sampling mask installed when the model loads, before anything compiles, and
verified on the served model).
"""

from __future__ import annotations

import gc
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import common, gc_policy
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg

needs_vllm = pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM is not installed")


class _EagerMode:
    DEFER_NEVER = SimpleNamespace(name="DEFER_NEVER")
    DEFER_AND_FUSE = SimpleNamespace(name="DEFER_AND_FUSE")


@pytest.fixture
def fake_tpu(monkeypatch):
    """torch_tpu's public eager-mode API (``torch.tpu``) stand-in, with no snapshot taken yet."""
    seen = []

    class Scope:
        def __init__(self, mode):
            seen.append(mode)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    fake = SimpleNamespace(set_eager_mode=Scope, EagerMode=_EagerMode, eager_mode=_EagerMode.DEFER_NEVER, seen=seen)
    monkeypatch.setattr(torch, "tpu", fake, raising=False)
    monkeypatch.setattr(b, "TRAINER_EAGER_AT_START", None)
    monkeypatch.setattr(b, "TRAINER_EAGER_AT_START_NAME", "n/a")
    monkeypatch.setattr(b, "TRAINER_EAGER_SNAPSHOT_WHERE", "never")
    monkeypatch.setattr(b, "_ENGINE_MODE", None)
    return fake


# ------------------------------------------------------------------------------------------------ eager mode


def test_trainer_eager_snapshot_keeps_the_first_value(fake_tpu):
    assert b.snapshot_trainer_eager_mode("platform import") == "DEFER_NEVER"
    assert b.TRAINER_EAGER_SNAPSHOT_WHERE == "platform import"
    fake_tpu.eager_mode = _EagerMode.DEFER_AND_FUSE  # vLLM's platform activation before the build
    assert b.snapshot_trainer_eager_mode("later") == "DEFER_NEVER"
    assert b.TRAINER_EAGER_AT_START is _EagerMode.DEFER_NEVER and b.TRAINER_EAGER_SNAPSHOT_WHERE == "platform import"


def test_global_eager_mode_restores_the_trainer_snapshot_not_the_mode_at_entry(fake_tpu):
    b.snapshot_trainer_eager_mode("platform import")
    fake_tpu.eager_mode = _EagerMode.DEFER_AND_FUSE  # already the engine platform's mode when the build starts
    with b.GlobalEagerMode() as g:
        pass
    assert g.before == "DEFER_AND_FUSE" and g.restored == "DEFER_NEVER" and g.ok()
    assert g.target_source.startswith("trainer snapshot at platform import")
    assert fake_tpu.eager_mode is _EagerMode.DEFER_NEVER


def test_global_eager_mode_without_a_snapshot_names_the_late_fallback(fake_tpu):
    with b.GlobalEagerMode() as g:
        fake_tpu.eager_mode = _EagerMode.DEFER_AND_FUSE
    assert g.target_source.startswith("late snapshot") and g.restored == "DEFER_NEVER" and g.ok()


def test_eager_guard_restores_a_flipped_mode_and_counts_it(fake_tpu):
    b.snapshot_trainer_eager_mode("platform import")
    guard = b.EagerGuard()
    guard.check("update_actor")
    assert guard.fixed == 0
    fake_tpu.eager_mode = _EagerMode.DEFER_AND_FUSE
    guard.check("update_actor")
    guard.check("compute_log_prob")
    assert guard.fixed == 1 and fake_tpu.eager_mode is _EagerMode.DEFER_NEVER


def test_eager_guard_is_inert_without_a_snapshot(fake_tpu):
    fake_tpu.eager_mode = _EagerMode.DEFER_AND_FUSE
    b.EagerGuard().check("update_actor")
    assert fake_tpu.eager_mode is _EagerMode.DEFER_AND_FUSE


def test_engine_scope_resolves_torch_tpu_once_and_never_caches_its_absence(monkeypatch, fake_tpu):
    monkeypatch.delattr(torch, "tpu")
    with b.engine_mode_ctx("step") as scope:
        assert scope.cm is None and not torch.is_grad_enabled()
    assert b._ENGINE_MODE is None  # a call before torch_tpu exists pins no no-op
    monkeypatch.setattr(torch, "tpu", fake_tpu, raising=False)
    with b.engine_mode_ctx("step"):
        pass
    resolved = b._ENGINE_MODE
    monkeypatch.delattr(torch, "tpu")
    with b.engine_mode_ctx("step"):  # the cached resolution: no lookup on torch.tpu any more
        pass
    assert resolved is not None and fake_tpu.seen == [_EagerMode.DEFER_AND_FUSE, _EagerMode.DEFER_AND_FUSE]


# ------------------------------------------------------------------------------------------------ small invariants


def test_collective_counter_scope_is_reusable_and_never_swallows_an_exception():
    c = b.CollectiveCounter()
    assert c.arm() is c.arm()
    with pytest.raises(ValueError):
        with c.arm():
            assert c.armed
            raise ValueError("x")
    assert not c.armed


def test_agreement_without_a_process_group_is_the_local_flag():
    assert common.all_ranks_ok(True) is True and common.all_ranks_ok(False) is False


def test_gc_deferral_is_reentrant_and_keeps_a_threshold_set_meanwhile(monkeypatch):
    t0 = gc.get_threshold()
    d = gc_policy.GcDefer()
    try:
        gc.set_threshold(700, 10, 10)
        d.enter()
        d.enter()  # re-entrant: the caller's value is saved once
        assert gc.get_threshold() == (700, 10, gc_policy.GC_DEFER_THRESHOLD2) and d.pump_active
        d.exit()
        assert gc.get_threshold() == (700, 10, 10) and not d.pump_active
        d.enter()
        gc.set_threshold(700, 10, 77)  # someone else sets threshold 2 while the pump runs: kept
        d.exit()
        assert gc.get_threshold() == (700, 10, 77)
    finally:
        gc.set_threshold(*t0)


# ------------------------------------------------------------------------------------------------ logits at load


class _Model(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 16, bias=False, device=device)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


def _worker_cls(device="cpu"):
    """vllm-torchtpu's TPUWorker shape: load_model builds the model; verl's extension method sets the mask."""
    from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

    class Worker:
        def __init__(self):
            self.model_runner = SimpleNamespace(model=None)
            self.loads = 0

        def load_model(self):
            self.loads += 1
            self.model_runner.model = _Model(device)

        def monkey_patch_model(self, vocab_size, banned_token_ids=None):
            monkey_patch_compute_logits(self.model_runner.model, vocab_size, banned_token_ids)

    return Worker


@pytest.fixture
def logits(monkeypatch):
    """A fresh hook object in place of the package's one (the engine module holds its own reference)."""
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine

    la = cg.LogitsAtLoad()
    monkeypatch.setattr(cg, "LOGITS_AT_LOAD", la)
    monkeypatch.setattr(engine, "LOGITS_AT_LOAD", la)
    return la


@needs_vllm
def test_logits_hook_is_installed_once_and_inert_until_armed(logits):
    W = _worker_cls()
    assert logits.install(W) and logits.install(W)
    hook = W.__dict__["load_model"]
    assert getattr(hook, "_inproc_logits", False)
    w = W()
    w.load_model()  # not armed: nothing applied
    assert logits.applied is None and "compute_logits" not in w.model_runner.model.__dict__
    assert w.loads == 1


@needs_vllm
def test_logits_mask_is_applied_at_load_and_freezes_the_rpc(logits):
    W = _worker_cls()
    logits.install(W)
    logits.arm({"vocab_size": 12, "banned_token_ids": [3, 5]})
    w = W()
    try:
        w.load_model()
    finally:
        logits.disarm()
    logits.verify_built(w)
    model = w.model_runner.model
    assert logits.applied["via"] == "the worker's monkey_patch_model" and logits.applied["rpc_frozen"] == "frozen"
    out = model.compute_logits(torch.ones(2, 4))
    assert torch.isinf(out[:, 12:]).all() and torch.isinf(out[:, [3, 5]]).all() and not torch.isinf(out[:, 7]).any()
    assert getattr(w.monkey_patch_model, "_inproc_frozen", False)


@needs_vllm
def test_logits_verify_refuses_when_the_mask_never_landed(logits):
    W = _worker_cls()
    logits.install(W)
    w = W()
    w.load_model()  # loaded while not armed
    with pytest.raises(RuntimeError, match="refusing to serve") as err:
        logits.verify_built(w)
    assert "never applied" in str(err.value)


@needs_vllm
def test_logits_verify_refuses_another_model_object(logits):
    W = _worker_cls()
    logits.install(W)
    logits.arm({"vocab_size": 12, "banned_token_ids": []})
    w = W()
    w.load_model()
    logits.disarm()
    w.model_runner.model = _Model()  # the runner serves another object than the one patched at load
    with pytest.raises(RuntimeError, match="not the object patched at load"):
        logits.verify_built(w)


@needs_vllm
def test_logits_banned_ids_move_to_the_model_device_before_anything_compiles(logits):
    from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

    model = _Model(device="meta")
    monkey_patch_compute_logits(model, 12, [3, 5])
    cl = model.__dict__["compute_logits"]
    assert logits._place_banned(cl, model) == "cpu -> meta"
    assert logits._place_banned(cl, model) == "already on meta"
    plain = _Model()
    monkey_patch_compute_logits(plain, 12, [])
    assert logits._place_banned(plain.__dict__["compute_logits"], plain).startswith("none")


def _engine_args_stub(monkeypatch):
    """vLLM's EngineArgs resolving to a TP=1 config without a model (the factory's checks only read these)."""
    from vllm.engine.arg_utils import EngineArgs

    one = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)
    vc = SimpleNamespace(parallel_config=one)
    args = SimpleNamespace(create_engine_config=lambda usage_context=None: vc, disable_log_stats=True)
    monkeypatch.setattr(EngineArgs, "from_cli_args", classmethod(lambda cls, cli: args))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return SimpleNamespace(model_config=SimpleNamespace(tokenizer=[0] * 20, processor=None))


@needs_vllm
def test_build_arms_the_mask_before_the_engine_and_disarms_it_after(logits, monkeypatch):
    """The in-process engine factory arms verl's mask (the arguments of verl's monkey_patch_model RPC) before vLLM's
    LLMEngine is constructed, so the hook applies it inside the worker's load_model, and disarms it in a finally."""
    import vllm.v1.engine.llm_engine as llm_engine
    import vllm.v1.executor as vexec

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine

    seen = {}

    class Stop(Exception):
        pass

    def fake_llm_engine(*a, **k):
        seen["spec_at_construction"] = dict(logits.spec or {})
        raise Stop

    srv = _engine_args_stub(monkeypatch)
    monkeypatch.setattr(vexec.Executor, "get_class", staticmethod(lambda vllm_config: object))
    monkeypatch.setattr(llm_engine, "LLMEngine", fake_llm_engine)
    logits.hooked = True
    with pytest.raises(Stop):
        engine.build_vllm_engine(srv, SimpleNamespace())
    assert seen["spec_at_construction"] == {"vocab_size": 20, "banned_token_ids": []}
    assert logits.spec is None


@needs_vllm
def test_build_refuses_without_the_load_model_hook(logits, monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine

    srv = _engine_args_stub(monkeypatch)
    assert not logits.hooked
    with pytest.raises(RuntimeError, match="hook is not installed"):
        engine.build_vllm_engine(srv, SimpleNamespace())
    assert logits.spec is None
