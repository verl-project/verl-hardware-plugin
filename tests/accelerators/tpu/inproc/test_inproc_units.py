# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU unit tests of the in-process rollout's small decisions that the other suites rely on without asserting:
the trainer's eager-mode snapshot and its restore after the build, the engine scope's resolution of torch_tpu's API,
the collective counter's scope, the cross-rank agreement without a process group, and the GC deferral around the
pump.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import common, gc_policy


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
