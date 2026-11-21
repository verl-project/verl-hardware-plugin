# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the engine boundary (``rollout/inproc/boundary.py``): every engine call normalizes the calling thread's
state the engine's compiled programs are guarded on, and puts the caller's back afterwards."""

import contextlib
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b


@pytest.fixture(autouse=True)
def _fresh_drift(monkeypatch):
    monkeypatch.setattr(b, "GS_DRIFT", b.GlobalStateDrift())


def test_grad_mode_is_off_inside_and_restored():
    torch.set_grad_enabled(True)
    with b.EngineScope("t"):
        assert not torch.is_grad_enabled()
        assert b.ENGINE_TLS.depth == 1 and b.ENGINE_TLS.kind == "t"
        with b.EngineScope("inner"):
            assert b.ENGINE_TLS.depth == 2 and not torch.is_grad_enabled()
        assert b.ENGINE_TLS.kind == "t"
    assert torch.is_grad_enabled() and b.ENGINE_TLS.depth == 0
    with torch.no_grad():
        with b.EngineScope("t"):
            assert not torch.is_grad_enabled()
        assert not torch.is_grad_enabled()


def test_state_is_restored_when_the_engine_call_raises():
    torch.set_grad_enabled(True)
    with pytest.raises(ValueError):
        with b.EngineScope("t"):
            raise ValueError("engine failure")
    assert torch.is_grad_enabled() and b.ENGINE_TLS.depth == 0


def test_autocast_is_disabled_inside():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with b.EngineScope("t"):
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")


def test_torch_function_modes_are_popped_at_the_outer_entry():
    class Mode(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Mode():
        assert torch._C._len_torch_function_stack() == 1
        with b.EngineScope("t"):
            assert torch._C._len_torch_function_stack() == 0
        assert torch._C._len_torch_function_stack() == 1


def test_inference_mode_is_off_inside():
    with torch.inference_mode():
        assert torch.is_inference_mode_enabled()
        with b.EngineScope("t"):
            assert not torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()


def test_a_drifted_global_setting_is_restored_for_the_call():
    n = torch.get_num_threads()
    b.snapshot_engine_global_state()
    assert b.GS_DRIFT.snap is not None and b.GS_DRIFT.warm_values["num_threads"] == n
    torch.set_num_threads(max(1, n - 1) if n > 1 else 2)
    drifted = torch.get_num_threads()
    try:
        with b.EngineScope("t"):
            assert torch.get_num_threads() == n
        assert torch.get_num_threads() == drifted
    finally:
        torch.set_num_threads(n)


def test_the_eager_scope_comes_from_torch_tpu(monkeypatch):
    seen = []

    class EagerMode:
        DEFER_AND_FUSE = "DEFER_AND_FUSE"

    @contextlib.contextmanager
    def set_eager_mode(mode):
        seen.append(("enter", mode))
        yield
        seen.append(("exit", mode))

    fake = SimpleNamespace(set_eager_mode=set_eager_mode, EagerMode=EagerMode, eager_mode="EAGER")
    monkeypatch.setattr(b, "_execution_mode", lambda: fake)
    monkeypatch.setattr(b, "_ENGINE_MODE", None)
    with b.engine_mode_ctx("t"):
        assert seen == [("enter", "DEFER_AND_FUSE")]
    assert seen[-1] == ("exit", "DEFER_AND_FUSE")
    seen.clear()
    with b.engine_grad_ctx("build"):  # the build runs under the engine platform's own global mode
        assert seen == []


def test_global_eager_mode_is_put_back_after_the_build(monkeypatch):
    fake = SimpleNamespace(eager_mode=SimpleNamespace(name="EAGER"))
    monkeypatch.setattr(b, "_execution_mode", lambda: fake)
    monkeypatch.setattr(b, "TRAINER_EAGER_AT_START", None)
    assert b.snapshot_trainer_eager_mode("platform import") == "EAGER"
    trainer = fake.eager_mode
    with b.GlobalEagerMode() as g:
        fake.eager_mode = SimpleNamespace(name="DEFER_AND_FUSE")  # the engine platform's activation
    assert fake.eager_mode is trainer and g.ok() and g.after_build == "DEFER_AND_FUSE" and g.restored == "EAGER"
    fake.eager_mode = SimpleNamespace(name="DEFER_AND_FUSE")  # a later flip: the trainer's next entry puts it back
    b.EagerGuard().check("update_actor")
    assert fake.eager_mode is trainer


def test_identity_variables_are_restored(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    with b.IdentityEnv() as ienv:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["SOMETHING_ELSE"] = "x"
    try:
        assert os.environ["RANK"] == "3" and "LOCAL_WORLD_SIZE" not in os.environ and ienv.restored()
        assert ienv.changed == {"RANK": ("3", "0"), "LOCAL_WORLD_SIZE": (None, "1")} and ienv.other_changed == [
            "SOMETHING_ELSE"
        ]
    finally:
        os.environ.pop("SOMETHING_ELSE", None)


def test_device_threads():
    d = b.DeviceThreads()
    d.record("step")
    d.record("install")
    assert d.single()
    t = threading.Thread(target=lambda: d.record("step"))
    t.start()
    t.join()
    assert not d.single()


def test_collective_counter_counts_world_group_calls_while_armed():
    c = b.CollectiveCounter()
    calls = []
    wrapped = c._wrap("all_reduce", lambda self_, x: calls.append(x) or x)
    big, small = SimpleNamespace(world_size=8, unique_name="world"), SimpleNamespace(world_size=1)
    wrapped(big, 1)
    assert not c.big and c.small == 0
    with c.arm():
        wrapped(big, 2)
        wrapped(small, 3)
        with c.arm():
            pass
        assert c.armed
    assert not c.armed and c.big == {"world.all_reduce": 1} and c.small == 1 and calls == [1, 2, 3]


def test_engine_settings_that_leak_into_the_trainer_are_refused():
    assert b.full_determinism_problem({}, {}) is None
    assert b.full_determinism_problem({}, {"VERL_FULL_DETERMINISM": "0"}) is None
    assert "refusing" in b.full_determinism_problem({}, {"VERL_FULL_DETERMINISM": "1"})
    assert "refusing" in b.full_determinism_problem({"full_determinism": True}, {})
    assert b.prefix_caching_problem({"enable_prefix_caching": False}) is None
    assert "refusing" in b.prefix_caching_problem({"enable_prefix_caching": True})
