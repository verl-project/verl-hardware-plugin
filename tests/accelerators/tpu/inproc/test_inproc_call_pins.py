# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the call-site grad pins and the GC-time grad watch (``rollout/inproc/compile_guard.py``).

vllm-torchtpu compiles its runner's functions for a grid of shapes at build time; dynamo guards each program on the
calling thread's grad mode, and in process that thread is the trainer's. A grad-mode flip inside an engine step (a
finalizer run by a garbage collection, a stray ``torch.set_grad_enabled(True)``) would make every compiled function
called after it recompile. :class:`CallPins` sets grad mode off at the call of every compiled engine function;
:class:`GradFlipWatch` puts it back off when a collection inside an engine scope flipped it.
"""

from __future__ import annotations

import gc
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "support"))

import inproc_vt_runner as vt  # noqa: E402

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compat import RUNNER_MOD  # noqa: E402

DISPATCH_MOD, DISPATCH_FN = cg.PIN_DISPATCHERS[0]


def _engine_file(tmp_path, name: str) -> str:
    """a path under an engine package (the guards key engine code by its path; nothing is written there)"""
    return str(tmp_path / "standin" / "vllm_torchtpu" / name)


def _compiled(name: str, path: str, body: str = "return x * 2 + 1", line: int = 10):
    ns: dict = {"torch": torch}
    exec(compile("\n" * line + f"def {name}(x, *args, **kwargs):\n    {body}\n", path, "exec"), ns)
    return torch.compile(ns[name], backend="eager", fullgraph=True, dynamic=False)


def _plain(name: str, path: str, body: str, line: int = 10):
    ns: dict = {"torch": torch}
    exec(compile("\n" * line + f"def {name}(*args, **kwargs):\n    {body}\n", path, "exec"), ns)
    return ns[name]


@pytest.fixture
def pins(monkeypatch, tmp_path):
    """A fresh pin set and grad watch, and a fake vllm-torchtpu module tree in sys.modules (all put back after)."""
    p, w = cg.CallPins(), cg.GradFlipWatch()
    monkeypatch.setattr(cg, "CALL_PINS", p)
    monkeypatch.setattr(cg, "GRAD_WATCH", w)
    for name in [n for n in sys.modules if n == "vllm_torchtpu" or n.startswith("vllm_torchtpu.")]:
        monkeypatch.delitem(sys.modules, name)

    def module(name: str, file: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__file__ = file
        monkeypatch.setitem(sys.modules, name, m)
        return m

    yield types.SimpleNamespace(pins=p, watch=w, module=module, tmp=tmp_path)
    if w.installed and w._cb in gc.callbacks:
        gc.callbacks.remove(w._cb)


def _runner(env, compiled: bool = True):
    """vllm-torchtpu's runner module with the three functions the build requires pinned, plus an alias of the kernel
    in another engine module and the mamba-row dispatcher imported by name."""
    path = _engine_file(env.tmp, "runner/tpu_runner.py")
    for pkg in ("vllm_torchtpu", "vllm_torchtpu.runner", "vllm_torchtpu.layers"):
        env.module(pkg, path).__path__ = []
    runner = env.module(RUNNER_MOD, path)
    make = _compiled if compiled else (lambda n, p, line=10: _plain(n, p, "return args", line))
    runner._substitute_placeholder_token = make("_substitute_placeholder_token", path, line=30)
    runner.TPUModelRunner = type(
        "TPUModelRunner",
        (),
        {
            "__module__": RUNNER_MOD,
            "sample_from_logits": make("sample_from_logits", path, line=40),
            "gather_logprobs": make("gather_logprobs", path, line=50),
            "not_compiled": _plain("not_compiled", path, "return 1", line=60),
        },
    )
    return runner


def test_the_scan_pins_compiled_engine_functions_only(pins):
    runner = _runner(pins)
    other = pins.module("vllm_torchtpu.layers.ops", _engine_file(pins.tmp, "layers/ops.py"))
    other.kernel_alias = runner._substitute_placeholder_token  # the same compiled object under another engine module
    other.helper = staticmethod(_compiled("helper", other.__file__, line=70))
    other.Ops = type(
        "Ops", (), {"__module__": other.__name__, "s": staticmethod(_compiled("s", other.__file__, line=80))}
    )
    disp = pins.module(DISPATCH_MOD, _engine_file(pins.tmp, "layers/adapter/custom_ops/mamba_state_copy_op.py"))
    setattr(disp, DISPATCH_FN, _plain(DISPATCH_FN, disp.__file__, "return torch.is_grad_enabled()"))
    setattr(runner, DISPATCH_FN, getattr(disp, DISPATCH_FN))  # the runner imports the dispatcher by name
    outside = pins.module("not_the_engine", "/somewhere/else/not_the_engine.py")
    outside.f = _compiled("f", outside.__file__)
    originals = {"kernel": runner._substitute_placeholder_token, "outside": outside.f}

    scan = pins.pins.install()
    assert scan["errors"] == 0 and pins.pins.required_missing() == []
    kern = runner._substitute_placeholder_token
    assert kern is not originals["kernel"] and getattr(kern, "_inproc_pin", None)
    assert other.kernel_alias is kern  # one wrapper per compiled function, its aliases included
    assert getattr(vars(other.Ops)["s"].__func__, "_inproc_pin", None)
    assert getattr(vars(runner.TPUModelRunner)["sample_from_logits"], "_inproc_pin", None)
    assert not hasattr(vars(runner.TPUModelRunner)["not_compiled"], "_inproc_pin")
    assert outside.f is originals["outside"]  # not engine code: never pinned
    d = getattr(disp, DISPATCH_FN)
    assert (
        getattr(runner, DISPATCH_FN) is d
        and d._inproc_pin.startswith("dispatch:")
        and d._inproc_pin in pins.pins.dispatch
    )
    again = pins.pins.install()  # idempotent
    assert again["new_sites"] == 0 and runner._substitute_placeholder_token is kern


def test_a_pinned_call_runs_with_grad_off_and_puts_the_thread_mode_back(pins):
    runner = _runner(pins)
    pins.pins.install()
    kern = runner._substitute_placeholder_token
    code = vt.compiled_code(kern)
    x = torch.ones(4)
    torch.set_grad_enabled(True)
    try:
        y_on = kern(x)
        assert torch.is_grad_enabled()  # the caller's mode back after the call
        with torch.no_grad():
            y_off = kern(x)
            assert not torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        kern(x)
    finally:
        torch.set_grad_enabled(True)
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    assert torch.equal(y_on, y_off) and torch.equal(y_on, x * 2 + 1)
    assert len(_debug_get_cache_entry_list(code)) == 1  # one program for both caller modes
    assert pins.pins.keys[kern._inproc_pin] == [1, 2]  # calls with the thread's grad [off, on]


def test_the_thread_mode_is_put_back_when_a_pinned_call_raises(pins):
    _runner(pins)
    disp = pins.module(DISPATCH_MOD, _engine_file(pins.tmp, "layers/adapter/custom_ops/mamba_state_copy_op.py"))
    setattr(disp, DISPATCH_FN, _plain(DISPATCH_FN, disp.__file__, "raise ValueError(torch.is_grad_enabled())"))
    pins.pins.install()
    torch.set_grad_enabled(True)
    with pytest.raises(ValueError) as err:
        getattr(disp, DISPATCH_FN)()
    assert err.value.args == (False,) and torch.is_grad_enabled()


def test_a_pin_is_transparent_inside_a_compiled_region(pins):
    runner = _runner(pins)
    pins.pins.install()
    kern = runner._substitute_placeholder_token
    outer = torch.compile(lambda x: kern(x) * 3, backend="eager", fullgraph=True, dynamic=False)
    torch.set_grad_enabled(True)
    assert torch.equal(outer(torch.ones(3)), (torch.ones(3) * 2 + 1) * 3)


def test_the_pinned_wrapper_keeps_the_compiled_functions_identity(pins):
    runner = _runner(pins)
    orig = runner._substitute_placeholder_token
    pins.pins.install()
    w = runner._substitute_placeholder_token
    assert w.__name__ == orig.__name__ and w.__qualname__ == orig.__qualname__ and w.__module__ == orig.__module__
    assert w._torchdynamo_orig_callable is orig._torchdynamo_orig_callable  # introspection of the compiled function
    assert not hasattr(w, "__wrapped__")  # an unwrapper keeps the pin
    assert w._inproc_pinned is orig


def test_the_build_refuses_a_runner_whose_required_functions_are_not_pinned(pins):
    _runner(pins, compiled=False)
    with pytest.raises(RuntimeError, match="call-site grad pin does not cover") as err:
        cg.install_call_pins_and_watch()
    assert "refusing to build without it" in str(err.value)
    for name in cg.PIN_REQUIRED:
        assert name in str(err.value)


def test_the_pins_freeze_once_the_runner_is_loaded(pins):
    cg.install_call_pins_and_watch()  # no runner module yet: nothing required, nothing frozen
    assert not pins.pins.frozen and pins.watch.installed
    _runner(pins)
    cg.install_call_pins_and_watch()
    assert pins.pins.frozen and pins.pins.required_missing() == []


# ------------------------------------------------------------------------------------------------ the GC-time watch


def test_a_gc_flip_inside_an_engine_scope_is_put_back(pins):
    pins.watch.install()
    gc.collect()
    torch.set_grad_enabled(True)
    vt.stale_grad_garbage()
    with b.EngineScope("step"):
        assert not torch.is_grad_enabled()
        gc.collect()  # the finalizer sets grad on in the middle of the engine call; the watch puts it back
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()


def test_without_the_watch_the_flip_survives_the_collection(pins):
    """The control: the stale-grad garbage does flip grad mode on inside an engine call."""
    gc.collect()
    torch.set_grad_enabled(True)
    vt.stale_grad_garbage()
    try:
        with b.EngineScope("step"):
            gc.collect()
            flipped = torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(True)
    assert flipped


def test_outside_engine_scopes_the_watch_leaves_grad_mode_alone(pins):
    pins.watch.install()
    gc.collect()
    torch.set_grad_enabled(False)
    vt.stale_grad_garbage()
    try:
        gc.collect()  # a trainer-side collection: the trainer's autograd state is never touched
        assert torch.is_grad_enabled()
    finally:
        torch.set_grad_enabled(True)


def test_the_watch_leaves_a_guard_audit_flip_alone(pins):
    pins.watch.install()
    gc.collect()
    vt.stale_grad_garbage()
    with b.EngineScope("step"):
        b.ENGINE_TLS.audit = True  # a guard re-evaluating cache entries under the other grad mode on purpose
        try:
            gc.collect()
            audited = torch.is_grad_enabled()
        finally:
            b.ENGINE_TLS.audit = False
            torch.set_grad_enabled(False)
    assert audited
