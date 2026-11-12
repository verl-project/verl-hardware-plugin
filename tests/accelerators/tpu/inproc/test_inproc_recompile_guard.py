# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the recompile guard (``rollout/inproc/compile_guard.py``): after warm-up an engine function may
recompile only for new input shapes, up to its precompiled count, and a few times for any other reason; beyond that,
on a fresh compile or on ``torch._dynamo.reset`` the guard trips and the run stops. Engine code is simulated by
functions compiled with dynamo's eager backend from files under a directory the test marks as engine code."""

import asyncio
import importlib.util
import logging
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import logger


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    """The package's ledger (its dynamo hooks are process-wide), reset, with ``tmp_path`` marked as engine code."""
    monkeypatch.setattr(cg.EngineCode, "markers", (str(tmp_path) + "/",))
    monkeypatch.delenv("VERL_TPU_INPROC_RECOMPILE_GUARD", raising=False)
    torch._dynamo.reset()
    cg.RECOMPILES.__init__()
    assert cg.RECOMPILES.install()
    yield cg.RECOMPILES
    cg.RECOMPILES.__init__()
    torch._dynamo.reset()


_N = [0]


def _engine_fn(tmp_path, body: str = "return x * 2"):
    """A compiled function whose code lives under tmp_path (engine code for the ledger)."""
    _N[0] += 1
    name = f"engine_mod_{_N[0]}"
    p = tmp_path / f"{name}.py"
    p.write_text(textwrap.dedent(f"def f(x):\n    {body}\n"))
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return torch.compile(mod.f, backend="eager", fullgraph=True, dynamic=False), mod.f


def test_shape_class_recompiles_are_allowed_up_to_the_precompiled_count(ledger, tmp_path):
    f, raw = _engine_fn(tmp_path)
    for n in (1, 2, 3):  # the build's precompile grid: three shapes
        f(torch.zeros(n))
    w = ledger.mark_warm(0)
    key = cg.EngineCode.key(raw.__code__)
    assert (
        w["entries"] == {key: 3} and w["armed"] and w["installed"] and w["guard_n"] == 8 and w["fresh_mode"] == "fatal"
    )
    for n in (4, 5, 6):  # new shapes after warm-up: up to the precompiled count, reported only
        f(torch.zeros(n))
    assert ledger.tripped is None and ledger.post_shape == {key: 3}
    f(torch.zeros(7))
    assert ledger.tripped is not None and "size-class recompiles 4 > its precompiled count 3" in ledger.tripped
    rec = ledger.compile_record()
    assert (
        rec["tripped"]
        and rec["post_warm_shape"] == {key: 4}
        and rec["entries_warm"] == {key: 3}
        and rec["unaccounted"] == {}
    )
    with pytest.raises(RuntimeError, match="RECOMPILE GUARD rank 0"):
        ledger.check()


def test_other_recompiles_trip_above_n(ledger):
    code = SimpleNamespace(co_filename=cg.EngineCode.markers[0] + "runner.py", co_name="sample", co_firstlineno=7)
    ledger.mark_warm(0)
    reason = "GLOBAL_STATE changed: grad_mode"
    for _ in range(cg.RECOMPILE_GUARD_N):
        ledger.on_recompile(code, [reason], 2, grad_only=True)
    assert ledger.tripped is None and ledger.post_other == {"sample@runner.py:7": 8}
    ledger.on_recompile(code, [reason], 2, grad_only=True)
    assert (
        ledger.tripped is not None and "non-shape recompiles 9" in ledger.tripped and '"grad_mode": 9' in ledger.tripped
    )


def test_a_recompile_at_dynamos_limit_trips(ledger, monkeypatch):
    monkeypatch.setattr(cg, "_dynamo_limit", lambda: 64)
    code = SimpleNamespace(co_filename=cg.EngineCode.markers[0] + "runner.py", co_name="g", co_firstlineno=1)
    ledger.warm_entries = {"g@runner.py:1": 64}
    ledger.warm = True
    ledger.on_recompile(code, ["tensor 'x' size mismatch at index 0"], 64)
    assert ledger.tripped is not None and "at dynamo recompile_limit" in ledger.tripped


def test_the_trainers_recompiles_are_counted_never_guarded(ledger):
    code = SimpleNamespace(co_filename="/somewhere/trainer.py", co_name="loss", co_firstlineno=3)
    ledger.mark_warm(0)
    for _ in range(20):
        ledger.on_recompile(code, ["GLOBAL_STATE changed: grad_mode"], 3)
    assert ledger.tripped is None and ledger.post["loss@trainer.py:3"] == 20


def test_a_fresh_engine_compile_after_warm_up_trips(ledger, tmp_path):
    f, _raw = _engine_fn(tmp_path)
    f(torch.zeros(1))
    ledger.mark_warm(0)
    assert ledger.fresh_pre and ledger.compile_hooked
    g, _graw = _engine_fn(tmp_path, "return x + 1")
    g(torch.zeros(1))
    assert ledger.tripped is not None and "compiled fresh after warm-up" in ledger.tripped
    assert ledger.compile_record()["fresh_post_warm"]


def test_dynamo_reset_after_warm_up_is_refused(ledger):
    torch._dynamo.reset()  # before warm-up: recorded, performed
    ledger.mark_warm(0)
    with pytest.raises(RuntimeError, match="torch._dynamo.reset\\(\\) called after warm-up"):
        torch._dynamo.reset()
    assert ledger.tripped is not None and [r["refused"] for r in ledger.resets] == [False, True]
    assert ledger.compile_record()["resets_after_warm"] == 1


def test_report_mode_counts_without_tripping(ledger, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("VERL_TPU_INPROC_RECOMPILE_GUARD", "report")
    assert ledger.install() and not ledger.armed
    f, _raw = _engine_fn(tmp_path)
    f(torch.zeros(1))
    w = ledger.mark_warm(0)
    assert w["armed"] is False and w["fresh_mode"] == "report"
    caplog.set_level(logging.WARNING, logger=logger.name)
    for n in (2, 3, 4):
        f(torch.zeros(n))
    g, _graw = _engine_fn(tmp_path, "return x - 1")
    g(torch.zeros(1))
    torch._dynamo.reset()  # performed in report mode
    assert ledger.tripped is None
    assert any("fresh compile after warm-up" in m for m in caplog.messages)


def test_recompile_guard_switch_values(monkeypatch):
    assert (
        cg.recompile_guard_mode({}) == "fatal"
        and cg.recompile_guard_mode({"VERL_TPU_INPROC_RECOMPILE_GUARD": "report"}) == "report"
    )
    with pytest.raises(RuntimeError, match="expected one of"):
        cg.recompile_guard_mode({"VERL_TPU_INPROC_RECOMPILE_GUARD": "0"})


@pytest.mark.parametrize(
    "reason, grad_only, cls",
    [
        ("tensor 'x' size mismatch at index 0. expected 4, actual 5", False, "size"),
        ("GLOBAL_STATE changed: grad_mode", True, "grad_mode"),
        ("GLOBAL_STATE changed: grad_mode", False, "new_input+grad_mode"),
        ("GLOBAL_STATE changed: deterministic_algorithms", False, "global_state:deterministic_algorithms"),
        ("not ___dict_contains('compute_logits', self.model.__dict__)", False, "dict_absent:compute_logits@self.model"),
        ("tensor 'x' dtype mismatch. expected Float, actual Half", False, "dtype"),
        ("L['self'].k == 3", False, "equals:L['self'].k"),
    ],
)
def test_reason_classes(reason, grad_only, cls):
    assert cg._reason_class(reason, grad_only) == cls


def test_guard_trip_arms_the_guard(ledger):
    line = cg.guard_trip("monkey_patch_model re-sent with another mask")
    assert ledger.tripped == line and line.startswith("RECOMPILE GUARD rank")


def test_the_pump_raises_when_the_guard_tripped_in_a_step(ledger):
    pytest.importorskip("vllm.sampling_params")
    from vllm.sampling_params import SamplingParams

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.engine import InProcAsyncEngine

    class TripEngine:
        model_config = SimpleNamespace()

        def __init__(self):
            self.reqs = {}

        def add_request(self, request_id, prompt, params, **_):
            self.reqs[request_id] = 0
            return request_id

        def has_unfinished_requests(self):
            return bool(self.reqs)

        def get_num_unfinished_requests(self):
            return len(self.reqs)

        def step(self):
            ledger.tripped = "RECOMPILE GUARD rank 0: engine function f recompiled 9 times after warm-up"
            return []

        def abort_request(self, ids, internal=False):
            pass

    async def go():
        eng = InProcAsyncEngine(TripEngine(), 0)
        col = await eng.add_request("a", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=2))
        with pytest.raises(RuntimeError, match="in-process engine step failed.*RECOMPILE GUARD"):
            await col.get()
        return eng

    assert asyncio.run(go()).window_failed == 1


def test_the_warm_up_line(ledger, monkeypatch, caplog):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.rank = 0
    r.engine = SimpleNamespace(warm=False)
    caplog.set_level(logging.INFO, logger=logger.name)
    r._mark_warm()
    line = next(m for m in caplog.messages if m.startswith("inproc warm-up rank 0: "))
    assert r.engine.warm and '"armed": true' in line and '"guard_n": 8' in line and '"required_missing": []' in line
