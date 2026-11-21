# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""First-token stamps of in-process requests under traffic, on an engine with vLLM's step structure.

The in-process requests are FINAL_ONLY, so the timeline's request span would read first == end without the stamps.
The engine here is ``LLMEngine``-shaped: ``step()`` gets the step's ``EngineCoreOutput``-like records from its core (a
partial-prefill chunk carries no new token) and hands them to ``output_processor.process_outputs`` through the
instance attribute, so the package's scan-only wrapper (:func:`engine.scan_first_tokens`) sits in the path, as it does
on vLLM's engine when the lazy output path is off. It records the wall time at the end of every step and the step at
which each request emitted its first token.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as eng_mod

pytest.importorskip("vllm.sampling_params")  # the adapter sets vLLM's FINAL_ONLY output kind on every request


class _FakeStepEngine:
    def __init__(self):
        self.model_config = None
        self.reqs: dict = {}
        self.n = 0
        self.k = 0  # engine steps
        self.step_end_wall: list = []
        self.first_step: dict = {}  # caller id -> the step that emitted its first token
        self.engine_core = types.SimpleNamespace(get_output=self._core)
        self.output_processor = types.SimpleNamespace(process_outputs=self._proc)

    def add_request(self, rid, prompt, params, **_):
        self.n += 1
        internal = f"{rid}-int{self.n:04d}"  # vLLM's internal ids differ from the caller's
        self.reqs[internal] = {
            "ext": rid,
            "prefill": int(prompt["prefill_steps"]),
            "left": int(params.max_tokens),
            "out": [],
            "child": bool(prompt.get("child_ids")),  # the engine reports it under another id (an n > 1 fan-out)
        }
        return internal

    def has_unfinished_requests(self):
        return bool(self.reqs)

    def get_num_unfinished_requests(self):
        return len(self.reqs)

    def abort_request(self, ids, internal=False):
        for i in list(ids):
            self.reqs.pop(i, None)

    def _core(self):
        outs = []
        for internal, st in self.reqs.items():
            if st["prefill"] > 0:
                st["prefill"] -= 1
                outs.append(types.SimpleNamespace(request_id=internal, new_token_ids=[]))
            else:
                st["out"].append(100 + len(st["out"]))
                if len(st["out"]) == 1:
                    self.first_step[st["ext"]] = self.k
                rid = internal + ("#c0" if st["child"] else "")
                outs.append(types.SimpleNamespace(request_id=rid, new_token_ids=[st["out"][-1]]))
        return types.SimpleNamespace(outputs=outs)

    def _proc(self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None):
        fin = []
        for o in engine_core_outputs:
            rid = o.request_id.split("#")[0]
            st = self.reqs.get(rid)
            if st is not None and len(st["out"]) >= st["left"]:
                self.reqs.pop(rid)
                fin.append(
                    types.SimpleNamespace(
                        request_id=st["ext"],
                        finished=True,
                        outputs=[types.SimpleNamespace(token_ids=list(st["out"]), finish_reason="length")],
                    )
                )
        return fin

    def step(self):
        core = self.engine_core.get_output()
        fin = self.output_processor.process_outputs(core.outputs, engine_core_timestamp=None)
        self.step_end_wall.append(time.time())
        self.k += 1
        return fin


class _CountingWatch(eng_mod.FirstTokenWatch):
    __slots__ = ("calls", "eng")

    def __init__(self, eng):
        super().__init__()
        self.calls, self.eng = [], eng

    def scan(self, outputs):
        self.calls.append(self.eng.k)
        return super().scan(outputs)


def _setup(sink):
    eng = _FakeStepEngine()
    aeng = eng_mod.InProcAsyncEngine(eng, 0)
    aeng.first_watch = _CountingWatch(eng)
    aeng.first_watch.sink = sink
    assert eng_mod.scan_first_tokens(eng, aeng.first_watch)
    return eng, aeng


def _drive(aeng, specs, abort_one=None):
    async def main():
        cols = []
        for rid, prefill, max_tokens in specs:
            sp = types.SimpleNamespace(max_tokens=max_tokens, output_kind=None)
            cols.append(await aeng.add_request(rid, {"prefill_steps": prefill}, sp))
        if abort_one is not None:
            await asyncio.sleep(0)
            await aeng.abort(abort_one)
        return [await c.get() for c in cols]

    return asyncio.run(main())


def test_each_request_is_stamped_at_the_end_of_the_step_that_emitted_its_first_token():
    sink: dict = {}
    eng, aeng = _setup(sink)
    specs = [("r0", 0, 6), ("r1", 2, 5), ("r2", 3, 1), ("r3", 0, 9)]  # r2: its first token is its last
    outs = _drive(aeng, specs)
    walls = eng.step_end_wall
    assert sorted(sink) == ["r0", "r1", "r2", "r3"] and all(o.finished for o in outs)
    assert eng.first_step == {"r0": 0, "r3": 0, "r1": 2, "r2": 3} and eng.k == 9
    for rid, step in eng.first_step.items():
        nxt = walls[step + 1] if step + 1 < len(walls) else float("inf")
        assert walls[step] <= sink[rid] < nxt, (rid, step, sink[rid], walls)
    w = aeng.first_watch
    assert w.calls == [0, 1, 2, 3]  # the step's outputs are scanned only while a request awaits its first token
    assert (w.stamped, w.late, w.pending) == (4, 0, {})


def test_a_stamp_never_overwrites_an_earlier_record():
    sink = {"q0": 1.0}
    _, aeng = _setup(sink)
    _drive(aeng, [("q0", 0, 3), ("q1", 1, 3)])
    assert sink["q0"] == 1.0 and sink["q1"] > 1.0


def test_an_aborted_request_leaves_the_watch_without_a_stamp():
    sink: dict = {}
    _, aeng = _setup(sink)
    outs = _drive(aeng, [("x0", 50, 3), ("x1", 0, 4)], abort_one="x0")
    assert "x0" not in sink and "x1" in sink and not aeng.first_watch.pending
    assert outs[0].outputs == [] and len(aeng.first_watch.calls) <= 2


def test_with_the_timeline_off_nothing_is_armed_scanned_or_stamped():
    _, aeng = _setup(None)
    _drive(aeng, [("y0", 1, 3), ("y1", 0, 2)])
    w = aeng.first_watch
    assert w.calls == [] and not w.pending and w.stamped == 0


def test_a_request_reported_under_other_ids_leaves_the_watch_at_its_final_output():
    """vLLM's n > 1 fan-out reports a request under its children's ids: no stamp, and no stale pending entry that
    would make every later step scan"""
    sink: dict = {}
    eng, aeng = _setup(sink)
    _drive(aeng, [("z0", 0, 4)])

    async def fan():
        sp = types.SimpleNamespace(max_tokens=6, output_kind=None)
        col = await aeng.add_request("zc", {"prefill_steps": 1, "child_ids": True}, sp)
        return await col.get()

    out = asyncio.run(fan())
    k_before = eng.k
    _drive(aeng, [("z1", 0, 3)])  # scanned only while its own first token is awaited (one step)
    assert out.finished and "zc" not in sink and "z0" in sink and "z1" in sink and not aeng.first_watch.pending
    assert [c for c in aeng.first_watch.calls if c >= k_before] == [k_before]
