# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the first-token stamps of the in-process engine (``rollout/inproc/engine.py``): with FINAL_ONLY output
the timeline's request spans need the engine step that emitted each request's first token; the pump stamps it into the
timeline mixin's first-output dict before the step's finished outputs wake their consumers."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as E

pytest.importorskip("vllm.sampling_params")


def test_watch_scan_and_forget():
    w = E.FirstTokenWatch()
    w.pending = {"a-1": "a", "b-1": "b", "c-1": "c"}
    w.scan([SimpleNamespace(request_id="a-1", new_token_ids=[5]), SimpleNamespace(request_id="b-1", new_token_ids=[])])
    assert w.hits == ["a"] and set(w.pending) == {"b-1", "c-1"}
    w.forget(["b-1"])
    assert set(w.pending) == {"c-1"}


def test_scan_only_wrapper_reports_first_tokens():
    calls = []
    proc = SimpleNamespace(process_outputs=lambda outs, *a, **k: calls.append(len(outs)) or "processed")
    eng = SimpleNamespace(output_processor=proc)
    w = E.FirstTokenWatch()
    assert E.scan_first_tokens(eng, w)
    w.pending = {"r-1": "r"}
    assert proc.process_outputs([SimpleNamespace(request_id="r-1", new_token_ids=[1])]) == "processed"
    assert w.hits == ["r"] and calls == [1]
    assert not E.scan_first_tokens(SimpleNamespace(), E.FirstTokenWatch())


class StampEngine:
    """Reports each request's first token to the watch in its step (as the output processor does) and finishes it
    after ``steps`` steps."""

    model_config = SimpleNamespace()

    def __init__(self, watch_ref, steps=2):
        self.reqs = {}
        self.steps = steps
        self.watch_ref = watch_ref

    def add_request(self, request_id, prompt, params, **_):
        self.reqs[f"{request_id}-i"] = [request_id, 0]
        return f"{request_id}-i"

    def has_unfinished_requests(self):
        return bool(self.reqs)

    def get_num_unfinished_requests(self):
        return len(self.reqs)

    def abort_request(self, ids, internal=False):
        for i in ids:
            self.reqs.pop(i, None)

    def step(self):
        outs = []
        for internal, st in list(self.reqs.items()):
            st[1] += 1
            if st[1] == 1:
                self.watch_ref[0].scan([SimpleNamespace(request_id=internal, new_token_ids=[7])])
            if st[1] >= self.steps:
                del self.reqs[internal]
                outs.append(SimpleNamespace(request_id=st[0], finished=True, outputs=[]))
        return outs


def test_the_pump_stamps_first_tokens_into_the_timeline_sink():
    from vllm.sampling_params import SamplingParams

    ref = [None]

    async def go():
        eng = E.InProcAsyncEngine(StampEngine(ref, steps=3), 0)
        ref[0] = eng.first_watch
        sink: dict = {}
        eng.first_watch.sink = sink
        cols = [await eng.add_request(r, {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3)) for r in ("a", "b")]
        for c in cols:
            await c.get()
        return eng, sink

    eng, sink = asyncio.run(go())
    assert set(sink) == {"a", "b"} and eng.first_watch.stamped == 2 and eng.first_watch.pending == {}


def test_no_sink_no_watch():
    from vllm.sampling_params import SamplingParams

    ref = [None]

    async def go():
        eng = E.InProcAsyncEngine(StampEngine(ref, steps=1), 0)
        ref[0] = eng.first_watch
        col = await eng.add_request("a", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=1))
        await col.get()
        return eng

    eng = asyncio.run(go())
    assert eng.first_watch.pending == {} and eng.first_watch.stamped == 0


def test_an_aborted_request_leaves_the_watch():
    from vllm.sampling_params import SamplingParams

    ref = [None]

    async def go():
        eng = E.InProcAsyncEngine(StampEngine(ref, steps=100), 0)
        ref[0] = eng.first_watch
        eng.first_watch.sink = {}
        col = await eng.add_request("a", {"prompt_token_ids": [1]}, SamplingParams(max_tokens=3))
        await eng.abort(["a"])
        await col.get()
        return eng

    assert asyncio.run(go()).first_watch.pending == {}


def test_the_lazy_output_loop_reports_first_tokens():
    """With the lazy output path installed the first token is found on the request's own output in its loop."""
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
    from vllm.v1.engine.output_processor import OutputProcessor

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.lazy_output import LazyFinalOutputs

    proc = OutputProcessor(tokenizer=None, log_stats=False)
    lz = LazyFinalOutputs(SimpleNamespace(output_processor=proc), environ={})
    if not lz.installed:
        pytest.skip(f"the lazy output path is off here: {lz.why_not}")
    sp = SamplingParams(max_tokens=4, logprobs=0, detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY)
    req = EngineCoreRequest(
        request_id="q-int",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        external_req_id="q",
    )
    proc.add_request(req, None)
    w = E.FirstTokenWatch()
    w.pending = {"q-int": "q"}
    lz.first_watch = w
    proc.process_outputs([EngineCoreOutput(request_id="q-int", new_token_ids=[9])], engine_core_timestamp=0.0)
    assert w.hits == ["q"] and w.pending == {}
