# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process engine adapter, the replica, the rollout's step health and the GC policy
(``rollout/inproc/engine.py``, ``gc_policy.py``, the package's worker hooks). The engine is a small fake of vLLM's
``LLMEngine`` surface; no device, no Ray."""

import asyncio
import gc
import logging
from types import SimpleNamespace

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout import inproc
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as E
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import gc_policy
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

pytest.importorskip("vllm.sampling_params")


class FakeEngine:
    """vLLM's LLMEngine surface the adapter drives: each request finishes after ``steps`` engine steps."""

    def __init__(self, steps: int = 3, fail_at: int = -1):
        self.model_config = SimpleNamespace()
        self.reqs: dict = {}
        self.steps = steps
        self.fail_at = fail_at
        self.n = 0
        self.aborted: list = []

    def add_request(self, request_id, prompt, params, **_):
        internal = f"{request_id}-{len(self.reqs):04d}"
        self.reqs[internal] = [request_id, 0]
        return internal

    def has_unfinished_requests(self):
        return bool(self.reqs)

    def get_num_unfinished_requests(self):
        return len(self.reqs)

    def abort_request(self, ids, internal=False):
        assert internal
        self.aborted.extend(ids)
        for i in ids:
            self.reqs.pop(i, None)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [method]

    def step(self):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("engine step failed")
        outs = []
        for internal, st in list(self.reqs.items()):
            st[1] += 1
            if st[1] >= self.steps:
                del self.reqs[internal]
                outs.append(
                    SimpleNamespace(request_id=st[0], finished=True, outputs=[SimpleNamespace(token_ids=[1] * st[1])])
                )
        return outs


def _params():
    from vllm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=4)


def test_requests_are_served_by_the_pump():
    async def go():
        eng = E.InProcAsyncEngine(FakeEngine(steps=3), 0)
        c1 = await eng.add_request("a", {"prompt_token_ids": [1]}, _params())
        c2 = await eng.add_request("b", {"prompt_token_ids": [2]}, _params())
        o1, o2 = await c1.get(), await c2.get()
        assert o1.finished and o2.finished and len(o1.outputs[0].token_ids) == 3
        assert list(eng.output_processor.request_states) == []
        return eng

    eng = asyncio.run(go())
    assert eng.window_take() == {"aborted": 0, "failed": 0} and eng._error is None


def test_add_request_sets_final_only_output():
    from vllm.sampling_params import RequestOutputKind

    async def go():
        eng = E.InProcAsyncEngine(FakeEngine(steps=1), 0)
        sp = _params()
        col = await eng.add_request("a", {"prompt_token_ids": [1]}, sp)
        await col.get()
        return sp

    assert asyncio.run(go()).output_kind == RequestOutputKind.FINAL_ONLY


def test_abort_maps_external_ids_to_internal_ones():
    async def go():
        fake = FakeEngine(steps=100)
        eng = E.InProcAsyncEngine(fake, 0)
        col = await eng.add_request("ext-1", {"prompt_token_ids": [1]}, _params())
        await eng.abort(["ext-1"])
        out = await col.get()
        assert out.finished and out.outputs == [] and fake.aborted == [col.request_id]
        await eng.pause_generation()
        return eng

    eng = asyncio.run(go())
    assert eng.window_take() == {"aborted": 1, "failed": 0} and eng.window_take() == {"aborted": 0, "failed": 0}


def test_pause_aborts_every_request_in_flight():
    async def go():
        fake = FakeEngine(steps=100)
        eng = E.InProcAsyncEngine(fake, 0)
        cols = [await eng.add_request(f"r{i}", {"prompt_token_ids": [i]}, _params()) for i in range(3)]
        await eng.pause_generation()
        outs = [await c.get() for c in cols]
        await eng.resume_generation()
        return eng, outs, fake

    eng, outs, fake = asyncio.run(go())
    assert all(o.outputs == [] for o in outs) and len(fake.aborted) == 3 and eng.window_aborted == 3


def test_a_failed_engine_step_fails_every_request_in_flight():
    async def go():
        eng = E.InProcAsyncEngine(FakeEngine(steps=100, fail_at=2), 0)
        cols = [await eng.add_request(f"r{i}", {"prompt_token_ids": [i]}, _params()) for i in range(2)]
        errs = []
        for c in cols:
            with pytest.raises(RuntimeError, match="in-process engine step failed"):
                await c.get()
            errs.append(True)
        with pytest.raises(RuntimeError, match="failed earlier"):
            await eng.add_request("late", {"prompt_token_ids": [1]}, _params())
        return eng

    eng = asyncio.run(go())
    assert eng.window_failed == 2 and eng._error is not None


def test_collective_rpc_runs_in_the_engine_scope():
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import ENGINE_TLS

    seen = []

    class Rpc(FakeEngine):
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            seen.append(ENGINE_TLS.kind)
            return [method]

    assert asyncio.run(E.InProcAsyncEngine(Rpc(), 0).collective_rpc("ping")) == ["ping"] and seen == ["rpc"]


def _shell(engine) -> TPUInProcRollout:
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.engine, r.rank = engine, 0
    return r


def test_step_health():
    eng = E.InProcAsyncEngine(FakeEngine(), 0)
    r = _shell(eng)
    assert r.step_health() is None
    eng.window_aborted = 1
    assert "aborted" in r.step_health()
    eng.window_aborted, eng.window_failed = 0, 2
    assert "2 request(s) failed" in r.step_health()
    eng.window_failed, eng._error = 0, RuntimeError("x")
    assert "the in-process engine failed" in r.step_health()
    assert _shell(None).step_health() == "the in-process engine is not built"


def test_worker_methods_are_refused_when_absent_or_coroutines(monkeypatch):
    import verl.workers.engine_workers as ew

    class W:
        def init_model(self):
            return "built"

        async def update_actor(self):
            pass

        def compute_log_prob(self):
            pass

    monkeypatch.setattr(ew, "ActorRolloutRefWorker", W)
    with pytest.raises(RuntimeError, match="update_actor is a coroutine"):
        inproc.install_worker_methods()
    assert "generate" not in vars(W)

    class Ok:
        def init_model(self):
            return "built"

        def update_actor(self):
            return "trained"

        def compute_log_prob(self):
            return "lp"

        async def update_weights(self, *a, **k):
            return "synced"

    monkeypatch.setattr(ew, "ActorRolloutRefWorker", Ok)
    assert inproc.install_worker_methods() == ["init_model", "update_actor", "compute_log_prob"]
    wrapped = vars(Ok)["update_actor"]
    assert inproc.install_worker_methods() and vars(Ok)["update_actor"] is wrapped  # idempotent
    w = Ok()
    assert w.init_model() == "built" and w.update_actor() == "trained" and w.compute_log_prob() == "lp"
    assert asyncio.run(w.update_weights()) == "synced"
    assert asyncio.run(w.inproc_server_info()) == {"engine_ready": False, "rollout": "NoneType"}


def test_kv_capacity_check():
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_max_concurrency=200.0, num_gpu_blocks=10, block_size=16),
        scheduler_config=SimpleNamespace(max_num_seqs=128),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    ok = E.kv_capacity_check(vc, None, {})
    assert ok["ok"] and ok["mode"] == "fatal" and ok["headroom_requests"] == 72.0
    vc.cache_config.kv_cache_max_concurrency = 100.0
    assert not E.kv_capacity_check(vc, None, {})["ok"]
    assert E.kv_capacity_check(vc, None, {"VERL_TPU_INPROC_INTEGRITY": "report"})["mode"] == "report"
    vc.cache_config.kv_cache_max_concurrency = None
    assert not E.kv_capacity_check(vc, None, {})["ok"]


def test_gc_defer_restores_the_callers_thresholds():
    t = gc.get_threshold()
    d = gc_policy.GcDefer()
    d.enter()
    try:
        assert gc.get_threshold()[2] == (gc_policy.GC_DEFER_THRESHOLD2 if t[0] else t[2]) and d.pump_active
        import torch._dynamo.config as dc

        assert dc.run_gc_after_compile is False
    finally:
        d.exit()
    assert gc.get_threshold() == t and not d.pump_active


def test_rss_freeze_due(monkeypatch):
    monkeypatch.setattr(gc_policy, "rss_bytes", lambda: 20 * 2**30)
    assert gc_policy.rss_freeze_due(5, 2**30, 8 * 2**30)[0] is False  # not an aligned sync
    assert gc_policy.rss_freeze_due(10, 2**30, 0)[0] is False  # off
    due, now, why = gc_policy.rss_freeze_due(10, 2**30, 8 * 2**30)
    assert due and now == 20 * 2**30 and "rss grew 19.000 GiB" in why
    assert gc_policy.rss_freeze_due(10, 19 * 2**30, 8 * 2**30)[0] is False
    assert gc_policy.rss_freeze_due(10, None, 8 * 2**30) == (False, 20 * 2**30, None)


def test_replica_no_ops_and_server_check(caplog):
    replica_cls = E.TPUInProcReplica
    r = replica_cls.__new__(replica_cls)
    r.replica_rank, r.world_size = 3, 1

    class Remote:
        def __init__(self, value):
            self.value = value

        def remote(self):
            async def v():
                return self.value

            return v()

    worker = SimpleNamespace(inproc_server_info=Remote({"engine_ready": True, "replica_rank": 3, "rank": 3, "pid": 1}))
    r.workers = [worker]
    caplog.set_level(logging.INFO, logger=inproc.logger.name)
    asyncio.run(r.launch_servers())
    assert r._server_address == "inproc://3" and r.servers == [worker]
    asyncio.run(r.sleep())
    asyncio.run(r.sleep())
    assert sum("sleep is a no-op in process" in m for m in caplog.messages) <= 1
    worker.inproc_server_info = Remote({"engine_ready": False, "replica_rank": 3})
    with pytest.raises(RuntimeError, match="replica 3: worker reports"):
        asyncio.run(r.launch_servers())


# vLLMHttpServer methods that reach the engine only through replica calls the in-process replica answers itself (no
# sleep mode, no profiler, no snapshot of vLLM's metrics, no single-request abort): server method -> replica method
INTERCEPTED_SERVER_METHODS = {
    "sleep": "sleep",
    "_sleep_hybrid": "sleep",
    "wake_up": "wake_up",
    "release_kv_cache": "release_kv_cache",
    "resume_kv_cache": "resume_kv_cache",
    "clear_kv_cache": "clear_kv_cache",
    "start_profile": "start_profile",
    "stop_profile": "stop_profile",
    "snapshot": "snapshot",
    "_prometheus_logger": "snapshot",
    "abort_request": "abort_request",
}
# engine attributes verl's server uses only with a configuration the in-process rollout refuses before any request
REFUSED_BEFORE_ANY_REQUEST = {
    "list_loras": "generate lists LoRA adapters only with an unmerged adapter; the first sync refuses its peft_config",
}


def test_the_adapter_covers_every_engine_call_of_verls_server():
    """Every ``self.engine.X`` of verl's vLLMHttpServer is implemented by the adapter, sits in a server method reached
    only through a replica call the in-process replica answers itself, or needs a configuration refused before any
    request."""
    import ast
    import inspect
    import textwrap

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    uses: dict = {}
    for fn in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(vLLMHttpServer)))):
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            for n in ast.walk(fn):
                if (
                    isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Attribute)
                    and n.value.attr == "engine"
                    and isinstance(n.value.value, ast.Name)
                    and n.value.value.id == "self"
                ):
                    uses.setdefault(n.attr, set()).add(fn.name)
    assert {"add_request", "abort"} <= set(uses), uses  # positive control: the scan sees the serving path
    adapter = E.InProcAsyncEngine(FakeEngine(), 0)
    missing = {}
    for attr, methods in sorted(uses.items()):
        if hasattr(adapter, attr) or attr in REFUSED_BEFORE_ANY_REQUEST:
            continue
        reached = sorted(m for m in methods if m not in INTERCEPTED_SERVER_METHODS)
        if reached:
            missing[attr] = reached
    assert missing == {}, f"verl's server calls engine attributes the adapter lacks, from {missing}"
    for replica_method in set(INTERCEPTED_SERVER_METHODS.values()):
        assert replica_method in vars(E.TPUInProcReplica), f"{replica_method} is not answered by the in-process replica"
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    with pytest.raises(NotImplementedError, match="LoRA adapters are not supported in process"):
        asyncio.run(r.update_weights(iter(()), peft_config={"r": 8}))


def test_update_weights_refuses_until_the_verified_install_lands():
    """Weight sync is not part of this change: the first sync refuses before anything is installed, so a training run
    with the switch on stops there instead of rolling out on weights the trainer never synced."""
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    with pytest.raises(NotImplementedError, match="weight sync into the in-process engine is not available yet"):
        asyncio.run(r.update_weights(iter(())))
