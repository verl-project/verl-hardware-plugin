# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process engine adapter, the step guards and the GC policy (``rollout/inproc/engine.py``,
``sampler.py``, ``gc_policy.py``, the package's worker hooks). The engine is a small fake of vLLM's ``LLMEngine``
surface; no device, no Ray."""

import asyncio
import gc
import logging
from types import SimpleNamespace

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout import inproc
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as E
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import gc_policy, sampler
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
    eng._error = None
    r.stale = SimpleNamespace(fatal_pending="STALE WEIGHTS rank 0: sync step 3")
    assert r.step_health().startswith("STALE WEIGHTS")
    assert _shell(None).step_health() == "the in-process engine is not built"


def test_step_guard_refuses_update_actor_before_anything_trains(caplog):
    eng = E.InProcAsyncEngine(FakeEngine(), 0)
    worker = SimpleNamespace(rollout=_shell(eng))
    inproc.step_guard(worker)  # healthy: nothing
    inproc.step_guard(SimpleNamespace(rollout=object()))  # not an in-process rollout: nothing
    eng.window_failed = 1
    with pytest.raises(RuntimeError, match="STEP FAILED before update_actor rank 0: 1 request\\(s\\) failed"):
        inproc.step_guard(worker)


def _batch(n_groups: int, n: int, missing: int = 0, aborted: int = 0, steps=(4, 4), status="success"):
    keys, tags = [], []
    for g in range(n_groups):
        for s in range(n - (missing if g == 0 else 0)):
            keys.append(f"u{g}_{s}")
            tags.append(
                {
                    "response_len": 0 if len(tags) < aborted else 5,
                    "status": status,
                    "min_global_steps": steps[0],
                    "max_global_steps": steps[1],
                }
            )
    return SimpleNamespace(keys=keys, tags=tags)


def _rb(n=None):
    rb = SimpleNamespace(failure_keys={})
    if n is not None:
        rb.sampler_kwargs = {"rollout_n": n}
    return rb


@pytest.mark.parametrize(
    "batch, rb, want",
    [
        (_batch(4, 2), _rb(2), []),
        (_batch(4, 2, missing=1), _rb(2), ["miss 1 of their rollout.n=2"]),
        (_batch(3, 2), _rb(2), ["3 prompt groups carry trajectories, 4 were sampled"]),
        (_batch(4, 2, aborted=1), _rb(2), ["1 aborted trajectories"]),
        (_batch(4, 2, steps=(3, 3)), _rb(2), ["trajectory_staleness min 1 max 1"]),
        (_batch(4, 2, steps=(3, 4)), _rb(2), ["trajectory_spans max 2"]),
        (_batch(4, 2, status="failure"), _rb(2), ["trajectory status ['failure']"]),
        (_batch(4, 2, missing=1), _rb(None), ["rollout.n cannot be resolved for this training batch"]),
    ],
)
def test_batch_guard(batch, rb, want):
    problems, info = sampler.batch_guard(rb, batch, 5, "train", 4)
    assert len(problems) == len(want) and all(w in p for w, p in zip(want, problems, strict=True)), problems


def test_batch_guard_val_without_a_resolved_n_is_not_refused():
    problems, info = sampler.batch_guard(_rb(None), _batch(4, 2), 5, "val", 4)
    assert problems == [] and info["rollout_n"] is None


def test_rollout_n_comes_from_the_trainer_config():
    owner = SimpleNamespace(
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=16, val_kwargs=SimpleNamespace(n=1)))
        )
    )
    rb = SimpleNamespace(refill_fn=SimpleNamespace(__self__=owner))
    assert sampler.expected_rollout_n(rb, False) == 16 and sampler.expected_rollout_n(rb, True) == 1


def test_fast_poll_sampler_serves_sync_mode_only():
    cls = sampler.FastPollReplayBufferSync
    assert cls.__name__ == "FastPollReplayBufferSync" and cls.__module__ == sampler.__name__
    with pytest.raises(RuntimeError, match="serves trainer_mode=sync only"):
        cls(trainer_mode="separate_async")


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
