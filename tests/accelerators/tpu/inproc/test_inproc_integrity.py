# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Integrity tests of the TPU in-process rollout on CPU, each through the real verl / vLLM code on the path:

* verl's ``monkey_patch_model`` RPC is frozen once the mask went in at model load: the same mask again (a direct call,
  or vLLM's ``run_method`` on a ``WorkerWrapperBase``, the collective RPC path) is a no-op, another mask is refused
  with the recompile guard's marker and leaves ``compute_logits`` untouched;
* the engine entry puts the caller's thread state right for every call (a ``torch.device`` context or another
  TorchFunctionMode, inference mode, num_threads, the default dtype, deterministic algorithms, the float32 matmul
  precision): vllm-torchtpu's real ``compute_selected_logits`` keeps its precompiled programs for every bucket, and the
  caller's state is back after every call (needs ``INPROC_TEST_VLLM_TORCHTPU_SRC``);
* verl's real ``vLLMHttpServer.abort_all_requests(abort_only=True)`` on the in-process adapter removes the requests
  from an engine with vLLM's request-id semantics (internal ids only);
* a failed or aborted rollout on one trainer rank makes ``update_actor`` refuse on every rank (a gloo world of three
  rank processes), so nothing trains on a partial batch;
* verl's real sync ``ReplayBuffer.sample`` through ``FastPollReplayBufferSync``: a batch with lost, partially failed,
  aborted, stale or spanning groups fails the step before ``update_actor``; healthy train and validation batches pass;
* verl's full-determinism mode is refused at the build, and is shown to reseed the process's RNG (the hazard).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import types
import uuid

import pytest
import torch

# the environment the children start from, taken before this process imports vLLM (its import sets variables of
# its own, which would leak into children started after it)
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "support"))

import inproc_vt_runner as vt  # noqa: E402

PKG = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"
needs_vllm = pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM is not installed")
needs_vt = pytest.mark.skipif(vt.skip_reason() is not None, reason=str(vt.skip_reason()))
ENTRY_CASES = ("device", "tfmode", "inference", "num_threads", "default_dtype", "deterministic", "matmul")


def _clean_env(**extra) -> dict:
    env = dict(BASE_ENV)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", **extra})
    return env


def _child(args: list, env: dict, cwd: str, timeout: int = 1200) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, f"rc {proc.returncode}: {proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


# ------------------------------------------------------------------------------------------------ the frozen RPC


class _Head(torch.nn.Module):
    """a vLLM-shaped model for the logits path: an inner model and an LM head"""

    def __init__(self, h: int = 16, v: int = 64):
        super().__init__()
        self.model = torch.nn.Module()
        self.lm_head = torch.nn.Linear(h, v, bias=False)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@needs_vllm
def test_monkey_patch_model_is_frozen_once_the_mask_went_in_at_load(monkeypatch):
    from vllm.v1.serial_utils import run_method
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension as Ext
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg

    logits = cg.LogitsAtLoad()
    monkeypatch.setattr(cg, "LOGITS_AT_LOAD", logits)
    if hasattr(cg, "RecompileLedger"):  # the refusal below arms the recompile guard: a fresh one, kept to this test
        monkeypatch.setattr(cg, "RECOMPILES", cg.RecompileLedger())
    H, V = 16, 64

    class TPUWorker:  # vllm-torchtpu's TPUWorker shape, with verl's real extension methods mixed in as vLLM does
        _nvfp4_per_token_config = None

        def __init__(self):
            self.model_runner = types.SimpleNamespace(
                model=None, vllm_config=types.SimpleNamespace(speculative_config=None)
            )

        def load_model(self, *, load_dummy_weights: bool = False) -> None:
            torch.manual_seed(0)
            self.model_runner.model = _Head(H, V).requires_grad_(False)

    for n in ("monkey_patch_model", "_iter_all_models", "_use_mtp_drafter_weight_sync", "_get_drafter_model"):
        setattr(TPUWorker, n, getattr(Ext, n))
    mask = {"vocab_size": V - 4, "banned_token_ids": [3, 5]}
    assert logits.install(TPUWorker)
    logits.arm(mask)
    worker = TPUWorker()
    try:
        worker.load_model()
    finally:
        logits.disarm()
    logits.verify_built(worker)
    model = worker.model_runner.model
    cl0 = model.__dict__.get("compute_logits")
    assert cl0 is not None and str(logits.applied["rpc_frozen"]).startswith("frozen")
    assert getattr(worker.__dict__.get("monkey_patch_model"), "_inproc_frozen", False)

    worker.monkey_patch_model(**mask)  # the same mask, a direct call
    ww = WorkerWrapperBase.__new__(WorkerWrapperBase)  # vLLM's collective_rpc path: run_method on the wrapper
    ww.worker = worker  # -> WorkerWrapperBase.__getattr__ -> the worker instance
    run_method(ww, "monkey_patch_model", (), dict(mask))
    worker.monkey_patch_model(vocab_size=V - 4, banned_token_ids=[5, 3])  # the same ids in another order
    assert model.__dict__.get("compute_logits") is cl0 and logits.repatch_noop == 3

    with pytest.raises(RuntimeError, match="RECOMPILE GUARD rank 0: monkey_patch_model re-sent after the build"):
        run_method(ww, "monkey_patch_model", (), {"vocab_size": V - 4, "banned_token_ids": [3, 5, 7]})
    assert model.__dict__.get("compute_logits") is cl0 and logits.repatch_refused == 1
    out = model.compute_logits(torch.ones(2, H))
    assert torch.isinf(out[:, V - 4 :]).all() and torch.isinf(out[:, [3, 5]]).all() and not torch.isinf(out[:, 7]).any()


# ------------------------------------------------------------------------------------------------ the engine entry

ENTRY_METHODS = (
    "_precompile_timed",
    "_update_num_xla_graphs",
    "_dummy_logits",
    "compute_selected_logits",
    "_precompile_compute_selected_logits",
)


def child_entry(case: str, scoped: bool) -> dict:
    """vllm-torchtpu's real compute_selected_logits and its real precompile, then every bucket served while the
    caller's thread holds ``case``'s state, through the engine scope (``scoped``) or under no_grad only (control)"""
    import torch._dynamo.config as dc

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b

    root = vt.vt_root()
    H, V = 16, 64
    dc.recompile_limit = 1024  # every recompile visible
    mod = vt.build_runner_module(root, ENTRY_METHODS, functions=())
    r = mod.TPUModelRunner.__new__(mod.TPUModelRunner)
    r.num_tokens_paddings, r.num_reqs_paddings = vt.paddings(root, 4096, 128)
    r.is_pooling_model, r.max_num_reqs, r.device = False, 128, torch.device("cpu")
    r.model_config = types.SimpleNamespace(max_logprobs=0, get_hidden_size=lambda: H)
    r.check_recompilation, r.enforce_eager, r._hidden_states_dtype = False, False, torch.float32
    torch.manual_seed(0)
    r.model = _Head(H, V).requires_grad_(False)
    csl = vars(mod.TPUModelRunner)["compute_selected_logits"]

    @torch.no_grad()
    def execute(t, n):  # execute_model's call: the step's hidden states are made inside the engine call
        return r.compute_selected_logits(torch.full((t, H), 0.01), torch.zeros(n, dtype=torch.int32))

    torch.set_grad_enabled(True)  # the build runs inside the trainer's init_model
    with b.engine_grad_ctx("build"):
        r._precompile_compute_selected_logits()
    b.snapshot_engine_global_state()  # the warm-up snapshot the build takes
    out: dict = {"entries_build": vt.cache_entries(csl)}
    buckets = []
    for t in r.num_tokens_paddings:
        for n in r.num_reqs_paddings:
            buckets.append((t, n))
            if n >= min(t, r.max_num_reqs):
                break
    out["buckets"] = len(buckets)
    n0 = torch.get_num_threads()

    class PassMode(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types_, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    @contextlib.contextmanager
    def caller_state():
        if case == "device":
            with torch.device("cpu"):
                yield lambda: torch._C._len_torch_function_stack() == 1
        elif case == "tfmode":
            with PassMode():
                yield lambda: torch._C._len_torch_function_stack() == 1
        elif case == "inference":
            with torch.inference_mode():
                yield lambda: torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        elif case == "num_threads":
            want = n0 + 1 if n0 < 8 else n0 - 1
            torch.set_num_threads(want)
            try:
                yield lambda: torch.get_num_threads() == want
            finally:
                torch.set_num_threads(n0)
        elif case == "default_dtype":
            torch.set_default_dtype(torch.float64)
            try:
                yield lambda: torch.get_default_dtype() == torch.float64
            finally:
                torch.set_default_dtype(torch.float32)
        elif case == "deterministic":
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                yield lambda: torch.are_deterministic_algorithms_enabled()
            finally:
                torch.use_deterministic_algorithms(False)
        elif case == "matmul":
            prev = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("medium")
            try:
                yield lambda: torch.get_float32_matmul_precision() == "medium"
            finally:
                torch.set_float32_matmul_precision(prev)
        else:
            raise ValueError(case)

    kept, errors = [], []
    with caller_state() as caller_ok:
        for t, n in buckets:
            try:
                with b.engine_mode_ctx("step") if scoped else contextlib.nullcontext():
                    execute(t, n)
            except Exception as exc:  # noqa: BLE001 - e.g. a recompile whose trace fails under a float64 default
                errors.append(f"{type(exc).__name__}: {exc}"[:160])
            kept.append(bool(caller_ok()))
    out.update({"entries_after": vt.cache_entries(csl), "errors": len(errors), "caller_state_kept": all(kept)})
    return out


@needs_vt
@pytest.mark.parametrize("case", ENTRY_CASES)
def test_the_engine_entry_puts_the_callers_state_right_for_every_bucket(case, tmp_path):
    pytest.importorskip("vllm")
    scoped = _child(["entry", case, "scoped"], _clean_env(), str(tmp_path))
    control = _child(["entry", case, "control"], _clean_env(), str(tmp_path))
    assert scoped["entries_build"] == scoped["buckets"] > 0, scoped
    assert scoped["entries_after"] == scoped["entries_build"] and scoped["errors"] == 0, scoped
    assert scoped["caller_state_kept"] is True, scoped
    # the control (no engine scope) shows that this caller state does break the precompiled programs
    assert control["entries_after"] > control["entries_build"] or control["errors"] > 0, control


# ------------------------------------------------------------------------------------------------ abort by internal id


@needs_vllm
def test_abort_only_removes_the_requests_from_an_engine_with_vllms_request_ids():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine

    class StrictEngine:
        """vLLM's request ids: add_request returns '{external}-{uuid8}' and abort_request(ids, internal=True) matches
        internal ids only; nothing finishes (long requests)"""

        def __init__(self):
            self.model_config, self.reqs = types.SimpleNamespace(), {}

        def add_request(self, rid, prompt, params, **_):
            internal = f"{rid}-{uuid.uuid4().hex[:8]}"
            self.reqs[internal] = rid
            return internal

        def abort_request(self, ids, internal=False):
            for i in list(ids):
                if internal:
                    self.reqs.pop(i, None)
                else:
                    for k in [k for k, v in self.reqs.items() if v == i]:
                        self.reqs.pop(k, None)

        def has_unfinished_requests(self):
            return bool(self.reqs)

        def get_num_unfinished_requests(self):
            return len(self.reqs)

        def step(self):
            return []

    eng = StrictEngine()
    aeng = engine.InProcAsyncEngine(eng, 0)
    srv = types.SimpleNamespace(node_rank=0, engine=aeng)

    async def go():
        cols = [
            await aeng.add_request(f"req-{i}", {"prompt_token_ids": [1, 2]}, types.SimpleNamespace(output_kind=None))
            for i in range(3)
        ]
        for _ in range(3):
            await asyncio.sleep(0)  # the pump steps
        res = await vLLMHttpServer.abort_all_requests(srv, abort_only=True)  # verl's own abort_only path
        outs = [await c.get() for c in cols]
        await asyncio.sleep(0)
        return res, outs

    res, outs = asyncio.run(go())
    assert res["aborted_count"] == 3 and eng.get_num_unfinished_requests() == 0
    assert all(o.finished and o.outputs == [] for o in outs) and not aeng._collectors
    assert aeng.aborted == 3 and aeng.window_take() == {"aborted": 3, "failed": 0}


# --------------------------------------------------------------------------------------------- the step guard, 3 ranks

WORLD = 3


def child_rank(scenario: dict) -> dict:
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{os.environ['INPROC_TEST_PORT']}", rank=rank, world_size=world
    )
    import importlib

    inproc = importlib.import_module(PKG)
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    class StepEngine:
        """an LLMEngine-shaped engine without a model: every request finishes at its 3rd step; on the failing rank
        the 2nd step raises"""

        def __init__(self, fail: bool):
            self.model_config, self.reqs, self.n, self.fail = types.SimpleNamespace(), {}, 0, fail

        def add_request(self, rid, prompt, params, **_):
            self.reqs[f"{rid}-int"] = [rid, 0]
            return f"{rid}-int"

        def has_unfinished_requests(self):
            return bool(self.reqs)

        def get_num_unfinished_requests(self):
            return len(self.reqs)

        def abort_request(self, ids, internal=False):
            for i in ids:
                self.reqs.pop(i, None)

        def step(self):
            self.n += 1
            if self.fail and self.n == 2:
                raise RuntimeError("injected engine-step failure")
            fin = []
            for k, v in list(self.reqs.items()):
                v[1] += 1
                if v[1] >= 3:
                    fin.append(types.SimpleNamespace(request_id=v[0], finished=True, outputs=[types.SimpleNamespace()]))
                    self.reqs.pop(k)
            return fin

    aeng = engine.InProcAsyncEngine(StepEngine(fail=rank == scenario.get("fail_rank")), rank)
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.rank, r.engine = rank, aeng
    worker = types.SimpleNamespace(rollout=r)

    async def rollout():
        cols = [
            await aeng.add_request(f"r{rank}-{i}", {"prompt_token_ids": [1]}, types.SimpleNamespace(output_kind=None))
            for i in range(4)
        ]
        if scenario.get("abort_rank") == rank:
            await aeng.abort([f"r{rank}-0"])  # the external id verl passes
        res = []
        for c in cols:
            try:
                res.append("ok" if (await c.get()).outputs else "aborted")
            except Exception as exc:  # noqa: BLE001
                res.append(f"failed: {type(exc).__name__}")
        return res

    gen = asyncio.run(rollout())
    trained = []
    wrapped = inproc._wrap_worker_method(lambda self, data: trained.append(rank) or {"metrics": {}}, "update_actor")
    try:
        wrapped(worker, {"batch": rank})
        outcome = "trained"
    except RuntimeError as exc:
        outcome = str(exc)
    dist.barrier()
    dist.destroy_process_group()
    return {"rank": rank, "gen": gen, "outcome": outcome, "trained": bool(trained)}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _step_guard_world(tmp_path, **scenario) -> list:
    port = _free_port()
    procs = []
    for rank in range(WORLD):
        env = _clean_env(RANK=str(rank), WORLD_SIZE=str(WORLD), INPROC_TEST_PORT=str(port))
        procs.append(
            subprocess.Popen(
                [sys.executable, "-B", os.path.abspath(__file__), "--child", "rank", json.dumps(scenario)],
                env=env,
                cwd=str(tmp_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    out = []
    for p in procs:
        so, se = p.communicate(timeout=900)
        lines = [x for x in so.splitlines() if x.startswith("RESULT ")]
        out.append(json.loads(lines[-1][7:]) if lines else {"error": f"rc {p.returncode}: {se[-1500:]}"})
    return out


@needs_vllm  # the adapter sets vLLM's output kind on every request
def test_a_failed_engine_step_on_one_rank_refuses_update_actor_on_every_rank(tmp_path):
    res = _step_guard_world(tmp_path, fail_rank=1)
    assert len(res) == WORLD and all("error" not in r for r in res), res
    assert not any(r["trained"] for r in res), res
    assert all("STEP FAILED before update_actor rank" in r["outcome"] for r in res), res
    assert "the in-process engine failed" in res[1]["outcome"]
    assert all("another rank's rollout failed" in r["outcome"] for i, r in enumerate(res) if i != 1)
    assert all(g.startswith("failed") for g in res[1]["gen"]) and all(g == "ok" for g in res[0]["gen"] + res[2]["gen"])


@needs_vllm  # the adapter sets vLLM's output kind on every request
def test_healthy_rollouts_train_on_every_rank(tmp_path):
    res = _step_guard_world(tmp_path)
    assert [r.get("outcome") for r in res] == ["trained"] * WORLD and all(r["trained"] for r in res), res


@needs_vllm  # the adapter sets vLLM's output kind on every request
def test_an_aborted_request_on_one_rank_refuses_update_actor_on_every_rank(tmp_path):
    res = _step_guard_world(tmp_path, abort_rank=2)
    assert all("error" not in r for r in res), res
    assert not any(r["trained"] for r in res) and "1 request(s) aborted" in res[2]["outcome"], res


# ------------------------------------------------------------------------------------------------ the batch guard


class _FakeTQ:
    """TransferQueue's kv surface verl's sync ReplayBuffer.sample reads (kv_list / kv_clear), on one dict"""

    def __init__(self, data: dict):
        self.data = data

    def kv_list(self):
        return {p: dict(items) for p, items in self.data.items()}

    def kv_clear(self, keys, partition_id):
        for k in keys:
            self.data.get(partition_id, {}).pop(k, None)


def _tq_state(step: int, part="train", prompts=32, n=16, failed=None, aborted=0, stale=0, spans=1) -> dict:
    """prompt groups (status finished / failure) and their trajectories' tags as verl's agent loop writes them"""
    failed = failed or {}
    d: dict = {}
    for p in range(prompts):
        uid = f"u{p:03d}"
        d[uid] = {"is_prompt": True, "status": "failure" if p in failed else "finished", "global_steps": step}
        for s in range(failed.get(p, n)):
            mx = step - 1 - stale
            d[f"{uid}_{s}_0"] = {
                "status": "success",
                "prompt_len": 30,
                "response_len": 0 if (p == 0 and s < aborted) else 40,
                "seq_len": 70,
                "global_steps": step,
                "min_global_steps": mx - (spans - 1),
                "max_global_steps": mx,
            }
    return {part: d}


@pytest.fixture
def replay(monkeypatch):
    pytest.importorskip("transfer_queue")
    import verl.trainer.ppo.v1.replay_buffer as rbm
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import sampler

    trainer = types.SimpleNamespace(
        config=types.SimpleNamespace(
            actor_rollout_ref=types.SimpleNamespace(
                rollout=types.SimpleNamespace(n=16, val_kwargs=types.SimpleNamespace(n=1))
            )
        )
    )
    refills: list = []
    refill = types.MethodType(lambda self, k: refills.append(k), trainer)  # verl passes the trainer's bound method

    def make(cls, data: dict):
        monkeypatch.setattr(rbm, "tq", _FakeTQ(data))
        return cls(
            trainer_mode="sync",
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            refill_fn=refill,
        )

    def guarded(data: dict, step: int, part: str = "train", bs: int = 32):
        rb = make(sampler.FastPollReplayBufferSync, data)
        try:
            batch, _ = rb.sample(global_steps=step, partition_id=part, batch_size=bs)
            return {"keys": len(batch.keys), "err": None}
        except RuntimeError as exc:
            return {"err": str(exc)}

    def verl_only(data: dict, step: int, part: str = "train", bs: int = 32) -> int:
        batch, _ = make(rbm.ReplayBuffer, data).sample(global_steps=step, partition_id=part, batch_size=bs)
        return len(batch.keys)

    return types.SimpleNamespace(guarded=guarded, verl_only=verl_only, refills=refills)


def test_a_healthy_sync_batch_passes_verls_sampler_and_the_guard(replay):
    assert replay.guarded(_tq_state(4), 4) == {"keys": 512, "err": None}
    assert replay.refills == []


def test_lost_groups_pass_verls_sampler_and_fail_the_guard(replay):
    lost = dict.fromkeys(range(12, 32), 0)  # 20 of 32 groups lost every trajectory
    assert replay.verl_only(_tq_state(4, failed=lost), 4) == 192  # what update_actor would train on
    err = replay.guarded(_tq_state(4, failed=lost), 4)["err"]
    assert "STEP FAILED before update_actor: the train batch of step 4" in err
    assert "12 prompt groups carry trajectories, 32 were sampled" in err


@pytest.mark.parametrize(
    "kw, want",
    [
        ({"failed": {3: 8, 7: 15}}, "2 group(s) miss 9 of their rollout.n=16 sessions"),
        ({"failed": {3: 8}}, "status 'failure'"),
        ({"aborted": 2}, "2 aborted trajectories"),
        ({"stale": 1}, "trajectory_staleness min 1 max 1"),
        ({"spans": 2}, "trajectory_spans max 2"),
    ],
)
def test_partial_aborted_stale_and_spanning_batches_fail_the_step(replay, kw, want):
    err = replay.guarded(_tq_state(4, **kw), 4)["err"]
    assert err and "STEP FAILED before update_actor" in err and want in err, err


def test_validation_batches(replay):
    ok = replay.guarded(_tq_state(0, part="val", prompts=8, n=1), 0, part="val", bs=8)
    assert ok == {"keys": 8, "err": None}
    bad = replay.guarded(_tq_state(0, part="val", prompts=8, n=1, failed={2: 0}), 0, part="val", bs=8)
    assert "7 prompt groups carry trajectories, 8 were sampled" in bad["err"]


# ------------------------------------------------------------------------------------------------ full determinism


@pytest.mark.parametrize(
    "env, config, refused",
    [
        ("1", {}, True),
        ("true", {}, True),
        ("0", {}, False),  # what verl forwards to every actor when the mode is off
        (None, {}, False),
        ("0", {"full_determinism": True}, True),
    ],
)
def test_the_build_refuses_verls_full_determinism_mode(monkeypatch, env, config, refused):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", [])
    if env is None:
        monkeypatch.delenv("VERL_FULL_DETERMINISM", raising=False)
    else:
        monkeypatch.setenv("VERL_FULL_DETERMINISM", env)
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.config, r.rank = dict(config), 0
    try:
        r._prepare()
        msg = ""
    except RuntimeError as exc:  # a later check of the build may refuse too (no model config here)
        msg = str(exc)
    assert ("full_determinism" in msg and "refusing to build" in msg) is refused, msg


def test_verls_full_determinism_would_reseed_the_trainers_rng(tmp_path):
    """Why it is refused: verl's worker extension runs enable_full_determinism in the process that builds the engine,
    which in process is the trainer's (a fresh interpreter here: the call changes process-wide state)."""
    code = (
        "import json, os, torch\n"
        "from verl.workers.engine.utils import enable_full_determinism\n"
        "torch.manual_seed(1234); torch.rand(5)\n"
        "before, det, env = torch.random.get_rng_state().clone(), torch.are_deterministic_algorithms_enabled(), "
        "dict(os.environ)\n"
        "enable_full_determinism(seed=7)\n"
        "print('RESULT ' + json.dumps({'reseeded': not torch.equal(before, torch.random.get_rng_state()), "
        "'det_before': det, 'det_after': torch.are_deterministic_algorithms_enabled(), "
        "'env_changed': sorted(k for k in os.environ if os.environ.get(k) != env.get(k))}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code], env=_clean_env(), cwd=str(tmp_path), capture_output=True, text=True
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, proc.stderr[-2000:]
    res = json.loads(lines[-1][7:])
    assert res["reseeded"] and not res["det_before"] and res["det_after"], res
    assert "PYTHONHASHSEED" in res["env_changed"], res


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--child":
    if sys.argv[2] == "entry":
        res = child_entry(sys.argv[3], sys.argv[4] == "scoped")
    else:
        res = child_rank(json.loads(sys.argv[3]))
    print("RESULT " + json.dumps(res, sort_keys=True, default=str), flush=True)
