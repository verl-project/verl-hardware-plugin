# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Decision tests for the TPU in-process rollout's guards: each test pins one fail-closed or bookkeeping decision that
the end-to-end CPU suites exercise without asserting it, so that removing the decision fails a test.

* the call-site grad pins are frozen after the build's scan: a later scan reports a new compiled engine function and
  replaces no attribute (replacing it after the engine compiled would change the object its programs guard on);
* the trainer's update_actor / compute_log_prob entries read the last sync's staleness probe back, so the step guard
  refuses update_actor on a stale verdict before anything trains;
* a failed engine step counts the requests it failed (the step guard's window reads the counts);
* a direct weight install that raises stops the run (there is no fallback to vLLM's loaders), and no environment
  variable brings a fallback back.

Each scenario runs in a fresh interpreter (``python <this file> --child <name>``): the guards are process-wide
singletons.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys

import pytest

PKG = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}


# --------------------------------------------------------------------------------------------- child side


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{PKG}.{name}")


def _quiet(fn):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue().splitlines()


def _engine_file(name: str) -> str:
    """a file name under an engine package (the guards key engine code by its path; nothing is written there)"""
    import tempfile

    return os.path.join(tempfile.gettempdir(), "standin", "vllm_torchtpu", "runner", name)


def _compiled(name: str, line: int, path: str):
    import torch

    ns: dict = {"torch": torch}
    exec(compile("\n" * line + f"def {name}(x, *args, **kwargs):\n    return x * 2 + 1\n", path, "exec"), ns)
    return torch.compile(ns[name], backend="eager", fullgraph=True, dynamic=False)


def scenario_pin_freeze() -> dict:
    """the build's pin scan (vllm-torchtpu's runner module loaded), then a later scan that meets a compiled engine
    function loaded after the build"""
    import types

    cg = _mod("compile_guard")
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compat import RUNNER_MOD

    runner_file = _engine_file("tpu_runner.py")
    for pkg in ("vllm_torchtpu", "vllm_torchtpu.runner", "vllm_torchtpu.layers"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
    runner = types.ModuleType(RUNNER_MOD)
    runner.__file__ = runner_file
    runner._substitute_placeholder_token = _compiled("_substitute_placeholder_token", 30, runner_file)
    runner.TPUModelRunner = type(
        "TPUModelRunner",
        (),
        {
            "__module__": runner.__name__,
            "sample_from_logits": _compiled("sample_from_logits", 40, runner_file),
            "gather_logprobs": _compiled("gather_logprobs", 50, runner_file),
        },
    )
    sys.modules[runner.__name__] = runner
    originals = {
        "kernel": runner._substitute_placeholder_token,
        "sample": vars(runner.TPUModelRunner)["sample_from_logits"],
    }
    cg.install_call_pins_and_watch()
    pinned = {
        "kernel": runner._substitute_placeholder_token,
        "sample": vars(runner.TPUModelRunner)["sample_from_logits"],
    }
    late = types.ModuleType("vllm_torchtpu.layers.late_ops")
    late.__file__ = _engine_file("late_ops.py")
    original_late = late.late_op = _compiled("late_op", 60, late.__file__)
    sys.modules[late.__name__] = late
    scan = cg.CALL_PINS.install()
    warm = cg.CALL_PINS.mark_warm()
    return {
        "frozen": cg.CALL_PINS.frozen,
        "late_replaced": late.late_op is not original_late,
        "late_reported": [k for k in (scan.get("unpinned_after_freeze") or {}) if k.startswith("late_op@")],
        "warm_reported": [k for k in (warm.get("unpinned_after_freeze") or {}) if k.startswith("late_op@")],
        "pinned_kept": runner._substitute_placeholder_token is pinned["kernel"]
        and vars(runner.TPUModelRunner)["sample_from_logits"] is pinned["sample"],
        "required_pinned": all(pinned[k] is not originals[k] for k in originals),
        "required_missing": cg.CALL_PINS.required_missing(),
    }


def _stale_rollout():
    """a rollout whose staleness probe holds a pending read-back of a sync that changed no sampled weight"""
    import types

    import torch

    rollout = _mod("rollout")
    ws = _mod("weight_sync")
    r = rollout.TPUInProcRollout.__new__(rollout.TPUInProcRollout)
    r.rank = 0
    r.engine = types.SimpleNamespace(_error=None, window_failed=0, window_aborted=0)
    g = torch.Generator().manual_seed(0)
    named = {f"t{i}": torch.randn(64, 64, generator=g).to(torch.bfloat16) for i in range(4)}
    probe = ws.StaleProbe(0)
    probe.dispatch(4, named, list(named))
    probe.resolve("update_actor")
    probe.dispatch(5, named, list(named))  # sync 5 installed nothing new
    r.stale = probe
    return r, types.SimpleNamespace(rollout=r)


def scenario_trainer_entry() -> dict:
    import importlib

    inproc = importlib.import_module(PKG)
    out = {}
    r, worker = _stale_rollout()
    trained = []
    wrapped = inproc._wrap_worker_method(lambda self, data: trained.append(1) or "trained", "update_actor")
    try:
        res = wrapped(worker, None)
        out["update_actor"] = f"returned {res!r}"
    except RuntimeError as exc:
        out["update_actor"] = f"RuntimeError: {exc}"[:400]
    out["trained"] = len(trained)
    out["update_actor_read_back"] = r.stale.pending is None
    r2, worker2 = _stale_rollout()
    wrapped_lp = inproc._wrap_worker_method(lambda self, data: "logprobs", "compute_log_prob")
    out["compute_log_prob"] = wrapped_lp(worker2, None)
    out["compute_log_prob_read_back"] = r2.stale.pending is None
    out["compute_log_prob_verdict"] = r2.stale.fatal_pending
    r3, worker3 = _stale_rollout()
    wrapped_init = inproc._wrap_worker_method(lambda self: "built", "init_model")
    out["init_model"] = wrapped_init(worker3)
    out["init_model_read_back"] = r3.stale.pending is None
    return out


def scenario_failed_requests() -> dict:
    import asyncio
    import importlib.util
    import types

    if importlib.util.find_spec("vllm") is None:  # the adapter sets vLLM's output kind on every request
        return {"skipped": "vLLM is not installed"}
    engine = _mod("engine")

    class FailingEngine:
        def __init__(self):
            self.model_config, self.reqs = types.SimpleNamespace(), {}

        def add_request(self, rid, prompt, params, **_):
            self.reqs[f"{rid}-int"] = rid
            return f"{rid}-int"

        def has_unfinished_requests(self):
            return bool(self.reqs)

        def get_num_unfinished_requests(self):
            return len(self.reqs)

        def abort_request(self, ids, internal=False):
            for i in ids:
                self.reqs.pop(i, None)

        def step(self):
            raise RuntimeError("injected engine-step failure")

    eng = engine.InProcAsyncEngine(FailingEngine(), 0)

    async def go():
        cols = [
            await eng.add_request(f"r{i}", {"prompt_token_ids": [1]}, types.SimpleNamespace(output_kind=None))
            for i in range(4)
        ]
        res = []
        for c in cols:
            try:
                await c.get()
                res.append("ok")
            except Exception as exc:  # noqa: BLE001
                res.append(type(exc).__name__)
        task = eng._pump_task
        if task is not None:
            try:
                await task
            except Exception:  # noqa: BLE001
                pass
        return res

    res = asyncio.run(go())
    rollout = _mod("rollout")
    r = rollout.TPUInProcRollout.__new__(rollout.TPUInProcRollout)
    r.engine, r.stale = eng, None
    return {
        "requests": res,
        "failed": eng.failed,
        "step_health": r.step_health(),
        "window": eng.window_take(),
        "window_after_take": eng.window_take(),
        "add_after_failure": _add_after_failure(eng),
    }


def _add_after_failure(eng) -> str:
    import asyncio
    import types

    async def go():
        try:
            await eng.add_request("late", {"prompt_token_ids": [1]}, types.SimpleNamespace(output_kind=None))
            return "accepted"
        except RuntimeError as exc:
            return f"RuntimeError: {exc}"[:200]

    return asyncio.run(go())


def _tiny_model():
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 8)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)

    torch.manual_seed(0)
    return Tiny().requires_grad_(False)


def scenario_direct_install_raises() -> dict:
    """a direct install that raises at an unverified sync, with and without the old fallback variable set"""
    import types

    import torch

    ws = _mod("weight_sync")
    common = _mod("common")
    out = {}
    for policy in ("default", "allow"):
        if policy == "allow":
            os.environ["VERL_TPU_INPROC_LOADER_FALLBACK"] = "allow"
        inst = ws.InProcInstaller(
            types.SimpleNamespace(model_runner=types.SimpleNamespace(model=_tiny_model())),
            types.SimpleNamespace(model_type="qwen3_5_text"),
            0,
        )
        inst.mode = "qwen3_5 direct"  # what plan_once selects for a Qwen3.5 model with a direct layout
        calls = []

        def install_once(staging, verify, pre, _inst=inst, _calls=calls):
            _calls.append({"mode": _inst.mode, "verify": verify})
            if _inst.mode != ws.LOADER_FAMILY:
                raise RuntimeError("direct install failed (injected)")
            return {"mode": _inst.mode, "tensors": len(staging)}

        inst._install_once = install_once
        staging = {"lm_head.weight": torch.zeros(32, 8)}
        rec = {}
        for step in (3, 4):
            try:
                res = inst.install(staging, verify=False, pre=False, step=step)
                rec[step] = {"mode": res["mode"]}
            except RuntimeError as exc:
                rec[step] = {"error": str(exc)[:300]}
        unread = common.check_switches({**os.environ, common.INPROC_ENV: "1"})
        out[policy] = {"steps": rec, "calls": calls, "mode_after": inst.mode, "unread": unread}
        os.environ.pop("VERL_TPU_INPROC_LOADER_FALLBACK", None)
    return out


SCENARIOS = {
    "pin_freeze": scenario_pin_freeze,
    "trainer_entry": scenario_trainer_entry,
    "failed_requests": scenario_failed_requests,
    "direct_install_raises": scenario_direct_install_raises,
}


# --------------------------------------------------------------------------------------------- parent side


def _run_child(name: str, cwd: str) -> dict:
    env = dict(BASE_ENV)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", name],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if not lines:
        return {"child_error": f"{name}: rc {proc.returncode}: {proc.stderr[-3000:]}"}
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    cwd = str(tmp_path_factory.mktemp("guard-decisions"))
    workers = max(1, int(os.environ.get("INPROC_TEST_WORKERS", "2")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(_run_child, name, cwd) for name in SCENARIOS}
        return {name: f.result() for name, f in futures.items()}


def _ok(res: dict) -> dict:
    assert "child_error" not in res, res["child_error"]
    return res


def test_pins_are_frozen_after_the_build_scan(results):
    res = _ok(results["pin_freeze"])
    assert res["required_pinned"] and res["pinned_kept"] and res["required_missing"] == [], res
    assert res["frozen"] is True, res
    assert res["late_replaced"] is False, "a scan after the freeze replaced a module attribute"
    assert res["late_reported"] and res["warm_reported"], (
        "the frozen scan did not report the unpinned compiled function"
    )


def test_update_actor_reads_the_staleness_probe_before_the_step_guard(results):
    res = _ok(results["trainer_entry"])
    assert res["update_actor_read_back"] is True, res
    assert res["update_actor"].startswith("RuntimeError") and "STEP FAILED before update_actor" in res["update_actor"]
    assert "STALE WEIGHTS" in res["update_actor"], res
    assert res["trained"] == 0


def test_compute_log_prob_reads_the_staleness_probe_back(results):
    res = _ok(results["trainer_entry"])
    assert res["compute_log_prob"] == "logprobs", res  # read back, never raised at this collective entry
    assert res["compute_log_prob_read_back"] is True
    assert res["compute_log_prob_verdict"] and "STALE WEIGHTS" in res["compute_log_prob_verdict"], res


def test_init_model_does_not_read_the_probe(results):
    res = _ok(results["trainer_entry"])
    assert res["init_model"] == "built" and res["init_model_read_back"] is False, res


def test_failed_engine_step_counts_the_failed_requests(results):
    res = _ok(results["failed_requests"])
    if res.get("skipped"):
        pytest.skip(res["skipped"])
    assert res["requests"] == ["RuntimeError"] * 4, res
    assert res["failed"] == 4 and res["window"] == {"aborted": 0, "failed": 4}, res
    assert res["window_after_take"] == {"aborted": 0, "failed": 0}, res
    assert res["step_health"] and "the in-process engine failed" in res["step_health"], res
    assert res["add_after_failure"].startswith("RuntimeError") and "failed earlier" in res["add_after_failure"], res


@pytest.mark.parametrize("policy", ["default", "allow"])
def test_failing_direct_install_stops_the_run(results, policy):
    """No fallback to vLLM's loaders, whatever the environment says: the first raising direct install is fatal, the
    install mode stays the planned one, and the old fallback variable is named as not read."""
    res = _ok(results["direct_install_raises"])[policy]
    for step in ("3", "4"):
        err = res["steps"][step].get("error", "")
        assert "INTEGRITY FAILED rank 0" in err and f"the qwen3_5 direct install raised at step {step}" in err, res
    assert [c["mode"] for c in res["calls"]] == ["qwen3_5 direct", "qwen3_5 direct"], res
    assert res["mode_after"] == "qwen3_5 direct", res
    assert res["unread"] == (["VERL_TPU_INPROC_LOADER_FALLBACK"] if policy == "allow" else []), res


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    print("RESULT " + json.dumps(SCENARIOS[sys.argv[2]](), sort_keys=True, default=str), flush=True)
