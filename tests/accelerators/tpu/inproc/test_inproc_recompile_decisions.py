# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Decision tests of the in-process engine's recompile guard (``compile_guard.RecompileLedger``), each pinning one
fail-closed or bookkeeping decision so that removing it fails a test:

* the weight sync refuses engine cache entries that no recompile / fresh-compile hook accounted for (a compile path
  that bypassed both hooks); with ``VERL_TPU_INPROC_RECOMPILE_GUARD=report`` it reports them and goes on;
* fresh compiles and ``torch._dynamo.reset`` during the build are recorded, never refused; after warm-up a fresh engine
  compile trips the guard and a reset is refused with every compiled program kept;
* verl's ``monkey_patch_model`` sent again with another mask after the build arms the guard (its marker line is the
  guard's trip, so the pump and the sync raise it);
* the build refuses when the guard is armed but its hook cannot be installed; in report mode it goes on;
* on vllm-torchtpu's real placeholder-substitution kernel (``INPROC_TEST_VLLM_TORCHTPU_SRC``; skipped without it),
  grad-mode flips inside the engine steps: without the call-site pins the kernel recompiles in the other grad mode
  and the guard trips at the ninth such recompile, failing the engine step; with the pins nothing recompiles.

Most scenarios run in a fresh interpreter (``python <this file> --child <name>``): the ledger hooks dynamo for the
whole process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest
import torch

# the environment the children start from, taken before this process imports vLLM (its import sets variables of
# its own, which would leak into children started after it)
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "support"))

import inproc_vt_runner as vt  # noqa: E402

PKG = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"
GUARD_ENV = "VERL_TPU_INPROC_RECOMPILE_GUARD"


# --------------------------------------------------------------------------------------------- child side


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{PKG}.{name}")


def _engine_fn(name: str, line: int):
    """a compiled function whose code lies under an engine package's file name (nothing is written there)"""
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "standin", "vllm_torchtpu", "runner", "tpu_runner.py")
    ns: dict = {"torch": torch}
    exec(compile("\n" * line + f"def {name}(x):\n    return x * 2 + 1\n", path, "exec"), ns)
    return torch.compile(ns[name], backend="eager", fullgraph=True, dynamic=False)


def _sync_shell():
    """a TPUInProcRollout without its build: an idle engine behind the real adapter, a tiny model and an installer
    that copies the export into it (update_weights reads only these)"""
    rollout = _mod("rollout")
    engine = _mod("engine")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 8)
            self.lm_head = torch.nn.Linear(8, 32, bias=False)

    torch.manual_seed(0)
    model = Tiny().requires_grad_(False)

    class Idle:
        model_config = types.SimpleNamespace()

        def has_unfinished_requests(self):
            return False

        def get_num_unfinished_requests(self):
            return 0

    class Installer:
        family = None

        @torch.no_grad()
        def install(self, staging, verify, pre, step=None):
            params = dict(model.named_parameters())
            for k, v in staging.items():
                params[k].copy_(v)
            return {"mode": "test direct", "tensors": len(staging), **({"_verify": {"direct": True}} if verify else {})}

        def finish_parity(self, res, staging):
            return {
                "pre_checked": None,
                "pre_mismatched": None,
                "post_checked": len(staging),
                "post_mismatched": [],
                "unresolved": [],
            }

        def coverage(self, res, staging):
            n = len(list(model.named_parameters()))
            return {"checked": True, "ok": True, "via": "test", "params": n, "written": n, "allowlisted": 0}

    r = rollout.TPUInProcRollout.__new__(rollout.TPUInProcRollout)
    r.rank = 0
    r.engine = engine.InProcAsyncEngine(Idle(), 0)
    r.worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
    r.installer = Installer()
    r.server = types.SimpleNamespace(global_steps=None)
    r.global_steps, r.sync_log, r._pending_export_t0 = None, [], None
    r._syncs_started, r._last_wsum = 1, None
    r.integrity, r.last_step, r.rss_limit = "fatal", None, 0
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    return r, base


def _sync(r, base, step: int, delta: float):
    import asyncio

    async def go():
        try:
            await r.update_weights([(k, v + delta) for k, v in base.items()], global_steps=step)
            return None
        except RuntimeError as exc:
            return str(exc)

    return asyncio.run(go())


def scenario_unaccounted_entries() -> dict:
    """an engine function compiled after warm-up through a path no ledger hook sees, then a weight sync"""
    import torch._dynamo.convert_frame as convert_frame

    cg = _mod("compile_guard")
    cg.RECOMPILES.install()
    a = _engine_fn("engine_a", 10)
    with torch.no_grad():
        a(torch.zeros(2))
    cg.RECOMPILES.mark_warm(0)
    hooked = convert_frame._compile
    convert_frame._compile = getattr(hooked, "__wrapped__", hooked)
    try:
        c = _engine_fn("engine_c", 20)
        with torch.no_grad():
            c(torch.zeros(4))
    finally:
        convert_frame._compile = hooked
    unacc = cg.RECOMPILES.compile_record().get("unaccounted")
    r, base = _sync_shell()
    return {
        "unaccounted": unacc,
        "mode": cg.RECOMPILES.fresh_mode,
        "sync_error": _sync(r, base, 2, 0.5),
        "tripped": cg.RECOMPILES.tripped,
        "synced": len(r.sync_log),
    }


def scenario_build_fresh_and_reset() -> dict:
    """fresh compiles and a dynamo reset during the build, then a hit, a fresh engine compile and a reset after
    warm-up"""
    cg = _mod("compile_guard")
    out: dict = {"installed": cg.RECOMPILES.install()}
    out["hooks"] = {"compile": cg.RECOMPILES.compile_hooked, "reset": cg.RECOMPILES.reset_hooked}
    a = _engine_fn("engine_a", 30)
    with torch.no_grad():
        a(torch.zeros(2))
        torch._dynamo.reset()  # in the build (vllm-torchtpu's profile run resets dynamo): allowed, recorded
        a(torch.zeros(2))  # compiled again (fresh) after the reset, still before warm-up
    warm = cg.RECOMPILES.mark_warm(0)
    out["warm"] = {k: warm.get(k) for k in ("fresh_before_warm", "resets_before_warm", "fresh_mode")}
    with torch.no_grad():
        a(torch.zeros(2))  # a hit: nothing
    out["tripped_after_hit"] = cg.RECOMPILES.tripped
    b = _engine_fn("engine_b", 40)
    with torch.no_grad():
        b(torch.zeros(3))  # a new engine function after warm-up: a fresh compile
    out["tripped_after_fresh"] = cg.RECOMPILES.tripped
    out["fresh_post"] = dict(cg.RECOMPILES.fresh_post)
    before = cg.engine_cache_entries()
    try:
        torch._dynamo.reset()
        out["reset_after_warm"] = "performed"
    except RuntimeError as exc:
        out["reset_after_warm"] = str(exc)
    out["entries_kept"] = cg.engine_cache_entries() == before and bool(before)
    out["resets"] = list(cg.RECOMPILES.resets)
    return out


SCENARIOS = {
    "unaccounted_entries": ({}, scenario_unaccounted_entries),
    "unaccounted_entries_report": ({GUARD_ENV: "report"}, scenario_unaccounted_entries),
    "build_fresh_and_reset": ({}, scenario_build_fresh_and_reset),
    "build_fresh_and_reset_report": ({GUARD_ENV: "report"}, scenario_build_fresh_and_reset),
}


# --------------------------------------------------------------------------------------------- parent side


def _clean_env(extra: dict) -> dict:
    env = dict(BASE_ENV)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", **extra})
    return env


def _run(name: str, tmp_path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", name],
        env=_clean_env(SCENARIOS[name][0]),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, f"{name}: rc {proc.returncode}: {proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_the_sync_refuses_cache_entries_no_hook_accounted_for(tmp_path):
    res = _run("unaccounted_entries", tmp_path)
    assert any(k.startswith("engine_c@") for k in res["unaccounted"]), res
    assert res["mode"] == "fatal" and res["tripped"] is None, res
    assert res["sync_error"] and "no recompile / fresh-compile hook accounts for" in res["sync_error"], res
    assert res["synced"] == 0, res


def test_the_sync_reports_unaccounted_entries_in_report_mode(tmp_path):
    res = _run("unaccounted_entries_report", tmp_path)
    assert res["unaccounted"] and res["mode"] == "report", res
    assert res["sync_error"] is None and res["synced"] == 1, res


def test_fresh_compiles_and_resets_in_the_build_are_recorded_never_refused(tmp_path):
    res = _run("build_fresh_and_reset", tmp_path)
    assert res["installed"] and res["hooks"] == {"compile": True, "reset": True}, res
    fresh = res["warm"]["fresh_before_warm"]
    assert [v for k, v in fresh.items() if k.startswith("engine_a@")] == [2], res
    assert res["warm"]["resets_before_warm"] == 1 and res["resets"][0]["warm"] is False, res
    assert res["tripped_after_hit"] is None, res


def test_after_warm_up_a_fresh_compile_trips_and_a_reset_is_refused_with_the_programs_kept(tmp_path):
    res = _run("build_fresh_and_reset", tmp_path)
    assert "compiled fresh after warm-up" in str(res["tripped_after_fresh"]), res
    assert "engine_b@runner/tpu_runner.py" in str(res["tripped_after_fresh"]), res
    assert "torch._dynamo.reset() called after warm-up" in res["reset_after_warm"], res
    assert "refused (the caches stay)" in res["reset_after_warm"], res
    assert res["entries_kept"] is True and res["resets"][-1] == {**res["resets"][-1], "warm": True, "refused": True}


def test_in_report_mode_a_fresh_compile_is_counted_and_a_reset_performed(tmp_path):
    res = _run("build_fresh_and_reset_report", tmp_path)
    assert res["tripped_after_fresh"] is None and any(k.startswith("engine_b@") for k in res["fresh_post"]), res
    assert res["reset_after_warm"] == "performed" and res["resets"][-1]["refused"] is False, res


def test_a_refused_monkey_patch_model_arms_the_guard(monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg

    ledger, logits = cg.RecompileLedger(), cg.LogitsAtLoad()
    monkeypatch.setattr(cg, "RECOMPILES", ledger)
    monkeypatch.setattr(cg, "LOGITS_AT_LOAD", logits)

    class Worker:
        def __init__(self):
            self.model_runner = types.SimpleNamespace(model=None)

        def load_model(self):
            self.model_runner.model = torch.nn.Linear(2, 2)

        def monkey_patch_model(self, vocab_size, banned_token_ids=None):
            self.model_runner.model.compute_logits = lambda h: h  # an instance attribute, as verl's mask is

    logits.install(Worker)
    logits.arm({"vocab_size": 10, "banned_token_ids": []})
    w = Worker()
    w.load_model()
    logits.disarm()
    w.monkey_patch_model(vocab_size=10, banned_token_ids=[])  # the same mask: a no-op
    assert ledger.tripped is None
    with pytest.raises(RuntimeError, match="monkey_patch_model re-sent after the build with another mask"):
        w.monkey_patch_model(vocab_size=11, banned_token_ids=[])
    assert str(ledger.tripped).startswith("RECOMPILE GUARD rank 0: monkey_patch_model re-sent")
    with pytest.raises(RuntimeError, match="RECOMPILE GUARD"):
        ledger.check()  # what the pump and the sync do


@pytest.mark.parametrize("mode, refused", [("fatal", True), ("report", False)])
def test_the_build_refuses_an_armed_guard_without_its_hook(monkeypatch, mode, refused):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import rollout as rollout_mod

    ledger = cg.RecompileLedger()
    ledger.unavailable = "AttributeError: no get_and_maybe_log_recompilation_reasons (test)"

    def install():
        ledger.armed = cg.recompile_guard_mode() == "fatal"
        return False

    ledger.install = install
    monkeypatch.setattr(rollout_mod, "RECOMPILES", ledger)
    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", [])
    monkeypatch.setenv(GUARD_ENV, mode)
    r = rollout_mod.TPUInProcRollout.__new__(rollout_mod.TPUInProcRollout)
    r.config = {"engine_kwargs": {"vllm": {"kv_cache_memory_bytes": 1}}}
    hf = types.SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])
    r.model_config, r.rank = types.SimpleNamespace(hf_config=hf), 0
    with pytest.raises(RuntimeError) as err:
        r._prepare()
    if refused:
        assert "the recompile guard cannot be installed" in str(err.value), err.value
        assert f"reviewed on torch {cg.REVIEWED_TORCH}" in str(err.value), err.value  # names the reviewed torch
    else:  # past the guard's check: the next refusal is the runtime check's (no trainer runtime here)
        assert "the recompile guard" not in str(err.value) and "torch.distributed is not initialized" in str(err.value)


# --------------------------------------------------------------------------------------------- the real kernel

PUMP_METHODS = (
    "_precompile_timed",
    "_update_num_xla_graphs",
    "_dummy_logits",
    "_precompile_substitute_placeholder_token",
    "_apply_async_token_substitution",
    "sample_from_logits",
    "gather_logprobs",
    "_precompile_gather_logprobs",
)


def child_pump_guard(spec: dict) -> dict:
    """the ledger installed before the build (as the build does), the call-site pins on or off, then rollout phases
    through the real pump; each phase's flipped step substitutes into another token bucket (so a flipped call is a new
    program in the other grad mode, not a hit of an earlier one)"""
    import asyncio
    import gc

    import torch._dynamo.config as dc

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as eng_mod

    gc.disable()
    dc.recompile_limit = 1024
    root = vt.vt_root()
    mod = vt.build_runner_module(root, PUMP_METHODS, functions=("_pp_step", "_substitute_placeholder_token"))
    vt.attach_step_standins(mod)
    out: dict = {"ledger": cg.RECOMPILES.install()}
    if spec["pins"]:
        cg.install_call_pins_and_watch()
    r = mod.TPUModelRunner.__new__(mod.TPUModelRunner)
    r.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    r.num_tokens_paddings, r.num_reqs_paddings = vt.paddings(root, 4096, 128)
    r._is_async_drafter, r.speculative_config, r.max_num_reqs = False, None, 128
    r.device, r.check_recompilation, r.enforce_eager, r._pp_wave = torch.device("cpu"), False, False, None
    r._fast_token_substitution, r._pre_async_results = True, None
    r.vocab_size, r._hidden_states_dtype = 64, torch.float32
    r.model_config = types.SimpleNamespace(max_logprobs=0)
    r.k, r.grad_at_kernel = 0, []
    kernel = mod._substitute_placeholder_token
    torch.set_grad_enabled(True)
    with b.engine_grad_ctx("build"):
        r._precompile_substitute_placeholder_token()
        r._precompile_gather_logprobs()
        for nr in r.num_reqs_paddings:
            r.sample_from_logits(
                r._dummy_logits(nr),
                torch.ones((nr, 1)),
                torch.full((nr, r.vocab_size), 0.5),
                torch.zeros((nr, 1), dtype=torch.int32),
                torch.ones((nr, 1)),
                all_greedy=True,
            )
    warm = cg.RECOMPILES.mark_warm(0)
    key = cg.EngineCode.key(vt.compiled_code(kernel))
    out["warm_entries"] = warm["entries"].get(key)

    class RunnerEngine:
        model_config = types.SimpleNamespace()

        def __init__(self):
            self.reqs, self.script, self.i = {}, [], 0

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
            st = self.script[self.i]
            r.execute_model(st)
            r.sample_tokens(st.num_reqs)
            self.i += 1
            if self.i < len(self.script):
                return []
            done = [types.SimpleNamespace(request_id=e, finished=True, outputs=[]) for e in self.reqs.values()]
            self.reqs.clear()
            return done

    engine = RunnerEngine()
    aeng = eng_mod.InProcAsyncEngine(engine, 0)
    phases = []
    for p in range(spec["phases"]):
        t = r.num_tokens_paddings[p % len(r.num_tokens_paddings)]
        engine.script, engine.i = (
            [
                types.SimpleNamespace(chunks=((2048, 0),), num_reqs=8, collect=False, flip=False),
                types.SimpleNamespace(chunks=((t, 4),), num_reqs=8, collect=False, flip=True),
                types.SimpleNamespace(chunks=((16, 4),), num_reqs=8, collect=False, flip=False),
            ],
            0,
        )

        async def go(p=p):
            col = await aeng.add_request(f"p{p}", {"prompt_token_ids": [1]}, types.SimpleNamespace(output_kind=None))
            await col.get()

        try:
            asyncio.run(go())
            phases.append("ok")
        except Exception as exc:  # noqa: BLE001
            phases.append(f"{type(exc).__name__}: {exc}"[:300])
            break
    rec = cg.RECOMPILES.compile_record()
    out.update(
        {
            "phases": phases,
            "tripped": cg.RECOMPILES.tripped,
            "post_warm_other": rec["post_warm_other"],
            "post_warm_shape": rec["post_warm_shape"],
            "entries_now": rec["entries"].get(key),
        }
    )
    return out


@pytest.mark.skipif(vt.skip_reason() is not None, reason=str(vt.skip_reason()))
def test_without_the_pins_the_real_kernel_recompiles_until_the_guard_trips(tmp_path):
    pytest.importorskip("vllm")

    def run(spec):
        proc = subprocess.run(
            [sys.executable, "-B", os.path.abspath(__file__), "--child", "pump_guard", json.dumps(spec)],
            env=_clean_env({}),
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
        assert lines, f"rc {proc.returncode}: {proc.stderr[-3000:]}"
        return json.loads(lines[-1][len("RESULT ") :])

    pinned = run({"pins": True, "phases": 10})
    control = run({"pins": False, "phases": 10})
    assert pinned["ledger"] and pinned["warm_entries"] == control["warm_entries"] > 0
    assert pinned["phases"] == ["ok"] * 10 and pinned["tripped"] is None, pinned
    assert pinned["post_warm_other"] == {} and pinned["post_warm_shape"] == {}, pinned
    assert pinned["entries_now"] == pinned["warm_entries"], pinned
    # without the pins each flipped call compiles a program in the other grad mode: the ninth trips the guard and
    # the engine step that compiled it fails
    trip = str(control["tripped"])
    assert "_substitute_placeholder_token@runner/tpu_runner.py" in trip and "recompiled 9 times after warm-up" in trip
    assert control["phases"][:8] == ["ok"] * 8 and "RECOMPILE GUARD" in control["phases"][8], control["phases"]


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--child":
    if sys.argv[2] == "pump_guard":
        res = child_pump_guard(json.loads(sys.argv[3]))
    else:
        res = SCENARIOS[sys.argv[2]][1]()
    print("RESULT " + json.dumps(res, sort_keys=True, default=str), flush=True)
