# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Decision tests for the TPU in-process rollout's guards: each test pins one fail-closed or bookkeeping decision that
the end-to-end CPU suites exercise without asserting it, so that removing the decision fails a test.

* the call-site grad pins are frozen after the build's scan: a later scan reports a new compiled engine function and
  replaces no attribute (replacing it after the engine compiled would change the object its programs guard on);
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
