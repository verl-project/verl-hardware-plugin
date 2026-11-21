# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Decision tests for the TPU in-process rollout's guards: each test pins one fail-closed or bookkeeping decision that
the end-to-end CPU suites exercise without asserting it, so that removing the decision fails a test.

* the call-site grad pins are frozen after the build's scan: a later scan reports a new compiled engine function and
  replaces no attribute (replacing it after the engine compiled would change the object its programs guard on).

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


SCENARIOS = {
    "pin_freeze": scenario_pin_freeze,
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


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    print("RESULT " + json.dumps(SCENARIOS[sys.argv[2]](), sort_keys=True, default=str), flush=True)
