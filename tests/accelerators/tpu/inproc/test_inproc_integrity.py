# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Integrity tests of the TPU in-process rollout on CPU, each through the real verl / vLLM code on the path:

* verl's ``monkey_patch_model`` RPC is frozen once the mask went in at model load: the same mask again (a direct call,
  or vLLM's ``run_method`` on a ``WorkerWrapperBase``, the collective RPC path) is a no-op, another mask is refused
  with the recompile guard's marker and leaves ``compute_logits`` untouched;
* the engine entry puts the caller's thread state right for every call (a ``torch.device`` context or another
  TorchFunctionMode, inference mode, num_threads, the default dtype, deterministic algorithms, the float32 matmul
  precision): vllm-torchtpu's real ``compute_selected_logits`` keeps its precompiled programs for every bucket, and the
  caller's state is back after every call (needs ``INPROC_TEST_VLLM_TORCHTPU_SRC``).
"""

from __future__ import annotations

import contextlib
import importlib.util
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


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--child":
    res = child_entry(sys.argv[3], sys.argv[4] == "scoped")
    print("RESULT " + json.dumps(res, sort_keys=True, default=str), flush=True)
