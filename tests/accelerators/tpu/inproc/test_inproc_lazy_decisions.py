# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Decision tests of the lazy final-only output path (``rollout/inproc/lazy_output.py``), each pinning one decision
that the end-to-end suites exercise without asserting it:

* every weight sync re-arms the path's self-check, so the next rollout's first deferring request is verified again
  against vLLM's eager processing (not only the first rollout after the build);
* a request's first output (the one that ends its prefill, with the prefill's stats) goes to vLLM's own
  ``process_outputs``: the request state after it and the final output equal vLLM's own, bit for bit.

Each scenario runs in a fresh interpreter (``python <this file> --child <name>``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# the environment the children start from, taken before this process imports vLLM (its import sets variables of
# its own, which would leak into children started after it)
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}

pytest.importorskip("vllm")

PKG = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"


# --------------------------------------------------------------------------------------------- child side


def _mod(name: str):
    import importlib

    return importlib.import_module(f"{PKG}.{name}")


def _sync_shell(lazy):
    """a TPUInProcRollout without its build whose server holds ``lazy``: an idle engine behind the real adapter, a tiny
    model and an installer that copies the export into it"""
    import types

    import torch

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
    r.engine.lazy = lazy
    r.worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
    r.installer = Installer()
    r.server = types.SimpleNamespace(global_steps=None, _inproc_lazy=lazy)
    r.global_steps, r.sync_log, r._pending_export_t0 = None, [], None
    r._syncs_started, r._last_wsum = 1, None
    r.integrity, r.last_step, r.rss_limit = "fatal", None, 0
    base = {k: v.detach().clone() for k, v in model.named_parameters()}
    return r, base


def scenario_lazy_rearm() -> dict:
    import asyncio
    import types

    from vllm.v1.engine.output_processor import OutputProcessor

    lo = _mod("lazy_output")
    lazy = lo.LazyFinalOutputs(types.SimpleNamespace(output_processor=OutputProcessor(tokenizer=None, log_stats=False)))
    lazy.arm_verify()
    armed_at_build = lazy.verify_left
    r, base = _sync_shell(lazy)
    out = {"installed": lazy.installed, "why_not": lazy.why_not, "armed_at_build": armed_at_build, "syncs": []}
    for step in (2, 3):
        lazy.verify_left = 0  # the rollout's first deferring request took the self-check

        async def go(s=step):
            try:
                await r.update_weights([(k, v + 0.1 * s) for k, v in base.items()], global_steps=s)
                return None
            except RuntimeError as exc:
                return str(exc)

        err = asyncio.run(go())
        out["syncs"].append({"error": err, "armed_after_sync": lazy.verify_left})
    out["lazy_verify_n"] = lo.LAZY_VERIFY_N
    return out


def scenario_lazy_prefill() -> dict:
    """one FINAL_ONLY request whose first output carries prefill stats, through vLLM's OutputProcessor with and without
    the lazy output path"""
    import struct
    import types

    import numpy as np
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.metrics.stats import PrefillStats
    from vllm.v1.outputs import LogprobsLists

    lo = _mod("lazy_output")

    def run(lazy: bool):
        proc = OutputProcessor(tokenizer=None, log_stats=False)
        installed = None
        if lazy:
            lz = lo.LazyFinalOutputs(types.SimpleNamespace(output_processor=proc), environ={})
            installed = lz.installed
        sp = SamplingParams(
            temperature=1.0, max_tokens=6, logprobs=0, detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY
        )
        proc.add_request(
            EngineCoreRequest(
                request_id="r0-int",
                prompt_token_ids=[1, 2, 3, 4, 5],
                mm_features=None,
                sampling_params=sp,
                pooling_params=None,
                arrival_time=0.0,
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
                external_req_id="r0",
            ),
            None,
        )
        stats = PrefillStats()
        stats.set(num_prompt_tokens=5, num_local_cached_tokens=3, num_external_cached_tokens=0)
        after_first, final = None, None
        for k in range(6):
            lists = LogprobsLists(
                np.array([[100 + k]], dtype=np.int64),
                np.array([[-0.25 * (k + 1)]], dtype=np.float32),
                np.array([1 + k % 2], dtype=np.int64),
            )
            o = EngineCoreOutput(
                request_id="r0-int",
                new_token_ids=[100 + k],
                new_logprobs=lists.slice_request(0, 1),
                finish_reason=FinishReason.LENGTH if k == 5 else None,
                prefill_stats=stats if k == 0 else None,
            )
            res = proc.process_outputs([o])
            if k == 0:
                st = proc.request_states["r0-int"]
                after_first = {
                    "is_prefilling": st.is_prefilling,
                    "num_cached_tokens": st.num_cached_tokens,
                    "queued_tokens": proc.get_num_queued_tokens(),
                }
            for ro in res.request_outputs:
                c = ro.outputs[0]
                final = {
                    "finished": ro.finished,
                    "token_ids": list(c.token_ids),
                    "num_cached_tokens": ro.num_cached_tokens,
                    "cumulative_logprob": struct.pack("<d", c.cumulative_logprob).hex(),
                    "logprobs": repr(lo.lp_canon(c.logprobs)),
                }
        return {"after_first": after_first, "final": final, "installed": installed}

    return {"vllm": run(False), "lazy": run(True)}


SCENARIOS = {"lazy_rearm": scenario_lazy_rearm, "lazy_prefill": scenario_lazy_prefill}


# --------------------------------------------------------------------------------------------- parent side


def _run(name: str, tmp_path) -> dict:
    env = dict(BASE_ENV)
    env.update({"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", name],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=1200,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, f"{name}: rc {proc.returncode}: {proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def test_every_sync_rearms_the_lazy_output_self_check(tmp_path):
    res = _run("lazy_rearm", tmp_path)
    if not res["installed"]:
        pytest.skip(f"the lazy output path is off on this vLLM: {res['why_not']}")
    assert res["armed_at_build"] == res["lazy_verify_n"] > 0, res
    for rec in res["syncs"]:
        assert rec["error"] is None and rec["armed_after_sync"] == res["lazy_verify_n"], res


def test_the_lazy_path_hands_the_prefilling_output_to_vllm(tmp_path):
    res = _run("lazy_prefill", tmp_path)
    if not res["lazy"]["installed"]:
        pytest.skip("the lazy output path is off on this vLLM")
    assert res["vllm"]["after_first"] == {"is_prefilling": False, "num_cached_tokens": 3, "queued_tokens": 0}, res
    assert {k: v for k, v in res["lazy"].items() if k != "installed"} == {
        k: v for k, v in res["vllm"].items() if k != "installed"
    }, res


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    print("RESULT " + json.dumps(SCENARIOS[sys.argv[2]](), sort_keys=True, default=str), flush=True)
