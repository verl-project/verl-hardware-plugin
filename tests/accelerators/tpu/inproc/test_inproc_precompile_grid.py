# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Every sampler and log-prob variant a request can reach maps onto the in-process engine's precompiled grid.

The in-process engine's recompile guard stops the run when an engine function compiles after warm-up (a fresh compile
at once, a recompile past the guard's limit). That default is safe only if no request the engine accepts can leave
the grid vllm-torchtpu precompiles. This test serves requests with every sampling and log-prob setting of vLLM's
``SamplingParams`` through vllm-torchtpu's real sampling code and the package's real guard hooks, on CPU:

* the runner's sampling methods (``compute_selected_logits``, ``sample_from_logits``, ``gather_logprobs``, their
  precompiles and the dispatch helpers) are executed from vllm-torchtpu's source with ``torch.compile``'s backend set
  to ``eager`` (dynamo's guards and recompiles do not depend on the backend);
* the precompiled variants are read from the runner's own precompile (they differ between vllm-torchtpu versions: the
  plain-random graph without top-k / top-p is warmed only under speculative decoding upstream), with no speculative
  decoding configured;
* each request batch is vLLM's real ``InputBatch``; each request is checked first by vLLM's own validation and by
  vllm-torchtpu's ``validate_request``, as the engine would;
* the dispatch between the compiled variants mirrors ``TPUModelRunner.sample_tokens`` (non-speculative path), and a
  source check fails the test if that dispatch changes.

Engine settings: ``TPU_SAMPLER_TOPK_CANDIDATES`` in {0, 2} and ``max_logprobs`` in {0, 5, -1}. Each setting runs in a
fresh interpreter (dynamo's caches are process-wide). Dynamo's ``recompile_limit`` is the build context's throughout
(vllm-torchtpu's runner raises it to 1024 inside the build), so the guard's own limit is what the last test reaches.

vllm-torchtpu's source is located through ``INPROC_TEST_VLLM_TORCHTPU_SRC`` (the directory that holds
``vllm_torchtpu``) or an installed ``vllm_torchtpu`` package (located, never imported). Without either, the module is
skipped.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import types

import pytest

# the environment the children start from, taken before this process imports vLLM (its import sets variables of
# its own, which would leak into children started after it)
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}

pytest.importorskip("vllm")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "support"))

import inproc_vt_runner as vt  # noqa: E402

if vt.skip_reason() is not None:
    pytest.skip(str(vt.skip_reason()), allow_module_level=True)

VT_ROOT = vt.vt_root()
RUNNER = vt.runner_file(VT_ROOT)
H, V = 16, 256  # a vocabulary of two 128-logit candidate blocks
MAX_TOKENS, MAX_SEQS = 4096, 128  # the token / request bucket ranges of the engine under test
ENGINE_SETTINGS = {
    "recipe": {"topk_blocks": 0, "max_logprobs": 0},
    "max_logprobs_5": {"topk_blocks": 0, "max_logprobs": 5},
    "max_logprobs_unlimited": {"topk_blocks": 0, "max_logprobs": -1},
    "topk_candidates_2": {"topk_blocks": 2, "max_logprobs": 0},
}
# One list of SamplingParams keyword sets per scenario; a batch cycles through them.
SCENARIOS = {
    "greedy": [{"temperature": 0.0}],
    "random": [{"temperature": 1.0}],
    "random_t0.7": [{"temperature": 0.7}],
    "top_k_1": [{"temperature": 1.0, "top_k": 1}],
    "top_k_5": [{"temperature": 1.0, "top_k": 5}],
    "top_k_vocab": [{"temperature": 1.0, "top_k": V}],
    "top_k_above_vocab": [{"temperature": 1.0, "top_k": 4 * V}],
    "top_p_0.9": [{"temperature": 1.0, "top_p": 0.9}],
    "top_k_and_top_p": [{"temperature": 1.0, "top_k": 5, "top_p": 0.9}],
    "min_p_0.1": [{"temperature": 1.0, "min_p": 0.1}],
    "penalties": [{"temperature": 1.0, "repetition_penalty": 1.2, "presence_penalty": 0.5, "frequency_penalty": 0.5}],
    "logit_bias": [{"temperature": 1.0, "logit_bias": {3: 5.0}}],
    "logprobs_0": [{"temperature": 1.0, "logprobs": 0}],
    "logprobs_1": [{"temperature": 1.0, "logprobs": 1}],
    "logprobs_5": [{"temperature": 1.0, "logprobs": 5}],
    "logprobs_all": [{"temperature": 1.0, "logprobs": -1}],
    "greedy_logprobs_0": [{"temperature": 0.0, "logprobs": 0}],
    "prompt_logprobs_1": [{"temperature": 1.0, "prompt_logprobs": 1}],
    "mixed_greedy_random": [{"temperature": 0.0}, {"temperature": 1.0}],
    "mixed_greedy_top_k": [{"temperature": 0.0}, {"temperature": 1.0, "top_k": 5}],
    "mixed_logprobs": [{"temperature": 1.0, "logprobs": 0}, {"temperature": 1.0}],
    "mixed_top_k_1_and_5": [{"temperature": 1.0, "top_k": 1}, {"temperature": 1.0, "top_k": 5}],
    "n_4": [{"temperature": 1.0, "n": 4, "logprobs": 0}],
    "seed": [{"temperature": 1.0, "seed": 1234}],
    # an RL recipe's requests as verl sends them: rollout (temperature 1, the sampled token's log-prob) and validation
    "verl_rollout": [{"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": 0}],
    "verl_validation": [{"temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "logprobs": 0}],
}
# Parameters vllm-torchtpu's sampler does not apply; they must still map onto the grid (no compile).
UNAPPLIED = ("min_p_0.1", "penalties", "logit_bias")
OFF_GRID_REQUESTS = tuple(n for n in range(9, MAX_SEQS) if n & (n - 1))  # request counts that are not request buckets
SAMPLE_TOKENS_DISPATCH = (  # TPUModelRunner.sample_tokens, non-speculative branch, in order
    "needs_logprobs = max_num_logprobs is not None",
    "all_greedy = self.input_batch.all_greedy",
    "selected_token_ids = self._greedy_sample(logits)",
    "self._build_padded_sampling_params( cur_start_idx, cur_end_idx, logits )",
    "selected_token_ids = self.sample_from_logits_func(",
    "all_greedy=all_greedy,",
    "topk_candidates=self._sampler_topk_candidates( cur_start_idx, cur_end_idx, grammar_output ),",
    "apply_topk_topp=self._sampler_applies_topk_topp(),",
    "self.gather_logprobs(logits, selected_token_ids) if needs_logprobs else None",
)
RUNNER_METHODS = (
    "_precompile_timed",
    "_update_num_xla_graphs",
    "_dummy_logits",
    "compute_selected_logits",
    "_precompile_compute_selected_logits",
    "_precompile_sample_from_logits",
    "_precompile_gather_logprobs",
    "_apply_temperature",
    "_topk_candidate_blocks",
    "_sampler_applies_topk_topp",
    "_sampler_topk_candidates",
    "_greedy_sample",
    "_build_padded_sampling_params",
    "sample_from_logits",
    "gather_logprobs",
)


# --------------------------------------------------------------------------------------------- child side


def _child_main(cfg: dict) -> dict:
    import contextlib
    from copy import copy

    import torch
    import torch._dynamo.config as dynamo_config
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.parallel_sampling import ParentRequest
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary as b
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard as cg

    blocks, max_logprobs = int(cfg["topk_blocks"]), int(cfg["max_logprobs"])
    dynamo_config.recompile_limit = 1024  # what vllm-torchtpu's runner sets in the build (kept: module docstring)
    cg.RECOMPILES.install()
    mod = vt.build_runner_module(VT_ROOT, RUNNER_METHODS, topk_blocks=blocks)
    cg.install_call_pins_and_watch()

    class TinyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(H, V, bias=False)

        def compute_logits(self, hidden):
            return self.lm_head(hidden)

    torch.manual_seed(0)
    model = TinyHead().requires_grad_(False)
    monkey_patch_compute_logits(model, V - 4, [3, 5])  # verl's mask, set at load (before anything compiles)
    runner = mod.TPUModelRunner.__new__(mod.TPUModelRunner)
    runner.num_tokens_paddings, runner.num_reqs_paddings = vt.paddings(VT_ROOT, MAX_TOKENS, MAX_SEQS)
    runner.is_pooling_model, runner.max_num_reqs, runner.vocab_size = False, MAX_SEQS, V
    runner.model_config = types.SimpleNamespace(max_logprobs=max_logprobs, get_hidden_size=lambda: H)
    runner.model, runner.device, runner._hidden_states_dtype = model, torch.device("cpu"), torch.float32
    runner.check_recompilation, runner.enforce_eager = False, False
    runner.speculative_config = None  # no speculative decoding: the runner's non-speculative grid
    fns = {
        "compute_selected_logits": vars(mod.TPUModelRunner)["compute_selected_logits"],
        "sample_from_logits": vars(mod.TPUModelRunner)["sample_from_logits"],
        "gather_logprobs": vars(mod.TPUModelRunner)["gather_logprobs"],
    }
    codes = {k: vt.compiled_code(f) for k, f in fns.items()}
    keys = {k: cg.EngineCode.key(c) for k, c in codes.items()}

    def entries():
        return {k: len(_debug_get_cache_entry_list(c)) for k, c in codes.items()}

    recorded: set = set()

    def recording(*a, **k):  # the variants the runner's own precompile compiles
        recorded.add((bool(k["all_greedy"]), int(k["topk_candidates"]), bool(k["apply_topk_topp"])))
        return runner.sample_from_logits(*a, **k)

    runner.sample_from_logits_func = recording
    torch.set_grad_enabled(True)  # the build runs inside the trainer's init_model
    with b.engine_grad_ctx("build"):
        for _ in range(2):  # the profile run's precompile, then the graph capture's pass (all hits)
            runner._precompile_compute_selected_logits()
            runner._precompile_sample_from_logits()
            runner._precompile_gather_logprobs()
    runner.sample_from_logits_func = runner.sample_from_logits
    out: dict = {
        "entries_build": entries(),
        "topk_blocks": runner._topk_candidate_blocks(),
        "keys": keys,
        "precompiled": sorted(list(v) for v in recorded),
        "req_buckets": len(runner.num_reqs_paddings),
    }
    warm = cg.RECOMPILES.mark_warm(0)
    cg.CALL_PINS.mark_warm()
    out["entries_warm"] = {k: (warm.get("entries") or {}).get(keys[k]) for k in keys}
    out["guard"] = {
        "armed": cg.RECOMPILES.armed,
        "n": cg.RECOMPILES.n,
        "fresh_mode": cg.RECOMPILES.fresh_mode,
        "compile_hooked": cg.RECOMPILES.compile_hooked,
    }
    buckets = []
    for t in runner.num_tokens_paddings:
        for r in runner.num_reqs_paddings:
            buckets.append((t, r))
            if r >= min(t, runner.max_num_reqs):
                break
    out["buckets"] = len(buckets)
    mc = types.SimpleNamespace(max_logprobs=max_logprobs, get_vocab_size=lambda: V)
    validate_request = vt.validate_request_fn(VT_ROOT)

    def requests_for(kw_list):
        """the engine-core requests of a scenario: n > 1 fans out into n children (vLLM's ParentRequest)"""
        reqs = []
        for kw in kw_list:
            sp = SamplingParams(max_tokens=64, **kw)
            sp._validate_logprobs(mc)  # vLLM's request validation (max_logprobs)
            validate_request(None, None, sp)  # vllm-torchtpu's platform validation
            if sp.n > 1:
                parent = types.SimpleNamespace(sampling_params=sp, cached_child_sampling_params=None)
                reqs.extend(ParentRequest._get_child_sampling_params(parent, i) for i in range(sp.n))
            else:
                reqs.append(sp)
        return reqs

    def batch_for(params, n_active):
        b_ = InputBatch(
            max_num_reqs=MAX_SEQS,
            max_model_len=512,
            max_num_batched_tokens=MAX_TOKENS,
            device=torch.device("cpu"),
            vocab_size=V,
            block_sizes=[16],
            max_num_blocks_per_req=[32],
        )
        for i in range(n_active):
            b_.add_request(
                CachedRequestState(
                    req_id=f"r{i}",
                    prompt_token_ids=[1, 2, 3],
                    mm_features=[],
                    sampling_params=copy(params[i % len(params)]),
                    generator=None,
                    block_ids=([i],),
                    num_computed_tokens=0,
                    output_token_ids=[],
                )
            )
        return b_

    gen = torch.Generator().manual_seed(0)

    def serve(t, r, n_active, batch):
        """one engine step's sampling: execute_model's selected logits, then sample_tokens' non-speculative dispatch"""
        runner.input_batch = batch
        with b.engine_mode_ctx("step"):
            with torch.no_grad():
                logits = runner.compute_selected_logits(
                    torch.randn(t, H, generator=gen), torch.zeros(r, dtype=torch.int32)
                )
            max_num_logprobs = batch.max_num_logprobs
            if max_num_logprobs == -1:
                raise NotImplementedError("full logprobs")
            needs_logprobs = max_num_logprobs is not None
            if batch.all_greedy:
                sel = runner._greedy_sample(logits)
                variant = (True, 0, True)
            else:
                temps, top_k, top_p = runner._build_padded_sampling_params(0, n_active, logits)
                u = torch.rand_like(logits, dtype=torch.float32)
                variant = (
                    False,
                    runner._sampler_topk_candidates(0, n_active, None),
                    runner._sampler_applies_topk_topp(),
                )
                sel = runner.sample_from_logits_func(
                    logits,
                    temps,
                    u,
                    top_k,
                    top_p,
                    all_greedy=False,
                    topk_candidates=variant[1],
                    apply_topk_topp=variant[2],
                )
            lp = runner.gather_logprobs(logits, sel) if needs_logprobs else None
        return variant, (None if lp is None else int(lp.logprobs.shape[1])), max_num_logprobs

    scen_out = {}
    for name, kw_list in SCENARIOS.items():
        rec: dict = {}
        try:
            params = requests_for(kw_list)
        except Exception as exc:  # noqa: BLE001 - a refused request
            rec["refused"] = f"{type(exc).__name__}: {exc}"[:300]
            scen_out[name] = rec
            continue
        rec["requests_per_batch_unit"] = len(params)
        post0, fresh0 = dict(cg.RECOMPILES.post), dict(cg.RECOMPILES.fresh_post)
        variants, lp_cols, want = set(), set(), set()
        cache = {}
        try:
            for t, r in buckets:
                n_active = max(1, r - 3)
                if (r, n_active) not in cache:
                    cache[(r, n_active)] = batch_for(params, n_active)
                variant, cols, mnl = serve(t, r, n_active, cache[(r, n_active)])
                variants.add(variant)
                lp_cols.add(cols)
                want.add(mnl)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rec["variants"] = sorted(list(v) for v in variants)
        rec["logprob_columns"] = sorted(c for c in lp_cols if c is not None)
        rec["max_num_logprobs"] = sorted(w for w in want if w is not None)
        rec["recompiles"] = {k: v - post0.get(k, 0) for k, v in cg.RECOMPILES.post.items() if v != post0.get(k, 0)}
        rec["fresh"] = {k: v - fresh0.get(k, 0) for k, v in cg.RECOMPILES.fresh_post.items() if v != fresh0.get(k, 0)}
        rec["tripped"] = cg.RECOMPILES.tripped
        scen_out[name] = rec
    out["scenarios"] = scen_out
    out["sync"] = cg.RECOMPILES.compile_record()
    out["sync"].pop("entries", None)
    out["sync"].pop("entries_warm", None)
    out["entries_served"] = entries()

    # controls: what the guard does with compiles outside the grid
    ctl: dict = {}
    with b.engine_mode_ctx("step"), torch.no_grad():
        runner.compute_selected_logits(
            torch.randn(32, H, generator=gen), torch.zeros(OFF_GRID_REQUESTS[0], dtype=torch.int32)
        )
    ctl["off_grid_bucket"] = {"post": dict(cg.RECOMPILES.post), "tripped": cg.RECOMPILES.tripped}
    r0 = runner.num_reqs_paddings[0]
    logits = torch.zeros(r0, V)
    sample_args = (
        logits,
        torch.ones(r0, 1),
        torch.rand(r0, V),
        torch.zeros(r0, 1, dtype=torch.int32),
        torch.ones(r0, 1),
    )
    # the greedy graph reads none of the other arguments, so dynamo installs no guard on them: every greedy variant
    # hits it
    with b.engine_mode_ctx("step"):
        for cand, apply in ((0, False), (2, True), (2, False)):
            runner.sample_from_logits_func(*sample_args, all_greedy=True, topk_candidates=cand, apply_topk_topp=apply)
    ctl["greedy_variants"] = {"post": dict(cg.RECOMPILES.post), "tripped": cg.RECOMPILES.tripped}
    # a candidate-block count _sampler_topk_candidates never returns (it returns 0 or TPU_SAMPLER_TOPK_CANDIDATES)
    unreachable = 1 if blocks else 2
    with b.engine_mode_ctx("step"):
        runner.sample_from_logits_func(
            *sample_args, all_greedy=False, topk_candidates=unreachable, apply_topk_topp=True
        )
    ctl["unreachable_variant"] = {"post": dict(cg.RECOMPILES.post), "tripped": cg.RECOMPILES.tripped}
    # more new request shapes until the guard trips (round-robin over the off-grid counts; a budget, not a target)
    tried = 1
    for n in OFF_GRID_REQUESTS[1:]:
        if cg.RECOMPILES.tripped is not None:
            break
        with b.engine_mode_ctx("step"), torch.no_grad():
            runner.compute_selected_logits(torch.randn(128, H, generator=gen), torch.zeros(n, dtype=torch.int32))
        tried += 1
    ctl["past_the_limit"] = {
        "post": dict(cg.RECOMPILES.post),
        "tripped": cg.RECOMPILES.tripped,
        "shape_recompiles_to_trip": tried,
        "precompiled": out["entries_warm"]["compute_selected_logits"],
    }
    try:
        cg.RECOMPILES.check()
        ctl["check"] = "no raise"
    except RuntimeError as exc:
        ctl["check"] = f"RuntimeError: {exc}"[:400]
    out["controls"] = ctl
    with contextlib.suppress(Exception):
        out["runner_sha256_16"] = __import__("hashlib").sha256(open(RUNNER, "rb").read()).hexdigest()[:16]
    return out


# --------------------------------------------------------------------------------------------- parent side


def _run_child(label: str, cfg: dict, cwd: str) -> dict:
    env = dict(BASE_ENV)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", json.dumps(cfg)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if not lines:
        return {"child_error": f"{label}: rc {proc.returncode}: {proc.stderr[-4000:]}"}
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    cwd = str(tmp_path_factory.mktemp("precompile-grid"))
    workers = max(1, int(os.environ.get("INPROC_TEST_WORKERS", "2")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {label: pool.submit(_run_child, label, cfg, cwd) for label, cfg in ENGINE_SETTINGS.items()}
        results = {label: f.result() for label, f in futures.items()}
    evidence = os.environ.get("INPROC_TEST_EVIDENCE_DIR")
    if evidence:
        os.makedirs(evidence, exist_ok=True)
        with open(os.path.join(evidence, "precompile_grid.json"), "w") as f:
            json.dump({"settings": ENGINE_SETTINGS, "results": results}, f, indent=1, sort_keys=True, default=str)
    return results


def _ok(res: dict) -> dict:
    assert "child_error" not in res, res["child_error"]
    return res


def _precompiled(res: dict) -> set:
    return {tuple(v) for v in res["precompiled"]}


def _expected_refusals(max_logprobs: int) -> dict:
    """scenario -> the parameter its refusal message names (vLLM's validation, vllm-torchtpu's validate_request)"""
    cap = V if max_logprobs == -1 else max_logprobs
    out = {"seed": "per-request seed"}
    for name, k in (("logprobs_1", 1), ("logprobs_5", 5), ("logprobs_all", V)):
        if k > cap:
            out[name] = "sample logprobs"
    if 1 > cap:
        out["prompt_logprobs_1"] = "prompt logprobs"
    return out


def test_sample_tokens_dispatch_is_the_one_this_test_mirrors():
    text = open(RUNNER).read()
    node = vt.function_node(ast.parse(text), "sample_tokens", "TPUModelRunner")
    src = re.sub(r"\(\s+", "( ", " ".join(ast.get_source_segment(text, node).split()))
    pos = src.find("needs_logprobs = max_num_logprobs is not None")
    assert pos >= 0
    for marker in SAMPLE_TOKENS_DISPATCH:
        nxt = src.find(marker, pos)
        assert nxt >= 0, f"sample_tokens no longer contains (in order): {marker!r}"
        pos = nxt


@pytest.mark.parametrize("setting", ENGINE_SETTINGS)
def test_the_precompile_builds_the_variant_grid(grid, setting):
    res = _ok(grid[setting])
    blocks = ENGINE_SETTINGS[setting]["topk_blocks"]
    pre = _precompiled(res)
    assert res["topk_blocks"] == blocks
    assert {(True, 0, True), (False, 0, True)} <= pre, pre  # greedy and the full random path, always
    assert {v for v in pre if v[1]} == ({(False, blocks, True)} if blocks else set()), pre  # the top-k candidates graph
    want = {
        "compute_selected_logits": res["buckets"],
        "sample_from_logits": len(pre) * res["req_buckets"],
        "gather_logprobs": res["req_buckets"],
    }
    assert res["entries_build"] == want
    assert res["entries_warm"] == want
    assert res["guard"]["armed"] and res["guard"]["fresh_mode"] == "fatal" and res["guard"]["compile_hooked"]


@pytest.mark.parametrize("setting", ENGINE_SETTINGS)
def test_every_accepted_request_maps_onto_the_precompiled_grid(grid, setting):
    res = _ok(grid[setting])
    pre = _precompiled(res)
    served = {k: v for k, v in res["scenarios"].items() if "refused" not in v}
    assert served, "no scenario was served"
    for name, rec in served.items():
        assert "error" not in rec, (name, rec)
        assert {tuple(v) for v in rec["variants"]} <= pre, (name, rec["variants"])
        assert rec["recompiles"] == {} and rec["fresh"] == {} and rec["tripped"] is None, (name, rec)
    sync = res["sync"]
    assert sync["post_warm_shape"] == {} and sync["post_warm_other"] == {}, sync
    assert sync["fresh_post_warm"] == {} and sync["unaccounted"] == {} and sync["tripped"] is False, sync
    assert res["entries_served"] == res["entries_build"]


@pytest.mark.parametrize("setting", ENGINE_SETTINGS)
def test_every_precompiled_variant_is_reachable(grid, setting):
    res = _ok(grid[setting])
    used = set()
    for rec in res["scenarios"].values():
        used |= {tuple(v) for v in rec.get("variants", [])}
    assert used == _precompiled(res)


@pytest.mark.parametrize("setting", ENGINE_SETTINGS)
def test_requests_outside_the_grid_are_refused_with_a_message(grid, setting):
    res = _ok(grid[setting])
    want = _expected_refusals(ENGINE_SETTINGS[setting]["max_logprobs"])
    got = {k: v["refused"] for k, v in res["scenarios"].items() if "refused" in v}
    assert set(got) == set(want), (got, want)
    for name, phrase in want.items():
        assert phrase in got[name], (name, got[name])


def test_unapplied_sampling_parameters_stay_on_the_grid(grid):
    """min_p, penalties and logit_bias reach no compiled sampler input (vllm-torchtpu's sampler applies temperature,
    top-k and top-p only): they select the same variant as plain random sampling and compile nothing."""
    res = _ok(grid["recipe"])
    for name in UNAPPLIED:
        rec = res["scenarios"][name]
        assert rec["variants"] == res["scenarios"]["random"]["variants"], (name, rec)
        assert rec["recompiles"] == {} and rec["fresh"] == {}, (name, rec)
    fn = vt.function_node(ast.parse(open(RUNNER).read()), "_build_padded_sampling_params", "TPUModelRunner")
    read = {
        n.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "input_batch"
    }
    assert read == {"temperature_cpu_tensor", "top_k_cpu_tensor", "top_p_cpu_tensor"}, read


class _NoTopkLogprobColumns(AssertionError):
    """vllm-torchtpu's known gap: with max_logprobs=-1, gather_logprobs returns the sampled token's column only."""


def _logprob_settings() -> list:
    """the engine settings, the max_logprobs=-1 one marked strict xfail on exactly the known gap: any other failure
    fails, and a vllm-torchtpu that fixes the gap turns the xfail into a failure (XPASS strict): remove the mark then"""
    out = []
    for name, cfg in ENGINE_SETTINGS.items():
        if cfg["max_logprobs"] == -1:
            mark = pytest.mark.xfail(
                strict=True,
                raises=_NoTopkLogprobColumns,
                reason="vllm-torchtpu's gather_logprobs adds no top-k columns when max_logprobs=-1",
            )
            out.append(pytest.param(name, marks=mark))
        else:
            out.append(name)
    return out


@pytest.mark.parametrize("setting", _logprob_settings())
def test_logprob_columns_cover_each_accepted_request(grid, setting):
    res = _ok(grid[setting])
    short = {}
    for name, rec in res["scenarios"].items():
        if "refused" in rec or not rec.get("max_num_logprobs"):
            continue
        need = 1 + max(rec["max_num_logprobs"])
        if not rec["logprob_columns"] or min(rec["logprob_columns"]) < need:
            short[name] = {"columns": rec["logprob_columns"], "need": need}
    if short and ENGINE_SETTINGS[setting]["max_logprobs"] == -1 and all(s["columns"] == [1] for s in short.values()):
        raise _NoTopkLogprobColumns(f"only the sampled token's log-prob column: {short}")
    assert not short, short


@pytest.mark.parametrize("setting", ENGINE_SETTINGS)
def test_compiles_outside_the_grid_are_counted_and_the_guard_trips_past_its_limit(grid, setting):
    res = _ok(grid[setting])
    keys, ctl = res["keys"], res["controls"]
    csl, sample = keys["compute_selected_logits"], keys["sample_from_logits"]
    assert ctl["off_grid_bucket"]["post"] == {csl: 1} and ctl["off_grid_bucket"]["tripped"] is None, ctl
    assert ctl["greedy_variants"]["post"] == {csl: 1}, ctl
    assert ctl["unreachable_variant"]["post"] == {csl: 1, sample: 1} and ctl["unreachable_variant"]["tripped"] is None
    past = ctl["past_the_limit"]
    assert past["tripped"] is not None and csl in past["tripped"], ctl
    # new input shapes are allowed up to the function's precompiled count: the next one trips (under the build
    # context's recompile_limit; a smaller serving limit can fail the call first)
    assert past["shape_recompiles_to_trip"] == past["precompiled"] + 1, past
    assert past["post"][csl] == past["shape_recompiles_to_trip"], past
    assert ctl["check"].startswith("RuntimeError") and "RECOMPILE GUARD" in ctl["check"], ctl["check"]


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    print("RESULT " + json.dumps(_child_main(json.loads(sys.argv[2])), sort_keys=True, default=str), flush=True)
