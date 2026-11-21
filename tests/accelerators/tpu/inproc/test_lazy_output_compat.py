# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Tests of the guard that keeps the lazy final-only output path off unless vLLM's output processing is the reviewed,
trace-tested code (``rollout/inproc/lazy_output_compat.py``), and of the guarded path itself (``lazy_output.py``).

CPU only, no Ray, no TPU. Path-free: the vLLM under test is the one on ``sys.path``. Optional input (tests that need it
are skipped without it): ``INPROC_TEST_VLLM_TREES``: ``tag=dir,tag=dir,...``; each ``dir`` holds a ``vllm`` package
source tree (only the modules the guard reads are needed). ``v0.32.0`` is the reviewed version.
"""

from __future__ import annotations

import contextlib
import functools
import os
import random
import struct
import types

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import lazy_output_compat as L

REVIEWED_TAG = "v0.32.0"


def _trees() -> dict:
    raw = os.environ.get("INPROC_TEST_VLLM_TREES", "").strip()
    out = {}
    for item in filter(None, (x.strip() for x in raw.split(","))):
        tag, _, path = item.partition("=")
        out[tag] = path
    return out


def _vllm_version():
    try:
        import vllm

        return str(vllm.__version__)
    except Exception:  # noqa: BLE001
        return None


needs_trees = pytest.mark.skipif(not _trees(), reason="INPROC_TEST_VLLM_TREES not set")
needs_reviewed_vllm = pytest.mark.skipif(
    _vllm_version() != L.REVIEWED_VLLM, reason="the imported vLLM is not the reviewed version"
)


# ------------------------------------------------------------------------------------------------ the reviewed facts


@needs_trees
def test_reviewed_facts_are_the_v0320_sources():
    path = _trees().get(REVIEWED_TAG)
    if not path:
        pytest.skip(f"no {REVIEWED_TAG} tree")
    facts = L.collect_source_facts(path, REVIEWED_TAG)
    for key, want in L.REVIEWED.items():
        assert key in facts, key
        got = facts[key]
        assert (got == want) if key.startswith("digest:") else (tuple(got) == tuple(want)), (key, got, want)
    result = L.check(facts=facts)
    assert result.ok, result.reasons


@needs_reviewed_vllm
def test_runtime_check_passes_on_the_reviewed_vllm():
    result = L.check()
    assert result.ok, result.reasons
    assert result.vllm_version == L.REVIEWED_VLLM
    assert all(v == "ok" for v in result.checks.values())


@needs_reviewed_vllm
@needs_trees
def test_runtime_facts_equal_the_source_facts_of_the_same_version():
    path = _trees().get(REVIEWED_TAG)
    if not path:
        pytest.skip(f"no {REVIEWED_TAG} tree")
    runtime = L.collect_runtime_facts()
    static = L.collect_source_facts(path, REVIEWED_TAG)
    keys = (set(runtime) | set(static)) - {"vllm_version"}
    diff = {k: (runtime.get(k), static.get(k)) for k in keys if runtime.get(k) != static.get(k)}
    assert not diff, diff


# what each other version must trip (a subset of its reasons; the guard may find more)
EXPECTED_TRIPS = {
    "v0.31.0": (
        "params:vllm.v1.engine.output_processor:OutputProcessor.process_outputs",
        "digest:vllm.v1.engine.output_processor:OutputProcessor.process_outputs",
    ),
    "v0.33.0": (
        "digest:vllm.v1.engine.output_processor:OutputProcessor.process_outputs",
        "digest:vllm.v1.engine.output_processor:RequestState.make_request_output",
    ),
    "origin_main": (
        "fields:vllm.v1.engine:EngineCoreOutput",
        "digest:vllm.v1.engine.output_processor:OutputProcessor.process_outputs",
    ),
}


@needs_trees
def test_other_vllm_versions_disable_the_lazy_path():
    trees = _trees()
    seen = 0
    for tag, keys in EXPECTED_TRIPS.items():
        if tag not in trees:
            continue
        seen += 1
        result = L.check(facts=L.collect_source_facts(trees[tag], tag))
        assert not result.ok, tag
        for key in keys:
            assert result.checks.get(key, "").startswith("mismatch"), (tag, key, result.checks.get(key))
    if not seen:
        pytest.skip("no other vLLM tree given")


# ---------------------------------------------------------------------------------------- synthetic negatives (no vLLM)


def _reviewed_facts() -> dict:
    facts = {k: (tuple(v) if not k.startswith("digest:") else v) for k, v in L.REVIEWED.items()}
    facts["vllm_version"] = L.REVIEWED_VLLM
    module, qual = L.REQUEST_STATE_INIT
    facts[f"attrs:{module}:{qual}"] = tuple(sorted(L.REQUEST_STATE_ATTRS))
    module, qual = L.OUTPUT_KIND_ENUM
    facts[f"members:{module}:{qual}"] = ("CUMULATIVE", "DELTA", "FINAL_ONLY")
    for module, qual in L.DIGEST_TARGETS:
        facts[f"wrapped:{module}:{qual}"] = False
    return facts


def test_reviewed_facts_pass():
    assert L.check(facts=_reviewed_facts()).ok


@pytest.mark.parametrize("version", ["0.32.1", "0.33.0", None])
def test_a_version_without_a_trace_test_disables(version):
    """The reviewed code alone is not enough: the vLLM version must also be one the bitwise trace test ran on."""
    facts = _reviewed_facts()
    facts["vllm_version"] = version
    result = L.check(facts=facts)
    assert not result.ok and any("is not a trace-tested version" in r for r in result.reasons), result.reasons
    assert all(v == "ok" for v in result.checks.values())


@pytest.mark.parametrize(
    "mutation, expect",
    [
        ("added_field", "added ['watermarking']"),
        ("removed_field", "removed ['stop_reason']"),
        ("reordered_lists", "reordered"),
        ("digest", "mismatch: digest"),
        ("param", "removed ['weight_version']"),
        ("missing_module", "unavailable"),
        ("request_state_attr", "no longer assigns ['streaming_input']"),
        ("no_final_only", "no RequestOutputKind.FINAL_ONLY"),
        ("wrapped", "wrapped or replaced"),
    ],
)
def test_any_difference_disables(mutation, expect):
    facts = _reviewed_facts()
    eco = "fields:vllm.v1.engine:EngineCoreOutput"
    if mutation == "added_field":
        facts[eco] = facts[eco] + ("watermarking",)
    elif mutation == "removed_field":
        facts[eco] = tuple(x for x in facts[eco] if x != "stop_reason")
    elif mutation == "reordered_lists":
        k = "fields:vllm.v1.outputs:LogprobsLists"
        f = list(facts[k])
        f[0], f[1] = f[1], f[0]
        facts[k] = tuple(f)
    elif mutation == "digest":
        facts["digest:vllm.v1.engine.detokenizer:IncrementalDetokenizer"] = "0" * 16
    elif mutation == "param":
        k = "params:vllm.v1.engine.output_processor:OutputProcessor.process_outputs"
        facts[k] = facts[k][:-1]
    elif mutation == "missing_module":
        for k in [k for k in facts if ":vllm.logprobs:" in k]:
            del facts[k]
    elif mutation == "request_state_attr":
        module, qual = L.REQUEST_STATE_INIT
        k = f"attrs:{module}:{qual}"
        facts[k] = tuple(x for x in facts[k] if x != "streaming_input")
    elif mutation == "no_final_only":
        module, qual = L.OUTPUT_KIND_ENUM
        facts[f"members:{module}:{qual}"] = ("CUMULATIVE", "DELTA")
    elif mutation == "wrapped":
        facts["wrapped:vllm.v1.engine.output_processor:OutputProcessor.process_outputs"] = True
    result = L.check(facts=facts)
    assert not result.ok
    assert any(expect in r for r in result.reasons), result.reasons


def test_check_never_raises(monkeypatch):
    def boom():
        raise ImportError("no module named vllm")

    monkeypatch.setattr(L, "collect_runtime_facts", boom)
    result = L.check()
    assert not result.ok and "cannot inspect" in result.reasons[0]


def test_digest_ignores_docstrings_comments_and_formatting():
    a = "def f(x):\n    '''doc'''\n    # comment\n    return x + 1\n"
    b = "def f(x):\n    return (x +\n            1)  # other\n"
    c = "def f(x):\n    return x + 2\n"
    assert L.source_digest(a) == L.source_digest(b) != L.source_digest(c)


# ----------------------------------------------------------------------------- run-time replacement (the imported vLLM)


@needs_reviewed_vllm
def test_a_wrapped_process_outputs_disables(monkeypatch):
    from vllm.v1.engine.output_processor import OutputProcessor

    orig = OutputProcessor.process_outputs

    @functools.wraps(orig)
    def wrapper(self, *a, **k):
        return orig(self, *a, **k)

    monkeypatch.setattr(OutputProcessor, "process_outputs", wrapper)
    result = L.check()
    assert not result.ok
    assert result.checks["wrapped:vllm.v1.engine.output_processor:OutputProcessor.process_outputs"].startswith(
        "mismatch"
    )
    monkeypatch.undo()
    assert L.check().ok


@needs_reviewed_vllm
def test_a_replaced_detokenizer_method_disables(monkeypatch):
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    def update(self, new_token_ids, stop_terminated):
        self.token_ids.extend(new_token_ids)
        return None

    monkeypatch.setattr(IncrementalDetokenizer, "update", update)
    result = L.check()
    assert not result.ok
    assert result.checks["wrapped:vllm.v1.engine.detokenizer:IncrementalDetokenizer"].startswith("mismatch")


@needs_reviewed_vllm
def test_processor_check():
    from vllm.v1.engine.output_processor import OutputProcessor

    proc = OutputProcessor(tokenizer=None, log_stats=False)
    assert L.check_processor(proc) == []
    assert L.check(proc=proc).ok

    class Sub(OutputProcessor):
        def process_outputs(self, *a, **k):
            return super().process_outputs(*a, **k)

    assert any("overrides" in p for p in L.check_processor(Sub(tokenizer=None, log_stats=False)))
    proc2 = OutputProcessor(tokenizer=None, log_stats=False)
    proc2.process_outputs = lambda *a, **k: None
    assert any("already replaced" in p for p in L.check_processor(proc2))
    assert any("not vLLM's OutputProcessor" in p for p in L.check_processor(types.SimpleNamespace(request_states={})))


# ------------------------------------------------------------ the guarded install, end to end on vLLM's OutputProcessor
# The trace driver follows the randomized-trace test of the lazy path: requests of every kind the in-process engine
# can meet without a tokenizer, a changing batch composition, aborts and weight-version changes; every step's
# RequestOutputs canonical and bitwise (floats as their bytes).


def _inproc():
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import lazy_output

    return lazy_output


@contextlib.contextmanager
def _facts(facts):
    """the guard's run-time facts replaced by ``facts`` while the lazy path is built (None: the imported vLLM's)"""
    if facts is None:
        yield
        return
    orig = L.collect_runtime_facts
    L.collect_runtime_facts = lambda: facts
    try:
        yield
    finally:
        L.collect_runtime_facts = orig


def guarded_lazy(ti, proc, facts=None):
    """LazyFinalOutputs runs the guard itself: installed only when it passes; otherwise vLLM's process_outputs stays
    and the guard's reasons are recorded"""
    with _facts(facts):
        result = L.check(facts=facts, proc=proc)
        lz = ti.LazyFinalOutputs(types.SimpleNamespace(output_processor=proc), environ={})
    return lz, result


def _fb(x) -> bytes:
    return struct.pack("<d", float(x))


def _canon(ti, ro) -> tuple:
    outs = []
    for c in ro.outputs:
        re_ = getattr(c, "routed_experts", None)
        outs.append(
            (
                c.index,
                c.text,
                list(c.token_ids),
                None if c.logprobs is None else ti.lp_canon(c.logprobs),
                None if c.cumulative_logprob is None else _fb(c.cumulative_logprob),
                c.finish_reason,
                c.stop_reason,
                None if re_ is None else (tuple(re_.shape), re_.tobytes()),
                repr(getattr(c, "weight_versions", None)),
            )
        )
    return (
        ro.request_id,
        ro.finished,
        list(ro.prompt_token_ids or []),
        repr(ro.prompt_logprobs),
        ro.num_cached_tokens,
        outs,
    )


KINDS = ("lp0", "lp0", "lp0", "lpnone", "lp3", "lp2flat", "cumulative", "n2", "multi", "routed", "lp0abort")


def _trace(seed: int, n_req: int, steps: int) -> dict:
    rng = random.Random(seed)
    reqs = []
    for i in range(n_req):
        kind = KINDS[i % len(KINDS)] if i < len(KINDS) else rng.choice(KINDS)
        reqs.append(
            {
                "i": i,
                "kind": kind,
                "prompt": [1000 + (7 * i + j) % 3000 for j in range(4 + i % 5)],
                "length": rng.randint(3, 40),
                "start": rng.randint(0, steps // 3),
                "abort_at": rng.randint(4, 12) if kind == "lp0abort" else None,
            }
        )
    sched = [[r["i"] for r in reqs if r["start"] <= k and rng.random() < 0.85] for k in range(steps)]
    return {"reqs": reqs, "sched": sched, "seed": seed}


def _sp(kind: str, length: int):
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    base = dict(temperature=1.0, max_tokens=length, output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
    if kind in ("lp0", "multi", "routed", "lp0abort"):
        return SamplingParams(logprobs=0, **base), 1
    if kind == "lpnone":
        return SamplingParams(logprobs=None, **base), 1
    if kind == "lp3":
        return SamplingParams(logprobs=3, **base), 1
    if kind == "lp2flat":
        return SamplingParams(logprobs=2, flat_logprobs=True, **base), 1
    if kind == "cumulative":
        return SamplingParams(
            temperature=1.0, max_tokens=length, logprobs=0, detokenize=False, output_kind=RequestOutputKind.CUMULATIVE
        ), 1
    if kind == "n2":
        return SamplingParams(n=2, logprobs=0, **base), 2
    raise ValueError(kind)


def _drive(ti, trace: dict, mode: str, facts=None) -> dict:
    """mode: 'vllm' (untouched processor), 'guarded' (guarded_lazy with the given facts)"""
    import numpy as np
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.engine.parallel_sampling import ParentRequest
    from vllm.v1.outputs import LogprobsLists

    proc = OutputProcessor(tokenizer=None, log_stats=False)
    lz = result = None
    if mode == "guarded":
        lz, result = guarded_lazy(ti, proc, facts)
        lz.arm_verify(2)
    state, by_req = {}, {}
    for r in trace["reqs"]:
        sp, n = _sp(r["kind"], r["length"])
        base = EngineCoreRequest(
            request_id=f"q{r['i']}-int",
            prompt_token_ids=list(r["prompt"]),
            mm_features=None,
            sampling_params=sp,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id=f"q{r['i']}",
        )
        ids = []
        if n == 1:
            proc.add_request(base, None)
            ids.append(base.request_id)
        else:
            parent = ParentRequest(base)
            for idx in range(n):
                cid, csp = parent.get_child_info(idx)
                creq = EngineCoreRequest(
                    request_id=cid,
                    prompt_token_ids=list(r["prompt"]),
                    mm_features=None,
                    sampling_params=csp,
                    pooling_params=None,
                    arrival_time=0.0,
                    lora_request=None,
                    cache_salt=None,
                    data_parallel_rank=None,
                    external_req_id=f"q{r['i']}",
                )
                proc.add_request(creq, None, parent, idx)
                ids.append(cid)
        by_req[r["i"]] = ids
        for e in ids:
            state[e] = {
                "r": r,
                "out": 0,
                "ntok": 2 if r["kind"] == "multi" else 1,
                "done": False,
                "nlp": sp.logprobs if r["kind"] != "n2" else 0,
            }
    steps = []
    for k, sched in enumerate(trace["sched"]):
        wv = f"v{k // 9}" if k % 2 == 0 else None
        batch = []
        for i in sched:
            for e in by_req[i]:
                s = state[e]
                r = s["r"]
                if (
                    s["done"]
                    or e not in proc.request_states
                    or (r["abort_at"] is not None and s["out"] >= r["abort_at"])
                ):
                    continue
                batch.append((e, min(s["ntok"], r["length"] - s["out"])))
        for r in trace["reqs"]:
            if r["abort_at"] is not None:
                for e in by_req[r["i"]]:
                    s = state[e]
                    if not s["done"] and s["out"] >= r["abort_at"] and e in proc.request_states:
                        proc.abort_requests([e], internal=True)
                        s["done"] = True
        rows = sum(n for _, n in batch)
        wants = [state[e]["nlp"] for e, _ in batch if state[e]["nlp"] is not None]
        width = 1 + max([w for w in wants if w is not None and w >= 0] or [0])
        rng = random.Random(trace["seed"] * 100003 + k)
        ids_a = np.zeros((max(rows, 1), width), dtype=np.int64)
        lps_a = np.zeros((max(rows, 1), width), dtype=np.float32)
        rks_a = np.ones(max(rows, 1), dtype=np.int64)
        toks, row = [], 0
        for e, n_new in batch:
            s = state[e]
            tl = []
            for j in range(n_new):
                t = 1000 + (sum(s["r"]["prompt"]) * 31 + 7919 * (s["out"] + j) + 13 * s["r"]["i"]) % 30000
                tl.append(t)
                ids_a[row + j, 0] = t
                for c in range(1, width):
                    ids_a[row + j, c] = t + c
                v = -abs(rng.gauss(0.5, 0.4))
                if rng.random() < 0.02:
                    v = float("-inf")
                lps_a[row + j, 0] = v
                for c in range(1, width):
                    lps_a[row + j, c] = -0.1 * c - abs(rng.gauss(0, 0.1))
                rks_a[row + j] = 1 + (s["out"] + j) % 3
            toks.append(tl)
            row += n_new
        lists = LogprobsLists(ids_a, lps_a, rks_a)
        outs, row = [], 0
        for (e, n_new), tl in zip(batch, toks, strict=False):
            s = state[e]
            fin = FinishReason.LENGTH if s["out"] + n_new >= s["r"]["length"] else None
            kw = {}
            if s["r"]["kind"] == "routed" and s["out"] % 5 == 0:
                kw["routed_experts"] = np.full((n_new, 2), s["out"] + 3 * s["r"]["i"], dtype=np.int32)
            outs.append(
                EngineCoreOutput(
                    request_id=e,
                    new_token_ids=list(tl),
                    finish_reason=fin,
                    new_logprobs=lists.slice_request(row, n_new) if s["nlp"] is not None else None,
                    **kw,
                )
            )
            s["out"] += n_new
            row += n_new
            if fin is not None:
                s["done"] = True
        res = proc.process_outputs(outs, engine_core_timestamp=1000.0 + k, iteration_stats=None, weight_version=wv)
        for e in res.reqs_to_abort:
            state[e]["done"] = True
        steps.append(([_canon(ti, ro) for ro in res.request_outputs], sorted(res.reqs_to_abort)))
    return {
        "steps": steps,
        "left": sorted(proc.request_states),
        "lazy": lz.take() if lz is not None else None,
        "installed": None if lz is None else lz.installed,
        "why_not": None if lz is None else lz.why_not,
        "instance_override": "process_outputs" in proc.__dict__,
        "guard": result,
    }


@needs_reviewed_vllm
def test_guard_on_installs_the_lazy_path_and_outputs_stay_bitwise_equal():
    ti = _inproc()
    agg = {"outputs": 0, "deferred": 0, "rows_one_pass": 0, "lists_vllm": 0, "verified_equal": 0}
    for seed in range(6):
        tr = _trace(seed, 40, 60)
        ref = _drive(ti, tr, "vllm")
        got = _drive(ti, tr, "guarded")
        assert got["guard"].ok, got["guard"].reasons
        assert got["installed"] and got["instance_override"]
        assert got["steps"] == ref["steps"] and got["left"] == ref["left"], seed
        agg["outputs"] += sum(len(s[0]) for s in ref["steps"])
        st = got["lazy"] or {}
        agg["deferred"] += int(st.get("deferred_outputs", 0))
        for key in ("rows_one_pass", "lists_vllm", "verified_equal"):
            agg[key] += int(st.get(key, 0))
    assert (
        agg["outputs"] > 150 and agg["deferred"] > 500 and agg["rows_one_pass"] > 500 and agg["verified_equal"] >= 6
    ), agg


@needs_reviewed_vllm
def test_guard_off_leaves_vllms_process_outputs_in_place():
    ti = _inproc()
    facts = L.collect_runtime_facts()
    facts["fields:vllm.v1.engine:EngineCoreOutput"] = tuple(facts["fields:vllm.v1.engine:EngineCoreOutput"]) + (
        "watermarking",
    )
    tr = _trace(7, 30, 50)
    ref = _drive(ti, tr, "vllm")
    got = _drive(ti, tr, "guarded", facts=facts)
    assert not got["guard"].ok
    assert got["installed"] is False and got["instance_override"] is False
    assert "added ['watermarking']" in (got["why_not"] or "")
    assert got["steps"] == ref["steps"] and got["left"] == ref["left"]
