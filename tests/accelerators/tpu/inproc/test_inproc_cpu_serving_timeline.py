# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The timeline's request spans of in-process requests carry their first-token time, through vLLM's real engine.

The in-process requests are FINAL_ONLY (one ``RequestOutput`` per request, at its end), so without the first-token
stamps the timeline's ``request`` span would read first == end for every request. The requests run through the real
in-process rollout on CPU (``cpu_e2e/inproc_serve_child.py``, mode ``timeline``, ``VERL_TPU_TIMELINE=1``), with the lazy
output path on (the stamps come from its loop) and off (from the scan-only wrapper of ``process_outputs``). Every span
must carry a first-token time near its start (multi-token requests) and the watch must end with every request stamped
and none pending; the replica record says TP 1 on one node.

Opt-in like the end-to-end coverage test: ``INPROC_TEST_CPU_E2E=1`` and ``INPROC_TEST_TOKENIZER``.
"""

import os
import sys

import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_e2e")
sys.path.insert(0, HERE)

import inproc_tiny_models as tiny  # noqa: E402
from inproc_serve_cases import run_case  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("INPROC_TEST_CPU_E2E") != "1" or not os.environ.get("INPROC_TEST_TOKENIZER"),
    reason="opt-in CPU serving test: set INPROC_TEST_CPU_E2E=1 and INPROC_TEST_TOKENIZER",
)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("vllm.platforms.cpu")
    root = tmp_path_factory.mktemp("inproc-serve-timeline")
    model = tiny.make_model(str(root), "qwen3", os.environ["INPROC_TEST_TOKENIZER"])
    case = {"model_dir": model, "mode": "timeline"}
    on = run_case(tmp_path_factory.mktemp("tl-lazy"), case, 29660, {"VERL_TPU_TIMELINE": "1"})
    off = run_case(
        tmp_path_factory.mktemp("tl-scan"),
        case,
        29661,
        {"VERL_TPU_TIMELINE": "1", "VERL_TPU_INPROC_LAZY_OUTPUT": "0"},
    )
    return {"lazy": on, "scan": off}


@pytest.mark.parametrize("path", ["lazy", "scan"])
def test_every_request_span_carries_its_first_token_time(runs, path):
    res = runs[path]
    assert res["lazy_installed"] is (path == "lazy")
    want = [30 + 10 * i for i in range(12)] + [1]
    assert res["lengths"] == want
    spans = res["spans"]
    assert sorted(s["args"]["out"] for s in spans) == sorted(want), spans
    for s in spans:
        a = s["args"]
        assert a["stop"] == "length" and a["first"] >= a["admit"], s  # every request ran to its max_tokens
        if a["out"] >= 2:
            assert a["first"] < s["dur"], s  # stamped at the first token, not at the final output
        if a["out"] >= 60:
            assert a["first"] < 0.5 * s["dur"], s
        if a["out"] == 1:
            assert a["first"] <= s["dur"] + 1e-4, s


@pytest.mark.parametrize("path", ["lazy", "scan"])
def test_the_watch_stamps_every_request_and_keeps_none_pending(runs, path):
    assert runs[path]["watch"] == {"sink_set": True, "stamped": 13, "late": 0, "pending": 0}, runs[path]["watch"]


def test_the_replica_record(runs):
    rec = runs["lazy"]["replica"]
    assert len(rec) == 1 and rec[0]["args"]["tp"] == 1 and rec[0]["args"]["nnodes"] == 1, rec
    assert rec[0]["args"]["host"] == "inproc-rank0", rec
