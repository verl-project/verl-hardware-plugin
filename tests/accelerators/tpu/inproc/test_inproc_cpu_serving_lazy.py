# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The lazy final-only output path end to end, bit for bit, through vLLM's real engine on CPU.

The same requests run twice through the real in-process rollout (``cpu_e2e/inproc_serve_child.py``, mode ``lazy``):
once with vLLM's own per-step output processing (``VERL_TPU_INPROC_LAZY_OUTPUT=0``) and once with the lazy path. Three
waves of concurrent requests mix the sampled token's log-prob (verl's rollout), the top-2 head (flat log-probs, three
columns) and no log-probs, with lengths 20 to 144; the self-check is re-armed after each wave as every weight sync
does. Every final ``RequestOutput`` the pump hands to a request and every ``TokenOutput`` verl returns must be
identical between the two runs, bit for bit; the lazy run must have deferred outputs and a verified, equal request in
every wave.

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
    root = tmp_path_factory.mktemp("inproc-serve-lazy")
    model = tiny.make_model(str(root), "qwen3", os.environ["INPROC_TEST_TOKENIZER"])
    case = {"model_dir": model, "mode": "lazy"}
    off = run_case(tmp_path_factory.mktemp("lazy-off"), case, 29650, {"VERL_TPU_INPROC_LAZY_OUTPUT": "0"})
    on = run_case(tmp_path_factory.mktemp("lazy-on"), case, 29651)
    return {"off": off, "on": on}


def test_the_lazy_path_is_installed_only_when_not_opted_out(runs):
    assert runs["on"]["lazy_state"]["installed"] is True, runs["on"]["lazy_state"]
    assert runs["off"]["lazy_state"] == {"installed": False, "why_not": "VERL_TPU_INPROC_LAZY_OUTPUT=0"}


def test_final_outputs_are_bitwise_equal_with_and_without_the_lazy_path(runs):
    off, on = runs["off"], runs["on"]
    assert len(on["finals"]) == len(off["finals"]) == 36
    assert on["finals"] == off["finals"]
    assert [w["token_outputs"] for w in on["waves"]] == [w["token_outputs"] for w in off["waves"]]
    assert on["failed_lines"] == [] and off["failed_lines"] == []


def test_every_wave_defers_outputs_and_verifies_a_request(runs):
    for w in runs["on"]["waves"]:
        win = w["window"]
        assert win["installed"] is True and win["deferred_outputs"] > 0, win
        assert win["verified_requests"] >= 1 and win["verified_equal"] == win["verified_requests"], win
        assert win["flushes"] >= win["verified_requests"], win
    for w in runs["off"]["waves"]:
        assert w["window"]["installed"] is False and w["window"]["deferred_outputs"] == 0, w["window"]
