# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""verl's requests through the real in-process rollout and vLLM's real engine on CPU.

Each case runs ``cpu_e2e/inproc_serve_child.py`` in a fresh interpreter: ``TPUInProcRollout`` built with vLLM's real
``LLMEngine`` / ``InprocClient`` / ``UniProcExecutor`` / scheduler / output processor, vLLM's own model class for a
tiny dense checkpoint and a mocked device side (``cpu_e2e/inproc_serve_worker.py``). Requests go through the trainer
worker's direct rollout methods the plugin adds to verl's ``ActorRolloutRefWorker``, verl's own server code on the
in-process adapter, and the pump:

* the build: vLLM's client is ``InprocClient`` with ``UniProcExecutor`` in this process, the trainer's identity
  variables are put back, vLLM's tensor-parallel group is this rank alone, and the engine is constructed under the
  engine's build scope (grad mode off) with verl's sampling mask already on the model when vLLM sizes and warms the KV
  cache;
* generation: concurrent requests with and without log-probs (and with the top-k head) return the worker's tokens;
* aborts: ``abort_all_requests(abort_only=True)`` takes running requests out of the engine and leaves admission open;
  the default abort closes admission until ``resume_generation``; the aborts are counted in the window the step guard
  reads;
* invariants: no vLLM world-group collective while the engine steps, all device work on one thread.

Opt-in like the end-to-end coverage test (minutes of CPU; vLLM's CPU platform and transformers):
``INPROC_TEST_CPU_E2E=1`` and ``INPROC_TEST_TOKENIZER`` = a local HF tokenizer directory whose vocabulary covers
token ids 1000-41000.
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
def model_dir(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("vllm.platforms.cpu")
    return tiny.make_model(
        str(tmp_path_factory.mktemp("inproc-serve-model")), "qwen3", os.environ["INPROC_TEST_TOKENIZER"]
    )


@pytest.fixture(scope="module")
def served(model_dir, tmp_path_factory):
    return run_case(tmp_path_factory.mktemp("serve"), {"model_dir": model_dir, "mode": "serve"}, 29640)


def test_the_engine_is_built_in_process(served):
    assert served["engine"] == {"client": "InprocClient", "executor": "UniProcExecutor"}, served["engine"]
    assert served["engine_line"] and served["engine_line"].endswith("OK"), served["engine_line"]
    assert served["tp_group"] == [0]
    assert "restored True" in (served["identity_line"] or ""), served["identity_line"]
    assert served["info"]["engine_ready"] is True and served["info"]["replica_rank"] == 0, served["info"]


def test_the_engine_is_constructed_under_the_build_scope_with_the_mask_already_on(served):
    """vLLM's construction steps where vllm-torchtpu precompiles on TPU (the profile run, the KV-cache init, the graph
    capture) see grad mode off inside the engine's build scope, and verl's mask on the served model already."""
    con = served["construction"]
    for where in ("determine_available_memory", "initialize_from_config", "compile_or_warm_up_model"):
        assert con[where] == {"mask_on_model": True, "grad": False, "scope": "build"}, (where, con)
    assert con["load_model"]["grad"] is False and con["load_model"]["scope"] == "build", con


def test_concurrent_requests_return_the_engines_tokens(served):
    for name in ("batch", "after_abort", "after_resume"):
        for i, rec in enumerate(served[name]):
            assert rec["tokens_ok"], (name, i, rec)
            assert (rec["log_probs"] is not None) == (i % 3 != 2), (name, i, rec)


def test_abort_only_takes_running_requests_out_of_the_engine(served):
    assert served["abort_only"]["aborted_count"] == 3, served["abort_only"]
    assert all(o["stop"] == "aborted" for o in served["abort_only_outputs"]), served["abort_only_outputs"]
    assert served["engine_unfinished_after_abort"] is False
    assert served["window_after_abort"] == {"aborted": 3, "failed": 0}, served["window_after_abort"]


def test_the_default_abort_closes_admission_until_resume(served):
    assert served["abort"]["aborted_count"] == 2, served["abort"]
    assert served["abort_outputs"] == ["aborted", "aborted"], served["abort_outputs"]
    assert all(rec["tokens_ok"] for rec in served["after_resume"]), served["after_resume"]


def test_the_step_guard_sees_the_aborts_and_the_invariants_hold(served):
    assert "2 request(s) aborted" in (served["step_health"] or ""), served["step_health"]
    assert served["window_end"] == {"aborted": 2, "failed": 0}, served["window_end"]
    assert served["collectives_world_gt1"] == {} and served["collectives_size1"] >= 0
    assert served["device_threads"] == 1
    assert served["failed_lines"] == [], served["failed_lines"]
