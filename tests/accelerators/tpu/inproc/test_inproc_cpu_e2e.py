# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU end-to-end coverage of the in-process rollout across model architectures: every row of the tested-layout table
(``rollout/inproc/compat.py``) is covered here, tied and untied, and layouts outside it are shown to refuse.

Each case runs ``cpu_e2e/inproc_arch_child.py`` in a fresh interpreter: the real ``TPUInProcRollout`` built with vLLM's
real ``LLMEngine`` / ``InprocClient`` / ``UniProcExecutor`` and vLLM's own model class for a tiny-width checkpoint that
keeps the architecture's real layer structure and weight naming (``cpu_e2e/inproc_tiny_models.py``); the forward is
mocked, so the tests are about the weights the engine holds. A trainer stand-in (HF fp32, real AdamW steps) syncs
through the real ``update_weights`` at the production parity cadence; after every sync the engine is compared bitwise
with an oracle (vLLM's own ``load_weights`` of the same export). Then the mutants (a fused shard or a whole tensor
missing at unverified and verified syncs, a tied LM head that differs from the embedding, a corrupted fused parameter,
the TPU transposed layout, fresh processes with incomplete exports) and variables that once weakened the checks. Pass
rules: ``cpu_e2e/inproc_arch_checks.py``.

Opt-in (minutes of CPU; needs vLLM's CPU platform and transformers): ``INPROC_TEST_CPU_E2E=1`` and
``INPROC_TEST_TOKENIZER`` = a local HF tokenizer directory whose vocabulary covers token ids 1000-41000.
"""

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_e2e")
sys.path.insert(0, HERE)

import inproc_arch_checks as checks  # noqa: E402
import inproc_tiny_models as tiny  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("INPROC_TEST_CPU_E2E") != "1" or not os.environ.get("INPROC_TEST_TOKENIZER"),
    reason="opt-in CPU end-to-end test: set INPROC_TEST_CPU_E2E=1 and INPROC_TEST_TOKENIZER",
)

TESTED = [
    "qwen3_5",
    "qwen3_5_tied",
    "qwen3",
    "qwen3_tied",
    "llama",
    "llama_tied",
    "gemma3",
    "gemma3_tied",
    "mistral",
    "mistral_tied",
    "qwen2",
    "qwen2_tied",
]
UNTESTED = ["qwen3_moe", "phi3"]
LIBTPU_ARGS = "--xla_tpu_dvfs_p_state=7 --xla_tpu_use_dynamic_smem_negotiation=true"


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    pytest.importorskip("transformers")
    pytest.importorskip("vllm.platforms.cpu")
    root = tmp_path_factory.mktemp("inproc-models")
    tok = os.environ["INPROC_TEST_TOKENIZER"]
    return {k: tiny.make_model(str(root), k, tok) for k in TESTED + UNTESTED}


def _run(tmp_path, case: dict, port: int) -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    path = os.pathsep.join([HERE] + [p for p in sys.path if p and p != HERE])
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "",
        "VLLM_TARGET_DEVICE": "cpu",
        "VERL_PLATFORM": "cpu",
        "VERL_USE_EXTERNAL_MODULES": "inproc_cpu_platform,verl_hardware_plugin",
        "INPROC_TEST_PATH": path,
        "PYTHONPATH": path,  # vLLM inspects model classes in a subprocess of the case: it needs the same import path
        "VERL_TPU_ROLLOUT_INPROC": "1",
        "VERL_TPU_INPROC_ROLE": "trainer",
        "VERL_TPU_TIMELINE": "1",
        "VERL_LOGGING_LEVEL": "INFO",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "TPU_VISIBLE_DEVICES": "0",
        "TPU_VISIBLE_CHIPS": "0",
        "TORCH_TPU_TOPOLOGY": "1,1,1,1",
        "ALLOW_MULTIPLE_LIBTPU_LOAD": "0",
        "LIBTPU_INIT_ARGS": LIBTPU_ARGS,
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "1",
        "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": "0",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "INPROC_TEST_CASE": json.dumps(case),
    }
    for k in ("ONEDNN_JIT_PROFILE", "DNNL_JIT_PROFILE"):
        if k in os.environ:
            env[k] = os.environ[k]
    child = os.path.join(HERE, "inproc_arch_child.py")
    p = subprocess.run([sys.executable, "-I", "-B", child], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, f"the case printed no RESULT (rc {p.returncode}): {p.stderr[-3000:]}"
    res = json.loads(line[-1][len("RESULT ") :])
    dump = os.environ.get("INPROC_TEST_DUMP")  # optional: keep every case's RESULT
    if dump:
        os.makedirs(dump, exist_ok=True)
        tag = f"{case['key']}.{case['mode']}" + (".added" if case.get("add_layout") else "")
        with open(os.path.join(dump, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
    return res


def _assert_all(res: dict) -> None:
    failed = [(n, d) for n, ok, d in checks.evaluate(res) if not ok]
    assert not failed, "\n".join(f"FAIL {n}\n     {d}" for n, d in failed)


@pytest.mark.parametrize("key", TESTED)
def test_tested_layout(models, tmp_path, key):
    _assert_all(_run(tmp_path, {"model_dir": models[key], "key": key, "mode": "full"}, 29600 + TESTED.index(key)))


@pytest.mark.parametrize("key", UNTESTED)
def test_untested_layout_refuses_at_build(models, tmp_path, key):
    _assert_all(_run(tmp_path, {"model_dir": models[key], "key": key, "mode": "build"}, 29620 + UNTESTED.index(key)))


def test_layout_added_without_its_test_still_fails_closed(models, tmp_path):
    """Defense in depth: a mixture-of-experts entry added to the table without a test still cannot install unverified
    expert weights."""
    case = {"model_dir": models["qwen3_moe"], "key": "qwen3_moe", "mode": "build", "add_layout": True}
    _assert_all(_run(tmp_path, case, 29630))
