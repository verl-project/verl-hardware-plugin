# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The TPU in-process rollout serves one TP=1 engine per trainer process and refuses every other layout.

A TPU TensorCore can be opened by one process only, so the engine built inside a trainer process cannot shard over
the other trainer processes' cores, and independent TP>1 engines in one slice stall on libtpu's global barrier.
The rollout therefore refuses any tensor, data or pipeline parallel size other than 1, at three points:

* ``TPUInProcRollout.__init__``, on the rollout config (before anything is built);
* the engine build (``engine.build_vllm_engine``), on vLLM's *resolved* parallel config (whatever the CLI arguments
  said);
* ``TPUInProcReplica.launch_servers``, on the number of workers verl hands the replica.

CPU only: no device is opened; the engine build is stopped right after the check.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import types

import pytest

# the environment the children start from, taken before this process imports vLLM (its import sets variables of
# its own, which would leak into children started after it)
BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith(("VERL_TPU_INPROC_", "VERL_TPU_ROLLOUT_INPROC"))}

pytest.importorskip("vllm")

INPROC_ENV = "VERL_TPU_ROLLOUT_INPROC"
ROLLOUT_MOD = "verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout"
ENGINE_MOD = "verl_hardware_plugin.accelerators.tpu.rollout.inproc.engine"
ROLLOUT_MSG = "TP=DP=PP=1"
ENGINE_MSG = "TP=PP=DP=1"
REFUSED_LAYOUTS = [(2, 1), (1, 2), (2, 2), (4, 1), (8, 1)]  # (tensor, data); verl rejects pipeline > 1 for vLLM itself
ENGINE_LAYOUTS = [(2, 1, 1), (1, 2, 1), (1, 1, 2), (4, 1, 1)]  # (tensor, pipeline, data) as vLLM resolves them


@pytest.fixture
def inproc(monkeypatch):
    """The rollout module with the feature switch on and no other in-process switch set."""
    for name in list(os.environ):
        if name.startswith("VERL_TPU_INPROC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(INPROC_ENV, "1")
    import importlib

    return importlib.import_module(ROLLOUT_MOD)


def _rollout_config(tp: int, dp: int = 1, pp: int = 1):
    from verl.workers.config import RolloutConfig

    return RolloutConfig(
        name="vllm",
        mode="async",
        tensor_model_parallel_size=tp,
        data_parallel_size=dp,
        pipeline_model_parallel_size=pp,
        free_cache_engine=False,
        enable_prefix_caching=False,
    )


# --------------------------------------------------------------------------------------------- the rollout


@pytest.mark.parametrize("tp,dp", REFUSED_LAYOUTS)
def test_rollout_refuses_parallel_layouts(inproc, tp, dp):
    with pytest.raises(RuntimeError, match=ROLLOUT_MSG) as err:
        inproc.TPUInProcRollout(_rollout_config(tp, dp), None, None, defer_build=True)
    assert f"got world {tp * dp}" in str(err.value)


def test_rollout_refuses_verl_default_tensor_parallel_size(inproc):
    """verl's RolloutConfig defaults to tensor_model_parallel_size=2: a config that leaves it unset is refused."""
    from verl.workers.config import RolloutConfig

    cfg = RolloutConfig(name="vllm", mode="async", free_cache_engine=False)
    assert cfg.tensor_model_parallel_size == 2
    with pytest.raises(RuntimeError, match=ROLLOUT_MSG):
        inproc.TPUInProcRollout(cfg, None, None, defer_build=True)


def test_rollout_accepts_tp1(inproc):
    rollout = inproc.TPUInProcRollout(_rollout_config(1, 1), None, None, defer_build=True)
    assert rollout.replica_rank == rollout.rank


def test_verl_rejects_pipeline_parallel_vllm_rollout():
    """Pipeline parallelism never reaches the plugin through verl's config: verl refuses it for vLLM itself."""
    with pytest.raises(NotImplementedError):
        _rollout_config(1, 1, 2)


def test_rollout_refuses_pipeline_layout_given_directly(inproc):
    """A config object that bypasses verl's validation is refused by the plugin's own check."""
    cfg = types.SimpleNamespace(
        tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=2, free_cache_engine=False
    )
    with pytest.raises(RuntimeError, match=ROLLOUT_MSG):
        inproc.TPUInProcRollout(cfg, None, None, defer_build=True)


@pytest.mark.xfail(
    strict=True,
    reason="the constructor checks the product of the three sizes, so negative sizes whose product is 1 pass it; "
    "the engine build's check of vLLM's resolved parallel config still refuses such a layout",
)
def test_rollout_refuses_each_size_not_their_product(inproc):
    cfg = types.SimpleNamespace(
        tensor_model_parallel_size=-1, data_parallel_size=-1, pipeline_model_parallel_size=1, free_cache_engine=False
    )
    with pytest.raises(RuntimeError, match=ROLLOUT_MSG):
        inproc.TPUInProcRollout(cfg, None, None, defer_build=True)


# --------------------------------------------------------------------------------------------- the engine build


class _EngineReached(Exception):
    """Raised by a stand-in for vLLM's LLMEngine: the build got past the parallel-config check."""


@pytest.fixture
def engine_build(inproc, monkeypatch, tmp_path):
    """``build_vllm_engine`` (the in-process server's engine factory) on vLLM's real CLI arguments for a tiny model
    config (no weights, no tokenizer); vLLM's LLMEngine is replaced by a stand-in that records the resolved parallel
    config."""
    import vllm.v1.engine.llm_engine as llm_engine
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compile_guard

    engine = __import__(ENGINE_MOD, fromlist=["build_vllm_engine"])
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float32",
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    def reached(*args, **kwargs):
        raise _EngineReached(kwargs.get("vllm_config").parallel_config)

    monkeypatch.setattr(llm_engine, "LLMEngine", reached)
    # the logits-at-load hook goes on vllm-torchtpu's TPUWorker, absent on CPU: mark it installed (it is armed after
    # the check and disarmed in a finally)
    monkeypatch.setattr(compile_guard.LOGITS_AT_LOAD, "hooked", True)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    server = types.SimpleNamespace(model_config=types.SimpleNamespace(tokenizer=[0] * 256, processor=None))

    def build(tp: int, pp: int, dp: int):
        args = parser.parse_args(
            [
                "--model",
                str(tmp_path),
                "--skip-tokenizer-init",
                "--load-format",
                "dummy",
                "--dtype",
                "float32",
                "--max-model-len",
                "256",
                "--enforce-eager",
                "--tensor-parallel-size",
                str(tp),
                "--pipeline-parallel-size",
                str(pp),
                "--data-parallel-size",
                str(dp),
            ]
        )
        engine.build_vllm_engine(server, args)

    yield build
    assert compile_guard.LOGITS_AT_LOAD.spec is None  # disarmed whatever happened


@pytest.mark.parametrize("tp,pp,dp", ENGINE_LAYOUTS)
def test_engine_build_refuses_resolved_parallel_config(engine_build, tp, pp, dp):
    with pytest.raises(RuntimeError, match=ENGINE_MSG) as err:
        engine_build(tp, pp, dp)
    assert f"got {tp}/{pp}/{dp}" in str(err.value)


def test_engine_build_accepts_tp1(engine_build):
    with pytest.raises(_EngineReached) as err:
        engine_build(1, 1, 1)
    pc = err.value.args[0]
    assert (pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.data_parallel_size) == (1, 1, 1)


# --------------------------------------------------------------------------------------------- the replica


def _replica(tp: int):
    engine = __import__(ENGINE_MOD, fromlist=["TPUInProcReplica"])
    return engine.TPUInProcReplica(0, _rollout_config(tp), None, gpus_per_node=8)


def test_replica_refuses_more_than_one_worker(inproc):
    replica = _replica(2)
    replica.workers = [object(), object()]
    with pytest.raises(RuntimeError, match="one worker"):
        asyncio.run(replica.launch_servers())


def test_replica_refuses_more_than_one_worker_under_optimize(tmp_path):
    """The replica's refusal is a raise, not an assert statement: it holds under ``python -O`` too."""
    code = (
        "import asyncio, os\n"
        f"os.environ[{INPROC_ENV!r}] = '1'\n"
        "from verl.workers.config import RolloutConfig\n"
        f"from {ENGINE_MOD} import TPUInProcReplica\n"
        "cfg = RolloutConfig(name='vllm', mode='async', tensor_model_parallel_size=2, free_cache_engine=False)\n"
        "r = TPUInProcReplica(0, cfg, None, gpus_per_node=8)\n"
        "class W:\n"
        "    class inproc_server_info:\n"
        "        @staticmethod\n"
        "        async def remote():\n"
        "            return {'engine_ready': True, 'replica_rank': 0, 'rank': 0, 'pid': 0}\n"
        "r.workers = [W(), W()]\n"
        "try:\n"
        "    asyncio.run(r.launch_servers())\n"
        "    print('REFUSED no')\n"
        "except RuntimeError:\n"
        "    print('REFUSED yes')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-B", "-c", code],
        env=dict(BASE_ENV),
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "REFUSED yes" in proc.stdout, (proc.stdout[-500:], proc.stderr[-1500:])


def test_replica_accepts_one_worker_with_a_ready_engine(inproc):
    """One worker whose in-process engine is built and owns this replica: the server is the worker itself."""
    replica = _replica(1)

    class W:
        class inproc_server_info:
            @staticmethod
            async def remote():
                return {"engine_ready": True, "replica_rank": 0, "rank": 0, "pid": os.getpid()}

    w = W()
    replica.workers = [w]
    asyncio.run(replica.launch_servers())
    assert replica.servers == [w] and replica._server_address == "inproc://0"
