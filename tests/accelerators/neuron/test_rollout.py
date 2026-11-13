# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU rollout contract tests using the same scoped server stand-ins as TPU."""

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from hydra.utils import get_class

from verl_hardware_plugin.accelerators.neuron.registration import load_rollout
from verl_hardware_plugin.accelerators.neuron.rollout.worker_extension import NeuronWorkerExtension


class _FakeHttpServer:
    def _preprocess_engine_kwargs(self, engine_kwargs):
        pass


@pytest.fixture
def neuron_vllm():
    """Exercise our server hooks without importing or starting a vLLM engine."""
    server_module = "verl.workers.rollout.vllm_rollout.vllm_async_server"
    target = "verl_hardware_plugin.accelerators.neuron.rollout.vllm_neuron"
    upstream = ModuleType(server_module)
    upstream.vLLMHttpServer = _FakeHttpServer
    upstream.vLLMReplica = type("_FakeReplica", (), {})
    with mock.patch.dict(sys.modules, {server_module: upstream}):
        sys.modules.pop(target, None)
        yield importlib.import_module(target)
        sys.modules.pop(target, None)


def test_rollout_entry_points_resolve(neuron_vllm):
    assert load_rollout() is neuron_vllm.NeuronVLLMReplica
    extension = neuron_vllm.NeuronVLLMHttpServer._get_worker_extension_cls(None)
    assert get_class(extension) is NeuronWorkerExtension


def server(returns):
    return SimpleNamespace(
        _submission_paused=True,
        _sync_connected=True,
        _sync_spec=SimpleNamespace(timeout_s=10),
        _sync_schedule=[],
        _sync_bucket_bytes=1024,
        _sync_checksums={"model.norm.weight": [2.0, 2.0]},
        config=SimpleNamespace(tensor_model_parallel_size=2),
        collective_rpc=AsyncMock(side_effect=returns),
        engine=SimpleNamespace(reset_prefix_cache=AsyncMock()),
        global_steps=0,
    )


def test_rollout_environment_does_not_autoload_training_backend(neuron_vllm):
    environment = neuron_vllm.serving_environment()
    assert environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert environment["VERL_NEURON_PROCESS_ROLE"] == "rollout"
    assert environment["VLLM_NEURON_BACKEND"] == "neuron_native"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_server_translates_ray_core_allocation_for_vllm_children(monkeypatch, neuron_vllm):
    import os

    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "6,7")
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "0,1")
    instance = object.__new__(neuron_vllm.NeuronVLLMHttpServer)
    kwargs = {}
    instance._preprocess_engine_kwargs(kwargs)
    assert "NEURON_RT_VISIBLE_CORES" not in os.environ
    assert os.environ["NEURON_VISIBLE_DEVICES"] == "6,7"
    assert kwargs["distributed_executor_backend"] == "mp"
    assert kwargs["async_scheduling"] is False
    assert kwargs["worker_cls"] == "verl_hardware_plugin.accelerators.neuron.rollout.worker.NeuronRLWorker"
    assert kwargs["additional_config"]["neuron_config"]["enable_structured_outputs"] is True


def test_server_rejects_sampling_options_native_generation_cannot_honor(neuron_vllm):
    preprocess = neuron_vllm.NeuronVLLMHttpServer._preprocess_sampling_params
    preprocess(None, {"temperature": 1.0, "top_k": -1, "top_p": 1.0})
    with pytest.raises(ValueError, match="repetition_penalty"):
        preprocess(None, {"repetition_penalty": 1.1})


def test_update_requires_pause(neuron_vllm):
    instance = server([])
    instance._submission_paused = False
    with pytest.raises(RuntimeError, match="paused"):
        asyncio.run(neuron_vllm.NeuronVLLMHttpServer.update_weights(instance, global_steps=1))
    instance.collective_rpc.assert_not_called()


def test_rollout_rejects_missing_sampled_token_logprobs(neuron_vllm):
    output = SimpleNamespace(token_ids=[7, 8], log_probs=[])
    with pytest.raises(RuntimeError, match="each sampled token"):
        neuron_vllm.NeuronVLLMHttpServer._validate_output(output, True)
    output.log_probs = [-0.5, -1.0]
    neuron_vllm.NeuronVLLMHttpServer._validate_output(output, True)


def test_collective_rpc_returns_worker_acknowledgements(neuron_vllm):
    instance = SimpleNamespace(engine=SimpleNamespace(collective_rpc=AsyncMock(return_value=[3, 4])))
    result = asyncio.run(neuron_vllm.NeuronVLLMHttpServer.collective_rpc(instance, "weight_sync_recv", timeout=12))
    assert result == [3, 4]
    instance.engine.collective_rpc.assert_awaited_once_with(method="weight_sync_recv", timeout=12, args=(), kwargs=None)


@pytest.mark.parametrize("config_value", [{"repetition_penalty": 1.2}, {"full_determinism": True}])
def test_generate_validates_configuration_defaults(neuron_vllm, config_value):
    from omegaconf import OmegaConf

    instance = object.__new__(neuron_vllm.NeuronVLLMHttpServer)
    instance.config = OmegaConf.create({"full_determinism": False, "seed": 42, **config_value})
    instance.replica_rank = 0
    with pytest.raises(ValueError, match="Native Neuron sampling"):
        asyncio.run(instance.generate([1], {}, "request"))


@pytest.mark.parametrize("tokens,logprobs", [([2], [-0.4]), ([], None)])
def test_generate_delegates_and_preserves_abort(neuron_vllm, monkeypatch, tokens, logprobs):
    from omegaconf import OmegaConf

    result = SimpleNamespace(token_ids=tokens, log_probs=logprobs)
    generate = AsyncMock(return_value=result)
    monkeypatch.setattr(_FakeHttpServer, "generate", generate, raising=False)
    instance = object.__new__(neuron_vllm.NeuronVLLMHttpServer)
    instance.config = OmegaConf.create({"full_determinism": False})
    params = {"logprobs": True}
    assert asyncio.run(instance.generate([1], params, "request")) is result
    assert params == {"logprobs": True}
    assert generate.call_args.args[1]["detokenize"] is False
    if tokens:
        result.log_probs = []
        with pytest.raises(RuntimeError, match="each sampled token"):
            asyncio.run(instance.generate([1], params, "request"))


@pytest.mark.parametrize("returns", [[[5, 0]], [[5, 5], [{}, {}]]])
def test_failed_update_does_not_advance_policy(returns, neuron_vllm):
    instance = server(returns)
    with pytest.raises(RuntimeError):
        asyncio.run(neuron_vllm.NeuronVLLMHttpServer.update_weights(instance, global_steps=1))
    assert instance.global_steps == 0
    assert instance._submission_paused
    instance.engine.reset_prefix_cache.assert_not_called()


def test_update_advances_only_after_all_workers_apply_and_verify(neuron_vllm):
    checksum = {"model.norm.weight": [2.0, 2.0]}
    instance = server([[5, 5], [checksum, checksum]])
    result = asyncio.run(neuron_vllm.NeuronVLLMHttpServer.update_weights(instance, global_steps=1))
    assert instance.global_steps == 1
    assert result["neuron_nixl/verified_replicated_parameters"] == 1
    instance.engine.reset_prefix_cache.assert_awaited_once()
