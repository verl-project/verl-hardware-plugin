# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Vocabulary restrictions must survive native LM-head sharding and grammars."""

import importlib
import os
from types import MethodType, SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.neuron.rollout.worker_extension import (
    NeuronWorkerExtension,
    _native_logprobs,
    native_sample_logprobs,
    resident_weight_model,
    restrict_token_mask,
)


def test_native_weight_names_share_compiled_storage_and_restore_on_error():
    model = torch.nn.Linear(4, 2, bias=False)
    compiled = torch.compile(model, backend="eager")
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=compiled))
    with pytest.raises(RuntimeError, match="sentinel"):
        with resident_weight_model(worker):
            assert list(dict(worker.model_runner.model.named_parameters())) == ["weight"]
            assert worker.model_runner.model.weight is compiled._orig_mod.weight
            raise RuntimeError("sentinel")
    assert worker.model_runner.model is compiled


def test_checksum_uses_unwrapped_parameter_names():
    worker = NeuronWorkerExtension()
    model = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    worker.model_runner = SimpleNamespace(model=torch.compile(model, backend="eager"))
    assert worker.weight_sync_checksum(["weight"]) == {"weight": [16.0, 16.0]}


def test_receiver_refuses_cpu_before_connecting_to_a_store():
    worker = SimpleNamespace(device="cpu")
    with pytest.raises(ValueError, match="real Trainium receiver"):
        NeuronWorkerExtension.weight_sync_connect(worker, {})


@pytest.mark.parametrize("offset,expected", [(0, [True, False, True, True]), (4, [True, True, False, False])])
def test_restrict_sharded_mask(offset, expected):
    actual = restrict_token_mask(torch.ones(2, 4, dtype=torch.bool), 7, (1, 6), offset)
    assert actual.tolist() == [expected, expected]


def test_restrictions_preserve_grammar_and_none():
    assert restrict_token_mask(None, 3, ()) is None
    mask = torch.tensor([[True, False, True, True]])
    assert restrict_token_mask(mask, 3, (2,)).tolist() == [[True, False, False, False]]


@pytest.mark.parametrize("incompatible_hook", [False, True])
def test_worker_installs_idempotent_native_mask_hooks(incompatible_hook):
    runner = SimpleNamespace(
        neuron_config=SimpleNamespace(
            enable_structured_outputs=True,
            on_device_sampling_config=SimpleNamespace(all_greedy=False, deterministic=False, max_top_k=256),
        ),
        sampler=SimpleNamespace(logprobs_mode="processed_logprobs"),
        use_async_scheduling=False,
        on_device_sampling=True,
        vocab_size=8,
        lm_head_tp_rank=1,
        _lm_head_vocab_shard_size=lambda: 4,
        _use_lm_head_sharded_logit_mask=lambda: True,
        _build_noop_logit_mask=lambda num_logit_rows, device, compiled_graph_input=False: torch.ones(
            num_logit_rows, 4, dtype=torch.bool, device=device
        ),
        _get_grammar_bitmask=lambda scheduler_output: torch.ones(1, 8, dtype=torch.bool),
        _sample=lambda model_output_tensor, spec_decode_metadata=None: None,
    )
    worker = SimpleNamespace(model_runner=runner)
    if incompatible_hook:
        runner._sample = lambda changed_signature: None
        original_mask = runner._build_noop_logit_mask
        with pytest.raises(RuntimeError, match="Incompatible native vLLM hook _sample"):
            NeuronWorkerExtension.monkey_patch_model(worker, 7, [1, 6])
        assert runner._build_noop_logit_mask is original_mask
        return
    NeuronWorkerExtension.monkey_patch_model(worker, 7, [1, 6])
    first_hook = runner._build_noop_logit_mask
    NeuronWorkerExtension.monkey_patch_model(worker, 7, [6, 1])
    assert runner._build_noop_logit_mask is first_hook
    assert first_hook(1, "cpu").tolist() == [[True, True, False, False]]
    assert runner._get_grammar_bitmask(None).tolist() == [[True, False, True, True, True, True, False, False]]
    with pytest.raises(ValueError, match="Cannot change"):
        NeuronWorkerExtension.monkey_patch_model(worker, 6)


def test_worker_rejects_graphs_without_mask_input():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(neuron_config=SimpleNamespace(enable_structured_outputs=False))
    )
    with pytest.raises(RuntimeError, match="before compilation"):
        NeuronWorkerExtension.monkey_patch_model(worker, 7)


def test_text_only_cache_reset_rejects_a_resident_vision_cache():
    worker = SimpleNamespace(model_runner=SimpleNamespace(encoder_cache=None))
    NeuronWorkerExtension.reset_encoder_cache(worker)
    worker.model_runner.encoder_cache = object()
    with pytest.raises(NotImplementedError, match="text-only"):
        NeuronWorkerExtension.reset_encoder_cache(worker)


@pytest.mark.parametrize("top_p,candidates", [(None, 256), (0.751, 192)])
def test_native_logprobs_match_capped_distribution_and_preserve_tied_sample(top_p, candidates):
    logits = torch.zeros(1, 512)
    # Choose an ID absent from CPU top-k to exercise different distributed tie ordering.
    selected = next(token for token in range(512) if token not in logits.topk(256).indices[0])
    metadata = SimpleNamespace(
        top_k=None,
        top_p=None if top_p is None else torch.tensor([top_p]),
        temperature=torch.ones(1),
    )
    logprobs = _native_logprobs(logits, torch.tensor([selected]), metadata, max_top_k=256)
    torch.testing.assert_close(logprobs[0, selected], torch.tensor(1.0 / candidates).log())
    torch.testing.assert_close(logprobs.exp().sum(-1), torch.ones(1))
    assert torch.isfinite(logprobs).sum() == candidates
    assert metadata.top_k is None


@pytest.fixture
def sampling_runner():
    """Share real vLLM sampling setup; tests supply only their runner hooks."""
    pytest.importorskip("vllm")
    device = os.environ.get("NEURON_TEST_SAMPLER_DEVICE", "cpu")
    if device != "cpu":
        importlib.import_module("libtorch_neuronx_lite")

    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler

    def make_runner(num_requests, greedy, *, temperature=1.0, top_k=None, max_num_logprobs=0, **hooks):
        metadata = SamplingMetadata(
            temperature=None if greedy else torch.full((num_requests,), temperature, device=device),
            all_greedy=greedy,
            all_random=not greedy,
            top_p=None,
            top_k=None if top_k is None else torch.full((num_requests,), top_k, dtype=torch.int32, device=device),
            generators={},
            max_num_logprobs=max_num_logprobs,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(num_requests),
            presence_penalties=torch.zeros(num_requests),
            repetition_penalties=torch.ones(num_requests),
            output_token_ids=[],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        return SimpleNamespace(
            device=device,
            input_batch=SimpleNamespace(num_reqs=num_requests, sampling_metadata=metadata),
            sampler=Sampler(logprobs_mode="processed_logprobs"),
            **hooks,
        )

    return make_runner


@pytest.mark.parametrize("topk", [0, 1])
@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("num_requests,padded_rows", [(1, 1), (3, 4)])
def test_native_logprobs_with_real_vllm_sampler(sampling_runner, topk, greedy, num_requests, padded_rows):
    runner = sampling_runner(
        num_requests,
        greedy,
        max_num_logprobs=topk,
        neuron_config=SimpleNamespace(on_device_sampling_config=SimpleNamespace(max_top_k=256)),
        _lm_head_vocab_shard_size=lambda: 3,
    )
    runner._on_device_logits = torch.tensor([[3.0, 1.0, 20.0]]).repeat(padded_rows, 1).to(runner.device)
    metadata = runner.input_batch.sampling_metadata
    output = native_sample_logprobs(runner, torch.ones(padded_rows, dtype=torch.long), 2, ())
    assert output.sampled_token_ids == [[1]] * num_requests
    assert output.logprobs_tensors.logprobs.shape[0] == num_requests
    assert output.logprobs_tensors.logprob_token_ids[0, 0] == 1
    torch.testing.assert_close(output.logprobs_tensors.logprobs[0, 0], torch.tensor([3.0, 1.0]).log_softmax(-1)[1])
    if metadata.temperature is not None:
        assert str(metadata.temperature.device) == runner.device


@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("num_requests", range(1, 9))
def test_cpu_sampling_scores_the_selected_token_and_preserves_batch(sampling_runner, greedy, num_requests, monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler

    # Exercise the production dispatch hazard, independent of test-host Triton.
    def unexpected_triton(*args, **kwargs):
        raise AssertionError("CPU sampling dispatched to Triton")

    monkeypatch.setattr(topk_topp_sampler, "HAS_TRITON", True)
    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p_triton", unexpected_triton, raising=False)
    vocab_size = 151936
    padded_rows = 1 << (num_requests - 1).bit_length()
    runner = sampling_runner(
        num_requests,
        greedy,
        temperature=0.5,
        top_k=2,
        neuron_config=SimpleNamespace(enable_structured_outputs=True, on_device_sampling_config=None),
        use_async_scheduling=False,
        on_device_sampling=False,
        vocab_size=vocab_size,
        _lm_head_vocab_shard_size=lambda: vocab_size,
        _use_lm_head_sharded_logit_mask=lambda: False,
        _build_noop_logit_mask=lambda num_logit_rows, device, compiled_graph_input=False: torch.ones(
            num_logit_rows, vocab_size, dtype=torch.bool, device=device
        ),
        _get_grammar_bitmask=lambda scheduler_output: None,
    )
    device, metadata = runner.device, runner.input_batch.sampling_metadata

    def original_sample(model_output_tensor, spec_decode_metadata=None):
        return runner.sampler(model_output_tensor, runner.input_batch.sampling_metadata)

    runner._sample = original_sample
    if device != "cpu":
        from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

        runner._sample = MethodType(NeuronModelRunner._sample, runner)
    NeuronWorkerExtension.monkey_patch_model(SimpleNamespace(model_runner=runner), vocab_size - 1, [2])
    # CPU sampling still passes an all-True mask into the Neuron graph. Token
    # restrictions apply only to host logits, including for multi-row masks.
    graph_mask = runner._build_noop_logit_mask(num_requests, device)
    assert graph_mask.to("cpu").all()
    logits = torch.full((padded_rows, vocab_size), -torch.inf)
    logits[:, :4] = torch.tensor([3.0, 1.0, 50.0, 0.0])
    logits[:, -1] = 30
    original_logits = logits.clone()
    output = runner._sample(logits)
    tokens = output.sampled_token_ids.reshape(-1)
    assert tokens.numel() == num_requests
    assert set(tokens.tolist()) <= {0, 1}
    if greedy:
        expected = torch.tensor([3.0, 1.0, -torch.inf, 0.0, -torch.inf]).log_softmax(-1)
        assert tokens.tolist() == [0] * num_requests
    else:
        expected = torch.tensor([6.0, 2.0]).log_softmax(-1)
    torch.testing.assert_close(output.logprobs_tensors.logprobs[:, 0], expected[tokens])
    assert runner.input_batch.sampling_metadata is metadata
    assert str(metadata.top_k.device) == device
    torch.testing.assert_close(logits, original_logits)
