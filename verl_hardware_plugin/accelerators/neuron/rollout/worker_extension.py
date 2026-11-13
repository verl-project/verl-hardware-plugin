# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Native worker hooks; never import the training runtime in a lite worker."""

import inspect
import json
import logging
import os
import time
from contextlib import contextmanager
from copy import copy
from functools import partial
from types import SimpleNamespace

import torch

from ..runtime import physical_core
from ..weight_sync.nixl_transport import NixlEndpoint, NixlTopology, control_store
from ..weight_sync.streaming import validate_box_coverage
from ..weight_sync.weight_layout import apply_box, plan_box, plan_buckets, slot_numel_for

logger = logging.getLogger(__name__)


def _check_native_hook(runner, name, expected):
    """Check the private runner contract pinned to vllm-neuron 533db849."""
    hook = getattr(runner, name, None)
    try:
        parameters = list(inspect.signature(hook).parameters) if callable(hook) else None
    except (TypeError, ValueError):
        parameters = None
    if parameters != expected:
        raise RuntimeError(f"Incompatible native vLLM hook {name}: expected {expected}, got {parameters}")
    return hook


def _validate_sampling_metadata(metadata):
    processors = metadata.logitsprocs
    if (
        not metadata.no_penalties
        or metadata.bad_words_token_ids
        or metadata.allowed_token_ids_mask is not None
        or processors.argmax_invariant
        or processors.non_argmax_invariant
        or getattr(metadata, "thinking_budget_state_holder", None) is not None
    ):
        raise ValueError("Native Neuron sampling does not support penalties or custom logits processors")


def _native_logprobs(logits: torch.Tensor, token_ids: torch.Tensor, metadata, max_top_k: int) -> torch.Tensor:
    """Mirror the pinned native sampler's candidate cap and nucleus filtering.

    Native top-p drops the threshold-crossing token, unlike vLLM's CPU sampler.
    Native top-k also keeps at most max_top_k candidates even when top_k=-1.
    Remove this adapter once the native runner returns its own sampled-token scores.
    """
    values, indices = logits.topk(min(max_top_k, logits.shape[-1]), dim=-1)
    original_values = values
    if metadata.top_k is not None:
        top_k = metadata.top_k.to("cpu").long().unsqueeze(-1)
        threshold = values.gather(-1, (top_k - 1).clamp(0, values.shape[-1] - 1))
        values = values.masked_fill((values < threshold) & (top_k > 0), -3000.0)
    if metadata.temperature is not None:
        temperature = metadata.temperature.to("cpu").float().unsqueeze(-1)
        values = values / torch.where(temperature < 1e-5, 1.0, temperature)
    probabilities = (values - values.amax(dim=-1, keepdim=True)).exp()
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    # Match the native cumulative sum implementation and strict threshold.
    cumulative = probabilities @ torch.triu(probabilities.new_ones((values.shape[-1], values.shape[-1])))
    top_p = 1.0 if metadata.top_p is None else metadata.top_p.to("cpu").unsqueeze(-1)
    excluded = cumulative > top_p
    excluded[:, 0] = False
    probabilities.masked_fill_(excluded, 0)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)

    # CPU and distributed top-k can choose different IDs at a tied cutoff.
    # Retain the actual device sample in an equally weighted surviving slot.
    live = probabilities > 0
    matches = (original_values == logits.gather(-1, token_ids[:, None])) & live
    if not matches.any(dim=-1).all():
        raise RuntimeError("Native sampled token is outside the reconstructed sampling distribution")
    actual = (indices == token_ids[:, None]) & live
    slots = torch.where(actual.any(dim=-1), actual.int().argmax(-1), matches.int().argmax(-1))
    rows = torch.arange(logits.shape[0])
    indices[rows, slots] = token_ids
    logprobs = torch.full_like(logits, -torch.inf)
    logprobs[rows[:, None].expand_as(indices)[live], indices[live]] = probabilities[live].log()
    return logprobs


@contextmanager
def resident_weight_model(worker):
    """Expose original parameter names without replacing any compiled weight storage."""
    runner = worker.model_runner
    compiled = runner.model
    model = compiled
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    runner.model = model
    try:
        yield
    finally:
        runner.model = compiled


def restrict_token_mask(mask, vocab_size, banned_token_ids, offset=0):
    """Intersect a full-vocabulary or TP-local native sampling mask in place."""
    if mask is None:
        return None
    width = mask.shape[-1]
    mask[..., max(0, min(width, vocab_size - offset)) :] = False
    for token_id in banned_token_ids:
        if offset <= token_id < offset + width:
            mask[..., token_id - offset] = False
    return mask


def native_sample_logprobs(runner, sampled_tokens, vocab_size, banned_token_ids):
    """Score the actual Trainium sample without another draw or sampler mutation."""
    from vllm.v1.outputs import LogprobsTensors, SamplerOutput

    if runner._on_device_logits is None:
        raise RuntimeError("Native rollout log-probabilities require gathered logits")
    metadata = runner.input_batch.sampling_metadata
    _validate_sampling_metadata(metadata)
    num_requests = runner.input_batch.num_reqs
    token_ids = sampled_tokens.to("cpu").long().reshape(-1)[:num_requests]
    logits = runner._on_device_logits.to("cpu")[:num_requests].float()
    if logits.shape[0] != num_requests or token_ids.numel() != num_requests:
        raise RuntimeError("Native sampling outputs must cover every live request")
    logits[..., vocab_size:] = -torch.inf
    for token_id in banned_token_ids:
        if 0 <= token_id < logits.shape[-1]:
            logits[..., token_id] = -torch.inf
    grammar = getattr(runner, "_verl_sampling_mask", None)
    if grammar is not None:
        logits.masked_fill_(~grammar[:num_requests], -torch.inf)
    sampling = runner.neuron_config.on_device_sampling_config
    cap = min(sampling.max_top_k, runner._lm_head_vocab_shard_size())
    logprobs = _native_logprobs(logits, token_ids, metadata, cap)
    if metadata.max_num_logprobs == -1:
        scored = LogprobsTensors(torch.empty(0), logprobs, torch.empty(0))
    else:
        scored = runner.sampler.gather_logprobs(logprobs, metadata.max_num_logprobs, token_ids)
    return SamplerOutput([[token_id] for token_id in token_ids.tolist()], scored)


def cpu_topk_topp(logits, generators, k, p, use_fp64_gumbel=False):
    """Keep host sampling on PyTorch even when vLLM detects installed Triton."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch, random_sample

    if logits.device.type != "cpu":
        raise ValueError("CPU sampling requires host logits")
    logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
    logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
    probabilities = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probabilities, generators, use_fp64_gumbel), logprobs


class NeuronWorkerExtension:
    def weight_sync_connect(self, spec):
        if getattr(self, "_weight_sync_state", None) is not None:
            raise RuntimeError("This worker already has a weight-sync connection")
        device = torch.device(self.device)
        if device.type != "neuron":
            raise ValueError("Weight synchronization requires a real Trainium receiver")
        topology = NixlTopology(**{name: value for name, value in spec.items() if name != "replica"})
        replica, rank = int(spec["replica"]), int(self.rank)
        with resident_weight_model(self):
            model = self.model_runner.model
            if (int(model.rank), int(model.world_size)) != (rank, topology.rollout_tp):
                raise ValueError("Resident model and rollout TP coordinates disagree")
            tensors = dict(model.named_parameters())
            for name, tensor in model.named_buffers():
                tensors.setdefault(name, tensor)
        if not tensors or any(tensor.dtype != torch.bfloat16 for tensor in tensors.values()):
            raise ValueError("Weight synchronization requires BF16 resident parameters and buffers")
        core = physical_core(os.environ["NEURON_RT_VISIBLE_CORES"], 0)
        store = control_store(topology)
        endpoint = NixlEndpoint(
            store,
            f"replica-{replica}-tp-{rank}",
            device,
            core,
            sender=False,
            lite=True,
            timeout_s=topology.timeout_s,
        )
        inventory = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
        endpoint.store.set("inventory", json.dumps(inventory))
        self._weight_sync_state = SimpleNamespace(
            endpoint=endpoint,
            tensors=tensors,
            inventory=inventory,
            rank=rank,
            size=topology.rollout_tp,
        )
        return {"tp_rank": rank, "n_params": len(tensors), "device": device.type}

    def weight_sync_recv(self, schedule, bucket_bytes):
        state = getattr(self, "_weight_sync_state", None)
        if state is None:
            raise RuntimeError("Connect the weight-sync receiver before receiving weights")
        started = time.monotonic()
        buckets = plan_buckets(schedule, bucket_bytes=bucket_bytes)
        shapes = {name: tuple(shape) for name, shape, dtype in schedule}
        plans = [plan_box(bucket, shapes, state.inventory, state.rank, state.size) for bucket in buckets]
        validate_box_coverage(plans, state.inventory)
        state.endpoint.register(slot_numel_for(plans))
        applied = 0
        for plan in plans:
            if plan.numel:
                applied += state.endpoint.receive(
                    lambda slot, plan=plan: apply_box(slot.to("cpu"), plan, state.tensors)
                )
        logger.info(
            "neuron_nixl TP rank %d applied %d pieces of %d buckets in %.2fs",
            state.rank,
            applied,
            len(plans),
            time.monotonic() - started,
        )
        return applied

    def weight_sync_checksum(self, names=None):
        with resident_weight_model(self):
            model = self.model_runner.model
            tensors = dict(model.named_parameters())
            for name, tensor in model.named_buffers():
                tensors.setdefault(name, tensor)
            checksums = {}
            for name in names if names is not None else sorted(tensors):
                if name in tensors:
                    host = tensors[name].detach().to("cpu").to(torch.float64)
                    checksums[name] = [host.sum().item(), host.abs().sum().item()]
            return checksums

    def weight_sync_close(self):
        state = getattr(self, "_weight_sync_state", None)
        if state is not None:
            state.endpoint.close()
            self._weight_sync_state = None

    def reset_encoder_cache(self):
        """Dense text models have no vision cache to invalidate on a policy update."""
        if self.model_runner.encoder_cache is not None:
            raise NotImplementedError("Native verl rollout supports text-only models without encoder caches")

    def monkey_patch_model(self, vocab_size, banned_token_ids=None):
        """Implement verl's token restrictions using compiled native mask inputs."""
        runner = self.model_runner
        if not runner.neuron_config.enable_structured_outputs:
            raise RuntimeError("Native vocabulary masking requires enable_structured_outputs before compilation")
        if runner.use_async_scheduling or getattr(runner, "lm_head_dp_size", 1) != 1:
            raise ValueError("Native verl sampling requires synchronous scheduling and LM-head DP=1")
        sampling = runner.neuron_config.on_device_sampling_config
        if sampling is not None and (sampling.all_greedy or sampling.deterministic or sampling.max_top_k < 1):
            raise ValueError("Native verl requires stochastic sampling with a positive max_top_k")
        if runner.sampler.logprobs_mode != "processed_logprobs":
            raise ValueError("Native verl requires logprobs_mode=processed_logprobs")
        if not 0 < vocab_size <= runner.vocab_size:
            raise ValueError("Tokenizer vocabulary must fit the resident model vocabulary")
        banned = tuple(sorted(set(banned_token_ids or ())))
        configuration = (vocab_size, banned)
        previous = getattr(runner, "_verl_token_restrictions", None)
        if previous is not None:
            if previous != configuration:
                raise ValueError("Cannot change token restrictions on a running rollout worker")
            return
        build_noop = _check_native_hook(
            runner, "_build_noop_logit_mask", ["num_logit_rows", "device", "compiled_graph_input"]
        )
        get_grammar = _check_native_hook(runner, "_get_grammar_bitmask", ["scheduler_output"])
        original_sample = _check_native_hook(runner, "_sample", ["model_output_tensor", "spec_decode_metadata"])
        _check_native_hook(runner, "_lm_head_vocab_shard_size", [])
        _check_native_hook(runner, "_use_lm_head_sharded_logit_mask", [])
        offset = (
            runner.lm_head_tp_rank * runner._lm_head_vocab_shard_size()
            if runner._use_lm_head_sharded_logit_mask()
            else 0
        )
        if not runner.on_device_sampling:
            # vLLM's automatic dispatch chooses Triton for batches >= 8 even
            # when these logits are on CPU and the active platform is Neuron.
            runner.sampler.topk_topp_sampler.forward = partial(
                cpu_topk_topp, use_fp64_gumbel=runner.sampler.use_fp64_gumbel
            )

        def build_mask(*args, **kwargs):
            mask = build_noop(*args, **kwargs)
            # CPU sampling applies restrictions to returned host logits. Its
            # graph mask lives on Neuron, whose eager sliced writes are limited.
            if not runner.on_device_sampling:
                return mask
            return restrict_token_mask(mask, vocab_size, banned, offset)

        def grammar_mask(*args, **kwargs):
            mask = restrict_token_mask(get_grammar(*args, **kwargs), vocab_size, banned)
            runner._verl_sampling_mask = mask
            return mask

        def sample(model_output_tensor, spec_decode_metadata=None):
            metadata = runner.input_batch.sampling_metadata
            _validate_sampling_metadata(metadata)
            if spec_decode_metadata is not None or getattr(runner, "_jf_active_tokens", None):
                raise NotImplementedError("Native verl log-probabilities require ordinary autoregressive decoding")
            if not runner.on_device_sampling:
                # The native CPU path returns logits, but InputBatch keeps its
                # sampling tensors on Neuron. Copy metadata without mutating
                # the persistent batch, and sample/score in the same operation.
                host_metadata = copy(metadata)
                for name, value in vars(metadata).items():
                    if isinstance(value, torch.Tensor):
                        setattr(host_metadata, name, value.to("cpu"))
                if any(generator.device.type != "cpu" for generator in metadata.generators.values()):
                    raise ValueError("CPU sampling requires CPU random generators")
                # Native graphs pad to a supported batch bucket (e.g. 3 -> 4).
                # Metadata describes only live requests, so discard padding.
                num_requests = runner.input_batch.num_reqs
                logits = model_output_tensor[:num_requests].to("cpu").float().clone()
                if logits.ndim != 2 or logits.shape[0] != num_requests:
                    raise RuntimeError("CPU sampling logits must cover every live request")
                logits[..., vocab_size:] = -torch.inf
                for token_id in banned:
                    if 0 <= token_id < logits.shape[-1]:
                        logits[..., token_id] = -torch.inf
                runner.input_batch.sampling_metadata = host_metadata
                try:
                    return original_sample(logits, spec_decode_metadata)
                finally:
                    runner.input_batch.sampling_metadata = metadata
            if metadata.max_num_logprobs is None:
                return original_sample(model_output_tensor, spec_decode_metadata)
            return native_sample_logprobs(runner, model_output_tensor, vocab_size, banned)

        runner._build_noop_logit_mask = build_mask
        runner._get_grammar_bitmask = grammar_mask
        runner._sample = sample
        runner._verl_token_restrictions = configuration
