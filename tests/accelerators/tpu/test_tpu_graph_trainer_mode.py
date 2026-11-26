# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``engine_kwargs.tpu.trainer_mode=graph_trainer`` (``engines/graph_trainer_mode.py``).

The TPU compile pass needs ``torch_tpu`` on TPU; the on-TPU parity check against the eager engine runs in verl's TPU CI.
These check the parts that decide numerics and TorchTitan integration: the option surface, the graph-trainer config,
the trace-safe PPO loss against verl's, the packed loss inputs against verl's response layout, the RL
``TpuGraphTrainer`` wrapper (loss metrics, packed stream as one row), and the engine step with gradient accumulation
across micro-batches.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.engines import fused_logprob, graph_trainer_mode
from verl_hardware_plugin.accelerators.tpu.losses.loss_inputs import _response_gather_index
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options


class _ActorConfig(dict):
    """The parts of verl's ``ActorConfig`` the policy loss reads (attribute and ``get`` access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _actor_config(**overrides):
    config = _ActorConfig(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        policy_loss=_ActorConfig(
            loss_mode="bypass_mode",
            rollout_correction=_ActorConfig(loss_type="ppo_clip", rollout_is="token", rollout_rs=None),
        ),
        global_batch_info={},
        loss_scale_factor=None,
    )
    config.update(overrides)
    return config


def _dense_fused_logprobs(hidden_states, labels, weight_t, *, temperature=1.0, **_):
    """CPU stand-in for the Tokamax ``fused_linear_logprobs_tpu`` (Pallas, TPU only): same math, dense logits."""
    logits = hidden_states.reshape(-1, hidden_states.shape[-1]).float() @ weight_t.float() / temperature
    return torch.log_softmax(logits, -1).gather(-1, labels.reshape(-1, 1).long()).squeeze(-1)


def _patch_fused_logprobs():
    return mock.patch.object(fused_logprob, "fused_linear_logprobs_tpu", _dense_fused_logprobs)


def _options(**tpu):
    return get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": tpu}))


def test_trainer_mode_options():
    assert _options().trainer_mode is None
    assert _options().graph_trainer is None
    assert _options(trainer_mode="Graph_Trainer").trainer_mode == "graph_trainer"
    assert graph_trainer_mode.is_graph_trainer_mode(_options(trainer_mode="graph_trainer"))
    assert not graph_trainer_mode.is_graph_trainer_mode(_options(trainer_mode="eager_blocks"))

    # When trainer_mode is not graph_trainer, graph_trainer config is ignored.
    assert not graph_trainer_mode.is_graph_trainer_mode(_options(graph_trainer={"memory_policy": "full"}))
    assert not graph_trainer_mode.is_graph_trainer_mode(
        _options(trainer_mode="eager_blocks", graph_trainer={"memory_policy": "full"})
    )

    opts = _options(
        trainer_mode="graph_trainer",
        graph_trainer={
            "memory_policy": "tpu_selective",
            "save_regions": "moe.routing,attention.inner_attention",
            "fsdp_barrier_policy": "interleave",
        },
    )
    assert opts.trainer_mode == "graph_trainer"
    assert opts.graph_trainer == {
        "memory_policy": "tpu_selective",
        "save_regions": "moe.routing,attention.inner_attention",
        "fsdp_barrier_policy": "interleave",
    }
    assert graph_trainer_mode.is_graph_trainer_mode(opts)

    with pytest.raises(ValueError, match="trainer_mode must be one of"):
        _options(trainer_mode="graph")


def test_validate_graph_trainer_engine():
    good = SimpleNamespace(forward_only=False, param_offload=False, data_parallel_shard_size=16)
    options = _options(trainer_mode="graph_trainer", static_micro_batch_len=8192)
    graph_trainer_mode.validate_graph_trainer_engine(good, options, use_simple_fsdp=True, tpu_kernels=frozenset())

    bad = SimpleNamespace(
        forward_only=True, param_offload=True, expert_parallel_size=2, context_parallel_size=1, tensor_parallel_size=2
    )
    with pytest.raises(ValueError) as e:
        graph_trainer_mode.validate_graph_trainer_engine(
            bad,
            _options(
                trainer_mode="graph_trainer",
                use_fused_linear_cross_entropy=False,
                kda_document_budget=1,
            ),
            use_simple_fsdp=False,
            tpu_kernels=frozenset({"kda"}),
        )
    message = str(e.value)
    for expected in (
        "forward_only",
        "use_simple_fsdp",
        "static_micro_batch_len",
        "use_fused_linear_cross_entropy",
        "kda_document_budget=1",
        "tensor_parallel_size=2",
    ):
        assert expected in message, (expected, message)
    assert "param_offload" in message

    # Context parallelism and divisible gdn_row_slot_len are supported; non-divisible gdn_row_slot_len raises.
    cp_engine = SimpleNamespace(forward_only=False, param_offload=False, context_parallel_size=2)
    graph_trainer_mode.validate_graph_trainer_engine(
        cp_engine,
        _options(trainer_mode="graph_trainer", static_micro_batch_len=8192, gdn_row_slot_len=4096),
        use_simple_fsdp=True,
        tpu_kernels=frozenset(),
    )
    with pytest.raises(ValueError, match="must be a multiple of gdn_row_slot_len=3000"):
        graph_trainer_mode.validate_graph_trainer_engine(
            good,
            _options(trainer_mode="graph_trainer", static_micro_batch_len=8192, gdn_row_slot_len=3000),
            use_simple_fsdp=True,
            tpu_kernels=frozenset(),
        )


def test_validate_graph_trainer_engine_accepts_kda():
    good = SimpleNamespace(forward_only=False, param_offload=False, data_parallel_shard_size=8)
    for budget in (None, 2, 80):
        graph_trainer_mode.validate_graph_trainer_engine(
            good,
            _options(trainer_mode="graph_trainer", static_micro_batch_len=32768, kda_document_budget=budget),
            use_simple_fsdp=True,
            tpu_kernels=frozenset({"moe", "kda", "lce"}),
        )


def test_initial_kda_document_budget_is_documents_plus_padding_tail_or_the_option():
    assert graph_trainer_mode.initial_kda_document_budget(_options(trainer_mode="graph_trainer"), 16) == 17
    assert graph_trainer_mode.initial_kda_document_budget(_options(trainer_mode="graph_trainer"), 63) == 64
    options = _options(trainer_mode="graph_trainer", kda_document_budget=24)
    assert graph_trainer_mode.initial_kda_document_budget(options, 16) == 24


def test_check_kda_document_budget_raises_only_when_the_documents_do_not_fit():
    graph_trainer_mode.check_kda_document_budget(17, 16)  # 16 documents + padding tail fit 17 segments
    graph_trainer_mode.check_kda_document_budget(17, 3)
    with pytest.raises(RuntimeError, match="pinned the Tokamax KDA document budget"):
        graph_trainer_mode.check_kda_document_budget(17, 17)


def test_memory_policy_defaults_follow_activation_checkpoint():
    options = _options(trainer_mode="graph_trainer")
    assert graph_trainer_mode.resolve_memory_policy(options, "full") == "full"
    assert graph_trainer_mode.resolve_memory_policy(options, "selective") == "default"
    assert graph_trainer_mode.resolve_memory_policy(options, "none") == "none"
    explicit = _options(trainer_mode="graph_trainer", graph_trainer={"memory_policy": "tpu_selective"})
    assert graph_trainer_mode.resolve_memory_policy(explicit, "full") == "tpu_selective"


def test_policy_loss_spec_reads_and_checks_the_actor_config():
    loss_fn = functools.partial(ppo_loss, config=_actor_config())
    spec = graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
        loss_fn, temperature=0.7, calculate_entropy=False, has_rollout_is_weights=True
    )
    assert spec == graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, 0.7)
    # clip_ratio_low / high fall back to clip_ratio.
    spec = graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
        functools.partial(ppo_loss, config=_actor_config(clip_ratio_low=None, clip_ratio_high=None, clip_ratio=0.1)),
        temperature=1.0,
        calculate_entropy=False,
        has_rollout_is_weights=False,
    )
    assert (spec.clip_ratio_low, spec.clip_ratio_high) == (0.1, 0.1)

    unsupported = [
        (dict(use_kl_loss=True), "use_kl_loss"),
        (dict(entropy_coeff=0.01), "entropy"),
        (dict(loss_agg_mode="seq-mean-token-mean"), "loss_agg_mode"),
        (dict(policy_loss=_ActorConfig(loss_mode="gspo")), "loss_mode='gspo'"),
        (
            dict(
                policy_loss=_ActorConfig(
                    loss_mode="bypass_mode",
                    rollout_correction=_ActorConfig(rollout_rs="unsupported_rs", rollout_rs_threshold="0.5"),
                )
            ),
            "rollout_rs",
        ),
        (
            dict(
                policy_loss=_ActorConfig(
                    loss_mode="bypass_mode", rollout_correction=_ActorConfig(loss_type="unknown_loss")
                )
            ),
            "unknown_loss",
        ),
    ]
    for overrides, match in unsupported:
        with pytest.raises(ValueError, match=match):
            graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
                functools.partial(ppo_loss, config=_actor_config(**overrides)),
                temperature=1.0,
                calculate_entropy=False,
                has_rollout_is_weights=False,
            )
    with pytest.raises(ValueError, match="rollout_is_weights"):
        graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
            functools.partial(ppo_loss, config=_actor_config(policy_loss=_ActorConfig(loss_mode="vanilla"))),
            temperature=1.0,
            calculate_entropy=False,
            has_rollout_is_weights=True,
        )
    with pytest.raises(ValueError, match="ppo_loss"):
        graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
            lambda **_: None, temperature=1.0, calculate_entropy=False, has_rollout_is_weights=False
        )


def test_graph_policy_loss_matches_verl_vanilla_ppo():
    """Traced loss x dp_size == verl's token-mean vanilla PPO loss, and so are its gradients."""
    torch.manual_seed(0)
    num_tokens, hidden, vocab, dp_size, global_tokens = 64, 16, 50, 4, 100.0
    config = _actor_config()
    spec = graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, temperature=0.9)
    loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)

    # Fused LM-head output (FusedLossLinearWrapper): (hidden_states, lm_head_weight_t), no logits.
    hidden_states = torch.randn(num_tokens, hidden, requires_grad=True)
    weight_t = torch.randn(hidden, vocab, requires_grad=True)
    target_ids = torch.randint(0, vocab, (num_tokens,))
    old_log_probs = torch.log_softmax(torch.randn(num_tokens, vocab), -1).gather(-1, target_ids[:, None]).squeeze(-1)
    advantages = torch.randn(num_tokens)
    loss_mask = (torch.rand(num_tokens) > 0.3).float()
    # A masked-out bucket-padding token with an extreme log-prob must not leak into the loss.
    old_log_probs[-1] = -1e4

    with _patch_fused_logprobs():
        loss, metrics = loss_fn(
            (hidden_states, weight_t),
            (target_ids, old_log_probs, advantages, loss_mask),
            global_valid_tokens=torch.tensor(global_tokens),
        )
    log_probs = metrics[graph_trainer_mode.LOG_PROBS_METRIC]
    loss.backward()
    graph_grads = (hidden_states.grad.clone(), weight_t.grad.clone())
    assert not log_probs.requires_grad

    hidden_states.grad = weight_t.grad = None
    logits = hidden_states @ weight_t
    log_prob = torch.log_softmax(logits / 0.9, -1).gather(-1, target_ids[:, None]).squeeze(-1)
    torch.testing.assert_close(log_probs, log_prob.detach())
    config.global_batch_info.update(dp_size=dp_size, batch_num_tokens=global_tokens)
    expected, _ = compute_policy_loss_vanilla(
        old_log_prob=old_log_probs[None],
        log_prob=log_prob[None],
        advantages=advantages[None],
        response_mask=loss_mask.bool()[None],
        loss_agg_mode="token-mean",
        config=config,
    )
    torch.testing.assert_close(loss * dp_size, expected)
    # The eager engine back-propagates verl's loss times 1/dp (SimpleFSDP sums gradients): same gradient.
    (expected / dp_size).backward()
    torch.testing.assert_close(graph_grads, (hidden_states.grad, weight_t.grad))
    # Graph mode has no dense-logits path: a plain logits tensor is rejected.
    with pytest.raises(TypeError, match="fused LM head"):
        loss_fn.log_probs(logits.detach(), target_ids)
    assert graph_trainer_mode.check_loss_consistency(loss.item(), expected.item(), dp_size)
    with pytest.raises(RuntimeError, match="traced loss x dp_size"):
        graph_trainer_mode.check_loss_consistency(loss.item(), 2 * expected.item(), dp_size)
    assert not graph_trainer_mode.check_loss_consistency(
        loss.item(), 2 * expected.item(), dp_size, raise_on_mismatch=False
    )


def _rl_micro_batch(prompt_lens, response_lens):
    torch.manual_seed(1)
    bsz, width = len(prompt_lens), max(response_lens) + 2
    seq_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    offsets = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, 100, (sum(seq_lens),)), offsets)
    response_mask = torch.zeros(bsz, width)
    for i, r in enumerate(response_lens):
        response_mask[i, :r] = 1
    response_mask[0, 1] = 0  # a masked token inside a response (e.g. a tool output)
    fields = {
        "input_ids": input_ids,
        "old_log_probs": -torch.rand(bsz, width) * response_mask,
        "advantages": torch.randn(bsz, 1).expand(bsz, width) * response_mask,
        "response_mask": response_mask,
    }
    fields["prompts"] = torch.nested.nested_tensor_from_jagged(
        torch.zeros(sum(prompt_lens), dtype=torch.int64),
        torch.tensor([0, *torch.tensor(prompt_lens).cumsum(0).tolist()]),
    )
    fields["responses"] = torch.nested.nested_tensor_from_jagged(
        torch.zeros(sum(response_lens), dtype=torch.int64),
        torch.tensor([0, *torch.tensor(response_lens).cumsum(0).tolist()]),
    )
    return TensorDict(fields, batch_size=[bsz])


def test_pack_rl_loss_targets_uses_verl_response_positions():
    prompt_lens, response_lens, padded_len = [3, 5, 2], [4, 1, 6], 32
    micro_batch = _rl_micro_batch(prompt_lens, response_lens)
    old_log_probs, advantages, loss_mask = graph_trainer_mode.pack_rl_loss_targets(micro_batch, padded_len)
    assert old_log_probs.shape == advantages.shape == loss_mask.shape == (padded_len,)

    # Same positions as the eager engine's loss-input gather (verl's no_padding_2_padding convention).
    width = micro_batch["response_mask"].shape[1]
    index, valid = _response_gather_index(micro_batch, width, padded_len, int(micro_batch.batch_size[0]))
    for packed, key in ((old_log_probs, "old_log_probs"), (advantages, "advantages"), (loss_mask, "response_mask")):
        torch.testing.assert_close(packed[index][valid], micro_batch[key][valid].float())
        outside = torch.ones(padded_len, dtype=torch.bool)
        outside[index[valid]] = False
        assert not packed[outside].any()
    assert loss_mask.sum() == micro_batch["response_mask"].sum()

    with pytest.raises(ValueError, match="padded length"):
        graph_trainer_mode.pack_rl_loss_targets(micro_batch, 8)

    # Dense (padded) prompts / responses are not the V1 trainer's layout.
    dense = micro_batch.copy()
    dense["prompts"] = torch.zeros(3, max(prompt_lens), dtype=torch.int64)
    with pytest.raises(ValueError, match="nested"):
        graph_trainer_mode.pack_rl_loss_targets(dense, padded_len)


def test_packed_graph_loss_matches_verl_ppo_loss_on_the_same_log_probs():
    """End to end on CPU: packed inputs + traced loss == verl's ppo_loss on the response layout, / dp_size."""
    prompt_lens, response_lens, padded_len, dp_size = [3, 5, 2], [4, 1, 6], 32, 2
    micro_batch = _rl_micro_batch(prompt_lens, response_lens)
    old_log_probs, advantages, loss_mask = graph_trainer_mode.pack_rl_loss_targets(micro_batch, padded_len)
    global_tokens = float(micro_batch["response_mask"].sum()) * dp_size
    log_probs = -torch.rand(padded_len)

    spec = graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, temperature=1.0)
    loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)
    with mock.patch.object(loss_fn, "log_probs", return_value=log_probs):
        graph_loss, _ = loss_fn(
            None, (None, old_log_probs, advantages, loss_mask), global_valid_tokens=torch.tensor(global_tokens)
        )

    num_tokens = sum(prompt_lens) + sum(response_lens)
    data = micro_batch.copy()
    tu.assign_non_tensor(
        data,
        dp_size=dp_size,
        batch_num_tokens=global_tokens,
        global_batch_size=None,
        max_response_len=micro_batch["response_mask"].shape[1],
    )
    model_output = {
        "log_probs": torch.nested.nested_tensor_from_jagged(log_probs[:num_tokens], data["input_ids"].offsets())
    }
    verl_loss, metrics = ppo_loss(_actor_config(), model_output=model_output, data=data)
    torch.testing.assert_close(graph_loss * dp_size, verl_loss)
    assert "actor/ppo_kl" in metrics and "actor/pg_clipfrac" in metrics


def test_prepare_model_for_graph_trainer_annotates_and_installs_preprocess_inputs():
    annotated = []
    common_utils = SimpleNamespace(annotate_module_fqns=annotated.append)
    modules = {
        name: SimpleNamespace()
        for name in ("torchtitan", "torchtitan.experiments", "torchtitan.experiments.graph_trainer")
    }
    modules["torchtitan.experiments.graph_trainer.common_utils"] = common_utils
    model = torch.nn.Module()
    with mock.patch.dict("sys.modules", modules):
        graph_trainer_mode.prepare_model_for_graph_trainer([model])
    assert annotated == [model]
    assert model.preprocess_inputs is graph_trainer_mode._plugin_preprocess_inputs


def test_validate_graph_trainer_engine_allows_ep():
    engine = SimpleNamespace(forward_only=False, param_offload=False, expert_parallel_size=4)
    options = _options(trainer_mode="graph_trainer", static_micro_batch_len=8192)
    graph_trainer_mode.validate_graph_trainer_engine(engine, options, use_simple_fsdp=True, tpu_kernels=frozenset())


def test_prepare_model_for_graph_trainer_annotates_ep_regions():
    calls = []
    common_utils = SimpleNamespace(
        annotate_module_fqns=lambda m: calls.append(("fqns", m)),
        annotate_moe_ep_regions=lambda: calls.append(("ep_regions", None)),
    )
    qwen3_5 = SimpleNamespace(annotate_expert_unshard_fqns=lambda m: calls.append(("expert_unshard", m)))
    modules = {
        name: SimpleNamespace()
        for name in (
            "torchtitan",
            "torchtitan.experiments",
            "torchtitan.experiments.graph_trainer",
            "torchtitan.experiments.tpu",
            "torchtitan.experiments.tpu.graph_trainer",
        )
    }
    modules["torchtitan.experiments.graph_trainer.common_utils"] = common_utils
    modules["torchtitan.experiments.tpu.graph_trainer.qwen3_5"] = qwen3_5
    model = torch.nn.Module()
    with mock.patch.dict("sys.modules", modules):
        graph_trainer_mode.prepare_model_for_graph_trainer([model], ep_enabled=True)
    assert calls == [("ep_regions", None), ("fqns", model), ("expert_unshard", model)]
    assert model.preprocess_inputs is graph_trainer_mode._plugin_preprocess_inputs


def test_plugin_preprocess_inputs_builds_no_masks():
    inputs, labels, extra = graph_trainer_mode._plugin_preprocess_inputs(
        {"input": "ids", "labels": ("t", "o"), "positions": "pos"}, parallel_dims=None, parallelism=None
    )
    assert (inputs, labels, extra) == ("ids", ("t", "o"), {"positions": "pos"})


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 100, dim: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens):
        # Fused LM-head output, as FusedLossLinearWrapper: (hidden_states, lm_head_weight_t), no logits.
        return torch.tanh(self.emb(tokens)), self.head.weight.t()

    def logits(self, tokens):
        hidden_states, weight_t = self(tokens)
        return hidden_states @ weight_t


class _FakeGraphTrainer:
    """Stands in for TpuGraphTrainer: runs the installed loss on a tiny model and its backward into .grad."""

    def __init__(self, model):
        self.model = model
        self.loss_fn = None
        self.last_loss_metrics = None
        self.calls = []

    def forward_backward_step(self, *, input_dict, global_valid_tokens):
        self.calls.append((tuple(input_dict["input"].shape), float(global_valid_tokens)))
        with _patch_fused_logprobs():
            loss, metrics = self.loss_fn(
                self.model(input_dict["input"]), input_dict["labels"], global_valid_tokens=global_valid_tokens
            )
        loss.backward()  # adds to param.grad, as TpuGraphTrainer accumulates the traced step's gradients
        self.last_loss_metrics = metrics
        return loss.detach()


def _graph_engine(model, padded_len):
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    engine = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    engine._tpu_eager_mode = None
    engine._tpu_options = _options(trainer_mode="graph_trainer", static_micro_batch_len=padded_len)
    engine._tpu_kernels = frozenset()
    engine._graph_policy_loss_spec = None
    engine._graph_trainer_traced_len = None
    engine.trainer = _FakeGraphTrainer(model)
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None

    def prepare_model_inputs(micro_batch):
        ids = micro_batch["input_ids"].values()
        n = int(ids.numel())
        pad = padded_len - n
        input_ids = torch.cat([ids, torch.zeros(pad, dtype=ids.dtype)])
        positions = torch.cat([micro_batch["position_ids"].values(), torch.zeros(pad, dtype=torch.int64)])
        labels = torch.cat([torch.roll(ids, -1), torch.zeros(pad, dtype=ids.dtype)])
        return (
            input_ids,
            {"positions": positions},
            {},
            {"labels": labels, "orig_seq_len": n, "padded_seq_len": padded_len},
        )

    engine.prepare_model_inputs = prepare_model_inputs
    return engine


def _engine_micro_batch(prompt_lens, response_lens, seed):
    mb = _rl_micro_batch(prompt_lens, response_lens)
    gen = torch.Generator().manual_seed(seed)
    mb["input_ids"] = torch.nested.nested_tensor_from_jagged(
        torch.randint(0, 100, (mb["input_ids"].values().numel(),), generator=gen), mb["input_ids"].offsets()
    )
    lens = mb["input_ids"].offsets().diff().tolist()
    mb["position_ids"] = torch.nested.nested_tensor_from_jagged(
        torch.cat([torch.arange(n) for n in lens]), mb["input_ids"].offsets()
    )
    width = mb["response_mask"].shape[1]
    tu.assign_non_tensor(
        mb,
        temperature=1.0,
        calculate_entropy=False,
        max_response_len=width,
        use_dynamic_bsz=False,
        dp_size=1,
        global_batch_size=None,
    )
    return mb


def test_graph_mode_step_matches_eager_ppo_loss_on_one_micro_batch():
    """One micro-batch through the graph-mode engine step == eager verl ppo_loss (global tokens) and its backward."""
    torch.manual_seed(0)
    padded_len = 48
    micro_batch = _engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    global_tokens = float(micro_batch["response_mask"].sum())
    tu.assign_non_tensor(micro_batch, batch_num_tokens=global_tokens)
    config = _actor_config()
    loss_fn = functools.partial(ppo_loss, config=config)

    model = _TinyLM()
    engine = _graph_engine(model, padded_len)
    data = micro_batch.copy()
    out = engine._graph_trainer_forward_backward(data, [micro_batch], None, loss_fn, global_tokens)
    graph_grads = [p.grad.clone() for p in model.parameters()]
    assert engine.trainer.calls == [((padded_len,), global_tokens)]
    assert len(out["loss"]) == 1 and "actor/ppo_kl" in out["metrics"]
    # Same trainer-vs-rollout gap metrics as the eager engine's forward_step.
    for name in ("actor/ppo_kl_token_mean", "actor/ppo_k3_token_mean", "actor/logprob_gap_abs_token_mean"):
        assert name in out["metrics"], name

    # Eager reference: verl's ppo_loss on the same log-probs, backward.
    for p in model.parameters():
        p.grad = None
    ids = micro_batch["input_ids"].values()
    logp = torch.log_softmax(model.logits(ids), -1).gather(-1, torch.roll(ids, -1)[:, None]).squeeze(-1)
    output = {"log_probs": torch.nested.nested_tensor_from_jagged(logp, micro_batch["input_ids"].offsets())}
    loss, _ = ppo_loss(config, model_output=output, data=micro_batch.copy())
    loss.backward()
    torch.testing.assert_close(float(loss), out["loss"][0], rtol=1e-5, atol=1e-6)
    for g, p in zip(graph_grads, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-6)

    # A micro-batch with another padded length cannot reuse the traced step.
    engine.prepare_model_inputs = lambda micro_batch: (
        torch.zeros(64, dtype=torch.int64),
        {"positions": torch.zeros(64)},
        {},
        {"labels": torch.zeros(64, dtype=torch.int64), "orig_seq_len": 1},
    )
    with pytest.raises(RuntimeError, match="TODO-2"):
        engine._graph_trainer_forward_backward(data, [micro_batch], None, loss_fn, global_tokens)


def test_graph_mode_step_accumulates_gradients_across_micro_batches():
    """Two micro-batches of one mini-batch: the gradients add up to eager verl's, each loss over the global tokens."""
    torch.manual_seed(0)
    padded_len = 48
    micro_batches = [_engine_micro_batch([3, 5, 2], [4, 6, 5], 1), _engine_micro_batch([4, 2], [5, 7], 2)]
    global_tokens = float(sum(mb["response_mask"].sum() for mb in micro_batches))
    for mb in micro_batches:
        tu.assign_non_tensor(mb, batch_num_tokens=global_tokens)
    config = _actor_config()
    loss_fn = functools.partial(ppo_loss, config=config)

    model = _TinyLM()
    engine = _graph_engine(model, padded_len)
    out = engine._graph_trainer_forward_backward(micro_batches[0].copy(), micro_batches, None, loss_fn, global_tokens)
    graph_grads = [p.grad.clone() for p in model.parameters()]
    assert engine.trainer.calls == [((padded_len,), global_tokens)] * 2
    assert len(out["loss"]) == 2

    for p in model.parameters():
        p.grad = None
    for mb, graph_loss in zip(micro_batches, out["loss"], strict=True):
        ids = mb["input_ids"].values()
        logp = torch.log_softmax(model.logits(ids), -1).gather(-1, torch.roll(ids, -1)[:, None]).squeeze(-1)
        output = {"log_probs": torch.nested.nested_tensor_from_jagged(logp, mb["input_ids"].offsets())}
        loss, _ = ppo_loss(config, model_output=output, data=mb.copy())
        loss.backward()
        torch.testing.assert_close(float(loss), graph_loss, rtol=1e-5, atol=1e-6)
    for g, p in zip(graph_grads, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-6)


def test_graph_mode_step_uses_the_mini_batch_device_token_count():
    """The traced step gets the engine's per-mini-batch 0-D token count (device_loss_scalars), the same tensor the
    eager verl loss divides by, instead of building another one per micro-batch."""
    padded_len = 48
    micro_batch = _engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    global_tokens = float(micro_batch["response_mask"].sum())
    tu.assign_non_tensor(micro_batch, batch_num_tokens=global_tokens)
    loss_fn = functools.partial(ppo_loss, config=_actor_config())

    losses, seen = [], []
    for scalars in (None, {"batch_num_tokens": torch.tensor(global_tokens)}):
        torch.manual_seed(0)
        engine = _graph_engine(_TinyLM(), padded_len)
        step = engine.trainer.forward_backward_step

        def record(*, input_dict, global_valid_tokens, step=step):
            seen.append(global_valid_tokens)
            return step(input_dict=input_dict, global_valid_tokens=global_valid_tokens)

        engine.trainer.forward_backward_step = record
        out = engine._graph_trainer_forward_backward(
            micro_batch.copy(), [micro_batch], None, loss_fn, global_tokens, scalars
        )
        losses.append(out["loss"][0])
    assert seen[1] is scalars["batch_num_tokens"]
    assert float(seen[0]) == float(seen[1]) == global_tokens
    assert losses[1] == pytest.approx(losses[0], rel=1e-6)


def test_graph_mode_step_handles_zero_batch_num_tokens():
    """``batch_num_tokens == 0`` (all response tokens masked out globally) gives finite ``0.0`` loss/grads and passes
    ``check_loss_consistency`` both via ``device_loss_scalars`` and via the CPU fallback."""
    from verl_hardware_plugin.accelerators.tpu.losses import loss_inputs

    padded_len = 48
    micro_batch = _engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    micro_batch["response_mask"] = torch.zeros_like(micro_batch["response_mask"])
    tu.assign_non_tensor(micro_batch, batch_num_tokens=0.0)
    loss_fn = functools.partial(ppo_loss, config=_actor_config(policy_loss=_ActorConfig(loss_mode="vanilla")))

    with mock.patch.object(loss_inputs, "accelerator_is_tpu", return_value=True):
        tpu_scalars = loss_inputs.device_loss_scalars(micro_batch, torch.device("cpu"))
    assert float(tpu_scalars["batch_num_tokens"]) == 1.0

    for scalars in (None, {}, tpu_scalars):
        torch.manual_seed(0)
        model = _TinyLM()
        engine = _graph_engine(model, padded_len)
        out = engine._graph_trainer_forward_backward(micro_batch.copy(), [micro_batch], None, loss_fn, 0.0, scalars)
        assert out["loss"] == [0.0]
        assert engine.trainer.calls == [((padded_len,), 1.0)]
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) == 0.0
        for name in ("actor/ppo_kl_token_mean", "actor/ppo_k3_token_mean", "actor/logprob_gap_abs_token_mean"):
            metric = out["metrics"][name]
            val = metric.aggregate() if hasattr(metric, "aggregate") else float(metric)
            assert val == 0.0


def test_build_graph_trainer_config():
    @dataclasses.dataclass
    class NewTPUConfig:
        use_simple_fsdp: bool = False
        bld_block_layout: bool = True

    @dataclasses.dataclass
    class TpuGraphTrainerCompileConfig:
        enable: bool = True
        mode: str = "aot_fx_trace"
        memory_policy: str = "default"
        save_regions: tuple[str, ...] = ()
        enable_remat_barriers: bool = False
        fsdp_barrier_policy: str = "none"

    class FakeTpuGraphTrainer:
        Config = lambda **kw: SimpleNamespace(**kw)  # noqa: E731

    class PallasCrossEntropyLoss:
        @dataclasses.dataclass
        class Config:
            soft_cap: float | None = None
            output_multiplier: float = 1.0

    opts = _options(
        trainer_mode="graph_trainer",
        graph_trainer={
            "memory_policy": "tpu_selective",
            "save_regions": "moe.routing, attention.inner_attention",
            "fsdp_barrier_policy": "interleave",
        },
    )
    new_modules = {
        "torchtitan": SimpleNamespace(),
        "torchtitan.experiments": SimpleNamespace(),
        "torchtitan.experiments.tpu": SimpleNamespace(),
        "torchtitan.experiments.tpu.tpu_job_config": SimpleNamespace(TPUConfig=NewTPUConfig),
        "torchtitan.experiments.tpu.graph_trainer": SimpleNamespace(),
        "torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer": SimpleNamespace(
            TpuGraphTrainer=FakeTpuGraphTrainer
        ),
        "torchtitan.experiments.tpu.graph_trainer.config": SimpleNamespace(
            TpuGraphTrainerCompileConfig=TpuGraphTrainerCompileConfig
        ),
        "torchtitan.experiments.tpu.components": SimpleNamespace(),
        "torchtitan.experiments.tpu.components.loss": SimpleNamespace(PallasCrossEntropyLoss=PallasCrossEntropyLoss),
    }
    with mock.patch.dict("sys.modules", new_modules):
        cfg = graph_trainer_mode.build_graph_trainer_config({"seed": 7}, opts, activation_checkpoint="full")
        assert cfg.tpu_config == NewTPUConfig(use_simple_fsdp=True, bld_block_layout=False)
        assert cfg.compile == TpuGraphTrainerCompileConfig(
            enable=False,
            mode="aot_fx_trace",
            memory_policy="tpu_selective",
            save_regions=("moe.routing", "attention.inner_attention"),
            enable_remat_barriers=True,
            fsdp_barrier_policy="interleave",
        )
        assert cfg.activation_checkpoint is None and cfg.seed == 7
        # The fused LM head needs a loss config with soft_cap / output_multiplier (TorchTitan checks them).
        assert cfg.loss == PallasCrossEntropyLoss.Config()

        # Simple string trainer_mode="graph_trainer" uses activation_checkpoint defaults
        opts_str = _options(trainer_mode="graph_trainer")
        cfg_str = graph_trainer_mode.build_graph_trainer_config({}, opts_str, activation_checkpoint="selective")
        assert cfg_str.compile == TpuGraphTrainerCompileConfig(
            enable=False,
            mode="aot_fx_trace",
            memory_policy="default",
            save_regions=(),
            enable_remat_barriers=False,
            fsdp_barrier_policy="none",
        )


def test_build_rl_graph_trainer_runs_the_step_as_one_packed_row():
    """The RL TpuGraphTrainer clears the unflatten-shim row length for its step; the loss metrics come back as is."""
    # TorchTitan's unflatten shim: a module-level row length set by a context manager.
    shim_state = {"seq_len": None}

    @contextlib.contextmanager
    def set_unflatten_seq_len(seq_len):
        old, shim_state["seq_len"] = shim_state["seq_len"], seq_len
        try:
            yield
        finally:
            shim_state["seq_len"] = old

    shim_mod = SimpleNamespace(
        set_unflatten_seq_len=set_unflatten_seq_len, get_unflatten_seq_len=lambda: shim_state["seq_len"]
    )
    max_context_length = 4  # smaller than the 12-token micro-batch, as max_seq_len < static_micro_batch_len

    class UpstreamTpuGraphTrainerStub:
        """TorchTitan's TpuGraphTrainer: sets the row length, runs the step, keeps its loss metrics, adds the grads."""

        def __init__(self, cfg):
            self.model = cfg.model
            self.loss_fn = None
            self.last_loss_metrics = None

        def forward_backward_step(self, *, input_dict, global_valid_tokens):
            params = [p for p in self.model.parameters() if p.requires_grad]
            with set_unflatten_seq_len(max_context_length):
                return self._make_fx_forward_backward_step(
                    self.model,
                    input_dict["input"],
                    input_dict["labels"],
                    global_valid_tokens,
                    params,
                    {"positions": input_dict["positions"]},
                )

        def _make_fx_forward_backward_step(self, model, inputs, labels, global_valid_tokens, params, extra_kwargs):
            loss, self.last_loss_metrics = self.loss_fn(
                model(inputs, **extra_kwargs), labels, global_valid_tokens=global_valid_tokens
            )
            for param, grad in zip(params, torch.autograd.grad(loss, params), strict=True):
                param.grad = grad if param.grad is None else param.grad + grad
            return loss.detach()

    class TinyPosLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(32, 8)
            self.head = torch.nn.Linear(8, 32, bias=False)
            self.seen_row_len: list[int | None] = []

        def forward(self, tokens, *, positions):
            # What a GatedDeltaNet would read to split the packed stream into rows.
            self.seen_row_len.append(shim_mod.get_unflatten_seq_len())
            # Fused LM-head output, as FusedLossLinearWrapper: (hidden_states, lm_head_weight_t), no logits.
            return torch.tanh(self.emb(tokens)) + 0.0 * positions.unsqueeze(-1).float(), self.head.weight.t()

    torch.manual_seed(0)
    model = TinyPosLM()
    modules = {
        "torchtitan": SimpleNamespace(),
        "torchtitan.experiments": SimpleNamespace(),
        "torchtitan.experiments.tpu": SimpleNamespace(),
        # Current (components/) and legacy layouts, so a real torchtitan on the path is not picked up instead.
        "torchtitan.experiments.tpu.components": SimpleNamespace(unflatten_shim=shim_mod),
        "torchtitan.experiments.tpu.components.unflatten_shim": shim_mod,
        "torchtitan.experiments.tpu.unflatten_shim": shim_mod,
        "torchtitan.experiments.tpu.graph_trainer": SimpleNamespace(),
        "torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer": SimpleNamespace(
            TpuGraphTrainer=UpstreamTpuGraphTrainerStub
        ),
    }
    with mock.patch.dict("sys.modules", modules), _patch_fused_logprobs():
        trainer = graph_trainer_mode.build_rl_graph_trainer(SimpleNamespace(model=model))
        spec = graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, temperature=1.0)
        trainer.loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)

        gen = torch.Generator().manual_seed(1)
        tokens = torch.randint(0, 32, (12,), generator=gen)
        labels = (torch.roll(tokens, -1), -torch.rand(12, generator=gen), torch.randn(12, generator=gen))
        labels = (*labels, (torch.rand(12, generator=gen) > 0.2).float())
        pos = torch.arange(12)
        global_tokens = torch.tensor(16.0)
        loss = trainer.forward_backward_step(
            input_dict={"input": tokens, "labels": labels, "positions": pos}, global_valid_tokens=global_tokens
        )
        log_probs = trainer.last_loss_metrics[graph_trainer_mode.LOG_PROBS_METRIC]
        ref_loss, ref_metrics = trainer.loss_fn(model(tokens, positions=pos), labels, global_valid_tokens=global_tokens)
        torch.testing.assert_close(loss, ref_loss.detach())
        torch.testing.assert_close(log_probs, ref_metrics[graph_trainer_mode.LOG_PROBS_METRIC])
        assert not log_probs.requires_grad

        # The step ran with the shim cleared, as in the eager engine, so a GatedDeltaNet would see the packed
        # micro-batch as one row, not max_context_length rows; the upstream value is restored afterwards.
        assert model.seen_row_len[0] is None
        assert shim_mod.get_unflatten_seq_len() is None
        with set_unflatten_seq_len(max_context_length):
            with graph_trainer_mode._packed_stream_as_one_row():
                assert shim_mod.get_unflatten_seq_len() is None
            assert shim_mod.get_unflatten_seq_len() == max_context_length


def test_graph_policy_loss_bypass_reinforce_is_and_rollout_rs():
    """Traced bypass-mode REINFORCE + truncated token IS + seq_mean_k1/k2 rollout_rs matches analytical math."""
    torch.manual_seed(7)
    prompt_lens, response_lens, padded_len = [3, 4], [5, 4], 24
    mb = _rl_micro_batch(prompt_lens, response_lens)
    old_lp, adv, mask = graph_trainer_mode.pack_rl_loss_targets(mb, padded_len)
    seq_ids = graph_trainer_mode.pack_rl_seq_ids(mb, padded_len)

    cfg = _actor_config(
        policy_loss=_ActorConfig(
            loss_mode="bypass_mode",
            rollout_correction=_ActorConfig(
                loss_type="reinforce",
                rollout_is="token",
                rollout_is_threshold=5.0,
                rollout_rs="seq_mean_k1,seq_mean_k2",
                rollout_rs_threshold="0.998_1.002,0.25",
            ),
        )
    )
    spec = graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
        functools.partial(ppo_loss, config=cfg),
        temperature=1.0,
        calculate_entropy=False,
        has_rollout_is_weights=False,
    )
    assert spec.loss_type == "reinforce"
    assert spec.rollout_is == "token" and spec.rollout_is_upper == 5.0
    assert spec.needs_seq_ids

    # Sequence 0 matches rollout closely (kept); sequence 1 drifts far (rejected by seq_mean_k1/k2).
    log_p = old_lp.clone()
    log_p[seq_ids == 0] += 0.0005
    log_p[seq_ids == 1] += 1.2
    log_p.requires_grad_(True)

    loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)
    global_tokens = torch.tensor(float(mask.sum()))
    with mock.patch.object(loss_fn, "log_probs", return_value=log_p):
        loss, metrics = loss_fn(
            None,
            (None, old_lp, adv, mask, seq_ids),
            global_valid_tokens=global_tokens,
        )
    assert graph_trainer_mode.LOG_PROBS_METRIC in metrics
    loss.backward()

    # Only sequence 0 should contribute; sequence 1 gradients must be zeroed by rollout_rs.
    kept_mask = mask * (seq_ids == 0).float()
    expected_w = torch.clamp(torch.exp((log_p.detach() - old_lp).clamp(-20.0, 20.0)) * mask, max=5.0)
    expected_loss = (-adv * log_p.detach() * expected_w * kept_mask).sum() / global_tokens
    torch.testing.assert_close(loss.detach(), expected_loss, rtol=1e-5, atol=1e-6)
    assert torch.all(log_p.grad[seq_ids == 1] == 0.0)
    assert log_p.grad[seq_ids == 0].abs().sum() > 0.0


def test_ffd_bin_slot_packing_and_roundtrip():
    """FFD bin packing groups variable-length sequences into fixed bins and restores exact packed token order."""
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        bin_slot_permutation,
        pack_micro_batches_into_bins,
    )

    lengths = [10, 6, 4, 9, 5, 3]
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    tokens = torch.arange(sum(lengths), dtype=torch.int64)
    data = TensorDict(
        {"input_ids": torch.nested.nested_tensor_from_jagged(tokens, offsets)},
        batch_size=[len(lengths)],
    )
    mbs, mb_indices = pack_micro_batches_into_bins(data, dp_group=None, bin_len=16, num_bins=2)
    # 10+6=16, 9+5=14, 4+3=7 -> 3 bins -> 2 micro-batches of <=2 bins each.
    assert len(mbs) == 2
    assert sorted(i for grp in mb_indices for i in grp) == list(range(len(lengths)))

    # Round-trip bin_slot_permutation on the first micro-batch padded to 32 (2 x 16).
    mb0_offsets = mbs[0]["input_ids"].offsets()
    mb0_vals = mbs[0]["input_ids"].values()
    padded = torch.cat([mb0_vals, torch.full((32 - mb0_vals.numel(),), -1, dtype=torch.int64)])
    slot_src, contig_src, tail_ranges = bin_slot_permutation(mb0_offsets, padded_len=32, bin_len=16)
    permuted = padded[slot_src]
    restored = permuted[contig_src]
    torch.testing.assert_close(restored, padded)
    for start, end in tail_ranges:
        assert 0 <= start < end <= 32
        assert torch.all(permuted[start:end] == -1)


def test_cp_layer_transforms_accepts_torchtitan_200_and_legacy_fields():
    """GatedDeltaNetContextParallelTransform supports both (#200) cp_strategy/cp_reduce_dtype and legacy field names."""
    from verl_hardware_plugin.accelerators.tpu.engines import cp_layer_transforms

    @dataclasses.dataclass
    class Transform200:
        load_balancer: str | None = None
        cp_strategy: str = "all_to_all"
        cp_reduce_dtype: str | None = None

    @dataclasses.dataclass
    class TransformLegacy:
        load_balancer: str | None = None
        strategy: str = "all_to_all"
        reduce_dtype: str | None = None

    opts = _options(
        context_parallel_gdn_strategy="allgather_reducescatter",
        context_parallel_gdn_reduce_dtype="bfloat16",
    )
    spec = cp_layer_transforms.QWEN3_5_GATED_DELTA_NET
    for cls, s_attr, d_attr in (
        (Transform200, "cp_strategy", "cp_reduce_dtype"),
        (TransformLegacy, "strategy", "reduce_dtype"),
    ):
        fake_mod = SimpleNamespace(GatedDeltaNetContextParallelTransform=cls)
        with (
            mock.patch.object(cp_layer_transforms, "module_available", return_value=True),
            mock.patch.object(cp_layer_transforms.importlib, "import_module", return_value=fake_mod),
        ):
            t = cp_layer_transforms._build_transform(spec, load_balancer=None, options=opts, num_layers=3)
            assert getattr(t, s_attr) == "allgather_reducescatter"
            assert getattr(t, d_attr) == "bfloat16"


def _empty_micro_batch_setup(empty):
    micro_batches = [
        _engine_micro_batch([3, 5, 2], [4, 6, 5], 1),
        _engine_micro_batch([4, 2], [5, 7], 2),
        _engine_micro_batch([2, 3], [6, 4], 3),
    ]
    for index in empty:
        micro_batches[index]["response_mask"] = torch.zeros_like(micro_batches[index]["response_mask"])
    global_tokens = float(sum(mb["response_mask"].sum() for mb in micro_batches))
    for mb in micro_batches:
        tu.assign_non_tensor(mb, batch_num_tokens=global_tokens)
    return micro_batches, global_tokens


@pytest.mark.parametrize("empty", [(1,), (0,), (0, 2)])
def test_graph_mode_step_survives_micro_batches_without_valid_response_tokens(empty):
    """A micro-batch whose response tokens are all masked out (at 397B a micro-batch holds about one sequence, so one
    filtered trajectory is enough) must not fail the step: verl's bypass-mode rollout correction would raise
    "response_mask must contain at least one valid token". It must also report one value per metric like the others,
    since verl aggregates each metric across data-parallel ranks per micro-batch and needs equal counts on every rank,
    including when the empty micro-batch comes first."""
    torch.manual_seed(0)
    micro_batches, global_tokens = _empty_micro_batch_setup(empty)
    loss_fn = functools.partial(ppo_loss, config=_actor_config())
    engine = _graph_engine(_TinyLM(), 48)

    out = engine._graph_trainer_forward_backward(micro_batches[0].copy(), micro_batches, None, loss_fn, global_tokens)

    assert len(out["loss"]) == 3
    for index in empty:
        assert out["loss"][index] == pytest.approx(0.0, abs=1e-6)
    counts = {name: len(getattr(values, "values", values)) for name, values in out["metrics"].items()}
    assert counts and set(counts.values()) == {3}, counts


def test_empty_micro_batch_with_a_cross_rank_loss_fails_instead_of_hanging():
    """With rollout_is_batch_normalize the loss all-reduces across ranks, so one rank cannot skip it."""
    config = SimpleNamespace(policy_loss={"rollout_correction": {"rollout_is_batch_normalize": True}})
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    loss_fn = functools.partial(ppo_loss, config=config)
    assert torchtitan_tpu._loss_needs_every_rank(loss_fn)
    with pytest.raises(RuntimeError, match="rollout_is_batch_normalize"):
        torchtitan_tpu._check_empty_micro_batch_can_skip_loss(loss_fn)
    assert not torchtitan_tpu._loss_needs_every_rank(functools.partial(ppo_loss, config=_actor_config()))


def test_backfill_empty_micro_batch_metrics_uses_the_nearest_filled_micro_batch():
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    engine = SimpleNamespace()
    outputs = [{"metrics": None}, {"metrics": {"a": [1.0]}}, {"metrics": None}, {"metrics": {"a": [3.0]}}]
    torchtitan_tpu._backfill_empty_micro_batch_metrics(engine, outputs)
    assert [o["metrics"] for o in outputs] == [{"a": [1.0]}, {"a": [1.0]}, {"a": [1.0]}, {"a": [3.0]}]
    assert outputs[0]["metrics"] is not outputs[1]["metrics"]  # copies
    assert engine._last_micro_batch_metrics == {"a": [3.0]}

    # A later call whose micro-batches are all empty repeats the last metrics this engine saw.
    outputs = [{"metrics": None}]
    torchtitan_tpu._backfill_empty_micro_batch_metrics(engine, outputs)
    assert outputs[0]["metrics"] == {"a": [3.0]}
