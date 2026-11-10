# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of ``engine_kwargs.tpu.trainer_mode=graph_trainer`` (``engines/graph_trainer_mode.py``).

The TPU compile pass needs ``torch_tpu`` on TPU; the on-TPU parity check against the eager engine runs in verl's TPU CI.
These check the parts that decide numerics and TorchTitan integration: the option surface, the graph-trainer config,
the trace-safe PPO loss against verl's, the packed loss inputs against verl's response layout, the
out-of-place fused log-prob backward and fake-op shim, the RL ``TpuGraphTrainer`` wrapper's auxiliary outputs, and
the one-micro-batch engine step.
"""

from __future__ import annotations

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


def _options(**tpu):
    return get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": tpu}))


def test_trainer_mode_options():
    assert _options().trainer_mode is None
    assert _options(trainer_mode="Graph_Trainer").trainer_mode == "graph_trainer"
    assert graph_trainer_mode.is_graph_trainer_mode(_options(trainer_mode="graph_trainer"))
    assert not graph_trainer_mode.is_graph_trainer_mode(_options(trainer_mode="eager_blocks"))
    opts = _options(
        trainer_mode="graph_trainer",
        graph_trainer_memory_policy="tpu_selective",
        graph_trainer_save_regions="moe.routing,attention.inner_attention",
        graph_trainer_fsdp_schedule="Interleave",
    )
    assert opts.graph_trainer_fsdp_schedule == "interleave"
    with pytest.raises(ValueError, match="trainer_mode must be one of"):
        _options(trainer_mode="graph")
    with pytest.raises(ValueError, match="only apply with"):
        _options(graph_trainer_memory_policy="full")
    with pytest.raises(ValueError, match="graph_trainer_fsdp_schedule must be one of"):
        _options(trainer_mode="graph_trainer", graph_trainer_fsdp_schedule="eager")


def test_validate_graph_trainer_engine():
    good = SimpleNamespace(forward_only=False, param_offload=False, data_parallel_shard_size=16)
    options = _options(trainer_mode="graph_trainer", static_micro_batch_len=8192)
    graph_trainer_mode.validate_graph_trainer_engine(good, options, use_simple_fsdp=True, tpu_kernels=frozenset())

    bad = SimpleNamespace(forward_only=True, param_offload=True, expert_parallel_size=2, context_parallel_size=1)
    with pytest.raises(ValueError) as e:
        graph_trainer_mode.validate_graph_trainer_engine(
            bad, _options(trainer_mode="graph_trainer"), use_simple_fsdp=False, tpu_kernels=frozenset({"kda"})
        )
    message = str(e.value)
    for expected in ("forward_only", "use_simple_fsdp", "static_micro_batch_len", "expert_parallel_size=2", "KDA"):
        assert expected in message, (expected, message)
    assert "param_offload" in message


def test_memory_policy_defaults_follow_activation_checkpoint():
    options = _options(trainer_mode="graph_trainer")
    assert graph_trainer_mode.resolve_memory_policy(options, "full") == "full"
    assert graph_trainer_mode.resolve_memory_policy(options, "selective") == "default"
    assert graph_trainer_mode.resolve_memory_policy(options, "none") == "none"
    explicit = _options(trainer_mode="graph_trainer", graph_trainer_memory_policy="tpu_selective")
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
            dict(policy_loss=_ActorConfig(loss_mode="bypass_mode", rollout_correction=_ActorConfig(rollout_rs="seq"))),
            "rollout_rs",
        ),
        (
            dict(
                policy_loss=_ActorConfig(
                    loss_mode="bypass_mode", rollout_correction=_ActorConfig(loss_type="reinforce")
                )
            ),
            "reinforce",
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
    num_tokens, vocab, dp_size, global_tokens = 64, 50, 4, 100.0
    config = _actor_config()
    spec = graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, temperature=0.9)
    loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)

    logits = torch.randn(num_tokens, vocab, requires_grad=True)
    target_ids = torch.randint(0, vocab, (num_tokens,))
    old_log_probs = torch.log_softmax(torch.randn(num_tokens, vocab), -1).gather(-1, target_ids[:, None]).squeeze(-1)
    advantages = torch.randn(num_tokens)
    loss_mask = (torch.rand(num_tokens) > 0.3).float()
    # A masked-out bucket-padding token with an extreme log-prob must not leak into the loss.
    old_log_probs[-1] = -1e4

    loss, (log_probs,) = loss_fn(
        logits,
        (target_ids, old_log_probs, advantages, loss_mask),
        global_valid_tokens=torch.tensor(global_tokens),
    )
    loss.backward()
    graph_grad = logits.grad.clone()
    assert not log_probs.requires_grad

    logits.grad = None
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
    torch.testing.assert_close(graph_grad, logits.grad)
    assert graph_trainer_mode.check_loss_consistency(loss.item(), expected.item(), dp_size)
    assert not graph_trainer_mode.check_loss_consistency(loss.item(), 2 * expected.item(), dp_size)


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
    index, valid = _response_gather_index(micro_batch, width, padded_len)
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


def _reference_forward_op(b_block, h_block, v_block):
    def op(x, labels, weight_t):
        logits = x.float() @ weight_t.float()
        lse = torch.logsumexp(logits, -1)
        return lse - logits.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1), lse

    return op


def test_functional_fused_logprob_backward_matches_in_place_backward():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    weight_t = torch.randn(8, 30) * 0.3
    labels = torch.randint(0, 30, (10,))
    cotangent = torch.randn(10)
    grads = []
    for functional in (False, True):
        xk, wk = x.clone().requires_grad_(), weight_t.clone().requires_grad_()
        with (
            mock.patch.object(fused_logprob, "_forward_op", _reference_forward_op),
            # Several vocab chunks of 7 columns over V = 30, the last one ragged.
            mock.patch.object(fused_logprob, "backward_vocab_chunk", lambda num_tokens, vocab: 7),
        ):
            logp = fused_logprob.fused_linear_logprobs_tpu(
                xk, labels, wk, temperature=1.5, b_block_size=16, functional_backward=functional
            )
        (logp * cotangent).sum().backward()
        grads.append((logp.detach(), xk.grad, wk.grad))
    for eager, functional in zip(*grads, strict=True):
        torch.testing.assert_close(functional, eager)


def test_functional_fused_logprob_backward_traces_without_mutations():
    """make_fx of forward + backward records no in-place op from the functional backward."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def step(x, weight_t, labels):
        logp = fused_logprob.fused_linear_logprobs_tpu(x, labels, weight_t, b_block_size=16, functional_backward=True)
        return torch.autograd.grad(logp.sum(), (x, weight_t))

    with (
        mock.patch.object(fused_logprob, "_forward_op", _reference_forward_op),
        mock.patch.object(fused_logprob, "backward_vocab_chunk", lambda num_tokens, vocab: 7),
    ):
        gm = make_fx(step)(
            torch.randn(10, 8, requires_grad=True), torch.randn(8, 30, requires_grad=True), torch.randint(0, 30, (10,))
        )
    in_place = [
        n.target for n in gm.graph.nodes if n.op == "call_function" and str(n.target).split(".")[-2].endswith("_")
    ]
    assert not in_place, in_place


def test_forward_op_overrides_singleton_pallas_fake_for_make_fx():
    """Unpatched TorchTitan registers a (1,) fake loss for reduction='none'; _forward_op overrides it to (rows,)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    op_name = "linear_loss_fwd_test_singleton"
    qualname = f"pallas::{op_name}"
    if not hasattr(getattr(torch.ops, "pallas", None), op_name):

        @torch.library.custom_op(
            qualname, mutates_args=(), schema="(Tensor x, Tensor labels, Tensor weights) -> (Tensor, Tensor)"
        )
        def _singleton_fwd(x: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
            logits = x.float() @ weights.float()
            lse = torch.logsumexp(logits, -1)
            return lse - logits.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1), lse

        _singleton_fwd.register_fake(
            lambda x, labels, weights: (
                torch.empty(1, dtype=torch.float32, device=x.device),
                torch.empty(x.shape[0], dtype=torch.float32, device=x.device),
            )
        )

    fused_logprob._VERIFIED_FORWARD_OPS.discard(op_name)
    loss_mod = SimpleNamespace(_get_loss_op_names=lambda *args: (op_name, "unused_bwd"))
    with mock.patch.object(fused_logprob, "_loss_module", return_value=loss_mod):
        gm = make_fx(
            lambda x, labels, w: fused_logprob.fused_linear_logprobs_tpu(
                x, labels, w, b_block_size=8, h_block_size=4, v_block_size=16, functional_backward=True
            )
        )(torch.randn(10, 4), torch.randint(0, 16, (10,)), torch.randn(4, 16))
    out = gm(torch.randn(10, 4), torch.randint(0, 16, (10,)), torch.randn(4, 16))
    assert tuple(out.shape) == (10,)


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
        return self.head(torch.tanh(self.emb(tokens)))


class _FakeGraphTrainer:
    """Stands in for TpuGraphTrainer: runs the installed loss on a tiny model and its backward into .grad."""

    def __init__(self, model):
        self.model = model
        self.loss_fn = None
        self.last_loss_aux = ()
        self.calls = []

    def forward_backward_step(self, *, input_dict, global_valid_tokens):
        self.calls.append((tuple(input_dict["input"].shape), float(global_valid_tokens)))
        loss, aux = self.loss_fn(
            self.model(input_dict["input"]), input_dict["labels"], global_valid_tokens=global_valid_tokens
        )
        loss.backward()
        self.last_loss_aux = aux
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
    logp = torch.log_softmax(model(ids), -1).gather(-1, torch.roll(ids, -1)[:, None]).squeeze(-1)
    output = {"log_probs": torch.nested.nested_tensor_from_jagged(logp, micro_batch["input_ids"].offsets())}
    loss, _ = ppo_loss(config, model_output=output, data=micro_batch.copy())
    loss.backward()
    torch.testing.assert_close(float(loss), out["loss"][0], rtol=1e-5, atol=1e-6)
    for g, p in zip(graph_grads, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, rtol=1e-5, atol=1e-6)

    # TpuGraphTrainer assigns param.grad, so a second micro-batch would drop the first one's gradients (TODO-1).
    with pytest.raises(RuntimeError, match="TODO-1"):
        engine._graph_trainer_forward_backward(data, [micro_batch, micro_batch], None, loss_fn, global_tokens)

    # A micro-batch with another padded length cannot reuse the traced step.
    engine.prepare_model_inputs = lambda micro_batch: (
        torch.zeros(64, dtype=torch.int64),
        {"positions": torch.zeros(64)},
        {},
        {"labels": torch.zeros(64, dtype=torch.int64), "orig_seq_len": 1},
    )
    with pytest.raises(RuntimeError, match="TODO-2"):
        engine._graph_trainer_forward_backward(data, [micro_batch], None, loss_fn, global_tokens)


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
        graph_trainer_memory_policy="tpu_selective",
        graph_trainer_save_regions="moe.routing, attention.inner_attention",
        graph_trainer_fsdp_schedule="interleave",
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

    # TorchTitan 109bd618 keeps the loss in torchtitan/experiments/tpu/loss.py.
    legacy_modules = dict(new_modules)
    legacy_modules["torchtitan.experiments.tpu.components"] = None
    legacy_modules["torchtitan.experiments.tpu.components.loss"] = None
    legacy_modules["torchtitan.experiments.tpu.loss"] = SimpleNamespace(PallasCrossEntropyLoss=PallasCrossEntropyLoss)
    with mock.patch.dict("sys.modules", legacy_modules):
        cfg = graph_trainer_mode.build_graph_trainer_config({}, opts, activation_checkpoint="full")
        assert cfg.loss == PallasCrossEntropyLoss.Config()


def test_build_rl_graph_trainer_traces_aux_outputs():
    """Unpatched upstream TpuGraphTrainer: RLTpuGraphTrainer returns detached aux log-probs; the trace is reused."""
    from torch.fx.experimental.proxy_tensor import make_fx

    def upstream_make_fwd_bwd_step(model, loss_fn, *, collect_moe_routing_metrics: bool = False):
        def fwd_bwd(inputs, labels, global_valid_tokens, extra_kwargs):
            pred = model(inputs, **extra_kwargs)
            routing = [pred.detach().sum()] if collect_moe_routing_metrics else []
            res = loss_fn(pred, labels, global_valid_tokens=global_valid_tokens)
            loss = res[0] if isinstance(res, tuple) else res  # upstream drops res[1]!
            params = [p for _, p in model.named_parameters(remove_duplicate=False) if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss, *grads, *routing]

        return fwd_bwd

    def upstream_run_traced(traced_step, *, module):
        def run(*args):
            params_and_buffers = [t for _, t in module.named_parameters(remove_duplicate=False)]
            return traced_step.gm(*params_and_buffers, *args)

        return run

    trainer_mod = SimpleNamespace(make_fwd_bwd_step=upstream_make_fwd_bwd_step)
    tracer_mod = SimpleNamespace(run_traced=upstream_run_traced)

    class UpstreamTpuGraphTrainerStub:
        def __init__(self, cfg):
            self.model = cfg.model
            self.model_parts = [self.model]
            self.loss_fn = None
            self._traced_step = None
            self._collect_moe_routing_metrics = True
            self.routing_received: list[list[torch.Tensor]] = []
            self.optimizers = SimpleNamespace(optimizers=[torch.optim.SGD(self.model.parameters(), lr=0.1)])

        def forward_backward_step(self, *, input_dict, global_valid_tokens):
            params = [p for _, p in self.model.named_parameters(remove_duplicate=False) if p.requires_grad]
            return self._make_fx_forward_backward_step(
                self.model,
                input_dict["input"],
                input_dict["labels"],
                global_valid_tokens,
                params,
                {"positions": input_dict["positions"]},
            )

        def _make_fx_forward_backward_step(self, model, inputs, labels, global_valid_tokens, params, extra_kwargs):
            if self._traced_step is None:
                step_fn = trainer_mod.make_fwd_bwd_step(
                    model, self.loss_fn, collect_moe_routing_metrics=self._collect_moe_routing_metrics
                )

                def flat_fn(*p_and_args):
                    n_p = len(params)
                    param_dict = {
                        name: p
                        for (name, _), p in zip(
                            model.named_parameters(remove_duplicate=False), p_and_args[:n_p], strict=True
                        )
                    }
                    with torch.nn.utils.stateless._reparametrize_module(model, param_dict):
                        return step_fn(*p_and_args[n_p:])

                gm = make_fx(flat_fn)(*params, inputs, labels, global_valid_tokens, extra_kwargs)
                self._traced_step = SimpleNamespace(gm=gm)
            outputs = tracer_mod.run_traced(self._traced_step, module=model)(
                inputs, labels, global_valid_tokens, extra_kwargs
            )
            loss = outputs[0]
            grads = outputs[1 : 1 + len(params)]
            if self._collect_moe_routing_metrics:
                self.routing_received.append(list(outputs[1 + len(params) :]))
            for param, grad in zip(params, grads, strict=True):
                param.grad = grad
            return loss

    class TinyPosLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(32, 8)
            self.head = torch.nn.Linear(8, 32, bias=False)

        def forward(self, tokens, *, positions):
            return self.head(torch.tanh(self.emb(tokens)) + 0.0 * positions.unsqueeze(-1).float())

    torch.manual_seed(0)
    model = TinyPosLM()
    ref_model = TinyPosLM()
    ref_model.load_state_dict(model.state_dict())

    modules = {
        "torchtitan": SimpleNamespace(),
        "torchtitan.experiments": SimpleNamespace(),
        "torchtitan.experiments.graph_trainer": SimpleNamespace(trainer=trainer_mod, make_fx_tracer=tracer_mod),
        "torchtitan.experiments.graph_trainer.trainer": trainer_mod,
        "torchtitan.experiments.graph_trainer.make_fx_tracer": tracer_mod,
        "torchtitan.experiments.tpu": SimpleNamespace(),
        "torchtitan.experiments.tpu.graph_trainer": SimpleNamespace(),
        "torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer": SimpleNamespace(
            TpuGraphTrainer=UpstreamTpuGraphTrainerStub
        ),
    }
    with mock.patch.dict("sys.modules", modules):
        trainer = graph_trainer_mode.build_rl_graph_trainer(SimpleNamespace(model=model))
        spec = graph_trainer_mode.GraphPolicyLossSpec(0.2, 0.28, 3.0, temperature=1.0)
        trainer.loss_fn = graph_trainer_mode.GraphPolicyLoss(spec)

        global_tokens = torch.tensor(16.0)
        mbs = []
        for seed in (1, 2):
            gen = torch.Generator().manual_seed(seed)
            tokens = torch.randint(0, 32, (12,), generator=gen)
            targets = torch.roll(tokens, -1)
            old_lp = -torch.rand(12, generator=gen)
            adv = torch.randn(12, generator=gen)
            mask = (torch.rand(12, generator=gen) > 0.2).float()
            mbs.append((tokens, (targets, old_lp, adv, mask), torch.arange(12)))

        # Two training steps of one micro-batch each: the second reuses the traced graph.
        for tokens, labels, pos in mbs:
            for m in (model, ref_model):
                for p in m.parameters():
                    p.grad = None
            loss = trainer.forward_backward_step(
                input_dict={"input": tokens, "labels": labels, "positions": pos},
                global_valid_tokens=global_tokens,
            )
            (aux_logp,) = trainer.last_loss_aux
            ref_loss, (ref_logp,) = trainer.loss_fn(
                ref_model(tokens, positions=pos), labels, global_valid_tokens=global_tokens
            )
            ref_loss.backward()
            torch.testing.assert_close(loss, ref_loss.detach())
            torch.testing.assert_close(aux_logp, ref_logp)
            for (_, p), (_, rp) in zip(model.named_parameters(), ref_model.named_parameters(), strict=True):
                torch.testing.assert_close(p.grad, rp.grad)

        # Routing metrics list kept its exact length (1 tensor per step), without leaking aux_logp into it.
        assert [len(r) for r in trainer.routing_received] == [1, 1]
