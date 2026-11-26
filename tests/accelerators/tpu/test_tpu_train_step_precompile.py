# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the opt-in trainer precompile (``VERL_TPU_PRECOMPILE_TRAIN_STEP``)."""

import contextlib
import functools
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.accelerators.tpu import test_tpu_graph_trainer_mode as gt
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.engines import graph_trainer_mode, train_step_precompile
from verl_hardware_plugin.accelerators.tpu.engines.train_step_precompile import (
    build_synthetic_train_batch,
    fire_precompile,
    precompile_engine_train_step,
    synthetic_sequence_lengths,
    wait_for_precompile,
)


def test_synthetic_batch_layout():
    data = build_synthetic_train_batch(num_seqs=2, prompt_len=5, response_len=7, vocab_size=50)
    assert data.batch_size == torch.Size([2])
    for key in ("input_ids", "position_ids", "prompts", "responses"):
        assert data[key].is_nested, key
    assert data["input_ids"].offsets().tolist() == [0, 12, 24]
    assert data["prompts"].offsets().diff().tolist() == [5, 5]
    assert data["responses"].offsets().diff().tolist() == [7, 7]
    assert data["position_ids"].values().tolist()[:12] == list(range(12))
    assert int(data["input_ids"].values().max()) < 50
    for key in ("response_mask", "loss_mask", "old_log_probs", "rollout_log_probs", "advantages"):
        assert tuple(data[key].shape) == (2, 7), key
    assert data["response_mask"].dtype == torch.int64 and int(data["loss_mask"].sum()) == 14
    assert bool((data["old_log_probs"] < 0).all())
    torch.testing.assert_close(data["old_log_probs"], data["rollout_log_probs"])
    # The prompt/response split is the one the packed graph-loss targets use.
    old, adv, mask = graph_trainer_mode.pack_rl_loss_targets(data, 32)
    assert int(mask.sum()) == 14 and old.shape == (32,)


def test_synthetic_sequence_lengths_fit_every_limit():
    assert synthetic_sequence_lengths(static_len=131072, bin_slot_len=65536) == (64, 192)
    p, r = synthetic_sequence_lengths(static_len=128, bin_slot_len=0, max_seq_len=0)
    assert p + r == 128 and p > 0 and r > 0


def _precompile_engine(padded_len=48):
    model = gt._TinyLM()
    engine = gt._graph_engine(model, padded_len)
    engine._graph_trainer_mode = True
    engine.engine_config = SimpleNamespace(max_token_len_per_gpu=0)
    engine.module = [model]
    zero_calls = []

    def zero_grad():
        zero_calls.append(1)
        for p in model.parameters():
            p.grad = None

    engine.optimizer_zero_grad = zero_grad

    def forward_backward_batch(data, loss_function, forward_only=False):
        # The real engine's micro-batching reduces to one micro-batch here; run the graph-trainer path itself.
        tokens = float(data["loss_mask"].sum())
        tu.assign_non_tensor(data, batch_num_tokens=tokens, dp_size=1)
        with gt._patch_fused_logprobs():
            return engine._graph_trainer_forward_backward(data, [data], None, loss_function, tokens)

    engine.forward_backward_batch = forward_backward_batch
    return engine, model, zero_calls


def _non_tensor():
    return {"temperature": 1.0, "calculate_entropy": False, "global_batch_size": None, "use_dynamic_bsz": False}


@contextlib.contextmanager
def _options(static_len):
    with (
        mock.patch(
            "verl_hardware_plugin.accelerators.tpu.options.get_tpu_engine_options",
            return_value=SimpleNamespace(static_micro_batch_len=static_len, kda_document_budget=None),
        ),
        mock.patch.object(train_step_precompile, "_vocab_size", return_value=100),
    ):
        yield


def test_precompile_traces_and_discards_gradients_and_buffers():
    engine, model, zero_calls = _precompile_engine()
    model.register_buffer("counter", torch.zeros(3))
    orig_forward = model.forward

    def forward(tokens):  # a buffer the forward mutates (e.g. MoE tokens_per_expert)
        model.counter.add_(1.0)
        return orig_forward(tokens)

    model.forward = forward
    params_before = [p.detach().clone() for p in model.parameters()]
    loss_fn = functools.partial(ppo_loss, config=gt._actor_config())
    with _options(48):
        result = precompile_engine_train_step(engine, loss_fn, non_tensor=_non_tensor())
    assert result["status"] == "ok", result
    assert engine.trainer.calls and engine.trainer.calls[0][0] == (48,)
    assert engine._graph_trainer_traced_len == 48 and engine._graph_policy_loss_spec is not None
    assert getattr(engine, train_step_precompile.PROVISIONAL_ATTR) is True
    assert all(p.grad is None for p in model.parameters())
    assert len(zero_calls) == 2
    for before, p in zip(params_before, model.parameters(), strict=True):
        assert torch.equal(before, p.detach())
    assert torch.equal(model.counter, torch.zeros(3))
    assert not hasattr(engine, "_last_micro_batch_metrics") or "_last_micro_batch_metrics" not in vars(engine)

    # The first real step reuses the precompiled state and clears the provisional flag.
    mb = gt._engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    tokens = float(mb["response_mask"].sum())
    tu.assign_non_tensor(mb, batch_num_tokens=tokens)
    with gt._patch_fused_logprobs():
        out = engine._graph_trainer_forward_backward(mb.copy(), [mb], None, loss_fn, tokens)
    assert len(out["loss"]) == 1
    assert getattr(engine, train_step_precompile.PROVISIONAL_ATTR) is False


def test_real_step_with_other_loss_constants_retraces_instead_of_failing():
    engine, model, _ = _precompile_engine()
    loss_fn = functools.partial(ppo_loss, config=gt._actor_config())
    with _options(48):
        assert precompile_engine_train_step(engine, loss_fn, non_tensor=_non_tensor())["status"] == "ok"
    first_spec = engine._graph_policy_loss_spec
    mb = gt._engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    tu.assign_non_tensor(mb, temperature=0.7)
    tokens = float(mb["response_mask"].sum())
    tu.assign_non_tensor(mb, batch_num_tokens=tokens)
    with gt._patch_fused_logprobs():
        engine._graph_trainer_forward_backward(mb.copy(), [mb], None, loss_fn, tokens)
    assert engine._graph_policy_loss_spec.temperature == 0.7 != first_spec.temperature
    # Not provisional any more: a later change is the usual hard error.
    tu.assign_non_tensor(mb, temperature=0.5)
    with pytest.raises(RuntimeError, match="traced the policy loss"), gt._patch_fused_logprobs():
        engine._graph_trainer_forward_backward(mb.copy(), [mb], None, loss_fn, tokens)


def test_failed_precompile_restores_the_untraced_state():
    engine, model, _ = _precompile_engine()
    engine.trainer.forward_backward_step = mock.Mock(side_effect=RuntimeError("compile OOM"))
    loss_fn = functools.partial(ppo_loss, config=gt._actor_config())
    with _options(48):
        result = precompile_engine_train_step(engine, loss_fn, non_tensor=_non_tensor())
    assert result["status"] == "failed" and "compile OOM" in result["error"]
    assert engine._graph_policy_loss_spec is None and engine._graph_trainer_traced_len is None
    assert engine.trainer.loss_fn is None
    assert not getattr(engine, train_step_precompile.PROVISIONAL_ATTR, False)
    assert all(p.grad is None for p in model.parameters())


def test_precompile_skips_when_not_applicable():
    engine, _, _ = _precompile_engine()
    loss_fn = functools.partial(ppo_loss, config=gt._actor_config())
    engine._graph_trainer_mode = False
    assert precompile_engine_train_step(engine, loss_fn)["status"] == "skipped"
    engine._graph_trainer_mode = True
    assert precompile_engine_train_step(engine, None)["status"] == "skipped"
    with _options(0):
        assert precompile_engine_train_step(engine, loss_fn)["status"] == "skipped"
    engine._graph_trainer_traced_len = 48
    with _options(48):
        assert precompile_engine_train_step(engine, loss_fn)["reason"] == "graph step already traced"


def _fake_trainer(hybrid=False):
    cfg = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(
                policy_loss={"rollout_correction": {}},
                calculate_entropy=False,
                entropy_coeff=0.0,
                ppo_mini_batch_size=4,
            ),
            rollout=SimpleNamespace(temperature=1.0, n=8),
        )
    )
    wg = SimpleNamespace(precompile_train_step=mock.Mock(return_value=["ref0", "ref1"]))
    return SimpleNamespace(config=cfg, actor_rollout_wg=wg, _enable_hybrid_replicas=hybrid)


def test_fire_is_non_blocking_once_and_wait_collects():
    trainer = _fake_trainer()
    assert fire_precompile(trainer) is True
    trainer.actor_rollout_wg.precompile_train_step.assert_called_once_with(
        {"temperature": 1.0, "calculate_entropy": False, "score_centering": False, "global_batch_size": 32}
    )
    assert trainer._tpu_precompile_refs == ["ref0", "ref1"]
    assert fire_precompile(trainer) is False  # only once
    fake_ray = SimpleNamespace(get=mock.Mock(return_value=[{"status": "ok"}, {"status": "ok"}]))
    with mock.patch.dict("sys.modules", {"ray": fake_ray}):
        wait_for_precompile(trainer)
        wait_for_precompile(trainer)  # second call: nothing pending
    fake_ray.get.assert_called_once_with(["ref0", "ref1"])


def test_fire_skips_hybrid_replicas_and_wait_tolerates_worker_errors():
    trainer = _fake_trainer(hybrid=True)
    assert fire_precompile(trainer) is False
    trainer.actor_rollout_wg.precompile_train_step.assert_not_called()
    trainer = _fake_trainer()
    fire_precompile(trainer)
    fake_ray = SimpleNamespace(get=mock.Mock(side_effect=RuntimeError("worker died")))
    with mock.patch.dict("sys.modules", {"ray": fake_ray}):
        wait_for_precompile(trainer)  # logs, does not raise


def test_driver_hooks_fire_only_when_enabled(monkeypatch):
    class FakeBase:
        def _update_actor(self, batch, metrics):
            self.events.append("update")
            return "out"

    class FakeSeparateAsync(FakeBase):
        def on_train_begin(self):
            self.events.append("warmup")

    base_mod = SimpleNamespace(PPOTrainer=FakeBase)
    sep_mod = SimpleNamespace(PPOTrainerSeparateAsync=FakeSeparateAsync)
    v1 = SimpleNamespace(trainer_base=base_mod, trainer_separate_async=sep_mod)
    modules = {
        "verl.trainer.ppo.v1": v1,
        "verl.trainer.ppo.v1.trainer_base": base_mod,
        "verl.trainer.ppo.v1.trainer_separate_async": sep_mod,
    }
    with mock.patch.dict("sys.modules", modules):
        train_step_precompile.apply_driver_hooks()
        train_step_precompile.apply_driver_hooks()  # idempotent
    trainer = FakeSeparateAsync()
    trainer.events = []
    with (
        mock.patch.object(train_step_precompile, "fire_precompile", side_effect=lambda t: t.events.append("fire")),
        mock.patch.object(train_step_precompile, "wait_for_precompile", side_effect=lambda t: t.events.append("wait")),
    ):
        monkeypatch.delenv(train_step_precompile.ENV_VAR, raising=False)
        trainer.on_train_begin()
        assert trainer.events == ["warmup"]
        monkeypatch.setenv(train_step_precompile.ENV_VAR, "1")
        trainer.on_train_begin()
        assert trainer.events == ["warmup", "warmup", "fire"]
        assert trainer._update_actor("b", {}) == "out"
        assert trainer.events[-2:] == ["wait", "update"]


def test_worker_method_is_registered_non_blocking():
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers import engine_workers

    train_step_precompile.apply_worker_method()
    method = engine_workers.ActorRolloutRefWorker.precompile_train_step
    attrs = getattr(method, MAGIC_ATTR)
    assert attrs["blocking"] is False
