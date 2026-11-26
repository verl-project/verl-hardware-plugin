# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the opt-in trainer precompile (``engine_kwargs.tpu.precompile_train_step``)."""

import asyncio
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
    collect_precompile,
    precompile_engine_train_step,
    start_precompile,
    synthetic_sequence_lengths,
    worker_precompile_train_step,
)


def _loss_fn(**overrides):
    return functools.partial(ppo_loss, config=gt._actor_config(**overrides))


def _non_tensor(**overrides):
    values = {
        "temperature": 1.0,
        "calculate_entropy": False,
        "global_batch_size": None,
        "use_dynamic_bsz": True,
        "max_token_len_per_gpu": 48,
        "micro_batch_size_per_gpu": None,
    }
    values.update(overrides)
    return values


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
    p, r = synthetic_sequence_lengths(static_len=128)
    assert p + r == 128 and p > 0 and r > 0
    # num_seqs sequences share the micro-batch; each one fits a GatedDeltaNet row slot and the model length.
    assert sum(synthetic_sequence_lengths(static_len=512, num_seqs=4)) == 128
    assert sum(synthetic_sequence_lengths(static_len=8192, num_seqs=2, row_slot_len=100)) == 100
    assert sum(synthetic_sequence_lengths(static_len=8192, max_seq_len=40)) == 40
    assert synthetic_sequence_lengths(static_len=3, num_seqs=2) is None


def _precompile_engine(padded_len=48, **tpu):
    model = gt._TinyLM()
    engine = gt._graph_engine(model, padded_len)
    engine._graph_trainer_mode = True
    engine._tpu_options = gt._options(
        trainer_mode="graph_trainer", static_micro_batch_len=padded_len, precompile_train_step=True, **tpu
    )
    engine.module = [model]
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=100, hidden_size=16))  # _TinyLM's
    engine.trainer.ntokens_seen = 7
    zero_calls = []

    def zero_grad():
        zero_calls.append(1)
        for p in model.parameters():
            p.grad = None

    engine.optimizer_zero_grad = zero_grad

    def forward_backward_batch(data, loss_function, forward_only=False):
        # One micro-batch: the graph-trainer path itself (test_real_forward_backward_batch_path covers the rest).
        tokens = float(data["loss_mask"].sum())
        # One micro-batch in its original order (no dynamic-batch indices to restore).
        tu.assign_non_tensor(data, batch_num_tokens=tokens, dp_size=1, use_dynamic_bsz=False)
        engine.trainer.ntokens_seen += padded_len
        with gt._patch_fused_logprobs():
            return engine._graph_trainer_forward_backward(data, [data], None, loss_function, tokens)

    engine.forward_backward_batch = forward_backward_batch
    return engine, model, zero_calls


def test_precompile_traces_and_discards_its_effects():
    engine, model, zero_calls = _precompile_engine()
    model.register_buffer("counter", torch.zeros(3))
    orig_forward = model.forward

    def forward(tokens):  # a buffer the forward mutates (e.g. MoE tokens_per_expert)
        model.counter.add_(1.0)
        return orig_forward(tokens)

    model.forward = forward
    params_before = [p.detach().clone() for p in model.parameters()]
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)
    result = precompile_engine_train_step(engine, _loss_fn(), non_tensor=_non_tensor())
    assert result["status"] == "ok", result
    assert engine.trainer.calls and engine.trainer.calls[0][0] == (48,)
    assert engine._graph_trainer_traced_len == 48 and engine._graph_policy_loss_spec is not None
    assert getattr(engine, train_step_precompile.PROVISIONAL_ATTR) is True
    assert all(p.grad is None for p in model.parameters())
    assert len(zero_calls) == 2
    for before, p in zip(params_before, model.parameters(), strict=True):
        assert torch.equal(before, p.detach())
    assert torch.equal(model.counter, torch.zeros(3))
    # The trainer's per-step outputs and the RNG state are those of before the synthetic step.
    assert engine.trainer.ntokens_seen == 7 and engine.trainer.last_loss_metrics is None
    assert "_last_micro_batch_metrics" not in vars(engine)
    torch.testing.assert_close(torch.rand(4), expected_next)

    # The first real step reuses the precompiled state and clears the provisional flag.
    mb = gt._engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    tokens = float(mb["response_mask"].sum())
    tu.assign_non_tensor(mb, batch_num_tokens=tokens)
    with gt._patch_fused_logprobs():
        out = engine._graph_trainer_forward_backward(mb.copy(), [mb], None, _loss_fn(), tokens)
    assert len(out["loss"]) == 1
    assert getattr(engine, train_step_precompile.PROVISIONAL_ATTR) is False


def test_real_step_with_other_loss_constants_retraces_instead_of_failing():
    engine, _, _ = _precompile_engine()
    assert precompile_engine_train_step(engine, _loss_fn(), non_tensor=_non_tensor())["status"] == "ok"
    first_spec = engine._graph_policy_loss_spec
    mb = gt._engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    tu.assign_non_tensor(mb, temperature=0.7)
    tokens = float(mb["response_mask"].sum())
    tu.assign_non_tensor(mb, batch_num_tokens=tokens)
    with gt._patch_fused_logprobs():
        engine._graph_trainer_forward_backward(mb.copy(), [mb], None, _loss_fn(), tokens)
    assert engine._graph_policy_loss_spec.temperature == 0.7 != first_spec.temperature
    # Not provisional any more: a later change is the usual hard error.
    tu.assign_non_tensor(mb, temperature=0.5)
    with pytest.raises(RuntimeError, match="traced the policy loss"), gt._patch_fused_logprobs():
        engine._graph_trainer_forward_backward(mb.copy(), [mb], None, _loss_fn(), tokens)


def test_real_step_with_another_padded_length_retraces():
    engine, _, _ = _precompile_engine()
    assert precompile_engine_train_step(engine, _loss_fn(), non_tensor=_non_tensor())["status"] == "ok"
    longer = gt._graph_engine(gt._TinyLM(), 64)
    engine.prepare_model_inputs = longer.prepare_model_inputs
    mb = gt._engine_micro_batch([3, 5, 2], [4, 6, 5], 1)
    tokens = float(mb["response_mask"].sum())
    tu.assign_non_tensor(mb, batch_num_tokens=tokens)
    with gt._patch_fused_logprobs():
        engine._graph_trainer_forward_backward(mb.copy(), [mb], None, _loss_fn(), tokens)
    assert engine._graph_trainer_traced_len == 64
    assert getattr(engine, train_step_precompile.PROVISIONAL_ATTR) is False


def test_failed_precompile_restores_the_untraced_state():
    engine, model, _ = _precompile_engine()
    engine.trainer.forward_backward_step = mock.Mock(side_effect=RuntimeError("compile OOM"))
    result = precompile_engine_train_step(engine, _loss_fn(), non_tensor=_non_tensor())
    assert result["status"] == "failed" and "compile OOM" in result["error"]
    assert engine._graph_policy_loss_spec is None and engine._graph_trainer_traced_len is None
    assert engine.trainer.loss_fn is None
    assert not getattr(engine, train_step_precompile.PROVISIONAL_ATTR, False)
    assert all(p.grad is None for p in model.parameters())
    assert engine.trainer.ntokens_seen == 7


def test_precompile_skips_when_not_applicable():
    engine, _, _ = _precompile_engine()

    def reason(**kwargs):
        loss_fn = kwargs.pop("loss_fn", _loss_fn())
        result = precompile_engine_train_step(engine, loss_fn, non_tensor=_non_tensor(**kwargs))
        assert result["status"] == "skipped", result
        assert not engine.trainer.calls
        return result["reason"]

    engine._graph_trainer_mode = False
    assert "graph_trainer" in reason()
    engine._graph_trainer_mode = True
    options = engine._tpu_options
    engine._tpu_options = gt._options(trainer_mode="graph_trainer", static_micro_batch_len=48)
    assert "precompile_train_step is off" in reason()
    engine._tpu_options = options
    assert "no loss function" in reason(loss_fn=None)
    # A loss the graph step cannot run is found before any synthetic data is built.
    assert "entropy" in reason(loss_fn=_loss_fn(entropy_coeff=0.01))
    assert "do not fit" in reason(use_dynamic_bsz=False, micro_batch_size_per_gpu=48)
    engine.trainer._optimizer_in_graph = True
    assert "in_graph" in reason()
    engine.trainer._optimizer_in_graph = False
    engine._graph_trainer_traced_len = 48
    assert reason() == "graph step already traced"


def _real_engine(use_dynamic_bsz, micro_batch_size_per_gpu=None, padded_len=48):
    """The engine of ``_precompile_engine`` running the real ``TorchTitanTPUEngine.forward_backward_batch``."""
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    engine, model, _ = _precompile_engine(padded_len, seq_bucket_size=16)
    del engine.forward_backward_batch
    tpu = {
        "trainer_mode": "graph_trainer",
        "static_micro_batch_len": padded_len,
        "seq_bucket_size": 16,
        "precompile_train_step": True,
    }
    engine.engine_config = SimpleNamespace(
        engine_kwargs={"tpu": tpu},
        tensor_parallel_size=1,
        context_parallel_size=1,
        use_dynamic_bsz=use_dynamic_bsz,
        max_token_len_per_gpu=padded_len,
        micro_batch_size_per_gpu=micro_batch_size_per_gpu,
        use_fused_kernels=False,
    )
    engine._use_simple_fsdp = True
    engine._bin_slot_len = 0
    assert isinstance(engine, torchtitan_tpu.TorchTitanTPUEngine)
    return engine, model


@pytest.mark.parametrize(("use_dynamic_bsz", "micro_batch_size_per_gpu", "num_seqs"), [(True, None, 1), (False, 2, 2)])
def test_real_forward_backward_batch_path(use_dynamic_bsz, micro_batch_size_per_gpu, num_seqs):
    """The engine's own micro-batching: without dynamic batch sizes a micro-batch holds micro_batch_size_per_gpu
    sequences, which the synthetic batch must match."""
    engine, model = _real_engine(use_dynamic_bsz, micro_batch_size_per_gpu)
    non_tensor = _non_tensor(use_dynamic_bsz=use_dynamic_bsz, micro_batch_size_per_gpu=micro_batch_size_per_gpu)
    with gt._patch_fused_logprobs():
        result = precompile_engine_train_step(engine, _loss_fn(), non_tensor=non_tensor)
    assert result["status"] == "ok", result
    assert result["synthetic_tokens"] <= 48 and result["synthetic_tokens"] % num_seqs == 0
    assert len(engine.trainer.calls) == 1 and engine.trainer.calls[0][0] == (48,)
    assert engine._graph_trainer_traced_len == 48
    assert all(p.grad is None for p in model.parameters())


def _worker(engine, *, is_rollout=False, entropy_coeff=0.0):
    actor_cfg = SimpleNamespace(
        calculate_entropy=False,
        entropy_coeff=entropy_coeff,
        ppo_mini_batch_size=4,
        policy_loss={"rollout_correction": {}},
    )
    actor = SimpleNamespace(
        engine=engine,
        loss_fn=_loss_fn(entropy_coeff=entropy_coeff),
        model_config={"use_remove_padding": True},
        engine_config=SimpleNamespace(
            use_dynamic_bsz=True, max_token_len_per_gpu=48, micro_batch_size_per_gpu=None, use_fused_kernels=False
        ),
    )
    config = SimpleNamespace(actor=actor_cfg, rollout=SimpleNamespace(temperature=1.0, n=8))
    return SimpleNamespace(actor=actor, config=config, _is_rollout=is_rollout, rollout=None)


def test_worker_builds_update_actor_metadata_from_its_config():
    engine, _, _ = _precompile_engine()
    engine.train_mode = lambda **_: contextlib.nullcontext()
    seen = {}

    def fake_precompile(engine_, loss_fn, *, non_tensor):
        seen.update(non_tensor)
        return {"status": "ok"}

    with mock.patch.object(train_step_precompile, "precompile_engine_train_step", side_effect=fake_precompile):
        assert worker_precompile_train_step(_worker(engine))["status"] == "ok"
    assert seen["temperature"] == 1.0 and seen["global_batch_size"] == 32
    assert seen["use_dynamic_bsz"] is True and seen["max_token_len_per_gpu"] == 48
    assert seen["calculate_entropy"] is False and seen["score_centering"] is False

    # A worker with a rollout engine on its devices (hybrid replicas) skips.
    assert worker_precompile_train_step(_worker(engine, is_rollout=True))["status"] == "skipped"
    assert worker_precompile_train_step(SimpleNamespace(actor=None))["status"] == "skipped"


class _FakeWorkerGroup:
    def __init__(self, refs=("ref0", "ref1")):
        self._dispatch_info = {}
        self.queries = []
        self.calls = 0
        self._refs = list(refs)

    def _query_dispatch_info(self, mesh_name):
        self.queries.append(mesh_name)
        return [mesh_name]

    def precompile_train_step(self):
        self.calls += 1
        return self._refs


def test_start_is_non_blocking_once_and_collect_logs_the_results():
    wg = _FakeWorkerGroup()
    manager = SimpleNamespace(actor_wg=wg)
    assert start_precompile(manager) is True
    assert wg.calls == 1
    # The actor dispatch info is cached before the workers start compiling (_balance_batch reads it on step 1).
    assert wg._dispatch_info == {"actor": ["actor"]}
    assert start_precompile(manager) is False and wg.calls == 1  # only once per manager
    fake_ray = SimpleNamespace(get=mock.Mock(return_value=[{"status": "ok"}, {"status": "ok"}]))
    with mock.patch.dict("sys.modules", {"ray": fake_ray}):
        assert asyncio.run(collect_precompile(manager)) == [{"status": "ok"}, {"status": "ok"}]
        assert asyncio.run(collect_precompile(manager)) is None  # nothing pending
    fake_ray.get.assert_called_once_with(["ref0", "ref1"])


def test_start_skips_without_the_worker_method_and_collect_tolerates_worker_errors():
    assert start_precompile(SimpleNamespace(actor_wg=None)) is False
    assert start_precompile(SimpleNamespace(actor_wg=SimpleNamespace())) is False
    manager = SimpleNamespace(actor_wg=_FakeWorkerGroup())
    start_precompile(manager)
    fake_ray = SimpleNamespace(get=mock.Mock(side_effect=RuntimeError("worker died")))
    with mock.patch.dict("sys.modules", {"ray": fake_ray}):
        assert asyncio.run(collect_precompile(manager)) is None  # logs, does not raise


def test_tpu_weight_sync_starts_the_precompile_after_the_sync(monkeypatch):
    import verl.checkpoint_engine.base as ckpt_base
    from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine

    events = []

    async def fake_update(manager, global_steps=None):
        events.append(("sync", global_steps))
        return {"timing_s/tpu-sync/total_sync": 1.0}

    async def fake_collect(manager):
        events.append("collect")

    monkeypatch.setattr(tpu_sync_checkpoint_engine, "update_tpu_sync_weights", fake_update)
    monkeypatch.setattr(train_step_precompile, "collect_precompile", fake_collect)
    monkeypatch.setattr(train_step_precompile, "start_precompile", lambda manager: events.append("start"))
    manager = SimpleNamespace(backend="tpu-sync")
    result = ckpt_base.CheckpointEngineManager.update_weights(manager, global_steps=0)
    assert result == {"timing_s/tpu-sync/total_sync": 1.0}
    assert events == ["collect", ("sync", 0), "start"]


def test_options_validate_precompile_train_step():
    assert gt._options(trainer_mode="graph_trainer", precompile_train_step=True).precompile_train_step is True
    with pytest.raises(ValueError, match="precompile_train_step needs trainer_mode=graph_trainer"):
        gt._options(precompile_train_step=True)
    engine_config = SimpleNamespace(forward_only=False, param_offload=False)
    for tpu, kernels, match in (
        ({}, frozenset({"kda"}), "needs kda_document_budget"),
        ({"graph_trainer": {"optimizer_compile": "in_graph"}}, frozenset(), "optimizer_compile='in_graph'"),
    ):
        options = gt._options(
            trainer_mode="graph_trainer", static_micro_batch_len=8192, precompile_train_step=True, **tpu
        )
        with pytest.raises(ValueError, match=match):
            graph_trainer_mode.validate_graph_trainer_engine(
                engine_config, options, use_simple_fsdp=True, tpu_kernels=kernels
            )


def test_worker_method_is_registered_non_blocking():
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers import engine_workers

    train_step_precompile.register_worker_method()
    method = engine_workers.ActorRolloutRefWorker.precompile_train_step
    attrs = getattr(method, MAGIC_ATTR)
    assert attrs["blocking"] is False
