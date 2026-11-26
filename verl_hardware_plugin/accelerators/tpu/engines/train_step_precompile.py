# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in trainer precompile: trace and compile the graph-trainer step while the first rollout runs.

With ``engine_kwargs.tpu.trainer_mode=graph_trainer`` the first ``update_actor`` traces the joint forward/backward
step and compiles it with XLA (~400 s for Qwen3.5-35B-A3B at 128K tokens per micro-batch). In ``separate_async`` the
trainer is idle while it waits for its first rollout batch, so ``VERL_TPU_PRECOMPILE_TRAIN_STEP=1`` runs that step
once on a synthetic micro-batch in that window:

* driver: ``PPOTrainerSeparateAsync.on_train_begin`` fires ``precompile_train_step`` on every trainer worker without
  blocking; ``PPOTrainer._update_actor`` waits for it (normally long finished) before the first real update.
* worker: ``precompile_engine_train_step`` runs the engine's own ``forward_backward_batch`` (same micro-batching,
  static padding, bin-slot permutation, policy-loss spec) on synthetic data, then drops the gradients and restores
  the module buffers. No optimizer step runs, so parameters and optimizer state are untouched, and no metrics leave
  the worker.

If the real step then needs a different graph (other padded length or loss constants), the provisional trace is
dropped and the real step traces as without this option. A failed precompile restores the untraced state.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

ENV_VAR = "VERL_TPU_PRECOMPILE_TRAIN_STEP"
PROVISIONAL_ATTR = "_graph_precompile_provisional"
_MISSING = object()

# Engine / trainer attributes pinned by the first traced step.
_ENGINE_PINNED_ATTRS = ("_graph_policy_loss_spec", "_graph_trainer_traced_len", "_kda_pinned_budget")
_TRAINER_PINNED_ATTRS = ("loss_fn", "_traced_step", "_graph_runner", "active_slot_len")


def is_enabled() -> bool:
    return os.getenv(ENV_VAR, "0").strip().lower() in ("1", "true", "yes")


def build_synthetic_train_batch(
    *,
    num_seqs: int = 1,
    prompt_len: int = 64,
    response_len: int = 192,
    vocab_size: int = 1000,
    seed: int = 0,
) -> TensorDict:
    """A V1 no-padding mini-batch (``left_right_2_no_padding`` layout) of ``num_seqs`` identical-length sequences.

    ``input_ids`` / ``position_ids`` / ``prompts`` / ``responses`` are jagged nested tensors; the response-level
    fields (``response_mask``, ``loss_mask``, ``old_log_probs``, ``rollout_log_probs``, ``advantages``) are
    ``[num_seqs, response_len]``. Every response token is valid, so the loss has a non-zero token count.
    """
    if num_seqs <= 0 or prompt_len <= 0 or response_len <= 0:
        raise ValueError(f"need positive sizes, got {num_seqs=} {prompt_len=} {response_len=}")
    gen = torch.Generator().manual_seed(seed)
    seq_len = prompt_len + response_len
    high = max(2, min(int(vocab_size), 32000))
    tokens = torch.randint(0, high, (num_seqs, seq_len), generator=gen, dtype=torch.int64)

    def jagged(values: torch.Tensor, length: int) -> torch.Tensor:
        offsets = torch.arange(0, (num_seqs + 1) * length, length, dtype=torch.int64)
        return torch.nested.nested_tensor_from_jagged(values.reshape(-1).contiguous(), offsets)

    positions = torch.arange(seq_len, dtype=torch.int64).expand(num_seqs, seq_len)
    response_mask = torch.ones(num_seqs, response_len, dtype=torch.int64)
    old_log_probs = -torch.rand(num_seqs, response_len, generator=gen, dtype=torch.float32) * 0.5 - 0.01
    advantages = torch.randn(num_seqs, 1, generator=gen, dtype=torch.float32).expand(num_seqs, response_len)
    fields = {
        "input_ids": jagged(tokens, seq_len),
        "position_ids": jagged(positions, seq_len),
        "prompts": jagged(tokens[:, :prompt_len], prompt_len),
        "responses": jagged(tokens[:, prompt_len:], response_len),
        "response_mask": response_mask,
        "loss_mask": response_mask.clone(),
        "old_log_probs": old_log_probs,
        "rollout_log_probs": old_log_probs.clone(),
        "advantages": advantages.contiguous(),
    }
    return TensorDict(fields, batch_size=[num_seqs])


def synthetic_sequence_lengths(
    *, static_len: int, bin_slot_len: int = 0, max_seq_len: int = 0, target: int = 256
) -> tuple[int, int]:
    """``(prompt_len, response_len)`` of one sequence that fits a micro-batch, a bin slot and the model length."""
    limits = [target] + [int(x) for x in (static_len, bin_slot_len, max_seq_len) if x and int(x) > 0]
    total = max(4, min(limits))
    prompt_len = max(1, total // 4)
    return prompt_len, total - prompt_len


def _snapshot_buffers(modules: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    saved = []
    for module in modules or ():
        for buf in module.buffers():
            saved.append((buf, buf.detach().clone()))
    return saved


@torch.no_grad()
def _restore_buffers(saved: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for buf, value in saved:
        buf.copy_(value)


def _save_pinned_state(engine: Any) -> dict[str, Any]:
    trainer = getattr(engine, "trainer", None)
    state = {f"engine.{k}": getattr(engine, k, _MISSING) for k in _ENGINE_PINNED_ATTRS}
    state.update({f"trainer.{k}": getattr(trainer, k, _MISSING) for k in _TRAINER_PINNED_ATTRS})
    slot_cache = getattr(trainer, "_slot_cache", None)
    state["trainer._slot_cache"] = dict(slot_cache) if isinstance(slot_cache, dict) else _MISSING
    return state


def _restore_pinned_state(engine: Any, state: dict[str, Any]) -> None:
    trainer = getattr(engine, "trainer", None)
    for key, value in state.items():
        owner_name, attr = key.split(".", 1)
        owner = engine if owner_name == "engine" else trainer
        if owner is None:
            continue
        if attr == "_slot_cache" and value is not _MISSING:
            cache = getattr(owner, "_slot_cache", None)
            if isinstance(cache, dict):
                cache.clear()
                cache.update(value)
                continue
        if value is _MISSING:
            if hasattr(owner, attr) and attr in vars(owner):
                delattr(owner, attr)
        else:
            setattr(owner, attr, value)


def reset_provisional_trace(engine: Any, reason: str) -> bool:
    """Drops a precompile-only trace so the next step traces from scratch; True if there was one to drop."""
    if not getattr(engine, PROVISIONAL_ATTR, False):
        return False
    state = getattr(engine, "_graph_precompile_untraced_state", None)
    logger.warning("trainer precompile: the real step needs another graph (%s); retracing", reason)
    if state is not None:
        _restore_pinned_state(engine, state)
    setattr(engine, PROVISIONAL_ATTR, False)
    return True


def _vocab_size(engine: Any) -> int:
    try:
        from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import get_text_config

        return int(getattr(get_text_config(engine.model_config.hf_config), "vocab_size", 1000) or 1000)
    except Exception:  # noqa: BLE001 - only picks the synthetic token range
        return 1000


def precompile_engine_train_step(
    engine: Any,
    loss_function: Optional[Callable],
    *,
    non_tensor: Optional[dict[str, Any]] = None,
    num_seqs: int = 1,
) -> dict[str, Any]:
    """Runs one graph-trainer forward/backward on synthetic data and discards its effects (see module docstring).

    Every trainer rank must call this together (the step runs collectives). Returns a status dict.
    """
    t0 = time.perf_counter()
    if not getattr(engine, "_graph_trainer_mode", False):
        return {"status": "skipped", "reason": "trainer_mode is not graph_trainer"}
    if loss_function is None:
        return {"status": "skipped", "reason": "no loss function installed"}
    if getattr(engine, "_graph_policy_loss_spec", None) is not None or getattr(
        engine, "_graph_trainer_traced_len", None
    ):
        return {"status": "skipped", "reason": "graph step already traced"}
    from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options

    options = get_tpu_engine_options(engine.engine_config)
    static_len = int(getattr(options, "static_micro_batch_len", 0) or 0)
    if static_len <= 0:
        return {"status": "skipped", "reason": "needs engine_kwargs.tpu.static_micro_batch_len (fixed traced shape)"}
    if "kda" in getattr(engine, "_tpu_kernels", ()) and not getattr(options, "kda_document_budget", None):
        # The first step pins max_num_segments from its document count: a synthetic count would not fit real steps.
        return {"status": "skipped", "reason": "needs engine_kwargs.tpu.kda_document_budget with Tokamax KDA"}

    prompt_len, response_len = synthetic_sequence_lengths(
        static_len=static_len,
        bin_slot_len=int(getattr(engine, "_bin_slot_len", 0) or 0),
        max_seq_len=int(getattr(engine.engine_config, "max_token_len_per_gpu", 0) or 0),
    )
    data = build_synthetic_train_batch(
        num_seqs=num_seqs, prompt_len=prompt_len, response_len=response_len, vocab_size=_vocab_size(engine)
    )
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(data, **(non_tensor or {}))

    untraced = _save_pinned_state(engine)
    buffers = _snapshot_buffers(getattr(engine, "module", None))
    last_metrics = getattr(engine, "_last_micro_batch_metrics", _MISSING)
    cpu_rng = torch.get_rng_state()
    result: dict[str, Any]
    try:
        engine.optimizer_zero_grad()
        engine.forward_backward_batch(data, loss_function, forward_only=False)
        result = {"status": "ok", "padded_len": getattr(engine, "_graph_trainer_traced_len", None)}
    except Exception as exc:  # noqa: BLE001 - the real step must still run; it then traces from scratch.
        logger.warning("trainer precompile failed; the first step will trace as usual: %r", exc, exc_info=True)
        _restore_pinned_state(engine, untraced)
        result = {"status": "failed", "error": repr(exc)}
    finally:
        engine.optimizer_zero_grad()
        _restore_buffers(buffers)
        if last_metrics is _MISSING:
            if "_last_micro_batch_metrics" in vars(engine):
                delattr(engine, "_last_micro_batch_metrics")
        else:
            engine._last_micro_batch_metrics = last_metrics
        torch.set_rng_state(cpu_rng)
    if result["status"] == "ok":
        engine._graph_precompile_untraced_state = untraced
        setattr(engine, PROVISIONAL_ATTR, True)
    result["seconds"] = round(time.perf_counter() - t0, 3)
    result["synthetic_tokens"] = num_seqs * (prompt_len + response_len)
    logger.warning("trainer precompile: %s", result)
    return result


def _worker_precompile_train_step(worker: Any, spec: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """``ActorRolloutRefWorker.precompile_train_step`` body: the actor engine's precompile under ``train_mode``."""
    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None)
    if actor is None or engine is None:
        return {"status": "skipped", "reason": "worker has no actor engine"}
    spec = dict(spec or {})
    model_config, engine_config = actor.model_config, actor.engine_config
    non_tensor = {
        # As TrainingWorker.train_batch's default keys.
        "use_remove_padding": model_config.get("use_remove_padding", False),
        "use_dynamic_bsz": engine_config.use_dynamic_bsz,
        "max_token_len_per_gpu": engine_config.max_token_len_per_gpu,
        "micro_batch_size_per_gpu": engine_config.micro_batch_size_per_gpu,
        "use_fused_kernels": engine_config.use_fused_kernels,
        # As PPOTrainer._update_actor's extra_info.
        "calculate_entropy": bool(spec.get("calculate_entropy", False)),
        "distillation_use_topk": False,
        "distillation_only": False,
        "score_centering": bool(spec.get("score_centering", False)),
        "global_batch_size": spec.get("global_batch_size"),
        "temperature": float(spec["temperature"]),
    }
    with engine.train_mode(disable_auto_offload=False):
        return precompile_engine_train_step(engine, actor.loss_fn, non_tensor=non_tensor)


def apply_worker_method() -> None:
    """Adds ``precompile_train_step`` (ONE_TO_ALL, non-blocking) to verl's ``ActorRolloutRefWorker``.

    Always installed (it does nothing unless called), so driver and workers agree on the worker's methods.
    """
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers import engine_workers

    cls = engine_workers.ActorRolloutRefWorker
    if "precompile_train_step" in vars(cls):
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def precompile_train_step(self, spec: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return _worker_precompile_train_step(self, spec)

    cls.precompile_train_step = precompile_train_step


def _precompile_spec(trainer: Any) -> dict[str, Any]:
    cfg = trainer.config
    actor = cfg.actor_rollout_ref.actor
    rollout_correction = actor.policy_loss.get("rollout_correction", {}) or {}
    return {
        "temperature": float(cfg.actor_rollout_ref.rollout.temperature),
        "calculate_entropy": bool(actor.calculate_entropy or actor.entropy_coeff != 0.0),
        "score_centering": bool(rollout_correction.get("score_centering", False)),
        "global_batch_size": int(actor.ppo_mini_batch_size) * int(cfg.actor_rollout_ref.rollout.n),
    }


def fire_precompile(trainer: Any) -> bool:
    """Starts ``precompile_train_step`` on the trainer workers without waiting; True if it was started."""
    if getattr(trainer, "_tpu_precompile_refs", None) is not None or getattr(trainer, "_tpu_precompile_fired", False):
        return False
    trainer._tpu_precompile_fired = True
    if getattr(trainer, "_enable_hybrid_replicas", False):
        logger.warning("%s=1 ignored: hybrid rollout replicas share the trainer devices", ENV_VAR)
        return False
    wg = getattr(trainer, "actor_rollout_wg", None)
    if wg is None or not hasattr(wg, "precompile_train_step"):
        logger.warning("%s=1 ignored: the actor worker group has no precompile_train_step", ENV_VAR)
        return False
    try:
        spec = _precompile_spec(trainer)
        trainer._tpu_precompile_t0 = time.perf_counter()
        trainer._tpu_precompile_refs = wg.precompile_train_step(spec)
    except Exception as exc:  # noqa: BLE001 - never block training on the optional precompile.
        logger.warning("trainer precompile could not be started: %r", exc)
        trainer._tpu_precompile_refs = None
        return False
    logger.warning("trainer precompile started on the trainer workers (spec=%s)", spec)
    return True


def wait_for_precompile(trainer: Any) -> None:
    """Waits for a started precompile (once) and logs its outcome; errors only log."""
    refs = getattr(trainer, "_tpu_precompile_refs", None)
    if refs is None:
        return
    trainer._tpu_precompile_refs = None
    t_wait = time.perf_counter()
    try:
        import ray

        results = ray.get(refs) if isinstance(refs, list) else ray.get(refs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("trainer precompile failed on a worker: %r", exc)
        return
    now = time.perf_counter()
    first = results[0] if isinstance(results, list) and results else results
    logger.warning(
        "trainer precompile done: waited %.1fs before the first update_actor (%.1fs since start); rank0=%s",
        now - t_wait,
        now - getattr(trainer, "_tpu_precompile_t0", now),
        first,
    )


def apply_driver_hooks() -> None:
    """Wraps ``PPOTrainerSeparateAsync.on_train_begin`` and ``PPOTrainer._update_actor`` (no-ops unless enabled)."""
    from verl.trainer.ppo.v1 import trainer_base, trainer_separate_async

    sep_cls = trainer_separate_async.PPOTrainerSeparateAsync
    if not getattr(sep_cls, "_verl_tpu_precompile_patched", False):
        orig_begin = sep_cls.on_train_begin

        def on_train_begin(self, *args: Any, **kwargs: Any) -> Any:
            out = orig_begin(self, *args, **kwargs)
            if is_enabled():
                fire_precompile(self)
            return out

        sep_cls.on_train_begin = on_train_begin
        sep_cls._verl_tpu_precompile_patched = True

    base_cls = trainer_base.PPOTrainer
    if not getattr(base_cls, "_verl_tpu_precompile_patched", False):
        orig_update = base_cls._update_actor

        def _update_actor(self, *args: Any, **kwargs: Any) -> Any:
            wait_for_precompile(self)
            return orig_update(self, *args, **kwargs)

        base_cls._update_actor = _update_actor
        base_cls._verl_tpu_precompile_patched = True


def apply_precompile_hooks() -> None:
    """The worker method always (driver and workers must agree on it); the driver hooks only when enabled."""
    hooks = [apply_worker_method] + ([apply_driver_hooks] if is_enabled() else [])
    for fn in hooks:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - optional feature; keep plugin import working.
            log = logger.warning if is_enabled() else logger.debug
            log("trainer precompile hook %s not applied: %r", fn.__name__, exc)
