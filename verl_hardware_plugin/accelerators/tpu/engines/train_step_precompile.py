# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in trainer precompile: trace and compile the graph-trainer step while the first rollout runs.

With ``engine_kwargs.tpu.trainer_mode=graph_trainer`` the first ``update_actor`` traces the joint forward/backward
step and compiles it with XLA (~400 s for Qwen3.5-35B-A3B at 128K tokens per micro-batch). With standalone rollout
replicas (``separate_async``) the trainer is idle from the initial weight sync until its first rollout batch, so
``engine_kwargs.tpu.precompile_train_step=True`` runs that step once on a synthetic micro-batch in that window:

* driver: right after the first weight sync to the standalone rollout (``CheckpointEngineManager.update_weights``
  with a TPU backend, see ``weight_sync_backends``), ``start_precompile`` calls ``precompile_train_step`` on every
  trainer worker without waiting. Each worker is a Ray actor that runs its calls in order, so the trainer's next call
  (e.g. the first ``update_actor``) starts after the precompile; the next weight sync collects and logs the results.
  No trainer class is patched.
* worker: ``precompile_engine_train_step`` runs the engine's own ``forward_backward_batch`` (same micro-batching,
  static padding, bin-slot permutation, policy-loss spec) on synthetic data. Running it, not only tracing it, is what
  compiles the executable: ``StaticCompiler`` lowers the graph at trace time but PJRT compiles it on its first
  execution. Gradients are dropped and the module buffers, the trainer's per-step outputs and the CPU / TPU RNG
  states are restored. No optimizer step runs, so parameters and optimizer state are untouched, and no metrics leave
  the worker. A worker that also hosts a rollout engine (hybrid replicas on the trainer chips) skips it.

If the real step then needs a different graph (other padded length or loss constants), the provisional trace is
dropped and the real step traces as without this option. A failed precompile restores the untraced state.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl_hardware_plugin.accelerators.tpu import device_api
from verl_hardware_plugin.accelerators.tpu.profilers import timeline

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

PROVISIONAL_ATTR = "_graph_precompile_provisional"
_UNTRACED_STATE_ATTR = "_graph_precompile_untraced_state"
_MISSING = object()

# Engine / trainer attributes pinned by the first traced step: restored when the precompile fails or its trace is
# dropped.
_ENGINE_PINNED_ATTRS = ("_graph_policy_loss_spec", "_graph_trainer_traced_len", "_kda_pinned_budget")
_TRAINER_PINNED_ATTRS = ("loss_fn", "_traced_step", "_graph_runner", "active_slot_len")
# Per-step outputs the synthetic step writes: always restored after it.
_ENGINE_STEP_OUTPUT_ATTRS = ("_last_micro_batch_metrics",)
_TRAINER_STEP_OUTPUT_ATTRS = ("ntokens_seen", "last_loss_metrics", "_ep_drop_stats_L2")

# Driver-side state on the CheckpointEngineManager.
_STARTED_ATTR = "_tpu_precompile_started"
_REFS_ATTR = "_tpu_precompile_refs"
_T0_ATTR = "_tpu_precompile_t0"


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
    *,
    static_len: int,
    num_seqs: int = 1,
    row_slot_len: int = 0,
    bin_slot_len: int = 0,
    max_seq_len: int = 0,
    target: int = 256,
) -> Optional[tuple[int, int]]:
    """``(prompt_len, response_len)`` of each of ``num_seqs`` sequences that fit one micro-batch, or ``None``.

    ``num_seqs`` sequences must fit ``static_len`` together; each one must fit a GatedDeltaNet row slot, a bin slot and
    the model length.
    """
    limits = [target, int(static_len) // max(1, int(num_seqs))]
    limits += [int(x) for x in (row_slot_len, bin_slot_len, max_seq_len) if x and int(x) > 0]
    total = min(limits)
    if total < 2:
        return None
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


def _save_attrs(owner: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(owner, name, _MISSING) for name in names} if owner is not None else {}


def _restore_attrs(owner: Any, saved: dict[str, Any]) -> None:
    if owner is None:
        return
    for name, value in saved.items():
        if value is _MISSING:
            if name in vars(owner):
                delattr(owner, name)
        else:
            setattr(owner, name, value)


def _save_pinned_state(engine: Any) -> dict[str, Any]:
    trainer = getattr(engine, "trainer", None)
    slot_cache = getattr(trainer, "_slot_cache", None)
    return {
        "engine": _save_attrs(engine, _ENGINE_PINNED_ATTRS),
        "trainer": _save_attrs(trainer, _TRAINER_PINNED_ATTRS),
        "slot_cache": dict(slot_cache) if isinstance(slot_cache, dict) else None,
    }


def _restore_pinned_state(engine: Any, state: dict[str, Any]) -> None:
    trainer = getattr(engine, "trainer", None)
    _restore_attrs(engine, state["engine"])
    _restore_attrs(trainer, state["trainer"])
    cache = getattr(trainer, "_slot_cache", None)
    if state["slot_cache"] is not None and isinstance(cache, dict):
        cache.clear()
        cache.update(state["slot_cache"])
    # The dropped trace holds the compiled PJRT executable; graph modules keep reference cycles, so free it now
    # rather than while the next step compiles its own.
    gc.collect()


def reset_provisional_trace(engine: Any, reason: str) -> bool:
    """Drops a precompile-only trace so the next step traces from scratch; True if there was one to drop."""
    if not getattr(engine, PROVISIONAL_ATTR, False):
        return False
    logger.warning("trainer precompile: the real step needs another graph (%s); retracing", reason)
    state = getattr(engine, _UNTRACED_STATE_ATTR, None)
    setattr(engine, PROVISIONAL_ATTR, False)
    if state is not None:
        setattr(engine, _UNTRACED_STATE_ATTR, None)
        _restore_pinned_state(engine, state)
    return True


def _save_rng_state() -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CPU and (when the TPU runtime is up) TPU default-generator states: a traced step that draws random numbers
    advances the TPU generator (``StaticCompiler`` threads its state through the executable)."""
    tpu_state = device_api.get_rng_state() if device_api.is_runtime_initialized() else None
    return torch.get_rng_state(), tpu_state


def _restore_rng_state(state: tuple[torch.Tensor, Optional[torch.Tensor]]) -> None:
    cpu_state, tpu_state = state
    torch.set_rng_state(cpu_state)
    if tpu_state is not None:
        device_api.set_rng_state(tpu_state)


def _vocab_size(engine: Any) -> int:
    try:
        from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import get_text_config

        return int(getattr(get_text_config(engine.model_config.hf_config), "vocab_size", 1000) or 1000)
    except Exception:  # noqa: BLE001 - only picks the synthetic token range
        return 1000


def _skip_reason(engine: Any, loss_function: Optional[Callable], non_tensor: dict[str, Any]) -> Optional[str]:
    """Why ``precompile_engine_train_step`` does nothing for ``engine``, or ``None``."""
    from verl_hardware_plugin.accelerators.tpu.engines import graph_trainer_mode

    if not getattr(engine, "_graph_trainer_mode", False):
        return "trainer_mode is not graph_trainer"
    options = getattr(engine, "_tpu_options", None)
    if not getattr(options, "precompile_train_step", False):
        return "engine_kwargs.tpu.precompile_train_step is off"
    if loss_function is None:
        return "no loss function installed"
    if getattr(engine, "_graph_policy_loss_spec", None) is not None or getattr(
        engine, "_graph_trainer_traced_len", None
    ):
        return "graph step already traced"
    if getattr(getattr(engine, "trainer", None), "_optimizer_in_graph", False):
        # TpuGraphTrainer's in-graph AdamW steps the parameters inside forward_backward_step.
        return "graph_trainer.optimizer_compile='in_graph' updates the parameters in the step"
    if "kda" in getattr(engine, "_tpu_kernels", ()) and not getattr(options, "kda_document_budget", None):
        # The first step pins max_num_segments from its document count: a synthetic count would not fit real steps.
        return "needs engine_kwargs.tpu.kda_document_budget with Tokamax KDA"
    try:
        graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
            loss_function,
            temperature=float(non_tensor["temperature"]),
            calculate_entropy=bool(non_tensor.get("calculate_entropy", False)),
            has_rollout_is_weights=False,
        )
    except ValueError as exc:
        return f"the real step cannot run in graph_trainer mode: {exc}"
    return None


def precompile_engine_train_step(engine: Any, loss_function: Optional[Callable], *, non_tensor: dict[str, Any]) -> dict:
    """Runs one graph-trainer forward/backward on synthetic data and discards its effects (see module docstring).

    ``non_tensor`` holds the micro-batch metadata of a real ``update_actor`` call (``temperature``,
    ``use_dynamic_bsz``, ``micro_batch_size_per_gpu``, ...). Every trainer rank must call this together (the step runs
    collectives). Returns a status dict.
    """
    t0 = time.perf_counter()
    reason = _skip_reason(engine, loss_function, non_tensor)
    if reason is not None:
        return {"status": "skipped", "reason": reason}
    options = engine._tpu_options
    # Without dynamic batch sizes every micro-batch holds exactly micro_batch_size_per_gpu sequences.
    num_seqs = (
        1 if non_tensor.get("use_dynamic_bsz", True) else max(1, int(non_tensor.get("micro_batch_size_per_gpu") or 1))
    )
    lengths = synthetic_sequence_lengths(
        static_len=int(options.static_micro_batch_len or 0),
        num_seqs=num_seqs,
        row_slot_len=int(getattr(engine, "_gdn_row_slot_len", 0) or 0),
        bin_slot_len=int(getattr(engine, "_bin_slot_len", 0) or 0),
        max_seq_len=int(non_tensor.get("max_token_len_per_gpu") or 0),
    )
    if lengths is None:
        return {"status": "skipped", "reason": f"{num_seqs} synthetic sequences do not fit static_micro_batch_len"}
    prompt_len, response_len = lengths
    data = build_synthetic_train_batch(
        num_seqs=num_seqs, prompt_len=prompt_len, response_len=response_len, vocab_size=_vocab_size(engine)
    )
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(data, **non_tensor)

    trainer = getattr(engine, "trainer", None)
    untraced = _save_pinned_state(engine)
    engine_outputs = _save_attrs(engine, _ENGINE_STEP_OUTPUT_ATTRS)
    trainer_outputs = _save_attrs(trainer, _TRAINER_STEP_OUTPUT_ATTRS)
    buffers = _snapshot_buffers(getattr(engine, "module", None))
    rng_state = _save_rng_state()
    on_tpu = device_api.accelerator_is_tpu()
    result: dict[str, Any]
    with timeline.span("precompile_train_step", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()):
        try:
            engine.optimizer_zero_grad()
            engine.forward_backward_batch(data, loss_function, forward_only=False)
            # Surface a failed (asynchronously dispatched) execution here, while the untraced state can be restored.
            device_api.tpu_synchronize(wait=True, strict=on_tpu)
            result = {"status": "ok", "padded_len": getattr(engine, "_graph_trainer_traced_len", None)}
        except Exception as exc:  # noqa: BLE001 - the real step must still run; it then traces from scratch.
            logger.warning("trainer precompile failed; the first step will trace as usual: %r", exc, exc_info=True)
            _restore_pinned_state(engine, untraced)
            result = {"status": "failed", "error": repr(exc)}
        finally:
            engine.optimizer_zero_grad()
            _restore_buffers(buffers)
            _restore_attrs(engine, engine_outputs)
            _restore_attrs(trainer, trainer_outputs)
            _restore_rng_state(rng_state)
            device_api.tpu_synchronize(wait=True)
    if result["status"] == "ok":
        setattr(engine, _UNTRACED_STATE_ATTR, untraced)
        setattr(engine, PROVISIONAL_ATTR, True)
    result["seconds"] = round(time.perf_counter() - t0, 3)
    result["synthetic_tokens"] = num_seqs * (prompt_len + response_len)
    logger.warning("trainer precompile: %s", result)
    return result


def worker_precompile_train_step(worker: Any) -> dict[str, Any]:
    """``ActorRolloutRefWorker.precompile_train_step``: the actor engine's precompile, with ``update_actor``'s
    micro-batch metadata built from the worker's own ``actor_rollout_ref`` config."""
    if getattr(worker, "_is_rollout", False) or getattr(worker, "rollout", None) is not None:
        return {"status": "skipped", "reason": "the worker hosts a rollout engine on the trainer devices"}
    actor = getattr(worker, "actor", None)
    engine = getattr(actor, "engine", None)
    if engine is None:
        return {"status": "skipped", "reason": "worker has no actor engine"}
    actor_cfg, rollout_cfg = worker.config.actor, worker.config.rollout
    model_config, engine_config = actor.model_config, actor.engine_config
    rollout_correction = actor_cfg.policy_loss.get("rollout_correction", {}) or {}
    non_tensor = {
        # As TrainingWorker.train_batch's defaults.
        "use_remove_padding": model_config.get("use_remove_padding", False),
        "use_dynamic_bsz": engine_config.use_dynamic_bsz,
        "max_token_len_per_gpu": engine_config.max_token_len_per_gpu,
        "micro_batch_size_per_gpu": engine_config.micro_batch_size_per_gpu,
        "use_fused_kernels": engine_config.use_fused_kernels,
        # As PPOTrainer._update_actor's extra_info.
        "calculate_entropy": bool(actor_cfg.calculate_entropy or actor_cfg.entropy_coeff != 0.0),
        "distillation_use_topk": False,
        "distillation_only": False,
        "score_centering": bool(rollout_correction.get("score_centering", False)),
        "global_batch_size": int(actor_cfg.ppo_mini_batch_size) * int(rollout_cfg.n),
        "temperature": float(rollout_cfg.temperature),
    }
    with engine.train_mode(disable_auto_offload=False):
        return precompile_engine_train_step(engine, actor.loss_fn, non_tensor=non_tensor)


def register_worker_method() -> None:
    """Adds ``precompile_train_step`` (ONE_TO_ALL, non-blocking) to verl's ``ActorRolloutRefWorker``.

    Registered in every process, so the driver's worker group and the workers agree on the worker's methods; it does
    nothing until called, and the worker skips it unless ``engine_kwargs.tpu.precompile_train_step`` is set.
    """
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers import engine_workers

    cls = engine_workers.ActorRolloutRefWorker
    if "precompile_train_step" in vars(cls):
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def precompile_train_step(self) -> dict[str, Any]:
        return worker_precompile_train_step(self)

    cls.precompile_train_step = precompile_train_step


def _cache_actor_dispatch_info(worker_group: Any) -> None:
    """Queries the actor mesh's dispatch info now, while the workers are idle.

    ``PPOTrainer._balance_batch`` queries it lazily on step 1; queued behind a running precompile, that query would
    block the driver (untimed) instead of the first ``update_actor``.
    """
    dispatch_info = getattr(worker_group, "_dispatch_info", None)
    query = getattr(worker_group, "_query_dispatch_info", None)
    if isinstance(dispatch_info, dict) and callable(query) and "actor" not in dispatch_info:
        dispatch_info["actor"] = query("actor")


def start_precompile(manager: Any) -> bool:
    """Starts ``precompile_train_step`` on the trainer workers once per ``manager`` without waiting; True if started.

    Called by the TPU weight-sync backends after a sync (``weight_sync_backends``), i.e. right after the initial sync
    to the standalone rollout, when the trainer is idle until its first batch.
    """
    if getattr(manager, _STARTED_ATTR, False):
        return False
    setattr(manager, _STARTED_ATTR, True)
    worker_group = getattr(manager, "actor_wg", None)
    if worker_group is None or not hasattr(worker_group, "precompile_train_step"):
        return False
    try:
        _cache_actor_dispatch_info(worker_group)
        setattr(manager, _T0_ATTR, time.perf_counter())
        setattr(manager, _REFS_ATTR, worker_group.precompile_train_step())
    except Exception as exc:  # noqa: BLE001 - never block training on the optional precompile.
        logger.warning("trainer precompile could not be started: %r", exc)
        setattr(manager, _REFS_ATTR, None)
        return False
    return True


async def collect_precompile(manager: Any) -> Optional[list[Any]]:
    """Waits for a started precompile (once) and logs its outcome; worker errors only log. Returns the results."""
    refs = getattr(manager, _REFS_ATTR, None)
    if refs is None:
        return None
    setattr(manager, _REFS_ATTR, None)
    t_wait = time.perf_counter()
    try:
        import ray

        results = await asyncio.to_thread(ray.get, refs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("trainer precompile failed on a worker: %r", exc)
        return None
    results = results if isinstance(results, list) else [results]
    first = results[0] if results else None
    if isinstance(first, dict) and first.get("status") != "skipped":
        now = time.perf_counter()
        logger.warning(
            "trainer precompile done (collected after %.1fs, %.1fs after it started); rank0=%s",
            now - t_wait,
            now - getattr(manager, _T0_ATTR, now),
            first,
        )
    return results
