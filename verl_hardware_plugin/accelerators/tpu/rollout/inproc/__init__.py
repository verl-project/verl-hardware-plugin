# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""In-process (same-process) colocation of the vLLM rollout with the trainer on TPU. Opt-in, default off.

A TPU TensorCore can be opened by one process only, there is no cross-process device-memory sharing, and
vllm-torchtpu has no sleep mode, so verl's colocation with process separation cannot share TPU chips. With
``VERL_TPU_ROLLOUT_INPROC=1`` each trainer rank (one process per TensorCore) also builds one TP=1 vLLM engine, with
vLLM's synchronous ``LLMEngine`` and ``InprocClient`` on the trainer's own PJRT client; both stay resident, and the
engine's KV pool is set explicitly (``rollout.engine_kwargs.vllm.kv_cache_memory_bytes``).

* :class:`~.rollout.TPUInProcRollout` is verl's ``("vllm", "async")`` rollout, built by
  ``ActorRolloutRefWorker.init_model`` on every trainer rank (``trainer.v1.trainer_mode=sync``,
  ``actor_rollout_ref.hybrid_engine=True``). It serves requests through an in-process subclass of the plugin's
  ``TPUvLLMHttpServer``: verl's server code runs verbatim on an ``AsyncLLM``-shaped adapter (:mod:`.engine`).
* The replica's server is the trainer worker actor itself (``inproc://<rank>``); ``generate`` /
  ``abort_all_requests`` / ``resume_generation`` / ``inproc_server_info`` are added to ``ActorRolloutRefWorker`` as
  direct rollout methods (:func:`install_worker_methods`; verl itself is not patched).
* Weight sync: verl's ``naive`` checkpoint-engine path calls ``update_weights`` with the trainer's export, which is
  installed in place through the tpu-sync worker extension's own install and verified (:mod:`.weight_sync`).
* The engine's process-global side effects are contained and its compiled programs kept valid (:mod:`.boundary`,
  :mod:`.compile_guard`); a step whose rollout lost requests fails before ``update_actor`` on every rank
  (:func:`step_guard`, :mod:`.sampler`).

Scope: rollout TP=1, a single host, verl's sync trainer mode, the model families of the tested-layout table
(:data:`.compat.TESTED_LAYOUTS`). Anything else is refused at build.
"""

from __future__ import annotations

import functools
import inspect
import os
import time
from typing import Any, Callable

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    INPROC_ENV,
    all_ranks_ok,
    check_switches,
    inproc_enabled,
    log,
    logger,
    warn,
)

__all__ = ["INPROC_ENV", "check_switches", "inproc_enabled", "logger", "register", "step_guard"]

ROLLOUT_FQN = "verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout.TPUInProcRollout"
_WORKER_METHODS = ("generate", "abort_all_requests", "resume_generation", "inproc_server_info")
_MARK = "_inproc_method"


def _inproc_rollout(worker: Any):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    r = getattr(worker, "rollout", None)
    return r if isinstance(r, TPUInProcRollout) else None


def step_guard(worker: Any) -> None:
    """At the entry of update_actor on every trainer rank, before anything trains: this rank's rollout since the last
    weight sync is healthy (:meth:`TPUInProcRollout.step_health`), agreed across the trainer's ranks (one gloo
    all-reduce per step), so a failure on one rank makes every rank raise 'STEP FAILED before update_actor' instead of
    training on a partial batch (and no rank is left alone in the FSDP collectives)."""
    r = _inproc_rollout(worker)
    if r is None:
        return
    local = r.step_health()
    if all_ranks_ok(local is None):
        return
    line = (
        f"STEP FAILED before update_actor rank {r.rank}: "
        + (local or "another rank's rollout failed or aborted requests in this step")
        + "; update_actor refused on every rank"
    )
    warn(line)
    raise RuntimeError(line)


def trainer_entry(worker: Any, kind: str) -> None:
    """At the trainer's first entry after a sync (compute_log_prob / update_actor; the rollout is over, the device
    idle): read back that sync's staleness probe. Never raises here (these entries are collective): a stale verdict is
    logged at once, refused by the step guard and raised at the next sync."""
    r = _inproc_rollout(worker)
    if r is not None and r.stale is not None and r.stale.pending is not None:
        r.stale.resolve(kind)


def _wrap_worker_method(orig_fn: Callable, kind: str) -> Callable:
    """A synchronous ActorRolloutRefWorker method with the device-thread record and the eager guard at every entry;
    compute_log_prob / update_actor read the last sync's staleness probe back first, and update_actor runs the step
    guard before anything trains. The trainer engine's optimizer-step counter is installed once init_model built it."""
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import DEVICE_THREADS, EAGER_GUARD
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.weight_sync import OPT_STEPS

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        worker = args[0] if args else None
        DEVICE_THREADS.record(kind)
        EAGER_GUARD.check(kind)
        if kind == "update_actor":
            OPT_STEPS.install(worker)
        if kind != "init_model":
            trainer_entry(worker, kind)
        if kind == "update_actor":
            step_guard(worker)
        out = orig_fn(*args, **kwargs)
        if kind == "init_model":
            OPT_STEPS.install(worker)
        return out

    setattr(wrapper, _MARK, True)
    return wrapper


def install_worker_methods() -> list[str]:
    """The trainer worker serves the replica's requests: adds async ``generate`` / ``abort_all_requests`` /
    ``resume_generation`` / ``inproc_server_info`` to ``ActorRolloutRefWorker`` as DIRECT_ROLLOUT_METHOD methods
    (verl binds those on the fused worker class without the role prefix) and wraps its synchronous ``init_model`` /
    ``update_actor`` / ``compute_log_prob`` (refused when one is absent or a coroutine). Idempotent. Returns the
    wrapped methods."""
    from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch, Execute
    from verl.workers.engine_workers import ActorRolloutRefWorker as W
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compat import (
        WRAPPED_WORKER_METHODS,
        worker_methods_problem,
    )

    attrs = {"dispatch_mode": Dispatch.DIRECT_ROLLOUT_METHOD, "execute_mode": Execute.ALL, "blocking": True}

    async def generate(
        self,
        prompt_ids,
        sampling_params,
        request_id,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        priority=0,
        kv_transfer_params=None,
        session_id=None,
    ):
        return await self.rollout.generate(
            prompt_ids,
            sampling_params,
            request_id,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
            kv_transfer_params=kv_transfer_params,
            session_id=session_id,
        )

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ):
        return await self.rollout.abort_all_requests(
            reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
        )

    async def resume_generation(self):
        return await self.rollout.resume_generation()

    async def inproc_server_info(self):
        r = _inproc_rollout(self)
        if r is None:
            return {"engine_ready": False, "rollout": type(getattr(self, "rollout", None)).__name__}
        return r.server_info()

    unwrapped = [n for n in WRAPPED_WORKER_METHODS if not getattr(W.__dict__.get(n), _MARK, False)]
    problem = worker_methods_problem(W) if unwrapped else None
    if problem:
        raise RuntimeError(f"cannot wrap the trainer worker methods the in-process rollout needs: {problem}")
    for fn in (generate, abort_all_requests, resume_generation, inproc_server_info):
        existing = W.__dict__.get(fn.__name__)
        if existing is not None and not getattr(existing, _MARK, False):
            raise RuntimeError(f"ActorRolloutRefWorker already defines {fn.__name__}: refusing to replace it")
        fn.__qualname__ = f"ActorRolloutRefWorker.{fn.__name__}"
        setattr(fn, MAGIC_ATTR, dict(attrs))
        setattr(fn, _MARK, True)
        setattr(W, fn.__name__, fn)
    for name in unwrapped:
        setattr(W, name, _wrap_worker_method(W.__dict__[name], name))
    # verl's update_weights (a coroutine): the start of the trainer's export, recorded at its entry so the sync time
    # includes get_per_tensor_param (the state dict and its all-gathers)
    orig_uw = W.__dict__.get("update_weights")
    if orig_uw is not None and inspect.iscoroutinefunction(orig_uw) and not getattr(orig_uw, _MARK, False):

        @functools.wraps(orig_uw)
        async def update_weights(self, *args, **kwargs):
            r = _inproc_rollout(self)
            if r is not None:
                r._pending_export_t0 = time.time()
            return await orig_uw(self, *args, **kwargs)

        setattr(update_weights, _MARK, True)
        W.update_weights = update_weights
    return list(WRAPPED_WORKER_METHODS)


def register_rollout_class() -> None:
    from verl.workers.rollout import base as rollout_base

    rollout_base._ROLLOUT_REGISTRY[("vllm", "async")] = ROLLOUT_FQN


def register() -> None:
    """Called by the plugin's TPU registration with ``VERL_TPU_ROLLOUT_INPROC=1``, in every process that imports verl.
    Fails closed."""
    check_switches()
    register_rollout_class()
    wrapped = install_worker_methods()
    log(
        f"registered in pid {os.getpid()}: rollout ('vllm','async') -> {ROLLOUT_FQN}; replica loader -> "
        "TPUInProcReplica; "
        f"worker methods {list(_WORKER_METHODS)}; wrapped {wrapped}"
    )
