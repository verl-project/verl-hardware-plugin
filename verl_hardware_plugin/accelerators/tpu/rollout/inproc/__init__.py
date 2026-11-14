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
* Weight sync is not part of the package yet: ``update_weights`` refuses, so a run builds and serves, and stops
  at its first weight sync, before anything is installed.
* The engine's process-global side effects are contained and its compiled programs kept valid (:mod:`.boundary`,
  :mod:`.compile_guard`).

Scope: rollout TP=1 (other parallel sizes are refused at build). Only single-host runs have been tested; more
hosts are not refused.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    INPROC_ENV,
    check_switches,
    inproc_enabled,
    log,
    logger,
)

__all__ = ["INPROC_ENV", "check_switches", "inproc_enabled", "logger", "register"]

ROLLOUT_FQN = "verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout.TPUInProcRollout"
_WORKER_METHODS = ("generate", "abort_all_requests", "resume_generation", "inproc_server_info")
_MARK = "_inproc_method"


def _inproc_rollout(worker: Any):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    r = getattr(worker, "rollout", None)
    return r if isinstance(r, TPUInProcRollout) else None


def _wrap_worker_method(orig_fn: Callable, kind: str) -> Callable:
    """A synchronous ActorRolloutRefWorker method with the device-thread record and the eager guard at every entry."""
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import DEVICE_THREADS, EAGER_GUARD

    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        DEVICE_THREADS.record(kind)
        EAGER_GUARD.check(kind)
        return orig_fn(*args, **kwargs)

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
