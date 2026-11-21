# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""verl's ``("vllm", "async")`` rollout with one TP=1 vLLM engine inside each trainer process."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import time
from typing import Any, Callable, Optional

import torch

from verl.workers.rollout.base import BaseRollout
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary, compat
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import (
    COLLECTIVES,
    DEVICE_THREADS,
    GlobalEagerMode,
    IdentityEnv,
    engine_grad_ctx,
    full_determinism_problem,
    prefix_caching_problem,
    snapshot_engine_global_state,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    all_ranks_ok,
    cfg_get,
    check_switches,
    dist_rank_world,
    log,
    say_build_failure,
    warn,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compile_guard import (
    LOGITS_AT_LOAD,
    install_call_pins_and_watch,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.engine import (
    engine_core_children,
    kv_capacity_check,
    log_hbm,
    make_server,
    preinit_vllm_parallel_state,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.gc_policy import freeze_heap, gib, rss_bytes


class TPUInProcRollout(BaseRollout):
    """verl's ``("vllm", "async")`` rollout with ``VERL_TPU_ROLLOUT_INPROC=1``: one in-process TP=1 engine per trainer
    rank, built by ``ActorRolloutRefWorker.init_model`` on every rank in one worker-group call."""

    # tests inject an engine here: (server, parsed CLI args) -> (engine, worker, vllm_config, engine_args); None = vLLM
    engine_factory: Optional[Callable] = None

    def __init__(self, config: Any, model_config: Any, device_mesh: Any, *args, defer_build: bool = False, **kwargs):
        super().__init__(config, model_config, device_mesh)
        unread = check_switches()
        self.rank, self.world = dist_rank_world()
        if unread:
            warn(f"rank {self.rank}: {unread} are set but not read by the in-process rollout (they change nothing)")
        tp = (
            int(self.config.tensor_model_parallel_size)
            * int(self.config.data_parallel_size)
            * int(self.config.pipeline_model_parallel_size)
        )
        if tp != 1:
            raise RuntimeError(f"the in-process rollout needs TP=DP=PP=1 per replica, got world {tp}")
        self.replica_rank = self.rank  # one replica per trainer rank, in rank order
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % 8) or 0)
        self.kv_bytes: Optional[int] = None
        self.global_steps: Optional[int] = None
        self.preinit: dict = {}
        if not defer_build:
            self._build()

    # -- build
    def _prepare(self) -> None:
        """Local checks before the build (no collective)."""
        self._t0 = time.time()
        DEVICE_THREADS.record("engine_build")
        if boundary.LIBTPU_PROBLEMS:
            raise RuntimeError(
                "the platform-import libtpu check failed in this trainer process: "
                f"{'; '.join(boundary.LIBTPU_PROBLEMS)}"
            )
        for problem in (full_determinism_problem(self.config), prefix_caching_problem(self.config)):
            if problem:
                raise RuntimeError(problem)
        engine_kwargs = dict(cfg_get(cfg_get(self.config, "engine_kwargs", {}), "vllm", {}) or {})
        kv = int(engine_kwargs.get("kv_cache_memory_bytes") or 0)
        if kv <= 0:
            raise RuntimeError(
                "rollout.engine_kwargs.vllm.kv_cache_memory_bytes must be a positive byte count: the in-process engine "
                "shares the device memory with the trainer, so its KV budget is explicit"
            )
        self.kv_bytes = kv
        self._check_runtime_up()
        self._patch_vllm_in_process()
        install_call_pins_and_watch()

    def _build_engine(self) -> None:
        """The collective vLLM parallel-state pre-init (every rank reaches it: callers agree first), then the local
        build."""
        with IdentityEnv() as ienv, GlobalEagerMode() as gmode:
            # the collective step first, before anything that can fail on one rank only; inside the identity / eager
            # guards, because constructing a VllmConfig already runs the platform's environment rewrites
            self.preinit = preinit_vllm_parallel_state()
            log_hbm("pre_engine", self.rank)
            self.server = make_server(self.config, self.model_config, self.replica_rank, self.rank, self.engine_factory)
            # the engine's construction (model load, KV pool, the runner's precompile and warm-up) under the engine's
            # one grad mode (off), as every later engine call: a program compiled here is the one served
            with engine_grad_ctx("build"):
                self._launch()
        self.identity = ienv
        self.eager = gmode
        if not ienv.restored():
            raise RuntimeError(f"identity variables not restored: {ienv.changed}")
        if not gmode.ok():
            raise RuntimeError(f"global eager mode not restored after the build: {gmode.restored} != {gmode.target}")
        COLLECTIVES.install()
        self.engine = self.server.engine
        self.worker = self.server._inproc_worker
        self.build_s = time.time() - self._t0
        self._report_build()
        log_hbm("post_engine", self.rank)
        self._mark_warm()

    def _mark_warm(self) -> None:
        """The build is over: the warm-up baselines every later engine call is held to."""
        snapshot_engine_global_state()
        self.engine.warm = True

    def _build(self) -> None:
        """verl's init_model on every trainer rank. The same two agreements on every path (:func:`all_ranks_ok`), so a
        rank that fails a local step never leaves the others inside a collective: prepare -> agree -> (the collective
        vLLM parallel-state pre-init, then the local build) -> agree. A failing rank logs its own exception first."""
        ok, err = True, None
        try:
            self._prepare()
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
            say_build_failure("prepare", self.rank, e)
        if not all_ranks_ok(ok):
            raise RuntimeError(f"in-process engine build refused on some rank (this rank: {err!r})") from err
        ok, err = True, None
        try:
            self._build_engine()
        except Exception as e:  # noqa: BLE001
            ok, err = False, e
            say_build_failure("build", self.rank, e)
        if not all_ranks_ok(ok):
            raise RuntimeError(f"in-process engine build failed on some rank (this rank: {err!r})") from err
        # the heap the build leaves (the trainer's init, the engine and its programs) is collected and frozen once
        self._gc_freeze(0, warm=True)

    def _launch(self) -> None:
        """``launch_server`` (an async def) from init_model's synchronous context: the build only parses arguments and
        constructs the engine, it awaits nothing asynchronous."""
        coro = self.server.launch_server(master_address="127.0.0.1", master_port=1, dp_rpc_port=1)
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        if not running:
            asyncio.run(coro)
            return
        # init_model is a synchronous actor method: it runs on the event-loop thread while the loop is "running" but
        # blocked in this call. Drive the coroutine by hand: launch_server's awaits are all synchronous underneath.
        try:
            coro.send(None)
        except StopIteration:
            return
        coro.close()
        raise RuntimeError("launch_server awaited something asynchronous during the in-process build")

    def _check_runtime_up(self) -> None:
        """The trainer's torch_tpu runtime must already be up in this process (its PJRT client, bound to this rank's
        TensorCore) and torch.distributed initialized (the trainer's world)."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("torch.distributed is not initialized: the trainer runtime is not up")
        if getattr(torch, "tpu", None) is None:
            raise RuntimeError("torch.tpu missing: torch_tpu not imported in the trainer process")
        try:
            from verl_hardware_plugin.accelerators.tpu import device_api

            up = device_api._tpu_runtime_initialized()
            n = device_api.device_count() if up else 0
            cur = device_api.current_device() if up else None
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"the trainer's TPU runtime is not queryable: {e!r}") from e
        if not up or n < 1:
            raise RuntimeError(
                f"the trainer's torch_tpu runtime is not up in this process (initialized={up}, devices={n})"
            )
        log(
            f"runtime rank {self.rank}: trainer torch_tpu runtime up (initialized {up}, devices {n}, current {cur}, "
            f"TPU_VISIBLE_DEVICES={os.environ.get('TPU_VISIBLE_DEVICES')}, LOCAL_RANK={self.local_rank}, world "
            f"{self.world}); "
            f"pid {os.getpid()}"
        )

    def _patch_vllm_in_process(self) -> None:
        """What an in-process engine needs of the plugin's vLLM patches (the multiprocessing wrapper is not needed:
        no child process): the ``TPUWorker.load_model`` hook that installs verl's logits mask at model load, and vLLM's
        compile cache off (a program loaded from the cache is not a compile the recompile accounting sees)."""
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
        from vllm_torchtpu.worker.tpu_worker import TPUWorker

        if not LOGITS_AT_LOAD.install(TPUWorker):
            raise RuntimeError(
                "compute_logits at load: TPUWorker has no load_model of its own to hook: refusing to build"
            )

    def _compat_record(self, env_problems: list) -> dict:
        return {
            "runner": compat.runner_features(),
            "env_override": {
                "structure": (boundary.ENV_OVERRIDE_PROBE or {}).get("fingerprint"),
                "reviewed": (boundary.ENV_OVERRIDE_PROBE or {}).get("reviewed"),
                "after_build": env_problems or "unchanged within the reviewed effects",
            },
        }

    def _report_build(self) -> None:
        eng = self.server._inproc_engine
        vc = self.server._inproc_vllm_config
        args = self.server._inproc_engine_args
        seed = args.get("seed")
        pc, sc, cc = vc.parallel_config, vc.scheduler_config, vc.cache_config
        client = type(eng.engine_core).__name__
        executor = type(eng.model_executor).__name__
        tpd = f"TP={pc.tensor_parallel_size} PP={pc.pipeline_parallel_size} DP={pc.data_parallel_size}"
        sched = {
            k: getattr(sc, k, None)
            for k in ("async_scheduling", "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill", "policy")
        }
        cache = {
            k: getattr(cc, k, None)
            for k in ("kv_cache_memory_bytes", "num_gpu_blocks", "block_size", "enable_prefix_caching", "cache_dtype")
        }
        tp_ranks = None
        world_ranks: Any = None
        try:
            from vllm.distributed import parallel_state as ps

            tp_ranks = list(ps.get_tp_group().ranks)
            world_ranks = len(ps.get_world_group().ranks)
        except Exception as e:  # noqa: BLE001
            world_ranks = f"error {type(e).__name__}"
        children = engine_core_children()
        # the trainer created PJRT before the engine was imported: the engine import and build must not have changed
        # any libtpu flag, nor any other environment name outside the reviewed env_override effects
        env_problems = compat.env_after_build_problems(boundary.ENV_AT_CHECK, dict(os.environ))
        log(f"compat rank {self.rank}: {json.dumps(self._compat_record(env_problems), sort_keys=True, default=str)}")
        good = (
            client == "InprocClient"
            and executor == "UniProcExecutor"
            and tpd == "TP=1 PP=1 DP=1"
            and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
            and not children
            and tp_ranks == [self.rank]
            and int(cache.get("kv_cache_memory_bytes") or 0) == self.kv_bytes
            and seed == self.replica_rank + int(self.config.seed)
            and not env_problems
            and not cache.get("enable_prefix_caching")
        )
        text = (
            f"in-process engine pid {os.getpid()}; {tpd}; VLLM_ENABLE_V1_MULTIPROCESSING="
            f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')} (client {client}, executor {executor}); engine seed "
            f"{seed}; "
            f"vLLM TP group {tp_ranks} world group size {world_ranks}; EngineCore children {children or 'none'}; build "
            f"{self.build_s:.1f} s; vLLM parallel state pre-init {json.dumps(self.preinit, sort_keys=True)}"
            + (f"; ENVIRONMENT CHANGED by the engine import / build: {env_problems}" if env_problems else "")
            + ("; PREFIX CACHING ON in the built engine" if cache.get("enable_prefix_caching") else "")
            + f"; {'OK' if good else 'FAILED'}"
        )
        (log if good else warn)(f"rank {self.rank}: {text}")
        log(
            f"scheduler rank {self.rank}: {json.dumps(sched, default=str, sort_keys=True)} cache "
            f"{json.dumps(cache, default=str, sort_keys=True)}"
        )
        kvc = kv_capacity_check(vc, eng)
        (log if kvc["ok"] else warn)(f"kv capacity rank {self.rank}: {json.dumps(kvc, default=str, sort_keys=True)}")
        if not kvc["ok"] and kvc["mode"] != "report":
            raise RuntimeError(
                f"rank {self.rank}: KV CAPACITY FAILED: the KV pool holds {kvc['max_concurrency']} requests of "
                "max_model_len "
                f"{kvc['max_model_len']} tokens, fewer than max_num_seqs {kvc['max_num_seqs']}"
            )
        log(
            f"identity rank {self.rank}: rewritten by the build "
            f"{json.dumps({k: list(v) for k, v in self.identity.changed.items()}, sort_keys=True)}; "
            f"restored {self.identity.restored()}; other env changes {self.identity.other_changed}"
        )
        log(
            f"eager rank {self.rank}: global eager mode at trainer start {boundary.TRAINER_EAGER_AT_START_NAME} "
            f"({boundary.TRAINER_EAGER_SNAPSHOT_WHERE}); before the build {self.eager.before}, after the build "
            f"{self.eager.after_build}, restored to {self.eager.restored} (target {self.eager.target}: "
            f"{self.eager.target_source}); "
            "engine calls run under a DEFER_AND_FUSE scope"
        )
        if not good:
            raise RuntimeError(f"rank {self.rank}: in-process engine check FAILED (see the line above)")

    # -- verl's rollout interface
    async def resume(self, tags: list[str]):
        return None

    async def release(self):
        return None

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
        return await self.server.generate(
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
        return await self.server.abort_all_requests(
            reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
        )

    async def resume_generation(self):
        return await self.server.resume_generation()

    def server_info(self) -> dict:
        return {
            "rank": self.rank,
            "replica_rank": self.replica_rank,
            "pid": os.getpid(),
            "engine_ready": getattr(self, "engine", None) is not None,
            "global_steps": self.global_steps,
        }

    def step_health(self) -> Optional[str]:
        """Why this rank's rollout since the last weight sync must not be trained on, or None: the engine failed (every
        request in flight failed), or requests failed or were aborted (a sync-mode rollout aborts none)."""
        e = getattr(self, "engine", None)
        if e is None:
            return "the in-process engine is not built"
        if e._error is not None:
            return f"the in-process engine failed in this step's rollout ({type(e._error).__name__}: {e._error})"[:300]
        if e.window_failed:
            return f"{e.window_failed} request(s) failed in this step's rollout"
        if e.window_aborted:
            return f"{e.window_aborted} request(s) aborted in this step's rollout (a sync-mode rollout aborts none)"
        return None

    async def update_weights(self, weights, wire_format: str = "named_tensors", **kwargs):
        if wire_format != "named_tensors":
            raise NotImplementedError(f"wire_format {wire_format!r}")
        if kwargs.get("peft_config") is not None:
            raise NotImplementedError("LoRA adapters are not supported in process")
        raise NotImplementedError(
            "weight sync into the in-process engine is not available yet (the verified in-place install is a separate "
            "change): refusing the sync, nothing was installed"
        )

    def _gc_freeze(self, step: int, warm: bool = False, why: Optional[str] = None) -> float:
        """Collect and freeze the heap (vLLM's ``freeze_gc_heap``) at the end of the build. Later collections never
        traverse it."""
        counts0, frozen0, rss0 = list(gc.get_count()), gc.get_freeze_count(), rss_bytes()
        how, dt = freeze_heap()
        rss1 = rss_bytes()
        where = "warm-up" if warm else f"step {step}"
        log(
            f"gc freeze rank {self.rank} {where}: {how} in {dt:.3f} s; frozen objects {gc.get_freeze_count()}"
            + (f"; gc thresholds {json.dumps(list(gc.get_threshold()))}; gc enabled {gc.isenabled()}" if warm else "")
            + f"; newly frozen {gc.get_freeze_count() - frozen0}; gc counts before {json.dumps(counts0)}; rss "
            f"{gib(rss0)} -> "
            f"{gib(rss1)} GiB" + (f"; why {why}" if why else "")
        )
        return dt
