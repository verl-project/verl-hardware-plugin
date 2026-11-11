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
    engine_mode_ctx,
    full_determinism_problem,
    prefix_caching_problem,
    snapshot_engine_global_state,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    INTEGRITY_ENV,
    INTEGRITY_TAG,
    all_ranks_ok,
    cfg_get,
    check_switches,
    dist_rank_world,
    gc_rss_limit_bytes,
    integrity_mode,
    last_step,
    log,
    say_build_failure,
    warn,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compile_guard import (
    CALL_PINS,
    LOGITS_AT_LOAD,
    RECOMPILES,
    install_call_pins_and_watch,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.engine import (
    engine_core_children,
    kv_capacity_check,
    log_hbm,
    make_server,
    preinit_vllm_parallel_state,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.gc_policy import freeze_heap, gib, rss_bytes, rss_freeze_due
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.weight_sync import (
    CHECKSUM_EVERY,
    FULL_PARITY_STEPS,
    LOADER_FAMILY,
    OPT_STEPS,
    InProcInstaller,
    StaleProbe,
    checksum_names,
    export_pin_problem,
    export_signature,
    no_update_between,
    tensor_checksums,
    verify_reason,
)


class TPUInProcRollout(BaseRollout):
    """verl's ``("vllm", "async")`` rollout with ``VERL_TPU_ROLLOUT_INPROC=1``: one in-process TP=1 engine per trainer
    rank, built by ``ActorRolloutRefWorker.init_model`` on every rank in one worker-group call."""

    # defaults of the per-instance state ``__init__`` sets, for instances made without it
    stale: Optional[StaleProbe] = None
    _rss_base: Optional[int] = None
    _export_pin: Optional[tuple] = None  # (step, export signature) of the first verified sync that passed
    layout_entry: Optional[compat.LayoutEntry] = None
    last_step: Optional[int] = None
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
        self.sync_log: list[dict] = []
        self.preinit: dict = {}
        # the start of the trainer's export, set at verl's update_weights entry (before get_per_tensor_param)
        self._pending_export_t0: Optional[float] = None
        self._syncs_started = 0  # the first sync of this process is verified
        self._last_wsum: Optional[tuple] = None  # (step, wsum, optimizer-step mark) of the last sync whose sha256 ran
        self.integrity = integrity_mode()
        self.last_step = last_step()
        self.rss_limit = gc_rss_limit_bytes()
        probe = StaleProbe(
            self.rank, mode=self.integrity, barrier_after=bool(getattr(self.config, "free_cache_engine", False))
        )
        if probe.demoted:
            log(f"stale probe rank {self.rank}: verdicts reported, not raised ({probe.demoted})")
        self.stale: Optional[StaleProbe] = probe
        self._rss_base: Optional[int] = None
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
        # the install layout must be a tested one (every rank reads the same config: the agreement after this carries
        # a refusal to all of them, before the collective build)
        hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
        self.layout_entry, problem = compat.layout_before_build(hf_config)
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
        # the recompile accounting before anything of the engine compiles; an armed guard without its hook refuses
        if not RECOMPILES.install() and RECOMPILES.armed:
            raise RuntimeError(
                f"the recompile guard cannot be installed ({RECOMPILES.unavailable}): refusing to build without it"
            )
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
        self.hf_text_config = getattr(self.model_config.hf_config, "text_config", None) or self.model_config.hf_config
        model = getattr(getattr(self.worker, "model_runner", None), "model", None)
        vc = self.server._inproc_vllm_config
        self.layout_entry, problem = compat.layout_after_build(
            model, getattr(vc, "model_config", None), self.hf_text_config
        )
        if problem:
            raise RuntimeError(problem)
        problem = compat.tpu_sync_contract()
        if problem:
            raise RuntimeError(f"tpu-sync install contract: {problem}: refusing to build")
        self.installer = InProcInstaller(self.worker, self.hf_text_config, self.rank, layout=self.layout_entry)
        self.build_s = time.time() - self._t0
        self._report_build()
        log_hbm("post_engine", self.rank)
        self._mark_warm()

    def _mark_warm(self) -> None:
        """The build is over: the warm-up baselines every later engine call is held to (the recompile guard is armed
        from here on), logged in the warm-up line."""
        guard = RECOMPILES.mark_warm(self.rank)
        entries = guard.pop("entries")
        pins = CALL_PINS.mark_warm()
        snapshot_engine_global_state()
        self.engine.warm = True
        warm = {
            "guard": {k: guard[k] for k in ("installed", "armed", "guard_n", "fresh_mode", "shape_rule")},
            "entries": entries,
            "fresh_before_warm": guard["fresh_before_warm"],
            "resets_before_warm": guard["resets_before_warm"],
            "pins": {k: pins[k] for k in ("pinned_functions", "required_missing", "runner_loaded", "frozen")},
        }
        log(f"warm-up rank {self.rank}: {json.dumps(warm, sort_keys=True, default=str)}")

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
        model = getattr(getattr(self.worker, "model_runner", None), "model", None)
        cls = type(model)
        entry = self.layout_entry
        return {
            "runner": compat.runner_features(),
            "env_override": {
                "structure": (boundary.ENV_OVERRIDE_PROBE or {}).get("fingerprint"),
                "reviewed": (boundary.ENV_OVERRIDE_PROBE or {}).get("reviewed"),
                "after_build": env_problems or "unchanged within the reviewed effects",
            },
            "tpu_sync_contract": "ok",
            "layout": {
                "model_type": getattr(self.hf_text_config, "model_type", None),
                "model_class": f"{cls.__module__}.{cls.__qualname__}",
                "quantization": getattr(
                    getattr(self.server._inproc_vllm_config, "model_config", None), "quantization", None
                ),
                "family": entry.family if entry is not None else None,
                "tested": entry is not None,
                "tests": list(entry.tests) if entry is not None else [],
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
        request in flight failed), requests failed or were aborted (a sync-mode rollout aborts none), or this step's
        sync served stale weights."""
        e = getattr(self, "engine", None)
        if e is None:
            return "the in-process engine is not built"
        if e._error is not None:
            return f"the in-process engine failed in this step's rollout ({type(e._error).__name__}: {e._error})"[:300]
        if e.window_failed:
            return f"{e.window_failed} request(s) failed in this step's rollout"
        if e.window_aborted:
            return f"{e.window_aborted} request(s) aborted in this step's rollout (a sync-mode rollout aborts none)"
        st = getattr(self, "stale", None)
        if st is not None and st.fatal_pending is not None:
            return st.fatal_pending
        return None

    def _health_record(self, step: int, window: dict) -> dict:
        """The per-sync health line: the recompile record, the device-thread and collective invariants and the rollout
        window."""
        return {
            "compile": RECOMPILES.compile_record(),
            "threads": {
                "device_threads": len(DEVICE_THREADS.counts),
                "collectives_world_gt1": sum(COLLECTIVES.big.values()),
                "collectives_size1": COLLECTIVES.small,
            },
            "window": window,
        }

    async def update_weights(self, weights, wire_format: str = "named_tensors", **kwargs):
        if wire_format != "named_tensors":
            raise NotImplementedError(f"wire_format {wire_format!r}")
        if kwargs.get("peft_config") is not None:
            raise NotImplementedError("LoRA adapters are not supported in process")
        raw_step = kwargs.get("global_steps")
        # an unlabelled sync (global_steps=None) keeps the label of the previous sync (0 before any) and is never
        # pre-verified (the pre-install comparison holds only at the initial sync)
        step = int(raw_step) if raw_step is not None else (self.global_steps if self.global_steps is not None else 0)
        first_sync = self._syncs_started == 0
        self._syncs_started += 1
        if self.engine._collectors or self.engine._engine.has_unfinished_requests():
            raise RuntimeError(f"weight update at step {step} with requests in flight")
        if self.stale is not None and self.stale.pending is not None:  # the previous sync's probe, before this install
            self.stale.resolve("sync")
        opt_mark = (
            OPT_STEPS.mark()
        )  # the trainer's optimizer steps so far (every update_actor of this interval returned)
        t_iter = time.time()
        if self._pending_export_t0 is not None:
            t0, clock = float(self._pending_export_t0), "the update_weights entry"
        else:
            t0, clock = t_iter, "the tensor iteration"
        self._pending_export_t0 = None
        t0 = min(t0, t_iter)
        DEVICE_THREADS.record("export")
        staging: dict[str, torch.Tensor] = {}
        for name, tensor in weights:  # the trainer's export: FSDP all-gathers, in the trainer's own eager mode
            staging[name] = tensor
        t1 = time.time()
        if self.stale is not None:  # a stale verdict of an earlier sync, raised after the export's collectives
            self.stale.check_fatal()
        # the export must be the one the first verified sync of this process passed with, checked before the install
        # and at every sync (after the export's collectives: every rank exports the same names)
        sig = export_signature(staging)
        line = export_pin_problem(self._export_pin, sig, step, self.rank)
        if line is not None:
            warn(line)
            raise RuntimeError(line)
        why = verify_reason(step, first_sync, self.last_step)
        if why is None and self._export_pin is None:  # until a verified sync has passed, every sync is verified
            why = "until a verified sync pins the export"
        verify = why is not None
        full_sha = step in FULL_PARITY_STEPS or first_sync
        pre = raw_step is not None and step == 0  # only an explicit step 0 holds the checkpoint vLLM loaded
        with engine_mode_ctx("install"):
            DEVICE_THREADS.record("install")
            res = self.installer.install(staging, verify=verify, pre=pre, step=step)
        t2 = time.time()
        family = self.installer.family
        if family is not None and res.get("mode") != family:
            line = (
                f"{INTEGRITY_TAG} rank {self.rank} step {step}: the install ran in mode {res.get('mode')!r}, "
                f"not the tested family {family!r}"
            )
            warn(line)
            raise RuntimeError(line)
        self.server.global_steps = step
        self.global_steps = step
        # checks timed apart from the sync: the staleness probe at every sync, the strided checksum at the verified
        # syncs and every few syncs, the full sha256 at steps 0 / 1 and the first sync, the parity and the coverage
        model = self.worker.model_runner.model
        named = dict(model.named_parameters())
        names = checksum_names(list(named))
        par = cov = None
        with engine_mode_ctx("checksum"):
            DEVICE_THREADS.record("checksum")
            probe_s = self.stale.dispatch(step, named, names, mark=opt_mark) if self.stale is not None else None
            if probe_s is not None and full_sha and self.stale is not None:
                self.stale.host_reference(step, named, names)
            sha_due = verify or full_sha or probe_s is None or step % CHECKSUM_EVERY == 0
            wsum, wbytes = tensor_checksums(named, names, full=False) if sha_due else ("deferred", 0)
            full = tensor_checksums(named, names, full=True)[0] if full_sha else None
            t3 = time.time()
            if res.get("_verify") is not None:
                par = self.installer.finish_parity(res, staging)
                cov = self.installer.coverage(res, staging)
        t4 = time.time()
        direct = res["mode"] != LOADER_FAMILY
        # staleness on the strided checksum: compared only between syncs whose checksum ran; unchanged weights are
        # expected only when the trainer provably took no optimizer step in between (all skipped)
        prev = self._last_wsum
        stale = None
        unchanged_expected = False
        if sha_due:
            if prev is not None and step > prev[0] and wsum == prev[1]:
                if no_update_between(prev[2], opt_mark):
                    unchanged_expected = True
                else:
                    stale = (
                        f"the engine's weights are unchanged since sync {prev[0]} (wsum {wsum[:16]} at both): the "
                        "trainer's "
                        "update or this install did not land"
                    )
            self._last_wsum = (step, wsum, opt_mark)
        window = self.engine.window_take()
        rec = {
            "step": step,
            "export_s": round(t1 - t0, 4),
            "install_s": round(t2 - t1, 4),
            "sync_s": round(t2 - t0, 4),
            "checksum_s": round(t3 - t2, 4),
            "tensors": len(staging),
            "mode": res["mode"],
            "wsum": wsum[:16],
            "verified": why,
            "parity_s": round(t4 - t3, 4),
            "label_kept": raw_step is None,
            "first_sync": first_sync,
            "stale": stale is not None,
            "rollout_window": window,
            "sha": sha_due,
            "opt_mark": opt_mark,
            "unchanged_expected": unchanged_expected,
        }
        staleness = (
            "UNCHANGED"
            if stale
            else (
                f"unchanged as expected (no optimizer step since sync {prev[0] if prev is not None else None})"
                if unchanged_expected
                else ("changed" if sha_due and prev is not None and step > prev[0] else "n/a")
            )
        )
        log(
            f"sync step {step} rank {self.rank}: mode {res['mode']}, {len(staging)} tensors, export+gather "
            f"{t1 - t0:.4f} s, "
            f"install {t2 - t1:.4f} s, checksum {t3 - t2:.4f} s (not in the sync), wsum {wsum[:16]} over {len(names)} "
            f"tensors {wbytes / 2**20:.1f} MiB sampled; export clock from {clock}; verified {why or 'no'} (parity + "
            "coverage "
            f"{t4 - t3:.4f} s, not in the sync); label {'kept (global_steps=None)' if raw_step is None else 'given'}; "
            "first "
            f"sync in process {first_sync}; staleness {staleness}; optimizer steps real/skipped/unknown "
            f"{list(opt_mark) if opt_mark is not None else 'not counted'}; rollout window "
            f"{json.dumps(window, sort_keys=True)}; "
            f"strided sha256 {'run' if sha_due else f'deferred (every {CHECKSUM_EVERY} syncs)'}, staleness probe "
            + (
                f"dispatched in {1e3 * probe_s:.3f} ms"
                if probe_s is not None
                else f"off ({getattr(self.stale, 'unavailable', None)})"
            )
        )
        integrity = []
        parity_failed = False
        if verify and par is not None:
            post_bad = par.get("post_mismatched") or []
            pre_bad = par.get("pre_mismatched") or []
            unresolved = par.get("unresolved") or []
            exact = not post_bad and not pre_bad and not unresolved and (par.get("post_checked") or 0) > 0
            parity_failed = not exact
            coverage_bad = bool(cov is not None and cov.get("checked") and not cov.get("ok"))
            cv = cov or {}
            coverage = (
                "direct"
                if direct
                else f"loaders {par.get('post_checked') or 0}/{par.get('params')} engine parameters, packed-aware"
            )
            rec.update(
                {
                    "parity": {
                        "pre_checked": par.get("pre_checked"),
                        "pre_mismatched": len(pre_bad),
                        "post_checked": par.get("post_checked"),
                        "post_mismatched": len(post_bad),
                        "unresolved": len(unresolved),
                        "coverage": coverage,
                    },
                    "wsha": full[:16] if full else None,
                }
            )
            # the coverage check is part of the verdict: an exact comparison of the written parameters is not an exact
            # install when some engine parameter was never written
            verdict_ok = exact and not coverage_bad
            log_fn = log if verdict_ok else warn
            log_fn(
                f"parity step {step} rank {self.rank}: {'PARITY EXACT' if verdict_ok else 'PARITY FAILED'}: "
                f"pre-install {par.get('pre_checked')} checked {len(pre_bad)} mismatched, post-install "
                f"{par.get('post_checked')} "
                f"checked {len(post_bad)} mismatched, {len(unresolved)} unresolved; coverage {coverage}"
                + (
                    f", {cv.get('not_written_n')} not written {(cv.get('not_written') or [])[:4]}"
                    if coverage_bad
                    else ""
                )
                + (
                    f", not-exported patterns match received {cv.get('not_exported_received')}"
                    if coverage_bad and cv.get("not_exported_received")
                    else ""
                )
                + f"; full sha256 {(full or '')[:16]}"
                + (f"; first {(pre_bad + post_bad + unresolved)[:4]}" if not exact else "")
            )
        if cov is not None:
            rec["install_coverage"] = {
                k: cov.get(k) for k in ("checked", "via", "params", "written", "allowlisted", "not_written_n", "ok")
            }
            (log if cov.get("ok") else warn)(
                f"coverage step {step} rank {self.rank}: "
                + (
                    f"{'COVERAGE OK' if cov.get('ok') else 'COVERAGE FAILED'}: "
                    f"{cov.get('written')}/{cov.get('params')} engine "
                    f"parameters written ({cov.get('via')}), {cov.get('allowlisted')} not exported by the trainer "
                    f"{cov.get('allowlisted_first')}, tied copy {cov.get('tied_copy')}, not written "
                    f"{cov.get('not_written_n')} "
                    f"{(cov.get('not_written') or [])[:6]}, not-exported patterns matching received tensors "
                    f"{cov.get('not_exported_received')}"
                    if cov.get("checked")
                    else f"not checked: {cov.get('why')}"
                )
            )
            if not cov.get("ok"):
                if cov.get("not_written_n"):
                    integrity.append(
                        f"coverage: {cov.get('not_written_n')} engine parameter(s) never written by the install "
                        "(they keep the checkpoint's values while the trainer moves on), "
                        f"first {(cov.get('not_written') or [])[:4]}"
                    )
                if cov.get("not_exported_received"):
                    integrity.append(
                        "the tested layout's not-exported patterns match received tensors "
                        f"{cov.get('not_exported_received')}: "
                        "the table entry is wrong for this trainer"
                    )
                if not cov.get("checked"):
                    integrity.append(f"coverage not checked: {cov.get('why')}")
        if parity_failed:
            raise RuntimeError(f"in-process weight parity FAILED at step {step} on rank {self.rank}")
        if stale is not None:
            if self.integrity == "fatal":
                integrity.append(stale)
            else:
                warn(f"rank {self.rank} step {step}: {stale} (reported: {INTEGRITY_ENV}=report)")
        health = self._health_record(step, window)
        log(f"health step {step} rank {self.rank}: {json.dumps(health, sort_keys=True, default=str)}")
        problems = self._sync_problems(health)
        if problems:
            raise RuntimeError(f"rank {self.rank}: " + "; ".join(problems))
        if integrity:  # after every line of this sync is out (the evidence)
            line = f"{INTEGRITY_TAG} rank {self.rank} step {step}: " + "; ".join(integrity)
            warn(line)
            raise RuntimeError(line)
        log_hbm(f"post_sync_{step}", self.rank)
        if (
            verify and self._export_pin is None
        ):  # a verified sync passed: its export is the one every later sync must send
            self._export_pin = (step, sig)
        rec["export_pin"] = self._export_pin[0] if self._export_pin is not None else None
        self.sync_log.append(rec)
        if step == 1:  # after the first update: the optimizer state and the trainer's compiled programs exist
            self._gc_freeze(step)
        else:
            self._gc_rss_guard(step)

    def _sync_problems(self, health: dict) -> list:
        """The invariants every sync re-checks (fatal): device work on one thread, no world-group collective during
        engine steps, the recompile guard not tripped and no engine cache entry that no compile hook accounts for."""
        problems = []
        if not DEVICE_THREADS.single():
            problems.append(f"device work ran on more than one thread: {dict(DEVICE_THREADS.counts)}")
        if sum(COLLECTIVES.big.values()):
            problems.append(f"vLLM world-group collectives during engine steps: {dict(COLLECTIVES.big)}")
        if RECOMPILES.tripped is not None:
            problems.append(RECOMPILES.tripped)
        unaccounted = (health.get("compile") or {}).get("unaccounted") or {}
        if unaccounted and RECOMPILES.armed:
            problems.append(
                f"RECOMPILE GUARD rank {self.rank}: engine cache entries no recompile / fresh-compile hook accounts "
                f"for {unaccounted}"
            )
        return problems

    def _gc_rss_guard(self, step: int) -> Optional[float]:
        due, now, why = rss_freeze_due(step, self._rss_base, self.rss_limit)
        if not due:
            if now is not None and self._rss_base is None:
                self._rss_base = now
            return None
        return self._gc_freeze(step, why=why)

    def _gc_freeze(self, step: int, warm: bool = False, why: Optional[str] = None) -> float:
        """Collect and freeze the heap (vLLM's ``freeze_gc_heap``): at the end of the build, at the step-1 sync, and at
        an aligned sync when the RSS grew past ``VERL_TPU_INPROC_GC_RSS_GIB``. Later collections never traverse it."""
        counts0, frozen0, rss0 = list(gc.get_count()), gc.get_freeze_count(), rss_bytes()
        how, dt = freeze_heap()
        rss1 = rss_bytes()
        self._rss_base = rss1
        where = "warm-up" if warm else f"step {step}"
        log(
            f"gc freeze rank {self.rank} {where}: {how} in {dt:.3f} s; frozen objects {gc.get_freeze_count()}"
            + (f"; gc thresholds {json.dumps(list(gc.get_threshold()))}; gc enabled {gc.isenabled()}" if warm else "")
            + f"; newly frozen {gc.get_freeze_count() - frozen0}; gc counts before {json.dumps(counts0)}; rss "
            f"{gib(rss0)} -> "
            f"{gib(rss1)} GiB" + (f"; why {why}" if why else "")
        )
        return dt
