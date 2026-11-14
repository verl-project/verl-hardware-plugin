# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The in-process engine: an ``AsyncLLM``-shaped adapter over vLLM's synchronous ``LLMEngine``, the in-process
``vLLMHttpServer`` that serves verl's requests from it, and the replica whose server is the trainer worker itself.

verl's server code (``generate``, the admission gate, ``abort_all_requests``, ``resume_generation``) runs verbatim on
:class:`InProcAsyncEngine`. A stepping pump on the actor's event-loop thread calls ``engine.step()`` while requests are
in flight and yields to the loop between steps; there is no extra thread, so every device call of the trainer and of
the engine runs on one thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable, Optional

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compat
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import (
    COLLECTIVES,
    DEVICE_THREADS,
    NOT_ARMED,
    EngineScope,
    engine_mode_ctx,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import integrity_mode, log, warn
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compile_guard import LOGITS_AT_LOAD
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.gc_policy import GC_DEFER

AUDIT_EVERY = 64  # after warm-up the device-thread record and the collective counter run on every 64th pump step


class _Collector:
    """The per-request collector ``vLLMHttpServer`` drains (``RequestOutputCollector``'s surface)."""

    def __init__(self, request_id: str, external_id: str, loop: asyncio.AbstractEventLoop):
        self.request_id = request_id  # vLLM's internal id (abort)
        self.external_id = external_id
        self._fut: asyncio.Future = loop.create_future()

    def put(self, output: Any) -> None:
        if not self._fut.done():
            self._fut.set_result(output)

    def get_nowait(self) -> Any:
        return self._fut.result() if self._fut.done() else None

    async def get(self) -> Any:
        return await self._fut

    def close(self) -> None:
        pass


def _aborted_output(external_id: str) -> Any:
    """A finished output with no completions: verl turns it into ``TokenOutput(stop_reason="aborted")``."""
    return SimpleNamespace(
        request_id=external_id, finished=True, outputs=[], num_cached_tokens=None, prompt_logprobs=None
    )


class InProcAsyncEngine:
    """``AsyncLLM``'s surface that ``vLLMHttpServer`` uses, over a synchronous in-process engine.

    ``add_request`` registers the request in the engine (FINAL_ONLY output: one RequestOutput per request, at the end,
    the same final value AsyncLLM's cumulative stream ends with) and starts the stepping pump if it is not running.
    The pump runs on the caller's event loop: one ``engine.step()`` per iteration inside the engine scope, finished
    outputs resolved, then ``await asyncio.sleep(0)`` so new requests and RPCs get the loop between steps.
    """

    def __init__(self, engine: Any, rank: int):
        self._engine = engine
        self.rank = rank
        self.model_config = engine.model_config
        self._collectors: dict[str, _Collector] = {}
        self._pump_task: Optional[asyncio.Task] = None
        self._paused = False
        self.steps = 0
        self.output_processor = SimpleNamespace(request_states=_RequestStatesView(self), parent_requests={})
        self._error: Optional[BaseException] = None
        # requests this engine aborted / failed, in all and in the window since the last weight sync (the step guard
        # before update_actor reads the window: in sync mode a healthy rollout aborts and fails none)
        self.aborted = 0
        self.failed = 0
        self.window_aborted = 0
        self.window_failed = 0
        self.warm = False  # set at the end of the build: the pump then samples its audit-only checks

    async def add_request(
        self, request_id, prompt, sampling_params, lora_request=None, priority=0, data_parallel_rank=None, **_
    ):
        if self._error is not None:
            raise RuntimeError(f"the in-process engine failed earlier: {self._error!r}")
        from vllm.sampling_params import RequestOutputKind

        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        loop = asyncio.get_running_loop()
        with engine_mode_ctx("add_request"), COLLECTIVES.arm():
            DEVICE_THREADS.record("add_request")
            internal = self._engine.add_request(
                request_id, prompt, sampling_params, lora_request=lora_request, priority=priority
            )
        col = _Collector(internal if isinstance(internal, str) else request_id, request_id, loop)
        self._collectors[request_id] = col
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = loop.create_task(self._pump())
        return col

    async def _pump(self) -> None:
        lean = self.warm
        k = 0
        GC_DEFER.enter()
        try:
            while self._engine.has_unfinished_requests() and not self._paused:
                audit = not lean or k == 0 or self.steps % AUDIT_EVERY == 0
                k += 1
                with EngineScope("step"), COLLECTIVES.arm() if audit else NOT_ARMED:
                    if audit:
                        DEVICE_THREADS.record("step")
                    outs = self._engine.step()
                self.steps += 1
                for out in outs:
                    if getattr(out, "finished", False):
                        col = self._collectors.pop(out.request_id, None)
                        if col is not None:
                            col.put(out)
                await asyncio.sleep(0)
        except BaseException as e:  # noqa: BLE001 - fail every waiting request instead of hanging it
            self._error = e
            warn(f"engine rank {self.rank}: STEP FAILED: {type(e).__name__}: {e}")
            nfail = 0
            for col in list(self._collectors.values()):
                if not col._fut.done():
                    col._fut.set_exception(RuntimeError(f"in-process engine step failed: {e!r}"))
                    nfail += 1
            self.failed += nfail
            self.window_failed += nfail
            self._collectors.clear()
            raise
        finally:
            GC_DEFER.exit()

    async def abort(self, request_ids, internal: bool = False) -> None:
        """The ids verl passes are mapped to the engine's internal ids before the engine abort: vLLM's
        ``abort_request(ids, internal=True)`` matches internal ids only, and verl's
        ``abort_all_requests(abort_only=True)`` passes ``output_processor.request_states`` keys, which in process are
        the external ids. Internal ids are accepted as they are; an id no collector knows is passed through."""
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        by_int = {c.request_id: c for c in self._collectors.values()}
        cols: list = []
        engine_ids: list = []
        for rid in ids:
            col = self._collectors.get(rid) or by_int.get(rid)
            if col is None:
                if rid not in engine_ids:
                    engine_ids.append(rid)
            elif col not in cols:
                cols.append(col)
                engine_ids.append(col.request_id)
        with engine_mode_ctx("abort"):
            self._engine.abort_request(engine_ids, internal=True)
        self.aborted += len(cols)
        self.window_aborted += len(cols)
        for col in cols:
            self._collectors.pop(col.external_id, None)
            col.put(_aborted_output(col.external_id))

    async def pause_generation(self, mode: str = "abort", clear_cache: bool = True, **_) -> None:
        self._paused = True
        cols = list(self._collectors.values())
        if cols:
            with engine_mode_ctx("pause"):
                self._engine.abort_request([c.request_id for c in cols], internal=True)
            self.aborted += len(cols)
            self.window_aborted += len(cols)
            for col in cols:
                self._collectors.pop(col.external_id, None)
                col.put(_aborted_output(col.external_id))
        if self._pump_task is not None:
            with contextlib.suppress(BaseException):
                await self._pump_task

    async def resume_generation(self) -> None:
        self._paused = False

    async def wait_for_requests_to_drain(self) -> None:
        while self._collectors:
            await asyncio.sleep(0.01)

    async def reset_mm_cache(self) -> None:
        return None

    async def reset_prefix_cache(self, *_, **__) -> bool:
        return True

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        with engine_mode_ctx("rpc"):
            return self._engine.collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)

    def window_take(self) -> dict:
        """The aborted / failed requests since the last weight sync, reset (called at every sync)."""
        out = {"aborted": self.window_aborted, "failed": self.window_failed}
        self.window_aborted = self.window_failed = 0
        return out


class _RequestStatesView:
    """``output_processor.request_states`` keys (the in-flight request ids) for vLLMHttpServer's abort bookkeeping."""

    def __init__(self, eng: InProcAsyncEngine):
        self._eng = eng

    def keys(self):
        return list(self._eng._collectors.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._eng._collectors)

    def get(self, key, default=None):
        return self._eng._collectors.get(key, default)


# ------------------------------------------------------------------------------------------------ the in-process server


def _server_base():
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm import TPUvLLMHttpServer

    return TPUvLLMHttpServer


def make_server(
    config: Any, model_config: Any, replica_rank: int, rank: int, engine_factory: Optional[Callable] = None
) -> Any:
    """An in-process ``TPUvLLMHttpServer``, built without its actor ``__init__`` (no ports, no HTTP), whose
    ``run_server`` builds the engine in this process instead of an AsyncLLM and an HTTP server. ``engine_factory``
    (default :func:`build_vllm_engine`) returns ``(engine, worker, vllm_config, engine_args)`` from the parsed CLI
    arguments of verl's ``launch_server``."""
    base = _server_base()

    class InProcServer(base):  # type: ignore[valid-type, misc]
        async def run_server(self, args):
            self._inproc_build(args)

        async def launch_server(self, *args, **kwargs):
            # verl's vLLMHttpServer.launch_server (its CLI arguments; run_server above builds the engine), without
            # the timeline mixin's sampler thread
            from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

            return await vLLMHttpServer.launch_server(self, *args, **kwargs)

    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    srv = object.__new__(InProcServer)
    srv._disaggregation_role = "null"
    srv._disaggregation_kv_transfer_config = None
    srv._pd_decode_peers = []
    srv._pd_prefill_side_channel_port = None
    srv._pd_prefill_engine_id = None
    srv._pd_peer_idx = 0
    srv.config = srv._init_config(config)
    srv.model_config = srv._init_model_config(model_config)
    srv._validate_configs()
    srv.rollout_mode = RolloutMode.HYBRID
    srv.workers = []
    srv.replica_rank = replica_rank
    srv.node_rank = 0
    srv.gpus_per_node = 1
    srv.nnodes = 1
    srv.global_steps = None
    srv._warned_missing_spec_decode_stats = False
    srv._submission_paused = False
    srv._admitting = 0
    srv._resume_event = asyncio.Event()
    srv._resume_event.set()
    srv._rejecting = False
    srv._server_address = "inproc"
    srv._server_port = None
    srv.profiler_controller = None
    srv.profiler_keep_global_ranks = None
    srv.replica_world_size = 1
    srv._master_address, srv._master_port, srv._dp_rpc_port, srv._dp_master_port = "127.0.0.1", None, None, None
    srv._inproc_rank = rank
    srv._inproc_engine_factory = engine_factory or build_vllm_engine
    srv._inproc_build = functools.partial(_build_engine_into, srv)
    srv._should_profile = lambda: False
    problem = compat.server_contract_problem(vLLMHttpServer, set(vars(srv)))
    if problem:
        raise RuntimeError(f"in-process server: {problem}: refusing to build (the attribute would be missing)")
    return srv


def build_vllm_engine(srv: Any, args: Any) -> tuple:
    """vLLM's synchronous ``LLMEngine`` in this process (``InprocClient``, ``multiprocess_mode=False``) from
    ``launch_server``'s parsed CLI arguments, with verl's logits mask installed at model load."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    from verl.workers.rollout.utils import get_vision_placeholder_token_ids

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.ENGINE_CONTEXT)
    pc = vllm_config.parallel_config
    if (pc.tensor_parallel_size, pc.pipeline_parallel_size, pc.data_parallel_size) != (1, 1, 1):
        raise RuntimeError(
            f"the in-process engine needs TP=PP=DP=1, got {pc.tensor_parallel_size}/{pc.pipeline_parallel_size}/"
            f"{pc.data_parallel_size}"
        )
    # verl's monkey_patch_model arguments (vLLMHttpServer.run_server's RPC), applied at model load
    mask = {
        "vocab_size": len(srv.model_config.tokenizer),
        "banned_token_ids": get_vision_placeholder_token_ids(srv.model_config.processor),
    }
    if not LOGITS_AT_LOAD.hooked:
        raise RuntimeError("compute_logits at load: the TPUWorker.load_model hook is not installed")
    LOGITS_AT_LOAD.arm(mask)
    try:
        eng = LLMEngine(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
            usage_context=UsageContext.ENGINE_CONTEXT,
            multiprocess_mode=False,
        )
    finally:
        LOGITS_AT_LOAD.disarm()
    client = type(eng.engine_core).__name__
    if client != "InprocClient":
        raise RuntimeError(f"the engine client is {client}, not InprocClient (EngineCore would run in another process)")
    eng.reset_mm_cache()
    worker = eng.model_executor.driver_worker.worker
    LOGITS_AT_LOAD.verify_built(worker)
    return eng, worker, vllm_config, {k: v for k, v in vars(args).items() if not callable(v)}


def _build_engine_into(srv: Any, args: Any) -> None:
    """``run_server`` of the in-process server."""
    eng, worker, vllm_config, engine_args = srv._inproc_engine_factory(srv, args)
    srv._inproc_engine = eng
    srv._inproc_worker = worker
    srv._inproc_vllm_config = vllm_config
    srv._inproc_engine_args = engine_args
    srv.engine = InProcAsyncEngine(eng, srv._inproc_rank)


def preinit_vllm_parallel_state(local_rank: int = 0) -> dict:
    """Collective (every trainer rank, in lockstep): vLLM's world group and its TP=PP=1 model-parallel groups on top of
    the trainer's torch.distributed world, with the arguments vllm-torchtpu's ``TPUWorker.init_device`` uses for a
    TP=1 engine (world_size=1, rank=0, the gloo backend). vLLM skips an initialized world and only asserts the group
    sizes later, so the engine's own calls become no-ops and nothing after this point in the build is a cross-rank
    collective: a rank that then fails locally cannot strand the others in vLLM's ``new_group`` calls."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return {"skipped": "torch.distributed is not initialized"}
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    t0 = time.time()
    with set_current_vllm_config(VllmConfig()):
        ps.init_distributed_environment(
            world_size=1, rank=0, local_rank=local_rank, distributed_init_method="env://", backend="gloo"
        )
        ps.ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    return {
        "s": round(time.time() - t0, 3),
        "world": len(ps.get_world_group().ranks),
        "tp": list(ps.get_tp_group().ranks),
    }


def engine_core_children() -> list[str]:
    """Child processes of this process that look like a vLLM EngineCore (must be none)."""
    try:
        import psutil

        out = []
        for c in psutil.Process(os.getpid()).children(recursive=True):
            with contextlib.suppress(Exception):
                name = c.name()
                cmd = " ".join(c.cmdline()[:3])
                if "EngineCore" in name or "EngineCore" in cmd:
                    out.append(f"{c.pid}:{name}")
        return out
    except Exception as e:  # noqa: BLE001
        return [f"error {type(e).__name__}"]


def kv_capacity_check(vc: Any, eng: Any = None, environ: Optional[Mapping[str, str]] = None) -> dict:
    """The KV pool against the scheduler's concurrency. vLLM's ``EngineCore._initialize_kv_caches`` stores its own
    group-aware capacity (``cache_config.kv_cache_max_concurrency``: requests of ``max_model_len`` the pool holds);
    validation serves up to ``max_num_seqs`` requests per engine, so the pool must hold that many full-length requests
    or vLLM preempts. The per-group recount from the scheduler's ``kv_cache_config`` is recorded beside it.
    ok = the concurrency (vLLM's value, else the recount) >= max_num_seqs > 0. Never raises (the caller decides by
    ``mode``)."""
    mode = integrity_mode(environ)
    cc, sc, mc = (
        getattr(vc, "cache_config", None),
        getattr(vc, "scheduler_config", None),
        getattr(vc, "model_config", None),
    )
    need = int(getattr(sc, "max_num_seqs", 0) or 0)
    conc = getattr(cc, "kv_cache_max_concurrency", None)
    src = "cache_config.kv_cache_max_concurrency" if conc is not None else None
    groups: Any = None
    recount = None
    kvcfg = None
    with contextlib.suppress(Exception):
        kvcfg = eng.engine_core.engine_core.scheduler.kv_cache_config
    if kvcfg is not None:
        try:
            from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

            groups = []
            for g in kvcfg.kv_cache_groups:
                spec = g.kv_cache_spec
                per = -(-int(spec.max_memory_usage_bytes(vc)) // int(spec.page_size_bytes))
                groups.append(
                    {
                        "spec": type(spec).__name__,
                        "layers": len(getattr(g, "layer_names", None) or ()),
                        "blocks_per_request": per,
                        "host_resident": bool(getattr(g, "host_resident", False)),
                    }
                )
            recount = float(get_max_concurrency_for_kv_cache_config(vc, kvcfg))
            if conc is None:
                conc, src = recount, "get_max_concurrency_for_kv_cache_config(the scheduler's kv_cache_config)"
        except Exception as e:  # noqa: BLE001 - recorded; the stored value (if any) still decides
            groups = f"error {type(e).__name__}: {e}"[:200]
    ok = conc is not None and need > 0 and float(conc) >= need
    return {
        "ok": bool(ok),
        "mode": mode,
        "max_concurrency": None if conc is None else round(float(conc), 3),
        "max_num_seqs": need,
        "headroom_requests": None if conc is None else round(float(conc) - need, 3),
        "max_model_len": getattr(mc, "max_model_len", None),
        "num_gpu_blocks": getattr(cc, "num_gpu_blocks", None),
        "block_size": getattr(cc, "block_size", None),
        "kv_cache_size_tokens": getattr(cc, "kv_cache_size_tokens", None),
        "kv_cache_memory_bytes": getattr(cc, "kv_cache_memory_bytes", None),
        "groups": groups,
        "blocks_per_request": (
            sum(x["blocks_per_request"] for x in groups if not x["host_resident"]) if isinstance(groups, list) else None
        ),
        "recount_concurrency": None if recount is None else round(recount, 3),
        "source": src or "unavailable",
    }


def hbm_stats() -> dict:
    """The device allocator's statistics for this process (bytes; {} where unavailable)."""
    out: dict[str, Any] = {}
    try:
        from verl.utils.device import get_torch_device

        dm = get_torch_device()
        for key in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
            fn = getattr(dm, key, None)
            if fn is not None:
                with contextlib.suppress(Exception):
                    out[key] = int(fn())
        fn = getattr(dm, "mem_get_info", None)
        if fn is not None:
            with contextlib.suppress(Exception):
                free, limit = fn()
                out["free"], out["limit"] = int(free), int(limit)
    except Exception:  # noqa: BLE001
        pass
    with contextlib.suppress(Exception):
        import torch

        stats = torch.accelerator.memory_stats()  # type: ignore[attr-defined]
        for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit", "largest_alloc_size", "bytes_reserved"):
            if k in stats:
                out[f"pjrt.{k}"] = int(stats[k])
    return out


def log_hbm(phase: str, rank: int) -> dict:
    st = hbm_stats()
    gib = {k: round(v / 2**30, 3) for k, v in st.items() if isinstance(v, int)}
    log(f"hbm {phase} rank {rank}: {json.dumps(gib, sort_keys=True)}")
    return st


# ------------------------------------------------------------------------------------------------ the replica


def _replica_class():
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm import TPUvLLMReplica

    class TPUInProcReplica(TPUvLLMReplica):
        """Hybrid replica whose server is the trainer worker actor itself (``inproc://<rank>``): verl's load balancer
        sends each request straight to the trainer rank that owns the engine. No sleep mode: sleep / wake_up / KV
        release are no-ops (logged once per call kind)."""

        _noop_logged: set = set()

        def _noop(self, what: str) -> None:
            if what not in TPUInProcReplica._noop_logged:
                TPUInProcReplica._noop_logged.add(what)
                log(
                    f"replica {self.replica_rank}: {what} is a no-op in process (no sleep mode; weights and KV stay "
                    "resident)"
                )

        async def launch_servers(self):
            if not (len(self.workers) == self.world_size == 1):
                raise RuntimeError(f"the in-process replica needs one worker, got {len(self.workers)}")
            worker = self.workers[0]
            info = await worker.inproc_server_info.remote()
            if not info.get("engine_ready") or info.get("replica_rank") != self.replica_rank:
                raise RuntimeError(f"replica {self.replica_rank}: worker reports {info}")
            self.servers = [worker]
            self._server_handle = worker
            self._server_address = f"inproc://{self.replica_rank}"
            log(
                f"replica {self.replica_rank}: server = trainer worker rank {info['rank']} pid {info['pid']}; address "
                f"{self._server_address}"
            )

        async def sleep(self):
            self._noop("sleep")

        async def wake_up(self):
            self._noop("wake_up")

        async def release_kv_cache(self):
            self._noop("release_kv_cache")

        async def resume_kv_cache(self):
            self._noop("resume_kv_cache")

        async def clear_kv_cache(self):
            self._noop("clear_kv_cache")

        async def start_profile(self, **kwargs):
            self._noop("start_profile")

        async def stop_profile(self):
            self._noop("stop_profile")

        async def snapshot(self):
            return {"inproc": True, "replica_rank": self.replica_rank}

        async def abort_all_requests(
            self, reject_request: bool = False, reset_prefix_cache: bool = True, abort_only: bool = False
        ):
            r = await self.servers[0].abort_all_requests.remote(
                reject_request=reject_request, reset_prefix_cache=reset_prefix_cache, abort_only=abort_only
            )
            return {
                "aborted_count": r.get("aborted_count", 0),
                "request_ids": r.get("request_ids", []),
                "server_results": [r],
            }

        async def resume_generation(self):
            await self.servers[0].resume_generation.remote()

        async def abort_request(self, request_id: str):
            return {"aborted": False, "request_id": request_id, "error": "not supported in process"}

    return TPUInProcReplica


_REPLICA_CLS = None


def __getattr__(name: str):  # the replica class imports verl's server module: built on first use
    global _REPLICA_CLS
    if name == "TPUInProcReplica":
        if _REPLICA_CLS is None:
            _REPLICA_CLS = _replica_class()
        return _REPLICA_CLS
    raise AttributeError(name)
