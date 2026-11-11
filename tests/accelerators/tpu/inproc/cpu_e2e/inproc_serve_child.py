# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""verl's requests through the real in-process rollout on CPU (one fresh interpreter per case). Test code.

A gloo world of one trainer rank. ``TPUInProcRollout`` is built in this process as in ``inproc_arch_child.py``: verl's
``vLLMHttpServer.launch_server`` arguments, vLLM's real ``LLMEngine`` with ``InprocClient`` + ``UniProcExecutor``, the
real scheduler and output processor; the worker is ``inproc_serve_worker.ServeWorker`` (vLLM's own model class for a
tiny dense checkpoint, mocked device side). Requests go through the trainer worker's direct rollout methods that the
plugin's registration adds to verl's ``ActorRolloutRefWorker`` (``generate`` / ``abort_all_requests`` /
``resume_generation`` / ``inproc_server_info``), into verl's own server code on the in-process adapter and its pump.

Case (env ``INPROC_TEST_CASE``, JSON): ``{"model_dir", "mode": "serve" | "lazy" | "timeline"}``. ``lazy``: three
waves of concurrent requests, the self-check re-armed after each as every weight sync does; every final
``RequestOutput`` the pump hands to a request and every ``TokenOutput`` verl returns, recorded canonically (each float
as its 8 bytes), and the lazy output path's window counts per wave. ``timeline`` (``VERL_TPU_TIMELINE=1``):
concurrent requests and the timeline records the server writes (its request spans and the replica record), with the
first-token watch's counts. Prints one line ``RESULT <json>``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback

# run with ``python -I`` (no implicit script dir / PYTHONPATH): the test passes its import path explicitly
sys.dont_write_bytecode = True
sys.path[:0] = [p for p in os.environ.get("INPROC_TEST_PATH", "").split(os.pathsep) if p and p not in sys.path]

import inproc_arch_child as arch  # noqa: E402

KV = 64 * 2**20
PROMPT = [151644, 872, 198, 1000, 2000, 151645, 198]


def serve_config(max_logprobs: int = 2, response_length: int = 600):
    from verl.workers.config import RolloutConfig
    from verl.workers.config.rollout import CheckpointEngineConfig

    return RolloutConfig(
        name="vllm",
        mode="async",
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        max_num_seqs=16,
        max_model_len=1024,
        prompt_length=256,
        response_length=response_length,
        seed=1,
        engine_kwargs={
            "vllm": {
                "kv_cache_memory_bytes": KV,
                "max_logprobs": max_logprobs,
                "seed": None,
                "no_enable_prefix_caching": True,
                "worker_cls": "inproc_serve_worker.ServeWorker",
                "distributed_executor_backend": "uni",
            }
        },
        calculate_log_probs=True,
        enable_prefix_caching=False,
        free_cache_engine=False,
        load_format="safetensors",
        gpu_memory_utilization=0.8,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        dtype="bfloat16",
        enforce_eager=False,
        max_num_batched_tokens=4096,
        checkpoint_engine=CheckpointEngineConfig(backend="naive"),
    )


def params(i: int, max_tokens: int) -> dict:
    """request i: a third ask for the sampled token's log-prob (verl's rollout), a third for the top-2 head as well,
    a third for none"""
    sp = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": max_tokens, "ignore_eos": True}
    if i % 3 == 0:
        sp["logprobs"] = True
    elif i % 3 == 1:
        sp.update({"logprobs": True, "topk_log_probs": 2})
    return sp


def prompt_of(i: int) -> list:
    return PROMPT[:3] + [1000 + 37 * i, 2000 + i] + PROMPT[5:]


def build(case: dict):
    cap = arch._Lines()
    R, _ = arch.setup({}, cap)
    from verl.workers.config import HFModelConfig
    from verl.workers.engine_workers import ActorRolloutRefWorker

    r = R.TPUInProcRollout(serve_config(), HFModelConfig(path=case["model_dir"]), None)
    wk = object.__new__(ActorRolloutRefWorker)  # the trainer worker the replica's server is
    wk.rollout = r

    import ray

    class _Ctx:  # verl's vLLMHttpServer.generate names its trace lane with ray.get_runtime_context(): a test stub
        def get_actor_name(self):
            return "inproc-trainer-rank0"

        def get_job_id(self):
            return "inproc"

    ray.get_runtime_context = lambda: _Ctx()
    return r, wk, ActorRolloutRefWorker, cap


def main_serve(case: dict) -> dict:
    import inproc_serve_worker as SW
    from inproc_arch_worker import mock_token

    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary

    r, wk, W, cap = build(case)
    out: dict = {"engine_line": arch.pick(cap.lines, "in-process engine pid")}
    out["identity_line"] = arch.pick(cap.lines, "identity rank 0: ")
    out["construction"] = dict(SW.ServeWorker.construction)
    eng = r.server._inproc_engine
    out["engine"] = {"client": type(eng.engine_core).__name__, "executor": type(eng.model_executor).__name__}
    from vllm.distributed import parallel_state as ps

    out["tp_group"] = list(ps.get_tp_group().ranks)

    async def batch(n: int, tag: str, max_tokens=lambda i: 5 + 3 * i):
        outs = await asyncio.gather(
            *[W.generate(wk, prompt_of(i), params(i, max_tokens(i)), f"{tag}-{i}") for i in range(n)]
        )
        res = []
        for i, o in enumerate(outs):
            want = [mock_token(prompt_of(i), k) for k in range(max_tokens(i))]
            res.append(
                {
                    "tokens_ok": list(o.token_ids) == want,
                    "log_probs": None if o.log_probs is None else len(o.log_probs),
                    "stop": o.stop_reason,
                }
            )
        return res

    async def scenario():
        rec: dict = {}
        rec["info"] = await W.inproc_server_info(wk)
        rec["batch"] = await batch(8, "a")
        # abort_only: requests in flight leave the engine, admission stays open
        longs = [asyncio.ensure_future(W.generate(wk, prompt_of(i), params(i, 500), f"long-{i}")) for i in range(3)]
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(r.engine._collectors) == 3 and r.engine.steps > 3:
                break
        rec["abort_only"] = await W.abort_all_requests(wk, abort_only=True)
        aborted = await asyncio.gather(*longs)
        rec["abort_only_outputs"] = [{"stop": o.stop_reason, "tokens": len(o.token_ids)} for o in aborted]
        rec["engine_unfinished_after_abort"] = eng.has_unfinished_requests()
        rec["window_after_abort"] = r.engine.window_take()
        rec["after_abort"] = await batch(2, "b")
        # the default abort (closes admission) and the resume
        longs = [asyncio.ensure_future(W.generate(wk, prompt_of(i), params(i, 500), f"long2-{i}")) for i in range(2)]
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(r.engine._collectors) == 2 and r.engine.steps > 3:
                break
        rec["abort"] = await W.abort_all_requests(wk)
        rec["abort_outputs"] = [o.stop_reason for o in await asyncio.gather(*longs)]
        await W.resume_generation(wk)
        rec["after_resume"] = await batch(2, "c")
        return rec

    out.update(asyncio.run(scenario()))
    out["step_health"] = r.step_health()
    out["window_end"] = r.engine.window_take()
    out["collectives_world_gt1"] = dict(boundary.COLLECTIVES.big)
    out["collectives_size1"] = boundary.COLLECTIVES.small
    out["device_threads"] = len(boundary.DEVICE_THREADS.counts)
    out["failed_lines"] = [x for x in cap.lines if "FAILED" in x][:5]
    return out


def _canon(x):
    """a value with every float as its 8 bytes (hex), recursively; tensors and arrays as lists"""
    import struct

    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, float):
        return struct.pack("<d", x).hex()
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, list | tuple):
        return [_canon(v) for v in x]
    return x


def main_lazy(case: dict) -> dict:
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as eng_mod
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.lazy_output import lp_canon

    finals: dict = {}
    orig_put = eng_mod._Collector.put

    def put(self, output):  # every final RequestOutput the pump hands to a request
        if not self._fut.done():
            c = output.outputs[0] if output.outputs else None
            finals[self.external_id] = {
                "finished": output.finished,
                "num_cached_tokens": output.num_cached_tokens,
                "token_ids": None if c is None else list(c.token_ids),
                "cumulative_logprob": None if c is None else _canon(c.cumulative_logprob),
                "logprobs": None if c is None else repr(lp_canon(c.logprobs)),
                "finish_reason": None if c is None else str(c.finish_reason),
            }
        return orig_put(self, output)

    eng_mod._Collector.put = put
    r, wk, W, cap = build(case)
    lazy = r.server._inproc_lazy
    out: dict = {"lazy_state": {"installed": lazy.installed, "why_not": lazy.why_not}, "waves": []}

    async def wave(w: int):
        n = 12
        lengths = [20 + 10 * i + 7 * w for i in range(n)]
        outs = await asyncio.gather(
            *[W.generate(wk, prompt_of(i + 100 * w), params(i, lengths[i]), f"w{w}-{i}") for i in range(n)]
        )
        return [
            {
                "token_ids": list(o.token_ids),
                "log_probs": _canon(o.log_probs),
                "stop": o.stop_reason,
                "extra": _canon(o.extra_fields),
            }
            for o in outs
        ]

    for w in range(3):
        token_outputs = asyncio.run(wave(w))
        out["waves"].append({"token_outputs": token_outputs, "window": r._lazy_window()})
        lazy.arm_verify()  # as every weight sync does
    out["finals"] = finals
    out["failed_lines"] = [x for x in cap.lines if "FAILED" in x or "MISMATCH" in x][:5]
    return out


def main_timeline(case: dict) -> dict:
    from verl_hardware_plugin.accelerators.tpu.profilers import timeline

    events: list = []
    timeline._write = events.append  # the records the server writes (test: kept in memory)
    r, wk, W, cap = build(case)
    lengths = [30 + 10 * i for i in range(12)] + [1]

    async def wave():
        outs = await asyncio.gather(
            *[W.generate(wk, prompt_of(i), params(i, n), f"t-{i}") for i, n in enumerate(lengths)]
        )
        return [len(o.token_ids) for o in outs]

    out: dict = {"lengths": asyncio.run(wave())}
    fw = r.engine.first_watch
    out["watch"] = {"sink_set": fw.sink is not None, "stamped": fw.stamped, "late": fw.late, "pending": len(fw.pending)}
    out["lazy_installed"] = bool(getattr(r.server._inproc_lazy, "installed", False))
    out["spans"] = [e for e in events if e.get("name") == "request"]
    out["replica"] = [e for e in events if e.get("name") == "rollout_replica"]
    return out


MODES = {"serve": main_serve, "lazy": main_lazy, "timeline": main_timeline}


if __name__ == "__main__":
    case = json.loads(os.environ["INPROC_TEST_CASE"])
    try:
        res = MODES[case["mode"]](case)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"[:3000], "traceback": traceback.format_exc()[-6000:]}
    logging.shutdown()
    print("RESULT " + json.dumps(res, default=str), flush=True)
