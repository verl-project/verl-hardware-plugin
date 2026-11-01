# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Timeline prints of the vLLM rollout server on TPU: requests, gate drain, engine pause, weight-sync RPCs.

:class:`vLLMServerTimelineMixin` goes in front of upstream ``vLLMHttpServer`` in ``TPUvLLMHttpServer``'s bases and
wraps the server's own hooks, so verl needs no change. Every hook is a plain pass-through unless
``VERL_TPU_TIMELINE`` is set (see ``verl_hardware_plugin.accelerators.tpu.profilers.timeline`` for the line format and
the report that reads it).

When enabled, ``launch_server`` additionally wraps the methods of this server's ``AsyncLLM`` instance:

* ``add_request`` / ``generate``: records when a request yields its first output (queueing and prefill done) and its
  finish reason.
* ``pause_generation``: records when ``abort_all_requests`` stops waiting for admitted requests (end of ``drain``).
"""

import time
from typing import Any, Callable, Optional

from verl_hardware_plugin.accelerators.tpu.profilers import timeline


class vLLMServerTimelineMixin:
    """Prints a timeline of a ``vLLMHttpServer`` when ``VERL_TPU_TIMELINE`` is set; a pass-through otherwise."""

    _tl_hooked = False
    engine: Any

    def _tl_on(self) -> bool:
        return self._tl_hooked and timeline.enabled()

    def _tl_group(self) -> str:
        return timeline.rollout_group(getattr(self, "replica_rank", 0))

    async def launch_server(self, *args, **kwargs):
        result = await super().launch_server(*args, **kwargs)
        # Only node rank 0 owns the AsyncLLM engine; the other actors of a multi-node replica run vLLM headless.
        if timeline.enabled() and getattr(self, "node_rank", 0) == 0 and getattr(self, "engine", None) is not None:
            self._tl_install_engine_hooks()
            config = getattr(self, "config", None)
            timeline.meta(
                "rollout_replica",
                group=self._tl_group(),
                max_num_seqs=getattr(config, "max_num_seqs", None),
                tp=getattr(config, "tensor_model_parallel_size", None),
                nnodes=getattr(self, "nnodes", None),
                host=getattr(self, "_server_address", None),
            )
            timeline.start_hbm_sampler(group=self._tl_group(), lane=timeline.HBM_LANE)
        return result

    def _tl_install_engine_hooks(self) -> None:
        engine = self.engine
        self._tl_admit: dict[str, float] = {}
        self._tl_first: dict[str, float] = {}
        self._tl_finish: dict[str, Optional[str]] = {}
        self._tl_pause_called: Optional[float] = None
        self._tl_paused_at: Optional[float] = None
        first_output, finish_reason = self._tl_first, self._tl_finish
        engine_generate = getattr(engine, "generate", None)
        engine_add_request = getattr(engine, "add_request", None)
        engine_pause: Callable = engine.pause_generation

        def _record_output(request_id: str, output: Any) -> Any:
            outputs = getattr(output, "outputs", None)
            if outputs is not None:
                first_output.setdefault(request_id, time.time())
                finish_reason[request_id] = outputs[0].finish_reason if outputs else None
            return output

        async def generate(*args, **kwargs):
            request_id = kwargs.get("request_id")
            last = None
            try:
                async for output in engine_generate(*args, **kwargs):
                    if last is None:
                        first_output[request_id] = time.time()
                    last = output
                    yield output
            finally:
                outputs = getattr(last, "outputs", None)
                finish_reason[request_id] = outputs[0].finish_reason if outputs else None

        async def add_request(request_id, *args, **kwargs):
            queue = await engine_add_request(request_id, *args, **kwargs)
            orig_get_nowait = getattr(queue, "get_nowait", None)
            orig_get = getattr(queue, "get", None)
            if callable(orig_get_nowait) and callable(orig_get):
                queue.get_nowait = lambda: _record_output(request_id, orig_get_nowait())

                async def get():
                    return _record_output(request_id, await orig_get())

                queue.get = get
            return queue

        async def pause_generation(*args, **kwargs):
            self._tl_pause_called = time.time()
            return await engine_pause(*args, **kwargs)

        # Instance attributes shadow the AsyncLLM methods; the upstream server looks them up on every call.
        if callable(engine_generate):
            engine.generate = generate
        if callable(engine_add_request):
            engine.add_request = add_request
        engine.pause_generation = pause_generation
        self._tl_hooked = True

    async def _park_until_admitted(self, request_id: str):
        rejected = await super()._park_until_admitted(request_id)  # type: ignore[misc]
        if rejected is None and self._tl_on():
            self._tl_admit[request_id] = time.time()
        return rejected

    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        if not self._tl_on():
            return await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)
        t_submit = time.time()
        n_in = len(prompt_ids) if hasattr(prompt_ids, "__len__") else None
        output = None
        try:
            output = await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)
            return output
        finally:
            t_end = time.time()
            t_admit = self._tl_admit.pop(request_id, None)
            t_first = self._tl_first.pop(request_id, None)
            finish = self._tl_finish.pop(request_id, None)
            if output is None or getattr(output, "stop_reason", None) == "aborted" or finish == "abort":
                stop = "abort"
            else:
                stop = finish or getattr(output, "stop_reason", None) or "none"
            token_ids = getattr(output, "token_ids", None)
            admit = (t_admit if t_admit is not None else t_end) - t_submit
            first = (t_first if t_first is not None else t_end) - t_submit
            timeline.emit(
                "request",
                t_submit,
                t_end - t_submit,
                group=self._tl_group(),
                lane=timeline.REQUESTS_LANE,
                admit=round(admit, 4),
                first=round(max(first, admit), 4),
                out=len(token_ids) if token_ids is not None else 0,
                stop=stop,
                wv=getattr(self, "global_steps", None),
                **{"in": n_in},
            )

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ) -> dict[str, Any]:
        if not self._tl_on() or abort_only or getattr(self, "node_rank", 0) != 0:
            return await super().abort_all_requests(  # type: ignore[misc]
                reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
            )
        self._tl_pause_called = None
        t_close = time.time()
        result = await super().abort_all_requests(  # type: ignore[misc]
            reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
        )
        t_end = time.time()
        t_pause = self._tl_pause_called if self._tl_pause_called is not None else t_end
        group = self._tl_group()
        # drain: the gate is closed, but the engine keeps running until every admitted request has its first token.
        timeline.emit("drain", t_close, t_pause - t_close, group=group, lane=timeline.SERVER_LANE)
        aborted = result.get("aborted_count") if isinstance(result, dict) else None
        timeline.emit("abort", t_close, t_end - t_close, group=group, lane=timeline.SERVER_LANE, aborted=aborted)
        self._tl_paused_at = t_pause
        return result

    async def resume_engine_generation(self):
        if not self._tl_on():
            return await super().resume_engine_generation()
        with timeline.span("resume_engine", group=self._tl_group(), lane=timeline.SERVER_LANE):
            return await super().resume_engine_generation()

    async def open_submission_gate(self):
        await super().open_submission_gate()
        paused_at = getattr(self, "_tl_paused_at", None)
        if paused_at is not None and self._tl_on():
            self._tl_paused_at = None
            timeline.emit(
                "paused", paused_at, time.time() - paused_at, group=self._tl_group(), lane=timeline.SERVER_LANE
            )

    async def set_global_steps(self, global_steps: int):
        await super().set_global_steps(global_steps)  # type: ignore[misc]
        if self._tl_on():
            timeline.instant("weights", group=self._tl_group(), lane=timeline.SERVER_LANE, wv=global_steps)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if not self._tl_on() or not isinstance(method, str):
            return await super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)  # type: ignore[misc]
        with timeline.span("rpc." + method, group=self._tl_group(), lane=timeline.SERVER_LANE):
            return await super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs)  # type: ignore[misc]
