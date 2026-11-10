# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Timeline prints of the SGLang rollout server on TPU: requests, engine pause/resume, weight-sync RPCs.

:class:`SGLangServerTimelineMixin` goes in front of upstream ``SGLangHttpServer`` in ``TPUSGLangHttpServer``'s bases.
Every hook is a plain pass-through unless ``VERL_TPU_TIMELINE`` is set (see
``verl_hardware_plugin.accelerators.tpu.profilers.timeline`` for the line format and the report that reads it).
"""

import time
from typing import Any, Callable, Optional

from verl_hardware_plugin.accelerators.tpu.profilers import timeline


class SGLangServerTimelineMixin:
    """Prints a timeline of an ``SGLangHttpServer`` when ``VERL_TPU_TIMELINE`` is set; a pass-through otherwise."""

    _tl_hooked = False
    tokenizer_manager: Any

    def _tl_on(self) -> bool:
        return self._tl_hooked and timeline.enabled()

    def _tl_group(self) -> str:
        return timeline.rollout_group(getattr(self, "replica_rank", 0))

    async def launch_server(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().launch_server(*args, **kwargs)  # type: ignore[misc]
        # Only the normalized leader owns TokenizerManager; follower hosts run SGLang schedulers headless.
        if timeline.enabled() and getattr(self, "tokenizer_manager", None) is not None:
            self._tl_paused_at: Optional[float] = None
            self._tl_hooked = True
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

    async def generate(self, prompt_ids, sampling_params, request_id, *args, **kwargs):
        if not self._tl_on():
            return await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)  # type: ignore[misc]
        t_submit = time.time()
        n_in = len(prompt_ids) if hasattr(prompt_ids, "__len__") else None
        output = None
        try:
            output = await super().generate(prompt_ids, sampling_params, request_id, *args, **kwargs)  # type: ignore[misc]
            return output
        finally:
            t_end = time.time()
            stop = getattr(output, "stop_reason", None) or ("abort" if output is None else "none")
            if stop == "aborted":
                stop = "abort"
            token_ids = getattr(output, "token_ids", None)
            timeline.emit(
                "request",
                t_submit,
                t_end - t_submit,
                group=self._tl_group(),
                lane=timeline.REQUESTS_LANE,
                admit=0.0,
                first=0.0,
                out=len(token_ids) if token_ids is not None else 0,
                stop=stop,
                wv=getattr(self, "global_steps", None),
                **{"in": n_in},
            )

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ):
        if not self._tl_on() or abort_only or getattr(self, "tokenizer_manager", None) is None:
            return await super().abort_all_requests(  # type: ignore[misc]
                reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
            )
        t_close = time.time()
        result = await super().abort_all_requests(  # type: ignore[misc]
            reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=abort_only
        )
        t_end = time.time()
        timeline.emit("abort", t_close, t_end - t_close, group=self._tl_group(), lane=timeline.SERVER_LANE)
        self._tl_paused_at = t_end
        return result

    async def resume_generation(self):
        if not self._tl_on() or getattr(self, "tokenizer_manager", None) is None:
            return await super().resume_generation()  # type: ignore[misc]
        with timeline.span("resume_engine", group=self._tl_group(), lane=timeline.SERVER_LANE):
            result = await super().resume_generation()  # type: ignore[misc]
        paused_at = getattr(self, "_tl_paused_at", None)
        if paused_at is not None:
            self._tl_paused_at = None
            timeline.emit(
                "paused", paused_at, time.time() - paused_at, group=self._tl_group(), lane=timeline.SERVER_LANE
            )
        return result

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
