# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The vLLM worker of the in-process serving CPU tests. Test code.

:class:`inproc_arch_worker.ArchModelWorker` (vLLM's own model class for the checkpoint, filled by its own loader; a
mocked device side), plus what the serving tests read:

* the top-k log-prob columns of every request that asks for them (the batch's width is 1 + the largest k it holds,
  as vLLM's sampler gives), with deterministic values;
* a record of the engine construction: at the end of ``load_model`` and at ``determine_available_memory`` /
  ``initialize_from_config`` / ``compile_or_warm_up_model`` (where vllm-torchtpu's profile run, KV-cache init and graph
  capture precompile on TPU), whether verl's sampling mask is on the served model, the grad mode and the engine scope;
* a device-time stand-in for long requests (``max_tokens`` above 300: 2 ms per step), so an abort finds them running.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from inproc_arch_worker import ArchModelWorker, mock_token
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

LONG_TOKENS, LONG_STEP_S = 300, 0.002


def topk_ids(tok: int, k: int) -> list:
    return [1000 + (tok * 7 + 13 * j) % 40000 for j in range(1, k + 1)]


def topk_lps(n_out: int, k: int) -> list:
    return [-1.0 - 0.125 * j - 0.01 * (n_out % 5) for j in range(1, k + 1)]


class ServeWorker(ArchModelWorker):
    construction: dict = {}

    def _record(self, where: str) -> None:
        from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import ENGINE_TLS

        model = getattr(getattr(self, "model_runner", None), "model", None)
        ServeWorker.construction[where] = {
            "mask_on_model": model is not None and "compute_logits" in model.__dict__,
            "grad": torch.is_grad_enabled(),
            "scope": ENGINE_TLS.kind,
        }

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._record("load_model")

    def determine_available_memory(self) -> int:
        self._record("determine_available_memory")
        return super().determine_available_memory()

    def initialize_from_config(self, kv_cache_config) -> None:
        self._record("initialize_from_config")
        return super().initialize_from_config(kv_cache_config)

    def compile_or_warm_up_model(self):
        self._record("compile_or_warm_up_model")
        return super().compile_or_warm_up_model()

    def execute_model(self, scheduler_output):
        for rid in scheduler_output.finished_req_ids:
            self.reqs.pop(rid, None)
        for nr in scheduler_output.scheduled_new_reqs:
            sp = nr.sampling_params
            self.reqs[nr.req_id] = {
                "prompt": list(nr.prompt_token_ids or []),
                "computed": int(nr.num_computed_tokens),
                "out": 0,
                "k": None if sp is None or sp.logprobs is None else int(sp.logprobs),
                "long": sp is not None and (sp.max_tokens or 0) > LONG_TOKENS,
            }
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            if rid in self.reqs:
                self.reqs[rid]["computed"] = int(cached.num_computed_tokens[i])
        scheduled = list(scheduler_output.num_scheduled_tokens.items())
        if any(self.reqs[r]["long"] for r, _ in scheduled if r in self.reqs):
            time.sleep(LONG_STEP_S)
        width = 1 + max([self.reqs[r]["k"] or 0 for r, _ in scheduled if r in self.reqs] or [0])
        req_ids, sampled, lp_ids, lps, ranks = [], [], [], [], []
        for rid, n in scheduled:
            st = self.reqs[rid]
            after = st["computed"] + int(n)
            req_ids.append(rid)
            if after >= len(st["prompt"]):
                tok = mock_token(st["prompt"], st["out"])
                st["out"] += 1
                sampled.append([tok])
                ids = [tok] + topk_ids(tok, width - 1)
                vals = [-0.25 - 0.01 * (st["out"] % 7)] + topk_lps(st["out"], width - 1)
            else:
                sampled.append([])
                ids, vals = [0] * width, [0.0] * width
            lp_ids.append(ids)
            lps.append(vals)
            ranks.append(1)
            st["computed"] = after
        logprobs = None
        if any(self.reqs[r]["k"] is not None for r in req_ids):
            logprobs = LogprobsLists(
                np.array(lp_ids, dtype=np.int64), np.array(lps, dtype=np.float32), np.array(ranks, dtype=np.int64)
            )
        self._pending = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=logprobs,
        )
        return None
