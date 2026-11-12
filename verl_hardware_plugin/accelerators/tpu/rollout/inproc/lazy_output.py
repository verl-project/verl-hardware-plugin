# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The lazy final-only output path of the in-process engine.

In process, vLLM's output processor runs on the stepping thread, on the decode's critical path. Every rollout request
is FINAL_ONLY (the adapter's ``add_request``) with detokenization off, so for a request that is neither prefilling nor
finishing, vLLM's per-output loop only appends the new token ids to the plain detokenizer, records the weight version,
turns the step's sample logprobs into ``Logprob`` objects and asks ``make_request_output``, which returns None for
FINAL_ONLY. Here the first two run as before, the logprob rows are kept on the request state, and the last is skipped.
Every other output (the request's first and final one, an abort, a stop string, n > 1, CUMULATIVE / DELTA output,
routed experts, sampling masks, prompt logprobs, a queue) goes to vLLM's own ``process_outputs``, after the request's
deferred rows were applied in order: the final ``RequestOutput`` is built by vLLM's own code from the same state, bit
for bit the one it would have built. With vLLM's stats on, every step goes the original way.

It is installed only when :func:`.lazy_output_compat.check` confirms vLLM's output-processing code is the reviewed,
trace-tested code (``VERL_TPU_INPROC_LAZY_OUTPUT=0`` opts out). A self-check guards it at run time: after
:meth:`LazyFinalOutputs.arm_verify` (the build, every sync) the next request that defers rows also runs vLLM's eager
update on a reference processor and is compared bitwise at its finish; a difference fails the engine step.
"""

from __future__ import annotations

import os
import struct
from collections import Counter
from collections.abc import Mapping
from typing import Any, Optional

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import lazy_output_compat
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import register_switch, register_switch_check, warn

LAZY_OUTPUT_ENV = register_switch("VERL_TPU_INPROC_LAZY_OUTPUT")  # "1" (default): on when the guard passes; "0": off
LAZY_VERIFY_N = 1  # requests per sync window whose deferred logprobs are also processed eagerly and compared
LAZY_VERIFY_ROWS = 256  # ... over their first 256 deferred steps
_LAZY_Q = "_inproc_lazy_lp"  # RequestState attribute: the request's deferred LogprobsLists, in step order
_LAZY_V = (
    "_inproc_lazy_verify"  # RequestState attribute: (eager reference processor, first deferred position, id, rows)
)


def lazy_output_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    raw = (env.get(LAZY_OUTPUT_ENV) or "").strip()
    if raw not in ("", "0", "1"):
        raise RuntimeError(f"{LAZY_OUTPUT_ENV}={raw!r}: expected '0' or '1'")
    return raw != "0"


register_switch_check(lazy_output_enabled)


def _fbits(x: Any) -> bytes:
    return struct.pack("<d", float(x))


def lp_canon(container: Any, start: int = 0) -> list:
    """A request's sample logprobs from position ``start`` on, canonical and bitwise (each float as its 8 bytes):
    vLLM's list[dict[token id, Logprob]] or its FlatLogprobs."""
    if container is None:
        return []
    if isinstance(container, list):
        return [
            None if d is None else [(t, _fbits(v.logprob), v.rank, v.decoded_token) for t, v in d.items()]
            for d in container[start:]
        ]
    return [
        [
            (container.token_ids[i], _fbits(container.logprobs[i]), container.ranks[i], container.decoded_tokens[i])
            for i in range(container.start_indices[p], container.end_indices[p])
        ]
        for p in range(start, len(container))
    ]


class LazyFinalOutputs:
    """vLLM's ``OutputProcessor.process_outputs`` with the per-step sample-logprob work of FINAL_ONLY requests deferred
    to the request's finish (see the module docstring). ``take()`` returns and resets the counts of the window since
    the last sync (the per-sync health line)."""

    def __init__(self, eng: Any, environ: Optional[Mapping[str, str]] = None):
        self.enabled = lazy_output_enabled(environ)
        self.installed = False
        self.why_not: Optional[str] = None
        self.verify_left = 0
        self.verify_rows = LAZY_VERIFY_ROWS
        self.mismatch: Optional[str] = None
        self.n_deferred = 0  # the window's outputs that took the deferred path / vLLM's (hot counters: plain ints)
        self.n_vllm = 0
        self.win: Counter = Counter()
        self.total: Counter = Counter()
        proc = getattr(eng, "output_processor", None)
        if not self.enabled:
            self.why_not = f"{LAZY_OUTPUT_ENV}=0"
            return
        if (
            proc is None
            or not callable(getattr(proc, "process_outputs", None))
            or not isinstance(getattr(proc, "request_states", None), dict)
        ):
            self.why_not = "no vLLM OutputProcessor"
            return
        compat = lazy_output_compat.check(proc=proc)  # never raises
        if not compat.ok:
            self.why_not = "vLLM output processing differs from the reviewed code: " + compat.summary()
            warn(f"lazy final-only output path off: {self.why_not}")
            return
        try:
            from vllm.logprobs import Logprob, create_sample_logprobs
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.detokenizer import IncrementalDetokenizer
            from vllm.v1.engine.logprobs import LogprobsProcessor
            from vllm.v1.engine.output_processor import OutputProcessorOutput
        except Exception as e:  # noqa: BLE001
            self.why_not = f"vLLM imports: {type(e).__name__}: {e}"[:200]
            return
        self.proc = proc
        self.orig = proc.process_outputs
        self._final_only = RequestOutputKind.FINAL_ONLY
        self._plain = IncrementalDetokenizer
        self._Logprob = Logprob
        self._new_samples = create_sample_logprobs
        self._LPP = LogprobsProcessor
        self._Out = OutputProcessorOutput
        proc.process_outputs = self.process_outputs
        self.installed = True

    def arm_verify(self, n: int = LAZY_VERIFY_N) -> None:
        self.verify_left = int(n) if self.installed else 0

    def process_outputs(
        self, engine_core_outputs, engine_core_timestamp=None, iteration_stats=None, weight_version=None
    ):
        states = self.proc.request_states
        if iteration_stats is not None:  # vLLM's stats on: every output its own way, each request's deferred rows first
            for o in engine_core_outputs:
                st = states.get(o.request_id)
                if st is not None and _LAZY_Q in st.__dict__:
                    self._flush(st)
            self.win["stats_steps"] += 1
            return self.orig(engine_core_outputs, engine_core_timestamp, iteration_stats, weight_version)
        final_only, plain = self._final_only, self._plain
        slow = None
        nfast = 0
        for o in engine_core_outputs:
            rid = o.request_id
            toks = o.new_token_ids
            st = states.get(rid)
            if st is None:
                continue  # an output of an already-aborted request: vLLM ignores it too
            if (
                o.finish_reason is None
                and not st.is_prefilling
                and st.output_kind is final_only
                and st.queue is None
                and st.parent_req is None
                and not st.streaming_input
                and type(st.detokenizer) is plain
                and o.pooling_output is None
                and o.routed_experts is None
                and o.new_sampling_mask is None
                and o.new_prompt_logprobs_tensors is None
                and o.prompt_token_id_logprobs is None
                and o.spec_decode_metrics is None
                and o.kv_transfer_params is None
            ):
                if toks:
                    if weight_version is not None:
                        st.track_weight_version(weight_version)
                    st.detokenizer.token_ids.extend(toks)
                lp = o.new_logprobs
                if lp is not None:
                    d = st.__dict__
                    q = d.get(_LAZY_Q)
                    if q is None:
                        q = d[_LAZY_Q] = []
                        if self.verify_left > 0:
                            self._arm_reference(st)
                    q.append(lp)
                    ref = d.get(_LAZY_V)
                    if ref is not None and len(q) <= ref[3]:
                        ref[0]._update_sample_logprobs(lp)  # the eager reference: vLLM's own code on this step's arrays
                nfast += 1
                continue
            if _LAZY_Q in st.__dict__:
                self._flush(st)
            if slow is None:
                slow = []
            slow.append(o)
        self.n_deferred += nfast
        if slow is None:
            return self._Out([], [])
        self.n_vllm += len(slow)
        return self.orig(slow, engine_core_timestamp, iteration_stats, weight_version)

    def _arm_reference(self, st: Any) -> None:
        lpp = st.logprobs_processor
        ref = self._LPP(
            tokenizer=lpp.tokenizer,
            logprobs=self._new_samples(not isinstance(lpp.logprobs, list)),
            prompt_logprobs=None,
            cumulative_logprob=lpp.cumulative_logprob,
            num_logprobs=lpp.num_logprobs,
            num_prompt_logprobs=None,
        )
        st.__dict__[_LAZY_V] = (ref, len(lpp.logprobs), st.request_id, self.verify_rows)
        self.verify_left -= 1
        self.win["verified_requests"] += 1

    def _flush(self, st: Any) -> None:
        """The request's deferred rows into its LogprobsProcessor, in step order (before vLLM's own code touches the
        request again). The common case (logprobs=0, no tokenizer, list container, one position per step: only column
        0 counts) in one pass with the same arithmetic; anything else through vLLM's ``_update_sample_logprobs``."""
        d = st.__dict__
        q = d.pop(_LAZY_Q, None)
        ref = d.pop(_LAZY_V, None)
        if not q:
            return
        lpp = st.logprobs_processor
        out = lpp.logprobs
        self.win["flushes"] += 1
        m = min(len(q), ref[3]) if ref is not None else -1  # the self-check covers the first m deferred lists
        cum_m = None
        if lpp.num_logprobs == 0 and lpp.tokenizer is None and type(out) is list:
            L = self._Logprob
            cum = lpp.cumulative_logprob
            one = 0
            for i, lists in enumerate(q):
                ids, lps, rks = lists[0], lists[1], lists[2]
                if lps.size == 1 and ids.size == 1:  # one position, one column (LogprobsLists' fields are numpy arrays)
                    x = lps.item()
                    cum += x
                    out.append({ids.item(): L(x, rks.item(), None)})
                    one += 1
                elif lps.ndim == 2 and lps.shape[0] == 1 and rks.size == 1:
                    # one position, several columns (the batch held a top-k request): at num_logprobs 0 vLLM
                    # keeps column 0 only, the same values tolist() gives
                    x = float(lps[0, 0])
                    cum += x
                    out.append({int(ids[0, 0]): L(x, int(rks[0]), None)})
                    one += 1
                else:  # several positions: vLLM's own per-position code, the running sum carried through it
                    lpp.cumulative_logprob = cum
                    lpp._update_sample_logprobs(lists)
                    cum = lpp.cumulative_logprob
                if i + 1 == m:
                    cum_m = cum
            lpp.cumulative_logprob = cum
            self.win["rows_one_pass"] += one
            self.win["lists_vllm"] += len(q) - one
        else:
            for i, lists in enumerate(q):
                lpp._update_sample_logprobs(lists)
                if i + 1 == m:
                    cum_m = lpp.cumulative_logprob
            self.win["lists_vllm"] += len(q)
        if ref is not None:
            self._check(st, ref, cum_m)

    def _check(self, st: Any, ref: tuple, cum_m: Optional[float]) -> None:
        eager, start, rid, _ = ref
        b = lp_canon(eager.logprobs, 0)
        a = lp_canon(st.logprobs_processor.logprobs, start)[: len(b)]
        if a == b and cum_m is not None and _fbits(cum_m) == _fbits(eager.cumulative_logprob):
            self.win["verified_equal"] += 1
            return
        first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        self.mismatch = (
            f"request {rid}: {len(a)} deferred positions vs {len(b)} eager, first difference at deferred position "
            f"{first}; "
            f"running sum {cum_m!r} vs {eager.cumulative_logprob!r}"
        )
        warn(
            f"LAZY OUTPUT MISMATCH: {self.mismatch}: the deferred sample logprobs differ from vLLM's per-step result; "
            "failing the step"
        )
        raise RuntimeError(f"lazy final-only output path mismatch ({self.mismatch})")

    def take(self) -> dict:
        self.win["deferred_outputs"] += self.n_deferred
        self.win["vllm_outputs"] += self.n_vllm
        self.n_deferred = self.n_vllm = 0
        out = {"installed": self.installed, **dict(self.win)}
        if not self.installed:
            out["why_not"] = self.why_not
        self.total.update(self.win)
        self.win = Counter()
        return out
