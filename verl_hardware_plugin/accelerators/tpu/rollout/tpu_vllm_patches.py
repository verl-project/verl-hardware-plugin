# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for the TPU rollout.

The rollout runs on vLLM's ``mp`` executor and vllm-torchtpu's ``mp`` multi-host backend, which need no
patch to span several hosts. Each remaining patch fills a gap in vLLM (v0.29.0 - v0.32.0) and vllm-torchtpu
(9faafb17 - 732c4346) and can be dropped once upstream closes it:

* ``TPUWorker.reset_encoder_cache``: no-op. verl resets vLLM's caches after every weight update,
  and vllm-torchtpu's worker does not implement this call. Upstream fix: implement it.
* ``TPUModelRunner.gather_logprobs``: normalize log probabilities in FP32, avoiding BF16
  cancellation. Skipped when vllm-torchtpu already does it (#2024).
* ``TPUModelRunner.sample_from_logits``: redraw BF16 sampling noise in FP32, avoiding sampling
  bias. vllm-torchtpu #1747 draws the noise in FP32 at the call sites, so there the wrapper
  passes it through unchanged.
* ``VLLM_DISABLE_COMPILE_CACHE=1``: a reloaded compile-cache artifact can make vLLM run the model
  eagerly, which crashes libtpu (b/501165531). Upstream fix: that bug.
* ``multiprocessing.process.BaseProcess.__init__``: install these patches in the EngineCore process and
  in vLLM's worker processes, which do not import the plugin.
"""

import importlib.util
import inspect
import logging
import multiprocessing.process
import os

import torch

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _source_upcasts_to_fp32(fn) -> bool:
    """Whether ``fn``'s source already computes in FP32 (vllm-torchtpu >= #2024's ``gather_logprobs`` does)."""
    try:
        return "torch.float32" in inspect.getsource(inspect.unwrap(fn))
    except (OSError, TypeError):
        return False


def patch_tpu_logprobs() -> None:
    """Use FP32 log_softmax when reporting vllm-torchtpu token log probabilities."""
    if importlib.util.find_spec("vllm_torchtpu") is None:
        return

    from vllm.v1.outputs import LogprobsTensors
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if getattr(TPUModelRunner, "_verl_fp32_logprobs_patched", False):
        return
    if _source_upcasts_to_fp32(getattr(TPUModelRunner, "gather_logprobs", None)):
        # vllm-torchtpu #2024 normalizes the log probabilities in FP32 itself.
        TPUModelRunner._verl_fp32_logprobs_patched = True
        logger.info("vllm-torchtpu already computes log probabilities in FP32; logprob patch not needed.")
        return

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor, sampled_tokens: torch.Tensor) -> LogprobsTensors:
        token_ids = sampled_tokens.to(torch.int64)
        token_logits = logits.gather(-1, token_ids)
        token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
        # Match the trainer's log_softmax + gather calculation. Upcast BEFORE
        # normalization: BF16 (token_logit - logsumexp(logits)).float() loses
        # small log probabilities to cancellation before the final cast.
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        token_logprobs = log_probs.gather(-1, token_ids)

        max_logprobs = self.model_config.max_logprobs
        if max_logprobs > 0:
            topk_indices = torch.topk(logits, max_logprobs, dim=-1).indices
            topk_logprobs = log_probs.gather(-1, topk_indices)
            logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
            logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        else:
            logprob_token_ids = token_ids
            logprobs = token_logprobs

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )

    TPUModelRunner.gather_logprobs = gather_logprobs
    TPUModelRunner._verl_fp32_logprobs_patched = True
    logger.info("Applied TPU generator FP32 log_softmax log-probability patch.")


# Read only by the plain-Python wrapper, never by compiled code.
_fp32_sampling_noise_logged = False


# TODO: Remove once the minimum supported vllm-torchtpu includes #1747.
def patch_tpu_sampler() -> None:
    """Draw TPU sampling noise from FP32 random values to reduce BF16 rounding bias.

    Older vllm-torchtpu draws the Gumbel-Max uniforms ``u`` in the logits dtype (BF16); the wrapper
    redraws them in FP32 with the runner's sampling generator when available. vllm-torchtpu #1747
    draws ``u`` in FP32 at the call sites rather than inside ``sample_from_logits``, so the
    function's own source cannot tell the versions apart; the wrapper checks ``u.dtype`` at run
    time instead and passes FP32 noise through unchanged.
    Apply this patch before the TPU model runner is created.
    """
    if importlib.util.find_spec("vllm_torchtpu") is None:
        return

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if getattr(TPUModelRunner, "_verl_fp32_sampler_patched", False):
        return

    original = getattr(TPUModelRunner, "sample_from_logits", None)
    # Later vllm-torchtpu releases append parameters (e.g. ``topk_candidates``); only the leading
    # ones are used here and the rest are forwarded unchanged.
    expected = ["self", "logits", "temperatures", "u", "top_k", "top_p", "all_greedy"]
    try:
        parameters = list(inspect.signature(original).parameters) if callable(original) else None
    except (TypeError, ValueError):
        # A compile wrapper that hides its signature: nothing to check, assume the known layout.
        parameters = expected
    if parameters in (["args", "kwargs"], ["self", "args", "kwargs"]):
        parameters = expected
    if parameters is None or parameters[: len(expected)] != expected:
        logger.error(
            "TPU sampler FP32 patch NOT applied: TPUModelRunner.sample_from_logits has parameters %s, "
            "expected %s. Sampling noise stays BF16, which biases policy-gradient training.",
            parameters,
            expected,
        )
        return

    def sample_from_logits(self, logits, temperatures, u, top_k, top_p, all_greedy=False, *args, **kwargs):
        # The all-greedy path never reads ``u``; preserve its compiled graph.
        if not all_greedy and u.dtype != torch.float32:
            global _fp32_sampling_noise_logged
            if not _fp32_sampling_noise_logged:
                _fp32_sampling_noise_logged = True
                logger.warning(
                    "verl TPU sampler patch active: sampling noise is drawn in float32 instead of %s "
                    "(logged once per process).",
                    str(u.dtype).removeprefix("torch."),
                )
            u = torch.rand(
                u.shape,
                dtype=torch.float32,
                device=u.device,
                # None during precompile warm-up, where the call site also uses the global RNG.
                generator=getattr(self, "_sampling_generator", None),
            )
        return original(self, logits, temperatures, u, top_k, top_p, all_greedy, *args, **kwargs)

    # functools.wraps copies torch.compile's bookkeeping attributes, which can let
    # Dynamo unwrap straight to the compiled function and skip the redraw.
    sample_from_logits.__doc__ = getattr(original, "__doc__", None)
    sample_from_logits.__wrapped__ = original  # type: ignore[attr-defined]
    TPUModelRunner.sample_from_logits = sample_from_logits
    TPUModelRunner._verl_fp32_sampler_patched = True
    logger.info("Applied TPU generator FP32 sampling-noise patch.")


def patch_vllm_for_tpu() -> None:
    """Install the patches in this process. Later calls return at once."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    patch_multiprocessing_for_tpu()
    # vLLM can reload a degenerate AOT compile artifact (num_artifacts=0) that silently runs the
    # model eagerly; eager execution then hits the XLA:TPU unaligned-DUS CHECK (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    # Imported here: CPU tests and processes outside the rollout may lack vLLM or vllm-torchtpu.
    try:
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    patch_tpu_logprobs()
    patch_tpu_sampler()
    TPUWorker.reset_encoder_cache = _reset_encoder_cache
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


def _reset_encoder_cache(self) -> None:
    """``TPUWorker.reset_encoder_cache``, which verl calls after every weight update: a no-op.

    The encoder cache holds multimodal encoder outputs; text models have none to reset.
    """


class PickleableProcessWrapper:
    """Process target that runs ``patch_vllm_for_tpu`` in the child before ``target``.

    This is how the patches reach vLLM's EngineCore and its ``mp`` workers. Inside a Ray actor vLLM always
    uses the ``spawn`` start method, and a spawned child imports only the modules its pickled target names
    (``EngineCoreProc.run_engine_core``, which vllm-torchtpu replaces with its own wrapper, or
    ``WorkerProc.worker_main``), so without this wrapper it would never import the plugin.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """Wrap the target of every ``multiprocessing`` process started from now on in ``PickleableProcessWrapper``."""
    base_process = multiprocessing.process.BaseProcess
    if getattr(base_process, "_tpu_patched", False):
        return
    original_init = base_process.__init__

    def patched_init(self, group=None, target=None, *args, **kwargs):
        if target is not None:
            target = PickleableProcessWrapper(target)
        original_init(self, group, target, *args, **kwargs)

    base_process.__init__ = patched_init  # type: ignore[method-assign]
    base_process._tpu_patched = True  # type: ignore[attr-defined]
