# Copyright (c) 2026 Google LLC. All rights reserved.
# Numerical patches adapted from verl: Copyright 2024-2025 BAAI and Google LLC.
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
* ``ApplyRotaryEmb`` and ``SiluAndMul``: compute TPU rotations and SiLU in FP32, retaining
  CUDA activation rounding. RoPE also avoids a concat that can crash TPU fusion (b/501165531).
* ``VERL_TPU_VLLM_ATTN_FP32_STATS=1`` and ``VERL_TPU_VLLM_RMSNORM_CUDA_ORDER=1``: opt in
  to FP32 attention statistics and CUDA RMSNorm rounding without changing BF16 model weights.
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
import sys

import torch

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _tpu_sign(x: torch.Tensor) -> torch.Tensor:
    """Builds [[-1], [1]] with the dtype/device of ``x``.

    This is deliberately built from tensor ops on every call instead of being
    memoized in a module-level dict. torch.compile specializes reads of Python
    globals into the serialized AOT prologue, so a cached constant resurfaces as
    a ``KeyError`` when the compiled artifact is replayed in a fresh process.
    XLA constant-folds this away, so there is no runtime cost.
    """
    return (torch.arange(2, dtype=x.dtype, device=x.device) * 2.0 - 1.0).unsqueeze(-1)


def _tpu_rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """cat((-x2, x1), -1) without a concat. Bitwise identical under IEEE754."""
    half = x.shape[-1] // 2
    swapped = x.unflatten(-1, (2, half)).flip(-2)
    return (swapped * _tpu_sign(x)).flatten(-2)


def _tpu_widen(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., half] -> [..., 1, 2 * half], i.e. cat((t, t), -1).unsqueeze(-2)."""
    lead = t.shape[:-1]
    half = t.shape[-1]
    return t.unsqueeze(-2).expand(*lead, 2, half).reshape(*lead, 2 * half).unsqueeze(-2).to(dtype)


# TODO: Remove this workaround once upstream vLLM PR #56879 is merged and released.
# https://github.com/vllm-project/vllm/pull/56879
def patch_tpu_rotary_emb():
    """Patches vLLM ApplyRotaryEmb and rotate_neox with concat-free implementations.

    On Google TPU, torch.cat along the last dimension lowers to a strided DynamicUpdateSlice (DUS).
    When fused with elementwise ops, it fails the unaligned DUS predicate inside the XLA:TPU
    fusion emitter (b/501165531) and causes a fatal crash:
        Check failed: fusion_util::IsFusibleUnalignedDUS(...)

    This patch replaces the concatenation with an unflatten + flip formulation that is
    bitwise identical and emits no DUS instructions.
    """
    try:
        import vllm.model_executor.layers.rotary_embedding.common as rotary_common
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_verl_tpu_rotary_patched", False):
            return

        def patched_forward_static(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> torch.Tensor:
            # Match CUDA: compute the rotation in FP32, then round once to the input dtype.
            if x.device.type == "tpu" and x.dtype in (torch.bfloat16, torch.float16):
                enable_fp32_compute = True
            origin_dtype = x.dtype
            if enable_fp32_compute:
                x = x.float()

            if is_neox_style:
                cos_f = _tpu_widen(cos, x.dtype)
                sin_f = _tpu_widen(sin, x.dtype)
                output = x * cos_f + _tpu_rotate_neox(x) * sin_f
            else:
                cos = cos.unsqueeze(-2).to(x.dtype)
                sin = sin.unsqueeze(-2).to(x.dtype)
                x1 = x[..., ::2]
                x2 = x[..., 1::2]
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                output = torch.stack((o1, o2), dim=-1).flatten(-2)

            if enable_fp32_compute:
                output = output.to(origin_dtype)
            return output

        ApplyRotaryEmb.forward_static = staticmethod(patched_forward_static)
        rotary_common.rotate_neox = _tpu_rotate_neox
        ApplyRotaryEmb._verl_tpu_rotary_patched = True
        logger.info("Successfully applied TPU concat-free RoPE patch to vLLM.")
    except Exception as e:
        logger.warning(f"Failed to apply TPU rotary embedding patch to vLLM: {e}")


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


ATTN_FP32_STATS_ENV = "VERL_TPU_VLLM_ATTN_FP32_STATS"
_RPA_KERNEL_MODULE = "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel"
_RPA_ALIAS_MODULES = ("vllm_torchtpu.layers.core.attention_interface", "vllm_torchtpu.layers.adapter.attention")


def patch_tpu_attention_fp32_stats() -> bool:
    """Opt-in (``VERL_TPU_VLLM_ATTN_FP32_STATS=1``): fp32 softmax statistics in the ragged-paged-attention kernel.

    vllm-torchtpu calls the v3 Pallas kernel with ``out_dtype=None``; on bf16 queries that keeps the running
    max/sum, the exp() probabilities and the cross-KV-block accumulator in bf16, where CUDA FlashAttention keeps
    them fp32. Measured on Qwen3-4B-Base (2026-10-09 capture), vLLM-TPU then carries ~1.25x the per-token
    log-prob error of the TPU trainer and of vLLM-CUDA against an fp32 reference. With the flag the kernel runs
    with ``out_dtype=float32`` and its output is cast back to the query dtype, so only the attention-internal
    rounding changes. Returns True when the patch is installed.
    """
    if os.environ.get(ATTN_FP32_STATS_ENV) != "1":
        return False
    try:
        import jax.numpy as jnp

        kernel = importlib.import_module(_RPA_KERNEL_MODULE)
    except ImportError as exc:
        logger.debug("Skipping the fp32 attention statistics patch (jax / vllm_torchtpu not installed): %s", exc)
        return False
    original = kernel.ragged_paged_attention
    if getattr(original, "_verl_fp32_stats", False):
        return True

    def ragged_paged_attention(*args, **kwargs):
        # The kernel lays out its output buffer in the query dtype, so fp32 statistics need fp32 queries
        # (bf16 queries pack two values per word and the fp32 result no longer fits that layout). Keys,
        # values and the KV cache stay bf16; QK^T/softmax/PV accumulate in fp32; the output returns to bf16.
        if args:
            queries, args = args[0], args[1:]
        else:
            queries = kwargs.pop("queries")
        query_dtype = queries.dtype
        kwargs["out_dtype"] = jnp.float32
        result = original(queries.astype(jnp.float32), *args, **kwargs)
        if isinstance(result, tuple):
            return (result[0].astype(query_dtype), *result[1:])
        return result.astype(query_dtype)

    # No functools.wraps: the kernel is a jax.jit object whose bookkeeping attributes must not be copied.
    setattr(ragged_paged_attention, "_verl_fp32_stats", True)  # noqa: B010 - marker on a plain function
    setattr(ragged_paged_attention, "_verl_original", original)  # noqa: B010
    setattr(kernel, "ragged_paged_attention", ragged_paged_attention)  # noqa: B010
    # attention_interface/adapter bind the kernel to a module-level name at import; rebind those aliases too.
    for name in _RPA_ALIAS_MODULES:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
        if getattr(module, "ragged_paged_attention", None) is original:
            setattr(module, "ragged_paged_attention", ragged_paged_attention)  # noqa: B010
    logger.warning("vLLM-TPU ragged paged attention: fp32 softmax statistics enabled (%s=1)", ATTN_FP32_STATS_ENV)
    return True


def _fp32_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Keep CUDA's BF16/FP16 activation rounding before the gated product."""
    d = x.shape[-1] // 2
    activated = torch.nn.functional.silu(x[..., :d].float()).to(x.dtype)
    return activated * x[..., d:]


def patch_tpu_silu_and_mul() -> None:
    """Patch the native path before vLLM constructs TPU SwiGLU layers."""
    if importlib.util.find_spec("vllm_torchtpu") is None:
        return

    from vllm.model_executor.layers.activation import SiluAndMul

    if getattr(SiluAndMul, "_verl_fp32_silu_and_mul_patched", False):
        return
    original = SiluAndMul.forward_native

    def forward_native(x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "tpu" or x.dtype not in (torch.bfloat16, torch.float16):
            return original(x)
        return _fp32_silu_and_mul(x)

    # Disabled CustomOps also call forward_native; patching only forward_tpu misses them.
    SiluAndMul.forward_native = staticmethod(forward_native)
    SiluAndMul._verl_fp32_silu_and_mul_patched = True
    logger.info("Applied TPU generator FP32 SiLU computation with CUDA activation rounding.")


RMSNORM_CUDA_ORDER_ENV = "VERL_TPU_VLLM_RMSNORM_CUDA_ORDER"


def _cuda_order_rms_norm(self, x: torch.Tensor, residual: torch.Tensor | None = None):
    """Match CUDA: round the residual sum first, and round after the affine multiply."""
    dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        value = x.float()
        if residual is not None:
            residual = (value + residual.float()).to(dtype)
            value = residual.float()
        size = self.variance_size_override
        variance_input = value if size is None else value[..., :size]
        variance = variance_input.square().mean(dim=-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        has_weight = self.pass_weight if residual is None else self.pass_weight_add
        if has_weight:
            value = value * self.weight.data.float()
        output = value.to(dtype)
    return output if residual is None else (output, residual)


def patch_tpu_rms_norm() -> None:
    """Opt in before TPU RMSNorm layers bind their native forward."""
    if os.environ.get(RMSNORM_CUDA_ORDER_ENV) != "1":
        return
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
    except ImportError:
        return
    if getattr(RMSNorm, "_verl_cuda_order_rms_norm_patched", False):
        return
    original = RMSNorm.forward_native

    def forward_native(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if x.device.type != "tpu" or x.dtype not in (torch.bfloat16, torch.float16):
            return original(self, x, residual)
        return _cuda_order_rms_norm(self, x, residual)

    RMSNorm.forward_native = forward_native
    RMSNorm._verl_cuda_order_rms_norm_patched = True
    logger.warning("Enabled TPU vLLM RMSNorm CUDA rounding (%s=1)", RMSNORM_CUDA_ORDER_ENV)


def _apply_tpu_precision_patches() -> None:
    """Install numerical patches before layers bind their native forwards."""
    patch_tpu_rotary_emb()
    patch_tpu_logprobs()
    patch_tpu_sampler()
    patch_tpu_silu_and_mul()
    patch_tpu_rms_norm()
    patch_tpu_attention_fp32_stats()


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

    _apply_tpu_precision_patches()
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
