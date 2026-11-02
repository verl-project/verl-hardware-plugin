# Copyright (c) 2026 Google LLC. All rights reserved.
# Numerical patches adapted from verl: Copyright 2024-2025 BAAI and Google LLC.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for multi-host TPU rollout.

``patch_vllm_for_tpu`` lets the vLLM engine of a ``TPUvLLMHttpServer`` run on all hosts of a TPU
slice through Ray. Each patch fills a gap in vLLM (v0.29.0 - v0.32.0) and vllm-torchtpu
(9faafb17 - 732c4346) and can be dropped once upstream closes it:

* ``EngineArgs.create_engine_config``, ``initialize_ray_cluster`` and
  ``RayDistributedExecutor._init_workers_ray``: run the engine on vLLM's generic Ray executor,
  on the placement groups verl reserved for the replica (one per host), with one worker per chip
  that gets the TPU multi-host environment, and run ``current_platform.update_block_size_for_backend``
  after ``load_model`` so hybrid GatedDeltaNet + full-attention models align their unified KV pool
  block size. vllm-torchtpu's own Ray executor can use only a single placement group (and omits
  ``update_block_size_for_backend``). Upstream fix: let it use several and update block size.
* ``RayDistributedExecutor._execute_dag``: run each step with plain Ray calls instead of a
  compiled Ray graph. The graph runs the model on a background thread, where vllm-torchtpu's
  raised torch.compile recompile limit does not apply, so the first request fails. Upstream fix:
  raise the limit on every thread.
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
* ``multiprocessing.process.BaseProcess.__init__`` and ``RayWorkerWrapper.__init__``: install
  these patches in the EngineCore process and in vLLM's worker actors, neither of which imports
  the plugin.
"""

import importlib.util
import inspect
import logging
import multiprocessing.process
import os
import sys
from collections import Counter

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    TPU_V7X_DEVICES_PER_HOST,
    driver_tpu_env_for_workers,
    is_tpu_v7x,
    ray_tpu_accelerator_name,
    tpu_mesh_env,
)

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
        import vllm.v1.executor.ray_executor as ray_executor
        from vllm.engine.arg_utils import EngineArgs
        from vllm_torchtpu.executors.ray_distributed_executor import RayWorkerWrapper
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU executor patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    _apply_tpu_precision_patches()

    original_create_engine_config = EngineArgs.create_engine_config

    def create_engine_config(self, *args, **kwargs):
        # vllm-torchtpu picks its own Ray executor, which takes a single placement group, but verl
        # reserves one per host. Use vLLM's generic Ray executor, which the patches below adapt to
        # TPU. It does not support async scheduling, which vLLM turned on for the
        # external_launcher backend that TPUvLLMHttpServer passes in.
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        vllm_config.parallel_config.distributed_executor_backend = "ray"
        vllm_config.scheduler_config.async_scheduling = False
        return vllm_config

    original_worker_init = RayWorkerWrapper.__init__
    original_init_worker = RayWorkerWrapper.init_worker

    def worker_init(self, *args, **kwargs):
        # vLLM's worker actors never import the plugin. Ray sends this method to them with the
        # actor class, so it installs the patches there.
        patch_vllm_for_tpu()
        original_worker_init(self, *args, **kwargs)

    def init_worker(self, *args, **kwargs):
        # Ray installs worker environment variables after constructing the wrapper. Reapply the
        # opt-in numerical patches now, before the model binds its native forwards.
        _apply_tpu_precision_patches()
        return original_init_worker(self, *args, **kwargs)

    EngineArgs.create_engine_config = create_engine_config
    RayWorkerWrapper.__init__ = worker_init
    RayWorkerWrapper.init_worker = init_worker
    TPUWorker.reset_encoder_cache = _reset_encoder_cache
    ray_executor.initialize_ray_cluster = _initialize_ray_cluster
    ray_executor.RayDistributedExecutor._init_workers_ray = _init_workers_ray
    ray_executor.RayDistributedExecutor._execute_dag = _execute_dag
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


def _initialize_ray_cluster(parallel_config, ray_address=None) -> None:
    """Connect to Ray as vLLM does, but create no placement group: ``_init_workers_ray`` uses verl's."""
    if not ray.is_initialized():
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)


def _init_workers_ray(self, placement_group, **ray_remote_kwargs) -> None:
    """Start and initialize one vLLM worker per TPU chip of the replica.

    vLLM's version uses the single ``placement_group`` (unset here) and leaves the TPU environment
    to vllm-torchtpu's executor. This one places the workers on the placement groups verl reserved
    for the replica (``VERL_TPU_PG_IDS``, one per host) and gives each worker its chip's TPU
    environment.
    """
    from vllm.platforms import current_platform
    from vllm.ray.ray_env import get_env_vars_to_copy
    from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
    from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS
    from vllm_torchtpu import envs as vllm_torchtpu_envs
    from vllm_torchtpu.executors.ray_distributed_executor import RayWorkerWrapper

    pg_ids = os.environ["VERL_TPU_PG_IDS"].split(",")
    logger.info("Reusing %d verl placement group(s) for TPU rollout: %s", len(pg_ids), pg_ids)
    bundles = [
        (pg, bundle_index)
        for pg in (PlacementGroup(ray.PlacementGroupID.from_hex(pg_id)) for pg_id in pg_ids)
        for bundle_index, bundle in enumerate(pg.bundle_specs)
        if bundle.get(current_platform.ray_device_key, 0)
    ][: self.parallel_config.world_size]
    self.workers = [
        ray.remote(
            num_cpus=0,
            num_gpus=0,
            resources={current_platform.ray_device_key: 1},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            ),
            **ray_remote_kwargs,
        )(RayWorkerWrapper).remote(rpc_rank=rank)
        for rank, (pg, bundle_index) in enumerate(bundles)
    ]

    # Rank the workers as vLLM does: the driver's host first, because rank 0 serves the
    # torch.distributed store at the driver's address, then each host's workers together.
    driver_ip = get_ip()
    ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])
    workers_per_ip = Counter(ips)
    order = sorted(range(len(ips)), key=lambda i: (ips[i] != driver_ip, workers_per_ip[ips[i]], ips[i]))
    self.workers = [self.workers[i] for i in order]
    ips = [ips[i] for i in order]
    self.collective_rpc("adjust_rank", args=({created: rank for rank, created in enumerate(order)},))

    driver_env = {
        name: os.environ[name]
        for name in get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers",
        )
        if name in os.environ
    }
    # vLLM only forwards its registered environment names. Precision flags also need to reach
    # every worker, including explicit zero values such as VERL_TPU_FP32_LOGITS=0.
    driver_env.update(driver_tpu_env_for_workers())
    # torch_tpu's own rendezvous, separate from the torch.distributed one below.
    torch_tpu_master = {"MASTER_ADDR": ips[0], "MASTER_PORT": str(get_open_port())}
    pod_type = ray_tpu_accelerator_name(node_ips=ips)
    self._env_vars_for_all_workers = [
        {**driver_env, **torch_tpu_master, **tpu_env}
        for tpu_env in _tpu_worker_envs(ips, vllm_torchtpu_envs.TORCH_TPU_BASE_PORT, pod_type=pod_type)
    ]
    self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))
    logger.info("TPU rollout process addresses: %s", self._env_vars_for_all_workers[0]["TPU_PROCESS_ADDRESSES"])

    distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())
    local_ranks = _local_ranks(ips)
    all_kwargs = [
        dict(
            vllm_config=self.vllm_config,
            local_rank=local_ranks[rank],
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=rank % self.parallel_config.tensor_parallel_size == 0,
        )
        for rank in range(len(self.workers))
    ]
    self.collective_rpc("init_worker", args=(all_kwargs,))
    self.collective_rpc("init_device")
    self.collective_rpc("load_model")
    self.collective_rpc(_update_worker_block_size)


def _update_worker_block_size(worker) -> None:
    """Finalize ``worker.vllm_config`` KV-cache block sizing after ``load_model``.

    vLLM >= 0.31 moved ``current_platform.update_block_size_for_backend`` out of
    ``EngineArgs.create_engine_config`` and into the executor right after ``load_model`` (because
    ``_find_non_ssm_backends`` inspects the loaded model's ``static_forward_context``). Skipping it
    leaves hybrid GatedDeltaNet + full-attention models (Qwen3.5) with ``block_size=16`` instead of
    the aligned ``mamba_fit_block_size`` (e.g. ``272``), crashing the unified block pool with
    ``AssertionError: pool page mismatch``.
    """
    from vllm.platforms import current_platform

    update_fn = getattr(current_platform, "update_block_size_for_backend", None)
    if callable(update_fn):
        update_fn(worker.vllm_config)


def _local_ranks(worker_ips: list[str]) -> list[int]:
    """Index of each worker among the workers on its host, for workers in rank order."""
    return [worker_ips[:rank].count(ip) for rank, ip in enumerate(worker_ips)]


def _tpu_worker_envs(worker_ips: list[str], base_port: int, pod_type: str = "") -> list[dict[str, str]]:
    """The TPU multi-host environment of each vLLM worker, for workers in rank order.

    Ranks on one host must be contiguous. A worker drives chip ``local_rank`` of its host and serves
    the slice builder on ``base_port + local_rank``. These are the variables libtpu and torch_tpu
    read to join the workers into one TPU mesh. On TPU 7x (``pod_type`` or the accelerator-type
    environment) a worker drives one TensorCore, the mesh is laid out by process, and an engine must span
    whole hosts: a standalone process can open only a chip's first core, so an engine with fewer workers on
    a host than that host has TensorCores has no layout and is rejected here.
    """
    hosts = list(dict.fromkeys(worker_ips))
    local_ranks = _local_ranks(worker_ips)
    # The engine's worker count defines its mesh: an engine may use part of a host's chips (two TP=4 engines
    # on a v6e-8, say), so the pod type only tells TPU 7x apart and never selects a v6e layout.
    v7x = is_tpu_v7x(pod_type)
    pod_type = pod_type if v7x else ""
    if v7x:
        partial = {ip: n for ip, n in Counter(worker_ips).items() if n != TPU_V7X_DEVICES_PER_HOST}
        if partial:
            raise ValueError(
                f"On TPU 7x a vLLM engine spans whole hosts of {TPU_V7X_DEVICES_PER_HOST} TensorCore workers; "
                f"this engine has {partial} workers per host. Set tensor_model_parallel_size to a multiple of "
                f"{TPU_V7X_DEVICES_PER_HOST} (one engine per host, or per group of hosts)."
            )
    addresses = ",".join(
        f"{ip}:{base_port + local_rank}" for ip, local_rank in zip(worker_ips, local_ranks, strict=True)
    )
    envs = []
    for rank, (ip, local_rank) in enumerate(zip(worker_ips, local_ranks, strict=True)):
        env = {
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "TPU_PROCESS_ADDRESSES": addresses,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
            "CLOUD_TPU_TASK_ID": str(hosts.index(ip)),
            "TPU_WORKER_HOSTNAMES": ",".join(hosts),
        }
        mesh_env = tpu_mesh_env(len(worker_ips), len(hosts), rank=rank, local_rank=local_rank, pod_type=pod_type)
        mesh_env.pop("CHIPS_PER_HOST")  # a trainer-side variable; the rollout workers never carried it
        env.update(mesh_env)
        envs.append(env)
    return envs


def _execute_dag(self, scheduler_output, grammar_output, non_block=False):
    """Run one step on every worker with plain Ray calls instead of vLLM's compiled Ray graph.

    The graph runs the model on a background thread, where torch.compile's recompile limit is
    still the default 8: vllm-torchtpu raises it to 1024 on the main thread only (dynamo config
    overrides are per thread), and the first request fails with FailOnRecompileLimitHit. verl uses
    no KV connector, so rank 0's output is the step's result.
    """
    from vllm.v1.executor.ray_utils import FutureWrapper, detach_zero_copy_from_model_runner_output

    refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
    if non_block:
        return FutureWrapper(refs[0])
    output = ray.get(refs)[0]
    detach_zero_copy_from_model_runner_output(output)
    return output


def _reset_encoder_cache(self) -> None:
    """``TPUWorker.reset_encoder_cache``, which verl calls after every weight update: a no-op.

    The encoder cache holds multimodal encoder outputs; text models have none to reset.
    """


class PickleableProcessWrapper:
    """Process target that runs ``patch_vllm_for_tpu`` in the child before ``target``.

    This is how the patches reach vLLM's EngineCore. Inside a Ray actor vLLM always uses the
    ``spawn`` start method, and the EngineCore target is ``EngineCoreProc.run_engine_core``, which
    vllm-torchtpu replaces with its own wrapper. A spawned child imports only the modules its pickled
    target names, so without this wrapper it would never import the plugin.
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
