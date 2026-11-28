# Copyright (c) 2026 Google LLC. All rights reserved.
# Numerical patches adapted from verl: Copyright 2024-2025 BAAI and Google LLC.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for the TPU rollout.

The rollout runs on vLLM's ``mp`` executor and vllm-torchtpu's ``mp`` multi-host backend, which need no
patch to span several hosts. Each remaining patch fills a gap in vLLM (v0.29.0 - v0.32.0) and vllm-torchtpu
(281b5266+) and can be dropped once upstream closes it:

* ``ApplyRotaryEmb`` and ``SiluAndMul``: compute TPU rotations and SiLU in FP32, retaining
  CUDA activation rounding. RoPE also avoids a concat that can crash TPU fusion (b/501165531).
* ``VERL_TPU_VLLM_ATTN_FP32_STATS=1`` and ``VERL_TPU_VLLM_RMSNORM_CUDA_ORDER=1``: opt in
  to FP32 attention statistics and CUDA RMSNorm rounding without changing BF16 model weights.
* ``multiprocessing.process.BaseProcess.__init__``: install these patches in the EngineCore process and
  in vLLM's worker processes, which do not import the plugin.
* ``vllm_torchtpu.worker.tpu_worker._configure_tpu_process_env``, only in an engine that shares its TPU 7x host
  with other engines (``tpu_vllm.subhost_engine_env``): per-worker MegaScale ports and libtpu log dir, and the
  GKE webhook's container-level TPU variables and ``TORCH_TPU_DEVICES_PER_HOST`` dropped. Upstream fix: a
  per-engine port base in vllm-torchtpu.
* ``vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states``: guard against missing
  ``req_data.all_token_ids[req_id]`` keys under ``VLLM_ASYNC_SCHEDULING=True`` during in-flight ``tpu-sync``
  abort/pause/resume weight updates (``KeyError`` -> ``EngineDeadError``).
* ``vllm_torchtpu.worker.tpu_worker.TPUWorker.init_device``, only in an engine that shares its TPU 7x host with
  other engines: serialize the ``tpu_dist`` / ``tpunetd`` ``StartSession`` handshake across co-located engines on
  the same host using a per-host ``fcntl.flock`` lockfile so concurrent ``StartSession`` calls over ``vbar.sock``
  do not collide (``GRPC_ERROR error message: START_SESSION failed``). Each engine's TP workers still initialize
  together, and the lock is released before model loading and XLA compilation. Upstream fix: concurrent
  ``StartSession`` handling in ``tpunetd`` / ``torch_tpu``.
"""

import contextlib
import fcntl
import functools
import importlib
import inspect
import logging
import multiprocessing.process
import os
import re
import sys
import time
import uuid

import torch

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def _tpu_sign(x: torch.Tensor) -> torch.Tensor:
    """Build rotation signs per call; cached tensors break compile-cache replay."""
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
    """Compute TPU RoPE in FP32 without concat fusion crashes (b/501165531)."""
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


ATTN_FP32_STATS_ENV = "VERL_TPU_VLLM_ATTN_FP32_STATS"
_RPA_KERNEL_MODULE = "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel"
_RPA_ALIAS_MODULES = ("vllm_torchtpu.layers.core.attention_interface", "vllm_torchtpu.layers.adapter.attention")


def patch_tpu_attention_fp32_stats() -> bool:
    """Opt in to FP32 attention statistics/accumulation, then restore the query dtype."""
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
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except ImportError:
        return

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
    patch_tpu_silu_and_mul()
    patch_tpu_rms_norm()
    patch_tpu_attention_fp32_stats()


def _patch_unquantized_routed_experts() -> None:
    """Apply vllm-torchtpu#2967 compat shim when ``VllmUnquantizedConfig`` lacks ``RoutedExperts`` support."""
    try:
        import vllm.model_executor.layers.fused_moe as fused_moe
        import vllm_torchtpu.layers.adapter.quantization.configs as quant_configs
        import vllm_torchtpu.layers.adapter.quantization.fp8 as fp8_mod
        import vllm_torchtpu.layers.adapter.quantization.unquantized as unquantized_mod
    except ImportError:
        return

    cfg_cls = getattr(unquantized_mod, "VllmUnquantizedConfig", None)
    routed_experts_cls = getattr(fused_moe, "RoutedExperts", None)
    should_quantize_on_load = getattr(quant_configs, "should_quantize_on_load", None)
    fp8_cfg_cls = getattr(fp8_mod, "VllmFp8Config", None)
    fp8_moe_cls = getattr(fp8_mod, "VllmFp8MoEMethodTPU", None)
    if cfg_cls is None or None in (routed_experts_cls, should_quantize_on_load, fp8_cfg_cls, fp8_moe_cls):
        return

    orig_get_quant = cfg_cls.__dict__.get("get_quant_method") or getattr(cfg_cls, "get_quant_method", None)
    if orig_get_quant is None or getattr(orig_get_quant, "_verl_pr2967_patched", False):
        return

    try:
        if "RoutedExperts" in inspect.getsource(orig_get_quant):
            return
    except (OSError, TypeError):
        pass

    @functools.wraps(orig_get_quant)
    def _get_quant_method_pr2967(self, layer, prefix):
        if isinstance(layer, routed_experts_cls) and should_quantize_on_load(prefix):
            moe_config = self.get_moe_config(layer)
            if moe_config.has_bias:
                raise NotImplementedError(
                    f"QUANTIZE_ON_LOAD_PREFIXES matched {prefix}, but FP8 MoE "
                    "on-load quantization does not support expert bias"
                )
            fp8_config = fp8_cfg_cls(is_checkpoint_fp8_serialized=False, activation_scheme="dynamic")
            return fp8_moe_cls(fp8_config, moe_config)
        return orig_get_quant(self, layer, prefix)

    _get_quant_method_pr2967._verl_pr2967_patched = True  # type: ignore[attr-defined]
    cfg_cls.get_quant_method = _get_quant_method_pr2967
    logger.info("Installed vllm-torchtpu#2967 compatibility shim on VllmUnquantizedConfig.")


def _create_dummy_online_fp8_moe_weights(
    fp8_mod,
    layer,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
) -> bool:
    """Allocate dummy FP8 MoE buffers directly; returns False if block requantization falls back to BF16."""
    _, requant_dtype, requant_block_size = fp8_mod._get_moe_quant_config(
        "[MoE dummy online FP8]: allocating dummy FP8 MoE weights"
    )
    if requant_block_size is not None:
        return False
    if not hasattr(layer, "has_bias") and hasattr(layer, "moe_config"):
        layer.has_bias = layer.moe_config.has_bias
    aligned = (intermediate_size_per_partition + 127) // 128 * 128
    w13 = torch.nn.Parameter(
        torch.zeros(num_experts, hidden_size, 2 * aligned, dtype=requant_dtype), requires_grad=False
    )
    w2 = torch.nn.Parameter(torch.zeros(num_experts, aligned, hidden_size, dtype=requant_dtype), requires_grad=False)
    w13_scale = torch.nn.Parameter(torch.ones(num_experts, 1, 1, 2 * aligned, dtype=torch.float32), requires_grad=False)
    w2_scale = torch.nn.Parameter(torch.ones(num_experts, 1, 1, hidden_size, dtype=torch.float32), requires_grad=False)
    w13.dummy_weight_value = 0  # type: ignore[attr-defined]
    w2.dummy_weight_value = 0  # type: ignore[attr-defined]
    w13_scale.dummy_weight_value = 1.0  # type: ignore[attr-defined]
    w2_scale.dummy_weight_value = 1.0  # type: ignore[attr-defined]
    layer.register_parameter("w13_weight", w13)
    layer.register_parameter("w2_weight", w2)
    layer.register_parameter("w13_weight_scale_inv", w13_scale)
    layer.register_parameter("w2_weight_scale_inv", w2_scale)
    if hasattr(layer, "weight_loader"):
        for param in (w13, w2, w13_scale, w2_scale):
            param.weight_loader = layer.weight_loader  # type: ignore[attr-defined]
    layer._tpu_dummy_fp8_initialized = True
    return True


def _finalize_dummy_online_fp8_moe_weights(method, fp8_mod, layer, fused_moe_ep_mod) -> None:
    """Finish MoE routing / kernel prebuild when dummy FP8 weights bypassed BF16->FP8 quantization."""
    method._use_adaptive_fused_moe = fp8_mod.envs.MOE_FUSED_EP_KERNEL_IMPL == "adaptive"
    activation_str = method._set_tpu_activation(layer)
    if layer.w13_weight.device.type == "tpu":
        fp8_mod.synchronize_tensors(
            [layer.w13_weight, layer.w2_weight, layer.w13_weight_scale_inv, layer.w2_weight_scale_inv]
        )
    if layer.moe_config.moe_parallel_config.use_ep:
        fp8_mod.moe_routing.validate_linear_ep_placement(layer)
    fp8_mod.moe_routing.register_experts_start_buffer(layer, device=layer.w13_weight.device)
    if method._use_adaptive_fused_moe:
        method._adaptive_fused_moe_op = fp8_mod.adaptive_moe.prebuild_adaptive_fused_moe(
            layer,
            topk=layer.moe_config.experts_per_token,
            renormalize=layer.renormalize,
            activation=activation_str,
        )
        return
    prebuild_kwargs = {
        "topk": layer.moe_config.experts_per_token,
        "activation": activation_str,
        "use_ep": layer.moe_config.moe_parallel_config.use_ep,
    }
    if "topk_weights_float32" in inspect.signature(fp8_mod.prebuild_fused_moe_kernel).parameters:
        prebuild_kwargs["topk_weights_float32"] = getattr(method, "_tpu_topk_weights_float32", False)
    fp8_mod.prebuild_fused_moe_kernel(**prebuild_kwargs)
    if fused_moe_ep_mod is None:
        raise ImportError("vllm_torchtpu.layers.adapter.fused_moe_ep is required for non-adaptive FP8 MoE")
    ep_op = fused_moe_ep_mod.prebuild_fused_moe_ep(
        layer,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        activation=activation_str,
    )
    setattr(method, fused_moe_ep_mod.FUSED_MOE_EP_OP_ATTR, ep_op)
    if ep_op is not None:
        fused_moe_ep_mod.register_score_bias_buffer(layer)


def _patch_fp8_moe_dummy_load() -> None:
    """Allocate dummy FP8 MoE buffers directly under ``load_format == 'dummy'`` if vllm-torchtpu lacks it."""
    try:
        import vllm_torchtpu.layers.adapter.quantization.fp8 as fp8_mod
    except ImportError:
        return

    try:
        from vllm.config import get_current_vllm_config
    except ImportError:
        get_current_vllm_config = None

    try:
        import vllm_torchtpu.layers.adapter.fused_moe_ep as fused_moe_ep_mod
    except ImportError:
        fused_moe_ep_mod = None

    moe_cls = getattr(fp8_mod, "VllmFp8MoEMethodTPU", None)
    if moe_cls is None or hasattr(moe_cls, "_create_dummy_online_fp8_weights"):
        return
    orig_create = moe_cls.__dict__.get("create_weights") or getattr(moe_cls, "create_weights", None)
    orig_process = moe_cls.__dict__.get("process_weights_after_loading") or getattr(
        moe_cls, "process_weights_after_loading", None
    )
    if orig_create is None or orig_process is None or getattr(orig_create, "_verl_dummy_fp8_patched", False):
        return

    @functools.wraps(orig_create)
    def _create_weights_dummy_fp8(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        vllm_config = getattr(self.quant_config, "vllm_config", None)
        if vllm_config is None and get_current_vllm_config is not None:
            try:
                vllm_config = get_current_vllm_config()
            except Exception:
                vllm_config = None
        load_format = getattr(getattr(vllm_config, "load_config", None), "load_format", None)
        if (
            not self.quant_config.is_checkpoint_fp8_serialized
            and load_format == "dummy"
            and _create_dummy_online_fp8_moe_weights(
                fp8_mod, layer, num_experts, hidden_size, intermediate_size_per_partition
            )
        ):
            return None
        return orig_create(
            self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
        )

    @functools.wraps(orig_process)
    def _process_weights_dummy_fp8(self, layer):
        if getattr(layer, "_tpu_dummy_fp8_initialized", False):
            _finalize_dummy_online_fp8_moe_weights(self, fp8_mod, layer, fused_moe_ep_mod)
            return None
        return orig_process(self, layer)

    _create_weights_dummy_fp8._verl_dummy_fp8_patched = True  # type: ignore[attr-defined]
    _process_weights_dummy_fp8._verl_dummy_fp8_patched = True  # type: ignore[attr-defined]
    moe_cls.create_weights = _create_weights_dummy_fp8
    moe_cls.process_weights_after_loading = _process_weights_dummy_fp8
    logger.info("Installed dummy online FP8 MoE allocation patch on VllmFp8MoEMethodTPU.")


def patch_tpu_fp8_moe_rollout() -> None:
    """Install FP8 MoE dummy-load and pre-vllm-torchtpu#2967 shims when needed."""
    _patch_unquantized_routed_experts()
    _patch_fp8_moe_dummy_load()


# TODO: Remove this workaround once upstream vLLM fixes the async-scheduling
# abort/pause/resume desync where GPUModelRunner._update_states unconditionally
# indexes req_data.all_token_ids[req_id] even when resumed_from_preemption is
# False (raising KeyError -> EngineDeadError during in-flight tpu-sync weight
# updates in multi-epoch runs).
def patch_tpu_async_scheduling_update_states() -> None:
    """Guard GPUModelRunner._update_states against missing req_data.all_token_ids keys under async scheduling."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return

    orig_update_states = GPUModelRunner.__dict__.get("_update_states") or getattr(
        GPUModelRunner, "_update_states", None
    )
    if orig_update_states is None or getattr(orig_update_states, "_verl_async_all_token_ids_patched", False):
        return

    @functools.wraps(orig_update_states)
    def _update_states_with_all_token_ids_guard(self, scheduler_output):
        if getattr(self, "use_async_scheduling", False):
            req_data = getattr(scheduler_output, "scheduled_cached_reqs", None)
            all_token_ids = getattr(req_data, "all_token_ids", None)
            num_output_tokens = getattr(req_data, "num_output_tokens", None)
            if isinstance(all_token_ids, dict) and num_output_tokens is not None:
                req_id_to_index = self.input_batch.req_id_to_index
                for i, req_id in enumerate(req_data.req_ids):
                    # Only the case upstream indexes: not in the persistent batch, async, with output tokens.
                    if num_output_tokens[i] <= 0 or req_id in all_token_ids or req_id in req_id_to_index:
                        continue
                    req_state = self.requests.get(req_id)
                    if req_state is not None:
                        # Same layout as the scheduler's Request.all_token_ids: prompt + output tokens.
                        all_token_ids[req_id] = list(req_state.prompt_token_ids or ()) + list(
                            req_state.output_token_ids
                        )
                        logger.warning(
                            "Backfilled missing req_data.all_token_ids[%s] in GPUModelRunner._update_states "
                            "(async scheduling + weight-sync desync guard).",
                            req_id,
                        )
        return orig_update_states(self, scheduler_output)

    _update_states_with_all_token_ids_guard._verl_async_all_token_ids_patched = True  # type: ignore[attr-defined]
    GPUModelRunner._update_states = _update_states_with_all_token_ids_guard
    logger.info("Installed async-scheduling all_token_ids KeyError guard on GPUModelRunner._update_states.")


def patch_vllm_for_tpu() -> None:
    """Install the patches in this process. Later calls return at once."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    patch_multiprocessing_for_tpu()

    # Imported here: CPU tests and processes outside the rollout may lack vLLM or vllm-torchtpu.
    try:
        import vllm_torchtpu  # noqa: F401
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    _apply_tpu_precision_patches()
    patch_tpu_fp8_moe_rollout()
    patch_tpu_async_scheduling_update_states()
    if os.environ.get(SUBHOST_MEGASCALE_PORT_ENV):
        patch_tpu_worker_subhost_env()
        if subhost_serialize_start_session_enabled() or _subhost_tp1_chip_order_active():
            patch_tpu_worker_subhost_start_session()
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


# Set by tpu_vllm.subhost_engine_env on an engine that shares its host: the first port of the engine's MegaScale block.
SUBHOST_MEGASCALE_PORT_ENV = "VERL_TPU_SUBHOST_MEGASCALE_PORT"
# Mirrors tpu_vllm.SUBHOST_OWNER_ENV (kept here to avoid importing tpu_vllm from a leaf module; tested equal).
_SUBHOST_OWNER_ENV = "VERL_TPU_SUBHOST_OWNER"
# Mirrors platform_tpu.TPU_V7X_PROCESS_BOUNDS (leaf module; tested equal).
_V7X_SINGLE_CORE_BOUNDS = "1,1,1,1"
# Keep per-chip liveness flock fds open until process exit so paired odd cores can test even-core liveness via flock.
_TP1_CHIP_ALIVE_FDS: list[int] = []
# Unique per-launch identifier for one sub-host engine, shared across all of its TP worker processes.
SUBHOST_ENGINE_ID_ENV = "VERL_TPU_SUBHOST_ENGINE_ID"
# Enabled by default for co-located sub-host engines; set to "0" to opt out of host-wide StartSession serialization.
SUBHOST_SERIALIZE_START_SESSION_ENV = "VERL_TPU_SUBHOST_SERIALIZE_START_SESSION"
# Host-wide flock serializing tpunetd StartSession across co-located sub-host engines. Never unlinked so flock inode
# identity stays shared across all processes on the host (matching tpu_vllm.SUBHOST_CLAIM_FILE).
SUBHOST_START_SESSION_LOCK_FILE = "/tmp/verl_tpu_subhost_start_session.lock"
# How long same-engine workers wait for rank 0 / siblings to reach the init_device gate during spawn.
# Once rank 0 holds its leader lock, siblings keep waiting as long as rank 0 is alive (bounded only by
# _SUBHOST_INIT_SAFETY_TIMEOUT_S), never falling through unserialized.
_SUBHOST_INIT_PEER_APPEAR_TIMEOUT_S = 120.0
# Optional long safety bound shared by rank 0 and siblings (30 min); if exceeded, every rank aborts consistently.
_SUBHOST_INIT_SAFETY_TIMEOUT_S = 1800.0
_SUBHOST_INIT_POLL_INTERVAL_S = 0.02
# Variables that describe the whole pod or host, which every engine in the container would inherit: the GKE TPU
# webhook's container-level variables (the second engine of a host then collides with the first on the pod's MegaScale
# ports), and torch_tpu's TORCH_TPU_DEVICES_PER_HOST (e.g. exported for the trainer): torch_tpu sizes its node-local
# store by it and checks local ranks against it, which does not fit an engine on part of a host.
_POD_TPU_ENV_VARS = (
    "MEGASCALE_NUM_SLICES",
    "MEGASCALE_SLICE_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_WORKER_ID",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_RUNTIME_METRICS_PORTS",
    "TORCH_TPU_DEVICES_PER_HOST",
)


def subhost_worker_env(rank: int, local_rank: int, megascale_port: int, world_size: int = 0) -> dict[str, str]:
    """The per-worker variables of an engine that shares its host; ``local_rank`` includes the engine's offset.

    For ``world_size == 1`` (TP=1 on TPU 7x), also restores explicit 4-D ``1,1,1,1`` bounds and per-slot localhost
    process addresses after ``_POD_TPU_ENV_VARS`` are stripped so both TensorCores of a chip can run independently.
    """
    env = {
        "TPU_VISIBLE_DEVICES": str(local_rank),
        "TPU_VISIBLE_CHIPS": str(local_rank),
        "MEGASCALE_PORT": str(megascale_port + 2 * rank),
        "MEGASCALE_DEBUG_PORT": str(megascale_port + 2 * rank + 1),
        "TPU_LOG_DIR": f"/tmp/tpu_logs/rank_{local_rank}",
    }
    if world_size == 1:
        process_port = megascale_port + 2
        env.update(
            {
                "TPU_HOST_BOUNDS": _V7X_SINGLE_CORE_BOUNDS,
                "TPU_PROCESS_BOUNDS": _V7X_SINGLE_CORE_BOUNDS,
                "TPU_CHIPS_PER_HOST_BOUNDS": _V7X_SINGLE_CORE_BOUNDS,
                "TPU_CHIPS_PER_PROCESS_BOUNDS": _V7X_SINGLE_CORE_BOUNDS,
                "CLOUD_TPU_TASK_ID": "0",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "localhost",
                "TPU_PROCESS_ADDRESSES": f"localhost:{process_port}",
                "TPU_PROCESS_PORT": str(process_port),
            }
        )
    return env


def patch_tpu_worker_subhost_env() -> None:
    """Give each worker of an engine that shares its host its own MegaScale ports and libtpu log dir.

    These depend on the worker's rank, so the server actor's environment cannot carry them. vllm-torchtpu's
    ``TPUWorker.init_device`` calls ``_configure_tpu_process_env(rank, local_rank, ...)`` once per worker, with the
    local rank already offset to the engine's devices, just before torch_tpu starts: the smallest hook that sees the
    rank. The pod- and host-wide variables (``_POD_TPU_ENV_VARS``) are dropped here too, as an actor's ``env_vars``
    can set a variable but not unset it.
    """
    try:
        import vllm_torchtpu.worker.tpu_worker as tpu_worker
    except ImportError:
        return
    original = tpu_worker._configure_tpu_process_env
    if getattr(original, "_verl_subhost_env", False):
        return
    megascale_port = int(os.environ[SUBHOST_MEGASCALE_PORT_ENV])

    def _configure_tpu_process_env(rank: int, local_rank: int, world_size: int, local_world_size: int) -> None:
        original(rank, local_rank, world_size, local_world_size)
        for name in _POD_TPU_ENV_VARS:
            os.environ.pop(name, None)
        env = subhost_worker_env(int(rank), int(local_rank), megascale_port, world_size=int(world_size))
        os.makedirs(env["TPU_LOG_DIR"], exist_ok=True)
        os.environ.update(env)

    _configure_tpu_process_env._verl_subhost_env = True  # type: ignore[attr-defined]
    tpu_worker._configure_tpu_process_env = _configure_tpu_process_env
    logger.info("Applied the TPU patch for engines that share a host (MegaScale ports from %d).", megascale_port)


def _subhost_tp1_chip_order_active() -> bool:
    """Whether this worker needs TPU 7x TP=1 even-before-odd per-chip StartSession ordering."""
    return (
        bool(os.environ.get(SUBHOST_MEGASCALE_PORT_ENV))
        and bool(os.environ.get(_SUBHOST_OWNER_ENV))
        and os.environ.get("TORCH_TPU_TOPOLOGY") == _V7X_SINGLE_CORE_BOUNDS
    )


def subhost_serialize_start_session_enabled() -> bool:
    """Whether co-located sub-host engines serialize ``TPUWorker.init_device`` (default on for sub-host engines)."""
    if not os.environ.get(SUBHOST_MEGASCALE_PORT_ENV):
        return False
    raw = os.environ.get(SUBHOST_SERIALIZE_START_SESSION_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _file_nonempty(path: str) -> bool:
    try:
        return os.stat(path).st_size > 0
    except OSError:
        return False


def _is_lock_held(path: str) -> bool:
    """Return True if another open file description currently holds an exclusive ``flock`` on ``path``."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _acquire_flock(fd: int, deadline: float | None, poll_interval_s: float) -> bool:
    """Acquire ``LOCK_EX`` on ``fd``, waiting indefinitely if ``deadline`` is ``None`` else until ``deadline``."""
    if deadline is None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return True
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_s)


def _wait_for_flock_release(path: str, deadline: float | None, poll_interval_s: float) -> bool:
    """Wait until no other open file description holds ``LOCK_EX`` on ``path`` (or ``deadline`` passes)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return True
    try:
        if not _acquire_flock(fd, deadline, poll_interval_s):
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _write_atomic(path: str, payload: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:6]}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _read_go_state(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


@contextlib.contextmanager
def serialize_subhost_start_session(
    *,
    rank: int,
    world_size: int,
    engine_id: str,
    lock_path: str = SUBHOST_START_SESSION_LOCK_FILE,
    coordination_dir: str | None = None,
    peer_appear_timeout_s: float = _SUBHOST_INIT_PEER_APPEAR_TIMEOUT_S,
    safety_timeout_s: float | None = _SUBHOST_INIT_SAFETY_TIMEOUT_S,
    poll_interval_s: float = _SUBHOST_INIT_POLL_INTERVAL_S,
):
    """Serialize ``tpunetd`` ``StartSession`` across co-located sub-host engines on one TPU host.

    All ``world_size`` TP workers of one engine must enter ``TPUWorker.init_device`` together because ``tpu_dist``
    performs a multi-rank rendezvous and ``PRE_START_SESSION_BARRIER`` / ``POST_START_SESSION_BARRIER``. A naive
    per-worker ``flock`` would deadlock sibling ranks of the same engine. Instead:

    1. Sibling ranks ``1..world_size-1`` acquire a per-rank ``flock`` and signal arrival before waiting for rank 0.
       Once rank 0 has locked its leader file, siblings keep waiting for ``GO`` as long as rank 0's leader lock stays
       held (bounded only by the optional long ``safety_timeout_s``), never falling through unserialized.
    2. Rank 0 locks its leader file, waits for its siblings to arrive (so it never holds the host lock idle while
       workers spawn), acquires ``LOCK_EX`` on ``lock_path``, verifies all siblings are still waiting, and atomically
       publishes ``GO`` so all workers of this engine run ``init_device`` together.
    3. On normal completion, rank 0 waits for every sibling's per-rank ``flock`` to drop before releasing ``lock_path``.
       If rank 0 raises before ``GO`` (including a missing peer or ``safety_timeout_s`` expiry), it publishes
       ``ABORT``; if it is killed (``SIGKILL`` / ``os._exit``), the kernel releases its leader lock. Either way,
       siblings notice within one ``poll_interval_s`` and raise rather than entering ``init_device`` unserialized.
    """
    if world_size <= 0 or not (0 <= rank < world_size):
        raise ValueError(f"Expected 0 <= rank < world_size, got rank={rank}, world_size={world_size}")

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    now = time.monotonic()
    safety_deadline = None if safety_timeout_s is None else now + safety_timeout_s

    if world_size == 1:
        single_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        single_locked = False
        try:
            if not _acquire_flock(single_fd, safety_deadline, poll_interval_s):
                raise TimeoutError(
                    f"Timed out after {safety_timeout_s}s waiting to acquire host StartSession lock {lock_path} "
                    f"for sub-host engine {engine_id}."
                )
            single_locked = True
            marker = f"{engine_id}:{os.getpid()}\n".encode()
            os.pwrite(single_fd, marker, 0)
            os.ftruncate(single_fd, len(marker))
            yield
        finally:
            try:
                if single_locked:
                    try:
                        os.ftruncate(single_fd, 0)
                    except OSError:
                        pass
                    fcntl.flock(single_fd, fcntl.LOCK_UN)
            finally:
                os.close(single_fd)
        return

    coord_dir = coordination_dir or lock_dir or "/tmp"
    os.makedirs(coord_dir, exist_ok=True)
    safe_id = "".join(c if (c.isalnum() or c in "._-") else "_" for c in engine_id)
    prefix = os.path.join(coord_dir, f".verl_tpu_subhost_init_{safe_id}")
    leader_path = f"{prefix}.leader"
    go_path = f"{prefix}.go"

    def rank_path(r: int) -> str:
        return f"{prefix}.r{r}"

    appear_deadline = now + peer_appear_timeout_s
    if safety_deadline is not None:
        appear_deadline = min(appear_deadline, safety_deadline)

    if rank > 0:
        r_path = rank_path(rank)
        rank_fd = os.open(r_path, os.O_RDWR | os.O_CREAT, 0o666)
        rank_locked = False
        try:
            # Lock first, then write non-empty bytes: rank 0 checks st_size > 0 before probing the lock.
            fcntl.flock(rank_fd, fcntl.LOCK_EX)
            rank_locked = True
            os.pwrite(rank_fd, f"{os.getpid()}\n".encode(), 0)

            leader_seen = False
            while True:
                state = _read_go_state(go_path)
                if state == "GO":
                    if _file_nonempty(leader_path) and _is_lock_held(leader_path):
                        break
                    raise RuntimeError(
                        f"Rank 0 of sub-host engine {engine_id} exited after signaling GO before rank {rank} "
                        "entered init_device; aborting to avoid unserialized StartSession."
                    )
                if state == "ABORT":
                    raise RuntimeError(
                        f"Rank 0 of sub-host engine {engine_id} aborted before acquiring {lock_path}; "
                        f"aborting rank {rank} to avoid unserialized StartSession."
                    )
                if not leader_seen and _file_nonempty(leader_path):
                    leader_seen = True
                if leader_seen and not _is_lock_held(leader_path):
                    if _read_go_state(go_path) == "ABORT":
                        raise RuntimeError(
                            f"Rank 0 of sub-host engine {engine_id} aborted before acquiring {lock_path}; "
                            f"aborting rank {rank} to avoid unserialized StartSession."
                        )
                    raise RuntimeError(
                        f"Rank 0 of sub-host engine {engine_id} exited before signaling GO; "
                        f"aborting rank {rank} to avoid unserialized StartSession."
                    )
                now = time.monotonic()
                if not leader_seen and now >= appear_deadline:
                    raise TimeoutError(
                        f"Timed out after {peer_appear_timeout_s:.1f}s waiting for rank 0 of sub-host engine "
                        f"{engine_id} to reach init_device gate; aborting rank {rank}."
                    )
                if safety_deadline is not None and now >= safety_deadline:
                    raise TimeoutError(
                        f"Timed out after {safety_timeout_s}s waiting for sub-host engine {engine_id} "
                        f"to acquire {lock_path}; aborting rank {rank}."
                    )
                time.sleep(poll_interval_s)

            yield
        finally:
            try:
                if rank_locked:
                    fcntl.flock(rank_fd, fcntl.LOCK_UN)
            finally:
                os.close(rank_fd)
        return

    # rank == 0 (engine leader)
    leader_fd = os.open(leader_path, os.O_RDWR | os.O_CREAT, 0o666)
    leader_locked = False
    host_fd: int | None = None
    host_locked = False
    succeeded = False
    arrived_ranks: set[int] = set()
    try:
        fcntl.flock(leader_fd, fcntl.LOCK_EX)
        leader_locked = True
        os.pwrite(leader_fd, f"{os.getpid()}\n".encode(), 0)

        while len(arrived_ranks) < world_size - 1 and time.monotonic() < appear_deadline:
            for r in range(1, world_size):
                if r not in arrived_ranks:
                    r_path = rank_path(r)
                    if _file_nonempty(r_path) and _is_lock_held(r_path):
                        arrived_ranks.add(r)
            if len(arrived_ranks) < world_size - 1:
                time.sleep(poll_interval_s)
        if len(arrived_ranks) < world_size - 1:
            missing = sorted(set(range(1, world_size)) - arrived_ranks)
            raise TimeoutError(
                f"Timed out waiting for sibling TPU worker(s) {missing} of sub-host engine {engine_id} "
                f"to reach init_device gate (arrived={sorted(arrived_ranks)}, world_size={world_size}); aborting."
            )

        host_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        if not _acquire_flock(host_fd, safety_deadline, poll_interval_s):
            raise TimeoutError(
                f"Timed out after {safety_timeout_s}s waiting to acquire host StartSession lock {lock_path} "
                f"for sub-host engine {engine_id}; aborting."
            )
        host_locked = True

        # Ensure every sibling is still alive and waiting at the gate before starting StartSession.
        departed = [r for r in range(1, world_size) if not _is_lock_held(rank_path(r))]
        if departed:
            raise RuntimeError(
                f"Sibling TPU worker(s) {departed} of sub-host engine {engine_id} exited while waiting for "
                f"{lock_path}; aborting StartSession."
            )

        marker = f"{engine_id}:{os.getpid()}\n".encode()
        os.pwrite(host_fd, marker, 0)
        os.ftruncate(host_fd, len(marker))

        _write_atomic(go_path, b"GO\n")

        yield
        succeeded = True
    finally:
        all_siblings_done = False
        exit_timed_out_rank: int | None = None
        try:
            if not succeeded:
                try:
                    _write_atomic(go_path, b"ABORT\n")
                except OSError:
                    pass
            else:
                exit_deadline = None if safety_timeout_s is None else time.monotonic() + safety_timeout_s
                all_siblings_done = True
                for r in range(1, world_size):
                    r_path = rank_path(r)
                    if r in arrived_ranks or _file_nonempty(r_path):
                        if not _wait_for_flock_release(r_path, exit_deadline, poll_interval_s):
                            all_siblings_done = False
                            exit_timed_out_rank = r
                            logger.warning(
                                "Timed out waiting for sibling rank %d of sub-host engine %s to finish "
                                "init_device; releasing %s.",
                                r,
                                engine_id,
                                lock_path,
                            )
        finally:
            if host_fd is not None:
                try:
                    if host_locked:
                        try:
                            os.ftruncate(host_fd, 0)
                        except OSError:
                            pass
                        fcntl.flock(host_fd, fcntl.LOCK_UN)
                finally:
                    os.close(host_fd)
            try:
                if leader_locked:
                    fcntl.flock(leader_fd, fcntl.LOCK_UN)
            finally:
                os.close(leader_fd)
            cleanup_paths = [leader_path]
            if succeeded and all_siblings_done and len(arrived_ranks) == world_size - 1:
                cleanup_paths.extend([go_path, *(rank_path(r) for r in range(1, world_size))])
            for path in cleanup_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        if exit_timed_out_rank is not None:
            raise TimeoutError(
                f"Timed out after {safety_timeout_s}s waiting for sibling rank {exit_timed_out_rank} of "
                f"sub-host engine {engine_id} to finish init_device."
            )


def _tp1_chip_handshake_paths(owner: str, chip: int, base_dir: str = "/tmp") -> tuple[str, str]:
    """Per-controller per-chip liveness lock and ready marker for TPU 7x TP=1 dual-core bootstrap."""
    safe_owner = re.sub(r"[^A-Za-z0-9_.-]", "_", owner)
    alive_path = os.path.join(base_dir, f"verl_tpu_subhost_chip{chip}_{safe_owner}.alive")
    ready_path = os.path.join(base_dir, f"verl_tpu_subhost_chip{chip}_{safe_owner}.ready")
    return alive_path, ready_path


def _acquire_tp1_chip_alive_lock(
    alive_path: str,
    ready_path: str,
    core: int,
    *,
    safety_timeout_s: float | None = _SUBHOST_INIT_SAFETY_TIMEOUT_S,
    poll_interval_s: float = _SUBHOST_INIT_POLL_INTERVAL_S,
) -> int:
    """Acquire the persistent per-chip liveness ``flock`` held by a TP=1 even core until process exit."""
    lock_dir = os.path.dirname(alive_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)
    fd = os.open(alive_path, os.O_RDWR | os.O_CREAT, 0o666)
    deadline = None if safety_timeout_s is None else time.monotonic() + safety_timeout_s
    if not _acquire_flock(fd, deadline, poll_interval_s):
        os.close(fd)
        raise TimeoutError(
            f"TPU 7x TP=1 even core {core} timed out after {safety_timeout_s}s acquiring chip liveness lock "
            f"{alive_path}."
        )
    # Remove any stale ready marker left by a previously exited even core on this chip before init_device runs.
    try:
        os.unlink(ready_path)
    except FileNotFoundError:
        pass
    os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
    return fd


def _wait_for_tp1_even_core_ready(
    alive_path: str,
    ready_path: str,
    core: int,
    *,
    peer_appear_timeout_s: float = _SUBHOST_INIT_PEER_APPEAR_TIMEOUT_S,
    safety_timeout_s: float | None = _SUBHOST_INIT_SAFETY_TIMEOUT_S,
    poll_interval_s: float = _SUBHOST_INIT_POLL_INTERVAL_S,
) -> None:
    """Block a TPU 7x TP=1 odd core until its chip's even core holds ``alive_path`` and publishes ``ready_path``.

    - ``alive_path`` locked and ``ready_path`` present: even core completed ``StartSession`` and is alive -> proceed.
    - ``alive_path`` locked without ``ready_path``: even core is queued or initializing -> keep waiting up to
      ``safety_timeout_s`` (reusing the host-serializer safety bound).
    - ``alive_path`` unlocked: even core has not started yet or exited -> wait up to ``peer_appear_timeout_s`` and
      raise ``TimeoutError`` (ignoring any stale ``ready_path`` left by an exited process).
    """
    now = time.monotonic()
    appear_deadline = now + peer_appear_timeout_s
    safety_deadline = None if safety_timeout_s is None else now + safety_timeout_s
    while True:
        alive = _is_lock_held(alive_path)
        if alive and os.path.exists(ready_path):
            return
        t = time.monotonic()
        if safety_deadline is not None and t >= safety_deadline:
            raise TimeoutError(
                f"TPU 7x TP=1 core {core} timed out after {safety_timeout_s}s waiting for chip "
                f"{core // 2}'s even core (core {core - 1}) to finish StartSession ({ready_path})."
            )
        if not alive and t >= appear_deadline:
            raise TimeoutError(
                f"TPU 7x TP=1 core {core} timed out after {peer_appear_timeout_s}s waiting for chip "
                f"{core // 2}'s even core (core {core - 1}) to appear and hold {alive_path}."
            )
        time.sleep(poll_interval_s)


def patch_tpu_worker_subhost_start_session() -> None:
    """Serialize ``TPUWorker.init_device`` across co-located sub-host engines and order TPU 7x TP=1 chip cores.

    Installed only when ``SUBHOST_MEGASCALE_PORT_ENV`` is set (sub-host engines where ``engines_per_host > 1``) and
    either host-wide serialization is enabled or TPU 7x TP=1 per-chip ordering is active. Engines without
    ``SUBHOST_MEGASCALE_PORT_ENV`` (whole-host, multi-host, and non-sub-host engines) never install this wrapper.

    On TPU 7x dual-core chips running two independent TP=1 engines, only the even core runs ``tpunetd``
    ``StartSession`` (the odd core logs ``process_owns_chips is false, using NoOpControl``). If an odd core enters
    TPU runtime initialization before its paired even core completes ``StartSession``, the odd core can observe a
    stale multi-chip id and abort with ``tpu_topology.cc: Check failed: index < ReduceProductDimensions(bounds)``.
    Keeping the odd-core wait inside this single wrapper *before* acquiring ``SUBHOST_START_SESSION_LOCK_FILE``
    guarantees an odd core arriving first cannot hold the host lock while waiting for its paired even core.
    ``VERL_TPU_SUBHOST_SERIALIZE_START_SESSION=0`` disables host-wide mutual exclusion only; TP=1 per-chip ordering
    remains enforced because odd-first initialization is a correctness failure (tracked upstream in
    ``google-pytorch/torch_tpu#7213``; remove this ordering workaround once fixed there).
    """
    if not subhost_serialize_start_session_enabled() and not _subhost_tp1_chip_order_active():
        return
    try:
        import vllm_torchtpu.worker.tpu_worker as tpu_worker
    except ImportError:
        return
    worker_cls = getattr(tpu_worker, "TPUWorker", None)
    if worker_cls is None:
        return
    original = getattr(worker_cls, "init_device", None)
    if original is None or getattr(original, "_verl_subhost_start_session", False):
        return

    def init_device(self, *args, **kwargs):
        rank = int(getattr(self, "rank", 0))
        pc = getattr(self, "parallel_config", None)
        world_size = int(getattr(pc, "world_size", 1))
        engine_id = (
            os.environ.get(SUBHOST_ENGINE_ID_ENV)
            or f"ppid{os.getppid()}_p{os.environ.get(SUBHOST_MEGASCALE_PORT_ENV, '0')}"
        )
        serialize = subhost_serialize_start_session_enabled()
        owner = os.environ.get(_SUBHOST_OWNER_ENV, "")
        tp1_order = world_size == 1 and bool(owner) and os.environ.get("TORCH_TPU_TOPOLOGY") == _V7X_SINGLE_CORE_BOUNDS

        def _run_serialized():
            if not serialize:
                return original(self, *args, **kwargs)
            with serialize_subhost_start_session(
                rank=rank,
                world_size=world_size,
                engine_id=engine_id,
                lock_path=SUBHOST_START_SESSION_LOCK_FILE,
            ):
                return original(self, *args, **kwargs)

        if not tp1_order:
            return _run_serialized()

        core = int(os.environ.get("TPU_LOCAL_RANK_OFFSET", "0"))
        alive_path, ready_path = _tp1_chip_handshake_paths(owner, core // 2)
        if core % 2 == 1:
            # Wait BEFORE acquiring the host StartSession lock so an odd core arriving first cannot deadlock.
            _wait_for_tp1_even_core_ready(alive_path, ready_path, core)
            return _run_serialized()

        alive_fd = _acquire_tp1_chip_alive_lock(alive_path, ready_path, core)
        try:
            ctx = (
                serialize_subhost_start_session(
                    rank=rank,
                    world_size=world_size,
                    engine_id=engine_id,
                    lock_path=SUBHOST_START_SESSION_LOCK_FILE,
                )
                if serialize
                else contextlib.nullcontext()
            )
            with ctx:
                result = original(self, *args, **kwargs)
                if hasattr(torch, "tpu"):
                    # Force TPU runtime initialization if deferred so this chip's StartSession finishes before
                    # the paired odd core enters init_device.
                    torch.empty(1, device="tpu")
                _write_atomic(ready_path, f"{os.getpid()}\n".encode())
        except BaseException:
            try:
                try:
                    os.unlink(ready_path)
                except OSError:
                    pass
                fcntl.flock(alive_fd, fcntl.LOCK_UN)
            finally:
                os.close(alive_fd)
            raise
        _TP1_CHIP_ALIVE_FDS.append(alive_fd)
        return result

    init_device._verl_subhost_start_session = True  # type: ignore[attr-defined]
    worker_cls.init_device = init_device
    logger.info(
        "Applied TPU sub-host StartSession serialization patch to TPUWorker.init_device (lock=%s).",
        SUBHOST_START_SESSION_LOCK_FILE,
    )


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
