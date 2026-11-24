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
"""

import functools
import importlib
import inspect
import logging
import multiprocessing.process
import os
import sys

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
    if None in (cfg_cls, routed_experts_cls, should_quantize_on_load, fp8_cfg_cls, fp8_moe_cls):
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
    if os.environ.get(SUBHOST_MEGASCALE_PORT_ENV):
        patch_tpu_worker_subhost_env()
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


# Set by tpu_vllm.subhost_engine_env on an engine that shares its host: the first port of the engine's MegaScale block.
SUBHOST_MEGASCALE_PORT_ENV = "VERL_TPU_SUBHOST_MEGASCALE_PORT"
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


def subhost_worker_env(rank: int, local_rank: int, megascale_port: int) -> dict[str, str]:
    """The per-worker variables of an engine that shares its host; ``local_rank`` includes the engine's offset."""
    return {
        "TPU_VISIBLE_DEVICES": str(local_rank),
        "TPU_VISIBLE_CHIPS": str(local_rank),
        "MEGASCALE_PORT": str(megascale_port + 2 * rank),
        "MEGASCALE_DEBUG_PORT": str(megascale_port + 2 * rank + 1),
        "TPU_LOG_DIR": f"/tmp/tpu_logs/rank_{local_rank}",
    }


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
        env = subhost_worker_env(int(rank), int(local_rank), megascale_port)
        os.makedirs(env["TPU_LOG_DIR"], exist_ok=True)
        os.environ.update(env)

    _configure_tpu_process_env._verl_subhost_env = True  # type: ignore[attr-defined]
    tpu_worker._configure_tpu_process_env = _configure_tpu_process_env
    logger.info("Applied the TPU patch for engines that share a host (MegaScale ports from %d).", megascale_port)


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
