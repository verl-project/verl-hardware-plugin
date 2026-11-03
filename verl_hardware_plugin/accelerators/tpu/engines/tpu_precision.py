# Copyright 2024-2025 BAAI and Google LLC
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TorchTitan precision corrections from jialei777/verl-upstream#76.

The existing parameter-storage configuration is preserved. Attention kernels use
BF16; RMSNorm (including its affine-gradient reduction), SiLU arithmetic, and
optionally LM-head accumulators use FP32.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from functools import cache
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


class _TPURMSNormAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, normalized: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(normalized, weight)
        return normalized * weight.float()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, weight = ctx.saved_tensors
        gradient = gradient.float()
        grad_normalized = gradient * weight.float()
        product = gradient * normalized
        dimensions = tuple(range(product.ndim - weight.ndim))
        grad_weight = product.sum(dim=dimensions, dtype=torch.float32) if dimensions else product
        # Autograd casts at the parameter edge. Casting here permits XLA to push
        # BF16 rounding inside the sum and defeats FP32 accumulation.
        return grad_normalized, grad_weight


def _tpu_fp32_rms_norm(self: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        weight = self.weight
        if weight is not None and weight.dtype == torch.bfloat16:
            normalized = F.rms_norm(x.float(), self.normalized_shape, None, self.eps)
            return _TPURMSNormAffine.apply(normalized, weight)
        return F.rms_norm(x.float(), self.normalized_shape, None if weight is None else weight.float(), self.eps)


def _tpu_fp32_feed_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate = self.w1(x)
    activated = F.silu(gate.float()).to(gate.dtype)
    return self.w2(activated * self.w3(x))


def apply_tpu_precision_patches(model_parts: list[nn.Module]) -> None:
    """Keep norm outputs FP32 through RoPE and use FP32 SiLU before BF16 rounding.

    Inspect the precise FeedForward type to preserve specialized subclasses' gates
    and kernels. SimpleFSDP dynamically subclasses FeedForward, so unwrap that type.
    All patches are installed before the first lazily compiled forward.
    """
    torch.set_float32_matmul_precision("highest")
    for model in model_parts:
        for module in model.modules():
            if isinstance(module, nn.RMSNorm):
                module.forward = MethodType(_tpu_fp32_rms_norm, module)
            module_type = type(module)
            if (
                module_type.__name__ == "SimpleFSDPFeedForward"
                and hasattr(module, "_simple_fsdp_parametrization_list")
                and len(module_type.__bases__) == 1
            ):
                module_type = module_type.__bases__[0]
            if module_type.__module__ != "torchtitan.models.common.feed_forward":
                continue
            from torchtitan.models.common.feed_forward import FeedForward

            if module_type is FeedForward:
                module.forward = MethodType(_tpu_fp32_feed_forward, module)


class _TPUFp32LogitsLinear(torch.autograd.Function):
    """Retain FP32 accumulators of BF16 dot products, with BF16 backward dots."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.x_dtype, ctx.w_dtype = x.dtype, weight.dtype
        x, weight = x.bfloat16(), weight.bfloat16()
        ctx.save_for_backward(x, weight)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium")
        try:
            return x.float() @ weight.float().t()
        finally:
            torch.set_float32_matmul_precision(previous)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight = ctx.saved_tensors
        gradient = gradient.bfloat16()
        grad_x = gradient @ weight
        grad_weight = gradient.reshape(-1, gradient.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        return grad_x.to(ctx.x_dtype), grad_weight.to(ctx.w_dtype)


def _tpu_fp32_logits_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if getattr(self, "bias", None) is not None:
        raise ValueError("fp32_logits supports a bias-free LM head only")
    with torch.autocast(device_type=x.device.type, enabled=False):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return _TPUFp32LogitsLinear.apply(x, weight)


def apply_fp32_logits_tpu(modules: Any) -> int:
    """Patch eager, bias-free LM heads while preserving parameter storage and tied weights."""
    model_list = modules if isinstance(modules, list | tuple | nn.ModuleList) else [modules]
    patched = 0
    for model in model_list:
        head = getattr(model, "lm_head", None)
        if not isinstance(head, nn.Module) or getattr(head.forward, "__func__", None) is _tpu_fp32_logits_forward:
            continue
        if getattr(head, "bias", None) is not None:
            raise ValueError("fp32_logits supports a bias-free LM head only")
        head.forward = MethodType(_tpu_fp32_logits_forward, head)
        patched += 1
    if patched:
        logger.warning("FP32 logits enabled on %d TPU LM heads", patched)
    return patched


_attention_dtype_fallback_logged = False


def attention_compute_dtype(x: torch.Tensor) -> torch.dtype | None:
    """Use the autocast dtype at the attention boundary, retaining FP32 otherwise."""
    global _attention_dtype_fallback_logged
    try:
        return torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else None
    except (RuntimeError, TypeError, ValueError) as error:
        if not _attention_dtype_fallback_logged:
            _attention_dtype_fallback_logged = True
            logger.warning("Cannot query %s autocast (%s); TPU attention will use BF16", x.device.type, error)
        return torch.bfloat16


@cache
def precision_splash_attention_cls(splash_cls: type[nn.Module]) -> type[nn.Module]:
    """Adapt torchtitan's SplashAttention while retaining its CP/layout handling."""

    class PrecisionSplashAttention(splash_cls):
        def forward(self, q, k, v, **kwargs):
            # Keep torchtitan's CPU fallback unrounded. TPU Splash always takes
            # BF16 Q/K/V after the FP32 norms and RoPE, then restores Q's dtype.
            if q.device.type == "cpu":
                return super().forward(q, k, v, **kwargs)
            input_dtype = q.dtype
            result = super().forward(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), **kwargs)
            return result.to(input_dtype)

    return PrecisionSplashAttention


@contextmanager
def tpu_precision_compiler_context():
    """Keep XLA from rounding RMSNorm affine products before their FP32 sum.

    Apply only while trainer operations are traced/materialized. The separate
    JAX/vLLM rollout compiler does not inherit this TorchTPU-local setting.
    """
    try:
        from torch_tpu._internal.compiler_options import custom_compiler_options
    except ImportError as error:
        # Hosts running CPU tests need neither TorchTPU nor its compiler options.
        if not hasattr(torch, "tpu") or "torch_tpu" not in sys.modules:
            yield
            return
        raise RuntimeError("TPU trainer precision requires TorchTPU's custom_compiler_options API") from error
    with custom_compiler_options({"xla_disable_hlo_passes": "bfloat16-propagation"}):
        yield
