# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bridge API gaps between Primus' Megatron patches and the Megatron-LM verl runs."""

from __future__ import annotations

import dataclasses
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def install_megatron_compat_shims() -> list[str]:
    """Install shims Primus patches need against the Megatron in this process."""
    installed = []
    if _install_te_grouped_mlp_submodules():
        installed.append("TEGroupedMLPSubmodules")
    if _install_spec_provider_eager_fallback():
        installed.append("PrimusTurboSpecProvider.fallback_to_eager_attn")
    if _install_spec_provider_residual_norm():
        installed.append("PrimusTurboSpecProvider.layer_norm(has_residual)")
    if _install_spec_provider_experts_builder():
        installed.append("PrimusTurboSpecProvider.grouped_mlp_modules")
    if _restore_triton_next_power_of_2():
        installed.append("triton.next_power_of_2")
    if installed:
        logger.info("Installed Primus Megatron compatibility shims: %s", ", ".join(installed))
    return installed


def _restore_triton_next_power_of_2() -> bool:
    try:
        import triton
        from triton.runtime.jit import ConstexprFunction
    except ImportError:
        return False
    fn = getattr(triton, "next_power_of_2", None)
    if fn is None or isinstance(fn, ConstexprFunction):
        return False
    triton.next_power_of_2 = triton.constexpr_function(fn)
    return True


def _install_te_grouped_mlp_submodules() -> bool:
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.moe import experts

    if hasattr(experts, "TEGroupedMLPSubmodules"):
        return False
    target = getattr(experts, "GroupedMLPSubmodules", MLPSubmodules)
    fields = {field.name for field in dataclasses.fields(target)}
    if not {"linear_fc1", "linear_fc2"} <= fields:
        return False
    experts.TEGroupedMLPSubmodules = target
    return True


def _install_spec_provider_residual_norm() -> bool:
    import inspect

    from megatron.core.extensions import transformer_engine_spec_provider as te_spec
    from megatron.core.extensions.transformer_engine import TENorm
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    residual_norm = getattr(te_spec, "_TENormWithResidual", None)
    if residual_norm is None:
        return False
    if "has_residual" in inspect.signature(PrimusTurboSpecProvider.layer_norm).parameters:
        return False

    original_layer_norm = PrimusTurboSpecProvider.layer_norm

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False, has_residual: bool = False):
        norm = original_layer_norm(self, rms_norm=rms_norm, for_qk=for_qk)
        return residual_norm if has_residual and norm is TENorm else norm

    PrimusTurboSpecProvider.layer_norm = layer_norm
    return True


def _install_spec_provider_experts_builder() -> bool:
    import functools

    try:
        from megatron.core.transformer.moe.moe_layer import ExpertsBuilder  # noqa: F401
    except ImportError:
        return False
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    original = PrimusTurboSpecProvider.grouped_mlp_modules
    if getattr(original, "_verl_experts_builder", False):
        return False

    @functools.wraps(original)
    def grouped_mlp_modules(self, moe_use_grouped_gemm: bool, *args, **kwargs):
        result = original(self, moe_use_grouped_gemm, *args, **kwargs)
        if not isinstance(result, tuple):
            return result
        module_class, submodules = result
        if submodules is None:
            return module_class
        fields = {field.name for field in dataclasses.fields(submodules)}
        if "activation_func" in fields and submodules.activation_func is None:
            submodules = dataclasses.replace(submodules, activation_func=self.activation_func())
        return functools.partial(module_class, submodules=submodules)

    grouped_mlp_modules._verl_experts_builder = True  # type: ignore[attr-defined]
    PrimusTurboSpecProvider.grouped_mlp_modules = grouped_mlp_modules
    return True


def _install_spec_provider_eager_fallback() -> bool:
    import inspect

    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    if "fallback_to_eager_attn" in inspect.signature(PrimusTurboSpecProvider.__init__).parameters:
        return False

    original_init = PrimusTurboSpecProvider.__init__
    original_core_attention = PrimusTurboSpecProvider.core_attention

    def __init__(self, *args, fallback_to_eager_attn: bool = False, **kwargs):
        original_init(self, *args, **kwargs)
        self.fallback_to_eager_attn = fallback_to_eager_attn

    def core_attention(self):
        if getattr(self, "fallback_to_eager_attn", False):
            return DotProductAttention
        return original_core_attention(self)

    PrimusTurboSpecProvider.__init__ = __init__
    PrimusTurboSpecProvider.core_attention = core_attention
    return True
