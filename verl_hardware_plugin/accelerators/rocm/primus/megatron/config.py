# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Engine config for Primus Megatron and the projection of the Primus YAML onto it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.workers.config import McoreEngineConfig

# Fields of McoreEngineConfig that the Primus YAML owns. The Hydra group fills them from the
# YAML through ``${primus:...}``; the engine re-reads the YAML and rejects a mismatch.
PARALLEL_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
    "use_distributed_optimizer",
)

TRANSFORMER_FIELDS = (
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "apply_rope_fusion",
    "masked_softmax_fusion",
    "bias_activation_fusion",
    "bias_dropout_fusion",
    "gradient_accumulation_fusion",
    "moe_grouped_gemm",
    "moe_permute_fusion",
    "moe_router_dtype",
    "moe_aux_loss_coeff",
    "attention_backend",
)

TP1_ONLY_TURBO_FLAGS = (
    "use_turbo_gemm",
    "use_turbo_parallel_linear",
    "use_turbo_grouped_gemm",
    "use_turbo_grouped_mlp",
)


@dataclass
class PrimusMcoreEngineConfig(McoreEngineConfig):
    """``McoreEngineConfig`` plus the Primus experiment that configures it.

    ``strategy`` stays ``megatron`` so every verl code path written for Megatron applies. With an
    empty ``primus_config_path`` the engine is plain verl Megatron.
    """

    primus_config_path: str = ""
    primus_module: str = "pre_trainer"
    primus_path: Optional[str] = None
    primus_turbo_path: Optional[str] = None
    primus_overrides: list[str] = field(default_factory=list)
    # "kernels" (validated kernel/model patches), "none", "all", or a comma-separated id list.
    primus_patches: str = "kernels"
    primus_strict_config: bool = True
    _primus_params: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @property
    def primus_enabled(self) -> bool:
        return bool(self.primus_config_path)


def apply_primus_params(config: PrimusMcoreEngineConfig, params: dict[str, Any]) -> None:
    """Check the YAML-owned fields and fold transformer settings into ``override_transformer_config``."""
    if config.primus_strict_config:
        mismatched = [
            f"{name}: config={getattr(config, name)!r} yaml={params[name]!r}"
            for name in PARALLEL_FIELDS
            if params.get(name) is not None and getattr(config, name) != params[name]
        ]
        if mismatched:
            raise ValueError(
                "verl Megatron settings differ from the Primus YAML (change them in the YAML or with "
                "primus_overrides): " + "; ".join(mismatched)
            )
    overrides = dict(config.override_transformer_config)
    for key in TRANSFORMER_FIELDS:
        if params.get(key) is not None:
            overrides.setdefault(key, params[key])
    if params.get("use_packed_attention") and overrides.get("attention_backend", "auto") in (None, "auto"):
        overrides["attention_backend"] = "fused"
    object.__setattr__(config, "override_transformer_config", overrides)
    object.__setattr__(config, "_primus_params", dict(params))


def validate_primus_capabilities(config: PrimusMcoreEngineConfig, params: dict[str, Any], model_config) -> None:
    """Reject settings that would silently drop packed THD / CP or use unvalidated Turbo paths."""
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    if any("Qwen3Moe" in name for name in architectures) and params.get("use_packed_attention", False):
        if not config.use_remove_padding:
            raise ValueError("Primus packed attention needs THD inputs: set actor_rollout_ref.model.use_remove_padding")
    backend = str(config.override_transformer_config.get("attention_backend", "fused"))
    if params.get("use_packed_attention") and "aiter" in backend.lower():
        raise ValueError("Packed THD attention must use TransformerEngine attention, not AITER")
    if params.get("use_turbo_attention", False):
        raise ValueError(
            "Primus-Turbo attention is not validated for RL training gradients; disable use_turbo_attention"
        )
    enabled = [name for name in TP1_ONLY_TURBO_FLAGS if params.get(name, False)]
    if config.tensor_model_parallel_size > 1 and enabled:
        raise ValueError(f"Primus-Turbo linear kernels require TP=1; enabled: {', '.join(enabled)}")
