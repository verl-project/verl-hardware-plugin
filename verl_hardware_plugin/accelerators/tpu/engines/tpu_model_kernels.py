# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Optional torchtitan TPU kernels of a model, selected by ``engine_kwargs.tpu.model_kernels``.

None of this is needed to train a model on torchtitan ``main``: there the TorchTitan engine runs the model as
torchtitan builds it. Some torchtitan builds add faster TPU kernels for a model that its benchmark recipe turns on,
and this module turns on the same kernels in the engine. For Qwen3.5 (the 35B-A3B benchmark recipe
``qwen35_moe_35b_split_tokamax_benchmark``) these come from the optimization PRs on google-pytorch/torchtitan:

* ``moe``: the TPU MoE converters (grouped experts, all-to-all dispatcher) with Pallas paired grouped matmuls on
  kernel-native expert weights (https://github.com/google-pytorch/torchtitan/pull/73, ``_use_tpu_moe_kernels``),
  with 512-row tiles (https://github.com/google-pytorch/torchtitan/pull/79).
* ``kda``: Tokamax's Mosaic KDA kernels for the GatedDeltaNet (https://github.com/google-pytorch/torchtitan/pull/74,
  ``TOKAMAX_GATED_DELTA_OVERRIDE``; needs a Tokamax build with the KDA op). Packed-document resets go through
  Tokamax's ``segment_ids``, see ``set_packed_document_budget``.
* ``lce``: fused linear cross-entropy tiles (1024, 2048, 640) (https://github.com/google-pytorch/torchtitan/pull/78).

The Pallas gated RMSNorm (https://github.com/google-pytorch/torchtitan/pull/80) runs on its own in such a build.

``model_kernels``: ``auto`` (default) enables what the installed torchtitan exposes, so the same plugin runs both
torchtitan ``main`` (nothing is enabled) and an optimized build (everything is enabled) with only the container
image changed; ``none`` disables all; ``all`` enables every kernel of the model; or a comma-separated subset.

Some models also need a mandatory TPU setup, independent of ``model_kernels``: ``model_setup`` looks it up in
``_MODEL_SETUPS`` (today Qwen3 MoE). Add a ``ModelSetup`` entry there for a new model; the engine stays unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Optional, TypedDict

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The optional kernels each torchtitan model supports.
_SUPPORTED_KERNELS = {"qwen3_5": frozenset({"moe", "kda", "lce"})}

# Not ``qwen2_moe``: torchtitan's qwen3 model has no Qwen2-MoE shared expert.
_QWEN3_MOE_HF_MODEL_TYPES = frozenset({"qwen3_moe"})
# Pallas grouped-matmul row tile and dWeight tiling; tokens * top_k must be a multiple of the row tile.
_QWEN3_MOE_GROUPED_MM_TILE_M = 256
_QWEN3_MOE_GROUPED_MM_TGMM_TILING = (256, 0, 0)


class LceBlockSizes(TypedDict, total=False):
    """Block sizes of ``fused_linear_logprobs_tpu``; unset keys keep the kernel's defaults."""

    b_block_size: int
    h_block_size: int
    v_block_size: int


# Fused linear cross-entropy block sizes of ``lce`` (google-pytorch/torchtitan#78).
_LCE_BLOCK_SIZES = LceBlockSizes(b_block_size=1024, h_block_size=2048, v_block_size=640)
# Pallas grouped-matmul tiles of ``moe`` (google-pytorch/torchtitan#79).
_MOE_GROUPED_MM_TILE_M = 512
_MOE_GROUPED_MM_TGMM_TILING = (512, 0, 0)


def _qwen3_5_config_registry() -> Any:
    from torchtitan.experiments.tpu.qwen3_5 import config_registry

    return config_registry


def _available_kernels(torchtitan_name: str) -> frozenset[str]:
    """The optional kernels that the installed ``torchtitan`` exposes for ``torchtitan_name``."""
    if torchtitan_name != "qwen3_5":
        return frozenset()
    try:
        config_registry = _qwen3_5_config_registry()
    except ImportError:
        return frozenset()
    kernels: set[str] = set()
    if hasattr(config_registry, "_use_tpu_moe_kernels"):  # google-pytorch/torchtitan#73
        kernels.update(("moe", "lce"))
    if hasattr(config_registry, "TOKAMAX_GATED_DELTA_OVERRIDE") and importlib.util.find_spec("tokamax") is not None:
        kernels.add("kda")  # google-pytorch/torchtitan#74
    return frozenset(kernels)


def resolve_tpu_model_kernels(torchtitan_name: str, requested: Optional[str] = None) -> frozenset[str]:
    """The optional kernels that torchtitan model ``torchtitan_name`` runs, from ``engine_kwargs.tpu.model_kernels``.

    Raises:
        ValueError: on a kernel name that ``torchtitan_name`` does not support, or that the installed torchtitan
            does not provide.
    """
    supported = _SUPPORTED_KERNELS.get(torchtitan_name, frozenset())
    raw = (requested or "").strip().lower()
    if not raw or raw == "auto":
        return _available_kernels(torchtitan_name) & supported
    names = {k.strip() for k in raw.split(",") if k.strip()}
    if names <= {"none", "off", "false", "0"}:
        return frozenset()
    if "all" in names:
        names = set(supported)
    unknown = names - supported
    if unknown:
        raise ValueError(f"engine_kwargs.tpu.model_kernels: {sorted(unknown)} not supported for {torchtitan_name}")
    missing = names - _available_kernels(torchtitan_name)
    if missing:
        raise ValueError(
            f"engine_kwargs.tpu.model_kernels: {sorted(missing)} not provided by the installed torchtitan "
            f"(see google-pytorch/torchtitan#73, #74) or Tokamax"
        )
    return frozenset(names)


def model_registry_kwargs(kernels: frozenset[str]) -> dict[str, Any]:
    """Extra ``model_registry`` kwargs of ``kernels``: the TPU MoE converters for ``moe``."""
    if "moe" not in kernels:
        return {}
    return {"converters": list(_qwen3_5_config_registry()._TPU_MOE_CONVERTERS)}


def configure_model_spec(model_spec: Any, kernels: frozenset[str]) -> None:
    """Sets the model-config options of ``kernels`` on ``model_spec.model`` in place (``moe``: Pallas grouped mm)."""
    if "moe" not in kernels:
        return
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

    _qwen3_5_config_registry()._use_tpu_moe_kernels(SimpleNamespace(model_spec=model_spec))
    for _, experts, _, _ in model_spec.model.traverse(TpuGroupedExperts.Config):
        experts.pallas_grouped_mm_tile_m = _MOE_GROUPED_MM_TILE_M
        experts.pallas_grouped_mm_tgmm_tiling = _MOE_GROUPED_MM_TGMM_TILING


def is_qwen3_moe(torchtitan_name: str, hf_config: Any) -> bool:
    """Whether ``configure_qwen3_moe`` applies (Qwen3 MoE, not dense Qwen3/Qwen2 or Qwen3.5)."""
    return torchtitan_name == "qwen3" and getattr(hf_config, "model_type", None) in _QWEN3_MOE_HF_MODEL_TYPES


def qwen3_moe_model_registry_kwargs() -> dict[str, Any]:
    """``model_registry`` kwargs for Qwen3 MoE: torchtitan's TPU token dispatcher and grouped-experts converters.

    Same order as torchtitan's Qwen3-30B-A3B TPU recipe. The default dispatcher's backward is ~5x slower on TPU.
    """
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExpertsConverter
    from torchtitan.experiments.tpu.token_dispatcher import TpuTokenDispatcherConverter

    return {"converters": [TpuTokenDispatcherConverter.Config(), TpuGroupedExpertsConverter.Config()]}


def configure_qwen3_moe(model_spec: Any, *, expert_parallel_size: int, seq_bucket_size: int) -> None:
    """Sets up a Qwen3 MoE ``model_spec.model`` for RL on TPU, in place.

    - Turns off expert-bias load balancing and the aux loss: the HF checkpoint the rollout serves has no expert bias,
      so the trainer must route like it.
    - Runs the routed experts with the Pallas grouped matmul in the canonical layout: the native
      ``torch._grouped_mm`` backward is wrong on TPU (torch_tpu#2668).

    Raises:
        ValueError: if ``expert_parallel_size > 1``, or if ``seq_bucket_size * top_k`` is not a multiple of the
            256-row tile.
    """
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    from torchtitan.models.common.moe import MoE

    if expert_parallel_size > 1:
        raise ValueError(
            f"expert_parallel_size={expert_parallel_size} is not supported for Qwen3 MoE on TPU: the Pallas grouped "
            "matmul needs EP=1 (exact, tile-aligned routed rows), and the native torch._grouped_mm fallback computes "
            "a wrong expert dWeight on TPU (torch_tpu#2668). Use expert_parallel_size=1 with FSDP "
            "(data_parallel_shard_size) instead."
        )
    model_config = model_spec.model
    moe_configs = [moe for _, moe, _, _ in model_config.traverse(MoE.Config)]
    if not moe_configs:
        raise ValueError("Qwen3 MoE: the torchtitan model config has no MoE layer")
    tile_m = _QWEN3_MOE_GROUPED_MM_TILE_M
    token_multiple = max(int(seq_bucket_size), 1)  # bucket sizes <= 1 disable bucketing
    for moe in moe_configs:
        top_k = moe.router.top_k
        if (token_multiple * top_k) % tile_m:
            raise ValueError(
                f"Qwen3 MoE on TPU: the Pallas grouped matmul needs tokens * top_k to be a multiple of {tile_m}, but "
                f"token counts are only guaranteed to be multiples of the sequence bucket {token_multiple} "
                f"(top_k={top_k}). Set engine_kwargs.tpu.seq_bucket_size so that it times top_k is a multiple of "
                f"{tile_m} (the default 256 is)."
            )
        moe.load_balance_coeff = None
        moe.router.aux_loss = None
    experts_configs = [experts for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config)]
    if len(experts_configs) != len(moe_configs):
        raise ValueError(
            f"Qwen3 MoE: {len(experts_configs)} of {len(moe_configs)} MoE layers use TpuGroupedExperts; build the "
            "model with qwen3_moe_model_registry_kwargs()"
        )
    for experts in experts_configs:
        experts.use_pallas_grouped_mm = True
        experts.use_pallas_paired_grouped_mm = False
        experts.use_kernel_native_weight_layout = False  # keep the canonical parameter names and layout
        experts.pallas_grouped_mm_tile_m = tile_m
        experts.pallas_grouped_mm_tgmm_tiling = _QWEN3_MOE_GROUPED_MM_TGMM_TILING
    logger.info(
        "Qwen3 MoE: %d layers route without expert bias or aux loss; TPU token dispatcher and Pallas grouped "
        "matmul experts (tile_m=%d)",
        len(moe_configs),
        tile_m,
    )


@dataclasses.dataclass(frozen=True)
class ModelSetupContext:
    """Engine settings a ``ModelSetup.configure`` may need."""

    expert_parallel_size: int
    seq_bucket_size: int


@dataclasses.dataclass(frozen=True)
class ModelSetup:
    """Mandatory TPU setup of one model family (not an optional ``model_kernels`` entry)."""

    name: str
    matches: Callable[[str, Any], bool]  # (torchtitan model name, HF config) -> bool
    registry_kwargs: Callable[[], dict[str, Any]] = dict  # extra ``model_registry`` kwargs
    configure: Optional[Callable[[Any, ModelSetupContext], None]] = None  # in-place ``model_spec`` fix-ups


_MODEL_SETUPS: tuple[ModelSetup, ...] = (
    ModelSetup(
        name="qwen3_moe",
        matches=is_qwen3_moe,
        registry_kwargs=qwen3_moe_model_registry_kwargs,
        configure=lambda spec, ctx: configure_qwen3_moe(
            spec, expert_parallel_size=ctx.expert_parallel_size, seq_bucket_size=ctx.seq_bucket_size
        ),
    ),
)


def model_setup(torchtitan_name: str, hf_config: Any) -> Optional[ModelSetup]:
    """The mandatory TPU setup of the model, or ``None`` if it needs none."""
    return next((setup for setup in _MODEL_SETUPS if setup.matches(torchtitan_name, hf_config)), None)


def override_imports(kernels: frozenset[str]) -> list[str]:
    """torchtitan ``override.imports`` of ``kernels``: Tokamax's GatedDeltaNet kernel for ``kda``."""
    if "kda" not in kernels:
        return []
    return [_qwen3_5_config_registry().TOKAMAX_GATED_DELTA_OVERRIDE]


def lce_block_sizes(kernels: frozenset[str], hidden_size: int, vocab_size: int) -> LceBlockSizes:
    """Fused linear cross-entropy block sizes of ``kernels`` (empty: the kernel's defaults)."""
    if (
        "lce" not in kernels
        or hidden_size % _LCE_BLOCK_SIZES["h_block_size"] != 0
        or vocab_size % _LCE_BLOCK_SIZES["v_block_size"] != 0
    ):
        return LceBlockSizes()
    return LceBlockSizes(**_LCE_BLOCK_SIZES)


def set_packed_document_budget(modules: Any, num_documents: int) -> int:
    """Sets ``max_num_segments`` of every Tokamax GatedDeltaNet kernel in ``modules`` to ``num_documents``.

    Only for ``kda``. The engine packs the sequences of a micro-batch into one row (``use_remove_padding``) and pads
    the row to the sequence bucket; each sequence is one document whose GatedDeltaNet state must start from zero.
    Tokamax's KDA kernels restart the recurrence at each of the first ``max_num_segments`` documents of a row, and
    later documents share one recurrence. The torchtitan default derives the budget from the row length (one
    document per 512 tokens), which is too small for RL rows: two ~300-token sequences pad to a 768-token row whose
    default budget is 1, so the second sequence would continue the first one's state. The engine knows how many
    sequences each micro-batch packs, so it sets the budget to that plus one for the padding tail. Returns the
    number of kernels updated (0 when unchanged, so no recompilation).
    """
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    updated = 0
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        for module in model.modules():
            # By class name, anywhere in the MRO (parallelization may subclass modules), so that torchtitan builds
            # without Tokamax need not import it.
            is_kernel = any(c.__name__ == "TokamaxGatedDeltaKernel" for c in type(module).__mro__)
            if is_kernel and hasattr(module, "max_num_segments") and module.max_num_segments != num_documents:
                module.max_num_segments = int(num_documents)
                updated += 1
    return updated
