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
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any, Optional, TypedDict

import torch

# The optional kernels each torchtitan model supports.
_SUPPORTED_KERNELS = {"qwen3_5": frozenset({"moe", "kda", "lce"})}


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
_MOE_GROUPED_MM_TILES_MAX_DIM = 2048  # the model width those tiles fit (VMEM)
# dWeight (TGMM) tiles of wider models (Qwen3.5-397B-A17B: hidden 4096, expert hidden 1024). The full-K/N tiles of
# ``_use_tpu_moe_kernels`` keep (K, N) FP32 accumulators in VMEM: 16 MB for the down projection and 2 x 16 MB for the
# paired gate/up projection, past the scoped VMEM limit ("CompileTimeScopedVmemOom" in moe_tgmm_wgrad). 1024-wide
# K/N slabs keep each accumulator at 4 MB. Paired gate/up needs a full N tile, so only K is tiled there.
_MOE_WIDE_TGMM_TILING = (256, 1024, 0)
_MOE_WIDE_W2_TGMM_TILING = (256, 1024, 1024)
# Default ``TpuQwen35TokenDispatcher.capacity_factor`` under EP > 1. torchtitan's pretraining default (1.25) drops
# skewed RL / math tokens at EP 32-64 while vLLM rollout is dropless, inflating logp_trainer - logp_vllm.
DEFAULT_EP_CAPACITY_FACTOR = 4.0


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
        # The 512-row tiles were tuned for Qwen3.5-35B-A3B (hidden 2048). At hidden 4096 (Qwen3.5-397B-A17B) the
        # weight-gradient kernels do not fit in VMEM with full K/N tiles: tile K/N of the dWeight instead.
        if (getattr(experts, "dim", None) or 0) > _MOE_GROUPED_MM_TILES_MAX_DIM:
            experts.pallas_grouped_mm_tgmm_tiling = _MOE_WIDE_TGMM_TILING
            experts.pallas_grouped_mm_w2_tgmm_tiling = _MOE_WIDE_W2_TGMM_TILING
            continue
        experts.pallas_grouped_mm_tile_m = _MOE_GROUPED_MM_TILE_M
        experts.pallas_grouped_mm_tgmm_tiling = _MOE_GROUPED_MM_TGMM_TILING


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


def set_ep_capacity_factor(modules: Any, capacity_factor: float) -> int:
    """Sets ``capacity_factor`` of every ``TpuQwen35TokenDispatcher`` in ``modules``.

    torchtitan's ``TpuQwen35Model`` switches MoE layers to ``TpuQwen35TokenDispatcher(capacity_factor=1.25)`` when
    ``expert_parallel_degree > 1``. ``RoutedExperts`` holds ``token_dispatcher`` as a ``Configurable`` object
    attribute (not an ``nn.Module``), so inspect both each submodule and its ``token_dispatcher`` attribute. On skewed
    RL / math batches at EP 32-64, a 1.25 per-rank cap drops real tokens while vLLM rollout is dropless, corrupting
    ``logp_trainer - logp_vllm``. Must be called before the first compiled forward pass traces ``dispatch``.
    """
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    cap = float(capacity_factor)
    seen: set[int] = set()
    updated = 0
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        for module in model.modules():
            for candidate in (module, getattr(module, "token_dispatcher", None)):
                if candidate is None or id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                is_dispatcher = any(c.__name__ == "TpuQwen35TokenDispatcher" for c in type(candidate).__mro__)
                if is_dispatcher and hasattr(candidate, "capacity_factor") and candidate.capacity_factor != cap:
                    candidate.capacity_factor = cap
                    updated += 1
    return updated
