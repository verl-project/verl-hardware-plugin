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

import contextlib
import dataclasses
import importlib
import importlib.util
import inspect
import logging
import os
import re
import sys
from collections.abc import Mapping
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
_MOE_GROUPED_MM_TILES_MAX_DIM = 2048  # the model width those tiles fit (VMEM)
# dWeight (TGMM) tiles of wider models (Qwen3.5-397B-A17B: hidden 4096, expert hidden 1024). The full-K/N tiles of
# ``_use_tpu_moe_kernels`` keep (K, N) FP32 accumulators in VMEM: 16 MB for the down projection and 2 x 16 MB for the
# paired gate/up projection, past the scoped VMEM limit ("CompileTimeScopedVmemOom" in moe_tgmm_wgrad). 1024-wide
# K/N slabs keep each accumulator at 4 MB. Paired gate/up needs a full N tile, so only K is tiled there.
_MOE_WIDE_TGMM_TILING = (256, 1024, 0)
_MOE_WIDE_W2_TGMM_TILING = (256, 1024, 1024)
# Under >= 60 MiB scoped VMEM (65472 KiB on TPU v7x), TorchTitan's 397B recipe (google-pytorch/torchtitan#142) runs
# 512-row grouped-MM tiles with (512, 1024, 0) gate/up + down dWeight tiling and (512, 0, 2048) transposed down-proj
# dWeight tiling.
_SCOPED_VMEM_64MIB_MIN_KIB = 61440
_MOE_WIDE_64M_VMEM_TGMM_TILING = (512, 1024, 0)
_MOE_WIDE_64M_VMEM_W2_TRANSPOSED_TGMM_TILING = (512, 0, 2048)
# Default ``TpuQwen35TokenDispatcher.capacity_factor`` under EP > 1. torchtitan's pretraining default (1.25) drops
# skewed RL / math tokens at EP 32-64 while vLLM rollout is dropless, inflating logp_trainer - logp_vllm.
DEFAULT_EP_CAPACITY_FACTOR = 4.0
_SCOPED_VMEM_RE = re.compile(r"--xla_tpu_scoped_vmem_limit_kib(?:=|\s+)(\d+)")


def has_64mib_scoped_vmem(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``LIBTPU_INIT_ARGS`` (or ``VERL_TPU_TRAINER_LIBTPU_INIT_ARGS``) configures >= 60 MiB scoped VMEM."""
    env = os.environ if environ is None else environ
    combined = f"{env.get('LIBTPU_INIT_ARGS', '')} {env.get('VERL_TPU_TRAINER_LIBTPU_INIT_ARGS', '')}"
    matches = _SCOPED_VMEM_RE.findall(combined)
    return bool(matches) and int(matches[-1]) >= _SCOPED_VMEM_64MIB_MIN_KIB


def _qwen3_5_config_registry() -> Any:
    from torchtitan.experiments.tpu.qwen3_5 import config_registry

    return config_registry


def _has_tokamax_kda() -> bool:
    """Whether an installed Tokamax build provides ``tokamax._src.ops.kda`` (PyPI 0.0.14 does not)."""
    try:
        return importlib.util.find_spec("tokamax._src.ops.kda") is not None
    except (ImportError, ValueError, SyntaxError):
        return False


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
    if hasattr(config_registry, "TOKAMAX_GATED_DELTA_OVERRIDE") and _has_tokamax_kda():
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


def configure_model_spec(
    model_spec: Any,
    kernels: frozenset[str],
    *,
    expert_parallel_degree: int = 1,
    ep_token_chunks: Optional[int] = None,
    ep_paired_grouped_mm: bool = False,
) -> None:
    """Sets the model-config options of ``kernels`` on ``model_spec.model`` in place (``moe``: Pallas grouped mm).

    Under expert parallelism, ``ep_token_chunks`` / ``ep_paired_grouped_mm`` configure torchtitan's Qwen3.5 fast
    dropless EP (``configure_qwen3_5_expert_parallel``) when the installed torchtitan provides it.
    """
    if "moe" not in kernels:
        return
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts

    _qwen3_5_config_registry()._use_tpu_moe_kernels(SimpleNamespace(model_spec=model_spec))
    paired_under_ep = expert_parallel_degree > 1 and configure_qwen3_5_expert_parallel(
        model_spec,
        expert_parallel_degree=expert_parallel_degree,
        num_token_chunks=ep_token_chunks,
        paired_grouped_mm=ep_paired_grouped_mm,
    )
    wide_64m_vmem = has_64mib_scoped_vmem()
    for _, experts, _, _ in model_spec.model.traverse(TpuGroupedExperts.Config):
        # ExpertParallel shards expert weights on dim 0, which paired (w1 + w3) Pallas grouped matmul disallows on
        # torchtitan builds without ``allow_ep_paired_grouped_mm``. Turn it off here too so builds with #73 but
        # without #6's TpuQwen35Model.Config do not fail at EP > 1.
        if expert_parallel_degree > 1 and not paired_under_ep and hasattr(experts, "use_pallas_paired_grouped_mm"):
            experts.use_pallas_paired_grouped_mm = False
        # The 512-row tiles were tuned for Qwen3.5-35B-A3B (hidden 2048). At hidden 4096 (Qwen3.5-397B-A17B) the
        # weight-gradient kernels do not fit in VMEM with full K/N tiles: tile K/N of the dWeight instead. Under
        # 64 MiB scoped VMEM (65472 KiB on TPU v7x), 512-row tiles fit with (512, 1024, 0) and transposed W2
        # (512, 0, 2048), matching TorchTitan's qwen35_moe_397b_split_tokamax_ep_benchmark recipe.
        if (getattr(experts, "dim", None) or 0) > _MOE_GROUPED_MM_TILES_MAX_DIM:
            if wide_64m_vmem:
                if hasattr(experts, "pallas_grouped_mm_tile_m"):
                    experts.pallas_grouped_mm_tile_m = _MOE_GROUPED_MM_TILE_M
                if hasattr(experts, "pallas_grouped_mm_tgmm_tiling"):
                    experts.pallas_grouped_mm_tgmm_tiling = _MOE_WIDE_64M_VMEM_TGMM_TILING
                if hasattr(experts, "pallas_grouped_mm_w2_tgmm_tiling"):
                    experts.pallas_grouped_mm_w2_tgmm_tiling = None
                if hasattr(experts, "pallas_grouped_mm_w2_transposed_tgmm_tiling"):
                    experts.pallas_grouped_mm_w2_transposed_tgmm_tiling = _MOE_WIDE_64M_VMEM_W2_TRANSPOSED_TGMM_TILING
            else:
                if hasattr(experts, "pallas_grouped_mm_tgmm_tiling"):
                    experts.pallas_grouped_mm_tgmm_tiling = _MOE_WIDE_TGMM_TILING
                if hasattr(experts, "pallas_grouped_mm_w2_tgmm_tiling"):
                    experts.pallas_grouped_mm_w2_tgmm_tiling = _MOE_WIDE_W2_TGMM_TILING
            continue
        if hasattr(experts, "pallas_grouped_mm_tile_m"):
            experts.pallas_grouped_mm_tile_m = _MOE_GROUPED_MM_TILE_M
        if hasattr(experts, "pallas_grouped_mm_tgmm_tiling"):
            experts.pallas_grouped_mm_tgmm_tiling = _MOE_GROUPED_MM_TGMM_TILING


def configure_qwen3_5_expert_parallel(
    model_spec: Any,
    *,
    expert_parallel_degree: int,
    num_token_chunks: Optional[int] = None,
    paired_grouped_mm: bool = False,
) -> bool:
    """Configures torchtitan's Qwen3.5 fast dropless EP on ``model_spec`` when the installed torchtitan has it.

    Calls the torchtitan recipe helper ``qwen3_5.config_registry._use_expert_parallel`` (fast dropless EP branch):
    it switches the routed experts / dispatcher configs to ``TpuQwen35RoutedExperts`` /
    ``TpuQwen35TokenDispatcher`` (``comm_mode="auto"``: Ring-of-Experts when ``ep <= top_k``, else
    ``ragged_all_to_all``), sets ``num_token_chunks`` (``1`` when ``num_token_chunks`` is unset, i.e. no chunking)
    and ``allow_ep_paired_grouped_mm``. Its SparseCore libtpu flags are left to the launcher (``LIBTPU_INIT_ARGS``),
    since libtpu reads them before the model is built. Returns whether the paired grouped matmul stays enabled under
    EP; ``False`` (nothing changed) on torchtitan builds without the helper.
    """
    try:
        config_registry = _qwen3_5_config_registry()
    except ImportError:
        return False
    use_expert_parallel = getattr(config_registry, "_use_expert_parallel", None)
    if use_expert_parallel is None:
        return False
    params = inspect.signature(use_expert_parallel).parameters
    if not {"num_token_chunks", "allow_ep_paired_grouped_mm"} <= set(params):
        return False
    kwargs: dict[str, Any] = {
        "ep_degree": expert_parallel_degree,
        "num_token_chunks": num_token_chunks or 1,
        "allow_ep_paired_grouped_mm": bool(paired_grouped_mm),
    }
    if "enable_sparsecore_offload" in params:
        kwargs["enable_sparsecore_offload"] = False
    # The helper edits a trainer recipe; only ``model_spec`` matters here, the rest is a scratch namespace.
    recipe = SimpleNamespace(
        model_spec=model_spec,
        parallelism=SimpleNamespace(expert_parallel_degree=expert_parallel_degree),
        compile=SimpleNamespace(compile_grad_norm=False),
        tpu_config=SimpleNamespace(eager_mode="DEFER_AND_FUSE"),
        libtpu_init_flags={},
    )
    use_expert_parallel(recipe, **kwargs)
    logger.info(
        "torchtitan Qwen3.5 fast EP: ep=%d, num_token_chunks=%d, paired grouped mm under EP=%s",
        expert_parallel_degree,
        kwargs["num_token_chunks"],
        kwargs["allow_ep_paired_grouped_mm"],
    )
    return bool(paired_grouped_mm)


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

    - Turns off torchtitan's expert-bias load balancing: vLLM and the HF checkpoint have no expert bias, so with it
      the trainer would pick different experts than the rollout. The aux loss keeps torchtitan's default (off).
    - Runs the routed experts on torchtitan's Pallas grouped matmul (``TpuGroupedExperts``), unpaired and in the
      canonical w1/w2/w3 layout that the weight export expects. Pallas is required: torchtitan's default
      ``torch._grouped_mm`` path has a wrong backward on TPU (torch_tpu#2668).

    Raises:
        ValueError: if ``expert_parallel_size > 1``, or if ``seq_bucket_size * top_k`` is not a multiple of the
            256-row tile.
    """
    from torchtitan.experiments.tpu.grouped_experts import TpuGroupedExperts
    from torchtitan.models.common.moe import MoE

    if expert_parallel_size > 1:
        raise ValueError(
            f"expert_parallel_size={expert_parallel_size} is not supported for Qwen3 MoE on TPU yet: the Pallas "
            "grouped matmul setup here is validated only at EP=1 (as in torchtitan's Qwen3 TPU recipe). Use "
            "expert_parallel_size=1 with FSDP (data_parallel_shard_size) instead."
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
        # TODO: pass load_balance_coeff=None to torchtitan's model_registry once it accepts it, and remove this line.
        moe.load_balance_coeff = None
    experts_configs = [experts for _, experts, _, _ in model_config.traverse(TpuGroupedExperts.Config)]
    if len(experts_configs) != len(moe_configs):
        raise ValueError(
            f"Qwen3 MoE: {len(experts_configs)} of {len(moe_configs)} MoE layers use TpuGroupedExperts; build the "
            "model with qwen3_moe_model_registry_kwargs()"
        )
    # Same values as torchtitan's Qwen3-30B-A3B TPU recipe, except unpaired and the canonical layout.
    # TODO: call torchtitan's _use_tpu_moe_kernels (as for Qwen3.5) once the pinned torchtitan has it.
    for experts in experts_configs:
        experts.use_pallas_grouped_mm = True
        experts.use_pallas_paired_grouped_mm = False
        experts.use_kernel_native_weight_layout = False  # keep the canonical parameter names and layout
        experts.pallas_grouped_mm_tile_m = tile_m
        experts.pallas_grouped_mm_tgmm_tiling = _QWEN3_MOE_GROUPED_MM_TGMM_TILING
    logger.info(
        "Qwen3 MoE: %d layers route without expert bias; TPU token dispatcher and Pallas grouped matmul experts "
        "(tile_m=%d)",
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
    """Fused linear cross-entropy block sizes of ``kernels`` (empty: the kernel's defaults).

    ``_LCE_BLOCK_SIZES`` ``(1024, 2048, 640)`` fits the entire hidden width of Qwen3.5-35B-A3B (``H=2048``) in a
    single H block; wider models such as Qwen3.5-397B-A17B (``H=4096``) keep Tokamax's ``(1024, 512, 2048)`` default
    (matching ``qwen35_moe_397b_split_tokamax_ep_benchmark`` in google-pytorch/torchtitan#142).
    """
    if (
        "lce" not in kernels
        or hidden_size != _LCE_BLOCK_SIZES["h_block_size"]
        or vocab_size % _LCE_BLOCK_SIZES["v_block_size"] != 0
    ):
        return LceBlockSizes()
    return LceBlockSizes(**_LCE_BLOCK_SIZES)


# MaxText's MAX_SEGMENTS_PER_PACKED_ROW: the smallest Tokamax KDA document budget the engine sets.
MIN_PACKED_DOCUMENT_BUDGET = 16


def packed_document_budget(max_sequences_per_micro_batch: int) -> int:
    """The Tokamax KDA ``max_num_segments`` for micro-batches packing up to ``max_sequences_per_micro_batch`` sequences.

    One document per sequence plus one for the bucket-padding tail, rounded up to a power of two and at least
    ``MIN_PACKED_DOCUMENT_BUDGET``. ``max_num_segments`` is a static kernel shape: with ``use_dynamic_bsz`` the number
    of sequences per token-budgeted micro-batch varies from step to step, and an exact budget would recompile every
    GatedDeltaNet layer whenever it changes. A larger budget only adds empty segments.
    """
    raw = int(max_sequences_per_micro_batch) + 1
    return max(MIN_PACKED_DOCUMENT_BUDGET, 1 << (raw - 1).bit_length())


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


_DYNAMO_CACHE_TARGETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torchtitan.experiments.tpu.qwen3_5.tokamax_gated_delta", ("_tpu_generation", "_import_kda")),
    ("torchtitan.experiments.tpu.kernels.ragged_gather_tc", ("_tpu_generation",)),
    ("torchtitan.experiments.tpu.kernels.causal_conv1d", ("_tpu_generation",)),
)


def _is_dynamo_constant_primitive(value: Any) -> bool:
    if isinstance(value, type(None) | bool | int | float | str):
        return True
    if isinstance(value, tuple):
        return all(_is_dynamo_constant_primitive(item) for item in value)
    return False


def _pin_dynamo_constant_zero_arg(module: Any, attr_name: str) -> bool:
    """Evaluates zero-arg ``module.<attr_name>()`` once and replaces ``@lru_cache`` with a Dynamo-safe getter.

    Scalar/primitive return values (such as ``_tpu_generation() -> int | None``) are wrapped in
    ``torch._dynamo.assume_constant_result``. Callable/object return values (such as
    ``_import_kda() -> (kimi_delta_attention, None)``) must NOT use ``assume_constant_result`` because PyTorch
    Dynamo tags the returned function with ``ConstantSource``, whose ``make_guard(GuardBuilder.CLOSURE_MATCH)``
    raises ``NotImplementedError`` under ``checkpoint_wrapper``; storing the pre-evaluated value on a module
    global gives it a regular ``GlobalSource`` that supports ``CLOSURE_MATCH`` without tracing ``@lru_cache``.
    """
    fn = getattr(module, attr_name, None)
    if not callable(fn) or getattr(fn, "_verl_dynamo_constant", False):
        return False
    value = fn()
    if _is_dynamo_constant_primitive(value):
        assume_constant = getattr(getattr(torch, "_dynamo", None), "assume_constant_result", None)

        def _const_fn(_val: Any = value) -> Any:
            return _val

        if callable(assume_constant):
            _const_fn = assume_constant(_const_fn)
    else:
        cache_attr = f"_verl_const_{attr_name}"
        setattr(module, cache_attr, value)
        ns: dict[str, Any] = {}
        exec(f"def {attr_name}():\n    return {cache_attr}\n", module.__dict__, ns)
        _const_fn = ns[attr_name]
    _const_fn._verl_dynamo_constant = True  # type: ignore[attr-defined]
    setattr(module, attr_name, _const_fn)
    return True


def patch_torchtitan_tpu_dynamo_caches() -> int:
    """Pins TorchTitan TPU hardware/import helpers as compile-time constants for ``torch.compile(fullgraph=True)``.

    ``tokamax_gated_delta._tpu_generation`` and ``_import_kda`` use ``@functools.lru_cache(maxsize=1)`` (and
    ``ragged_gather_tc._tpu_generation`` uses an un-annotated Python list cache). Under per-block
    ``torch.compile(backend='tpu', fullgraph=True)``, PyTorch Dynamo strips ``@lru_cache`` and traces into
    ``torch.tpu.get_device_properties()`` -> ``pathlib.Path('/sys/class/accel').iterdir()`` -> ``posix.listdir``,
    raising ``torch._dynamo.exc.Unsupported``. Pre-evaluating them once and wrapping the cached result in
    ``torch._dynamo.assume_constant_result`` keeps ``fullgraph=True`` compilation free of host-filesystem queries.
    """
    patched = 0
    for mod_name, attr_names in _DYNAMO_CACHE_TARGETS:
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
        for attr_name in attr_names:
            try:
                if _pin_dynamo_constant_zero_arg(mod, attr_name):
                    patched += 1
            except Exception:
                logger.debug("Skipping Dynamo constant pin for %s.%s", mod_name, attr_name, exc_info=True)
    return patched


_EP_CHUNK_UNIFORM_BUCKET = 2048
_EP_CHUNK_MIN_ALIGN = 256
_BARRIER_HOOK: Optional[Callable[[list[torch.Tensor]], list[torch.Tensor]]] = None
_BARRIER_ROUTED_EXPERTS_SUBCLASSES: dict[type, type] = {}


def _optimization_barrier_fake(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for value in inputs:
        output = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device=value.device,
            requires_grad=value.requires_grad,
        )
        if value.storage_offset():
            output = output.as_strided(value.shape, value.stride(), value.storage_offset())
        outputs.append(output)
    return outputs


def _ensure_optimization_barrier_fake() -> None:
    op_name = "tpu::_optimization_barrier"
    has_kernel = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if callable(has_kernel) and not has_kernel(op_name, "Meta"):
        torch.library.register_fake(op_name, _optimization_barrier_fake)


def _tpu_optimization_barrier_tensors(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Applies ``torch.tpu.ops.optimization_barrier`` on TPU tensors (cloning non-zero storage-offset slices)."""
    if not tensors:
        return []
    if _BARRIER_HOOK is not None:
        return _BARRIER_HOOK(tensors)
    tpu_ops = getattr(getattr(torch, "tpu", None), "ops", None)
    barrier_fn = getattr(tpu_ops, "optimization_barrier", None)
    if callable(barrier_fn) and all(t.device.type == "tpu" for t in tensors):
        _ensure_optimization_barrier_fake()
        # torch_tpu's ContiguousToView re-wraps contiguous barrier outputs using each input's storage_offset(),
        # so dim-0 chunk slices with storage_offset() > 0 must be materialized at offset 0 first.
        normalized = [t.clone() if t.storage_offset() else t for t in tensors]
        return list(barrier_fn(normalized))
    return [t.clone() for t in tensors]


class _EpChunkWindowBarrier(torch.autograd.Function):
    """Autograd-aware ``optimization_barrier`` bounding live MoE EP chunks to ``max_inflight_chunks`` (default 2).

    TorchTitan's ``TpuGraphTrainer`` lowers ``annotate({'ep_chunk_idx': k, 'ep_max_inflight_chunks': W})`` in
    ``TpuQwen35RoutedExperts.forward`` via an FX pass (``insert_ep_chunk_barriers.py``), which does not run under
    ``TorchTitanTPUEngine``'s per-block ``torch.compile(backend='tpu', fullgraph=True)``. Gating
    ``(x_k, scores_k, ids_k, out_{k-W})`` here orders chunk ``k`` after chunk ``k - W`` in forward, and its exact
    reverse-mode transpose holds ``grad_out_{k-W}`` until chunk ``k``'s backward exits ``(grad_x_k, grad_scores_k)``
    complete.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x_c: torch.Tensor,
        scores_c: torch.Tensor,
        ids_c: torch.Tensor,
        prev_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        gated = _tpu_optimization_barrier_tensors([x_c, scores_c, ids_c, prev_out])
        return gated[0], gated[1], gated[2], gated[3]

    @staticmethod
    def backward(
        ctx: Any,
        grad_x_c: Optional[torch.Tensor],
        grad_scores_c: Optional[torch.Tensor],
        grad_ids_c: Optional[torch.Tensor],
        grad_prev_out: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, Optional[torch.Tensor]]:
        del ctx, grad_ids_c
        grads: list[Optional[torch.Tensor]] = [grad_x_c, grad_scores_c, None, grad_prev_out]
        present_idx = [idx for idx in (0, 1, 3) if grads[idx] is not None]
        if len(present_idx) >= 2:
            gated = _tpu_optimization_barrier_tensors([grads[idx] for idx in present_idx])  # type: ignore[misc]
            for idx, g_out in zip(present_idx, gated, strict=True):
                grads[idx] = g_out
        return grads[0], grads[1], None, grads[3]


def _resolve_ep_token_chunks(num_tokens: int, requested_chunks: int) -> int:
    """Chunk count dividing ``num_tokens`` cleanly for dynamic-bsz buckets without falling back to unchunked EP."""
    if requested_chunks <= 1 or num_tokens <= 0:
        return 1
    if num_tokens % requested_chunks == 0 and (
        num_tokens < _EP_CHUNK_MIN_ALIGN or (num_tokens // requested_chunks) % _EP_CHUNK_MIN_ALIGN == 0
    ):
        return requested_chunks
    if (
        num_tokens >= _EP_CHUNK_UNIFORM_BUCKET
        and num_tokens % _EP_CHUNK_UNIFORM_BUCKET == 0
        and (num_tokens // _EP_CHUNK_UNIFORM_BUCKET) <= requested_chunks
    ):
        return num_tokens // _EP_CHUNK_UNIFORM_BUCKET
    for chunks in range(min(requested_chunks, num_tokens), 1, -1):
        if num_tokens % chunks == 0 and (
            num_tokens < _EP_CHUNK_MIN_ALIGN or (num_tokens // chunks) % _EP_CHUNK_MIN_ALIGN == 0
        ):
            return chunks
    return 1


def _default_tokens_per_expert(ids_TK: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(ids_TK.reshape(-1), minlength=num_experts)[:num_experts].to(torch.int32)


def _get_barrier_routed_experts_cls(cls: type) -> type:
    cached = _BARRIER_ROUTED_EXPERTS_SUBCLASSES.get(cls)
    if cached is not None:
        return cached
    base_cls = next((c for c in cls.__mro__ if c.__name__ == "TpuQwen35RoutedExperts"), cls)
    mod = sys.modules.get(base_cls.__module__)
    maybe_set_sparse_mesh = getattr(mod, "maybe_set_sparse_mesh", contextlib.nullcontext)
    tokens_per_expert = getattr(mod, "_tokens_per_expert", _default_tokens_per_expert)
    dispatcher_cls = getattr(mod, "TpuQwen35TokenDispatcher", None)

    def _forward_with_chunk_barriers(
        self: Any,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        num_chunks = _resolve_ep_token_chunks(x_TD.shape[0], int(getattr(self, "num_token_chunks", 1)))
        ep_mesh = getattr(self.token_dispatcher, "ep_mesh", None)
        is_qwen35_disp = (
            isinstance(self.token_dispatcher, dispatcher_cls)
            if dispatcher_cls is not None
            else any(c.__name__ == "TpuQwen35TokenDispatcher" for c in type(self.token_dispatcher).__mro__)
        )
        weights_on_activation = (
            ep_mesh is not None
            and is_qwen35_disp
            and bool(getattr(self.token_dispatcher, "roe_weights_on_activation", False))
            and bool(getattr(self.inner_experts, "use_pallas_grouped_mm", False))
        )
        if (
            (num_chunks <= 1 and not weights_on_activation)
            or ep_mesh is None
            or x_TD.shape[0] == 0
            or x_TD.shape[0] % num_chunks != 0
        ):
            return super(base_cls, self).forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                num_local_tokens_per_expert_E,
            )

        needs_chunk_counts = True
        resolve_comm_mode = getattr(self.token_dispatcher, "_resolve_comm_mode", None)
        if is_qwen35_disp and callable(resolve_comm_mode):
            needs_chunk_counts = resolve_comm_mode(ep_mesh.size()) != "ring_of_experts"

        with maybe_set_sparse_mesh():
            unsharded_weights = self.inner_experts(
                x_TD,
                num_local_tokens_per_expert_E,
                unshard_only=True,
            )

        x_chunks = list(x_TD.chunk(num_chunks, dim=0))
        score_chunks = list(topk_scores_TK.chunk(num_chunks, dim=0))
        id_chunks = list(topk_expert_ids_TK.chunk(num_chunks, dim=0))
        num_experts = num_local_tokens_per_expert_E.numel()
        max_inflight = max(1, int(getattr(self, "max_inflight_chunks", 2)))

        out_chunks: list[torch.Tensor] = []
        for chunk_idx in range(num_chunks):
            x_c = x_chunks[chunk_idx]
            scores_c = score_chunks[chunk_idx]
            ids_c = id_chunks[chunk_idx]
            if chunk_idx >= max_inflight:
                prev_idx = chunk_idx - max_inflight
                x_c, scores_c, ids_c, out_chunks[prev_idx] = _EpChunkWindowBarrier.apply(
                    x_c,
                    scores_c,
                    ids_c,
                    out_chunks[prev_idx],
                )
            counts_c = tokens_per_expert(ids_c, num_experts) if needs_chunk_counts else num_local_tokens_per_expert_E
            routed_input_c, num_tokens_per_expert_e, metadata_c = self.token_dispatcher.dispatch(
                x_c, scores_c, ids_c, counts_c
            )
            row_scale_c = getattr(metadata_c, "expert_row_scale_R", None) if weights_on_activation else None
            with maybe_set_sparse_mesh():
                routed_output_c = self.inner_experts(
                    routed_input_c,
                    num_tokens_per_expert_e,
                    row_scale_R=row_scale_c,
                    pre_unsharded_weights=unsharded_weights,
                )
            if row_scale_c is not None:
                metadata_c = dataclasses.replace(metadata_c, row_scale_applied=True)
            out_chunks.append(self.token_dispatcher.combine(routed_output_c, metadata_c, x_c))
        return torch.cat(out_chunks, dim=0)

    subcls = type(
        f"_Barrier{cls.__name__}",
        (cls,),
        {"forward": _forward_with_chunk_barriers, "_verl_ep_chunk_barrier_cls": True},
    )
    _BARRIER_ROUTED_EXPERTS_SUBCLASSES[cls] = subcls
    return subcls


def install_ep_chunk_barriers(modules: Any) -> int:
    """Installs sliding-window ``optimization_barrier`` + dynamic-bucket chunking on ``TpuQwen35RoutedExperts``."""
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    updated = 0
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        for module in model.modules():
            if getattr(type(module), "_verl_ep_chunk_barrier_cls", False):
                continue
            if any(c.__name__ == "TpuQwen35RoutedExperts" for c in type(module).__mro__):
                module.__class__ = _get_barrier_routed_experts_cls(type(module))
                updated += 1
    return updated
