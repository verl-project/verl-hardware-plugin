# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU helper utilities for the TorchTitan training engine.

Provides sequence-length bucketing, host-side metadata unwrapping, packed-input padding,
``torch.compile(backend="tpu")`` Dynamo configuration, torchtitan's TPU ``SplashAttention``, the torchtitan TPU
recipe that shards with ``SimpleFSDP``, cross-rank micro-batch shape alignment, and ``DEFER_AND_FUSE``
eager-mode contexts for ``torch_tpu`` training.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch._dynamo
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils as tu
from verl.workers.config import TorchtitanEngineConfig
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to attach the full bucket-padded device tensor to a CPU
# nested tensor returned from ``prepare_model_outputs``, so loss functions can
# consume the fixed-shape device tensor directly without XLA recompilation.
TPU_PADDED_VALUES_ATTR = "_tpu_padded_values"

_DEFAULT_TPU_SEQ_BUCKET_SIZE = 256

# Mapping from HuggingFace model_type to torchtitan model name.
# Torchtitan models not mapped here:
#   - flux: diffusion model, not applicable to verl's RL/SFT workflows
#   - llama3_ft: fault-tolerant variant of llama3, same HF models (mapped via "llama")
_HF_MODEL_TYPE_TO_TORCHTITAN_NAME = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen2_moe": "qwen3",
    "qwen3_moe": "qwen3",
    "llama": "llama3",
    "deepseek_v3": "deepseek_v3",
    "gpt_oss": "gpt_oss",
    # Qwen3.5 (hybrid GatedDeltaNet + full attention, dense or MoE). The multimodal ``qwen3_5[_moe]``
    # configs keep the language model in ``text_config``; only the text model is trained.
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5_moe": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5",
}

# torchtitan model packages that are not ``torchtitan.models.<name>``. Upstream ``torchtitan.models.qwen3_5``
# runs its GatedDeltaNet with CUDA/Triton kernels; the TPU package swaps in XLA GatedDeltaNet kernels that restart
# the causal conv1d and the recurrent state at packed-document boundaries (from ``positions``), uses
# document-masked attention and drops the vision encoder.
_TORCHTITAN_MODEL_MODULES = {
    "qwen3_5": "torchtitan.experiments.tpu.qwen3_5",
}

# torchtitan models whose ``model_registry`` only accepts ``attn_backend="sdpa"``. Their inner attention is
# replaced by ``TPUSplashAttention`` (or ``TPUVarlenAttention``) after the model is built anyway.
_SDPA_ONLY_TORCHTITAN_MODELS = frozenset({"qwen3_5"})

# torchtitan TPU recipes whose ``parallelize_fn`` shards with SimpleFSDP when ``parallelism.use_simple_fsdp`` is set.
_TPU_SIMPLE_FSDP_RECIPES = {
    "qwen3": ("torchtitan.experiments.tpu.qwen3.infra.parallelize", "parallelize_qwen3"),
    "llama3": ("torchtitan.experiments.tpu.llama3.infra.parallelize", "parallelize_llama"),
    "qwen3_5": ("torchtitan.experiments.tpu.qwen3_5.infra.parallelize", "parallelize_qwen3_5"),
}


def get_torchtitan_model_module_name(torchtitan_name: str) -> str:
    """Python package holding ``model_registry`` (and the flavor configs) of torchtitan model ``torchtitan_name``."""
    return _TORCHTITAN_MODEL_MODULES.get(torchtitan_name, f"torchtitan.models.{torchtitan_name}")


def torchtitan_model_attn_backend(torchtitan_name: str, attn_type: str) -> str:
    """``attn_backend`` to build torchtitan model ``torchtitan_name`` with, given the engine's ``attn_type``."""
    return "sdpa" if torchtitan_name in _SDPA_ONLY_TORCHTITAN_MODELS else attn_type


def get_text_config(hf_config: Any) -> Any:
    """The language-model config of ``hf_config``: ``text_config`` for multimodal configs, else ``hf_config``."""
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and getattr(text_config, "hidden_size", None) is not None:
        return text_config
    return hf_config


def create_noop_dataloader_config() -> Any:
    """Create a no-op TorchTitan ``BaseDataLoader.Config`` when verl manages its own data loading."""
    from torchtitan.components.data.loader import BaseDataLoader

    class NoOpDataLoader(BaseDataLoader):
        """A no-op dataloader for use when verl manages its own data loading.

        Satisfies the BaseDataLoader interface required by torchtitan's Trainer
        but does nothing. Its __iter__ yields nothing, and state_dict /
        load_state_dict are no-ops.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(BaseDataLoader.Config):
            pass

        def __init__(self, **kwargs: Any):
            pass

        def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
            return iter([])

        def state_dict(self) -> dict[str, Any]:
            return {}

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            pass

    return NoOpDataLoader.Config()


def derive_torchtitan_name_and_flavor(hf_config: Any) -> tuple[str, str]:
    """Derive torchtitan model name and flavor from a HuggingFace config.

    The name is mapped from ``hf_config.model_type``. The flavor is found by
    matching architecture parameters (dim, n_layers, vocab_size) against the
    known flavors registered in the torchtitan model package.

    Args:
        hf_config: A HuggingFace AutoConfig object.

    Returns:
        A ``(name, flavor)`` tuple.

    Raises:
        ValueError: If model_type is unsupported or no matching flavor is found.
    """
    model_type = getattr(hf_config, "model_type", None)
    if model_type is None:
        raise ValueError("HuggingFace config does not have 'model_type' field")

    name = _HF_MODEL_TYPE_TO_TORCHTITAN_NAME.get(model_type)
    if name is None:
        raise ValueError(
            f"Cannot derive torchtitan model name from HF model_type '{model_type}'. "
            f"Supported types: {list(_HF_MODEL_TYPE_TO_TORCHTITAN_NAME.keys())}."
        )

    # Import the model package and use model_registry to build each flavor's config.
    # model_registry has sensible defaults for all optional params (attn_backend, etc.).
    module_name = get_torchtitan_model_module_name(name)
    model_module = importlib.import_module(module_name)
    model_registry = model_module.model_registry

    # The configs dict name isn't derivable from the model name
    # (e.g. gpt_oss -> gptoss_configs), so we find it by convention: ``<name>_configs`` if the package
    # has it (the TPU qwen3_5 package also imports other ``*_configs`` dicts), else the first dict attribute
    # ending with ``_configs``.
    flavor_names = None
    preferred = getattr(model_module, f"{name}_configs", None)
    if isinstance(preferred, dict):
        flavor_names = list(preferred.keys())
    else:
        for attr, obj in vars(model_module).items():
            if attr.endswith("_configs") and isinstance(obj, dict):
                flavor_names = list(obj.keys())
                break

    if flavor_names is None:
        raise ValueError(
            f"Could not find model configs dict in {module_name}. Expected a dict attribute ending with '_configs'."
        )

    text_config = get_text_config(hf_config)
    hidden_size = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    vocab_size = text_config.vocab_size

    for flavor_name in flavor_names:
        cfg = model_registry(flavor_name).model
        n_layers = getattr(cfg, "n_layers", None) or len(getattr(cfg, "layers", []))
        if (
            getattr(cfg, "dim", None) == hidden_size
            and n_layers == num_layers
            and getattr(cfg, "vocab_size", None) == vocab_size
        ):
            logger.info(
                "Auto-derived torchtitan name=%r, flavor=%r from HF model_type=%r",
                name,
                flavor_name,
                model_type,
            )
            return name, flavor_name

    raise ValueError(
        f"No matching torchtitan flavor found for model_type='{model_type}' "
        f"(hidden_size={hidden_size}, num_hidden_layers={num_layers}, "
        f"vocab_size={vocab_size}). "
        f"Available flavors for '{name}': {flavor_names}."
    )


def enable_fsdp_gradient_division(model: nn.Module, dp_size: int) -> None:
    """
    Re-enable FSDP's automatic gradient division.

    TorchTitan calls disable_fsdp_gradient_division() which sets gradient_divide_factor=1.0.
    This re-enables it by setting the factor to the specified dp_size, so gradients are
    averaged across FSDP ranks. This is needed for verl's loss scaling (loss * dp_size)
    to work correctly.

    Args:
        model: The model (or model part) to enable gradient division on.
        dp_size: The data parallel size to use as the gradient divide factor.
    """

    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(float(dp_size))


@contextmanager
def spmd_no_typecheck_context():
    """Disable ``spmd_types`` type-checking when ``spmd_types`` is installed and active."""
    try:
        import spmd_types as spmd
    except ImportError:
        yield
        return
    with spmd.no_typecheck():
        yield


class ContextParallelGather(torch.autograd.Function):
    """All-gather along dim 0 whose backward keeps only the local slice.

    Every CP rank computes the same loss on the gathered tensor, so the upstream gradient is
    identical across CP ranks and its local slice is already the full gradient of the local shard.
    Summing it across ranks (the default all-gather backward) would scale gradients by the CP size.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        with spmd_no_typecheck_context():
            ctx.rank = torch.distributed.get_rank(group)
            ctx.world_size = torch.distributed.get_world_size(group)
            tensor = tensor.contiguous()
            gathered = tensor.new_empty((tensor.size(0) * ctx.world_size, *tensor.shape[1:]))
            torch.distributed.all_gather_into_tensor(gathered, tensor, group=group)
            return gathered

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        with spmd_no_typecheck_context():
            return grad_output.chunk(ctx.world_size, dim=0)[ctx.rank], None


def extend_torchtitan_engine_config() -> None:
    """Register TPU fields on ``TorchtitanEngineConfig`` for Hydra/OmegaConf.

    ``engine_kwargs`` holds the TPU options under ``engine_kwargs.tpu`` (see ``accelerators.tpu.options``). The other
    TPU fields are deprecated aliases, kept so existing recipes keep working.
    TODO(tpu): drop this patch once verl's ``TorchtitanEngineConfig`` has an ``engine_kwargs`` field.
    """
    extras: list[tuple[str, Any, Any]] = [
        ("engine_kwargs", dict[str, Any], field(default_factory=dict)),
        # Deprecated: use engine_kwargs.tpu.<name>.
        ("use_splash_attention", Optional[bool], None),
        ("use_simple_fsdp", Optional[bool], None),
        ("use_fused_linear_cross_entropy", Optional[bool], None),
        ("fuse_qkv", Optional[bool], None),
        ("tpu_eager_mode", Optional[str], None),
        ("pad_to_length", bool, False),
        ("pad_to_length_bucket", int, _DEFAULT_TPU_SEQ_BUCKET_SIZE),
    ]
    missing = [item for item in extras if item[0] not in TorchtitanEngineConfig.__dataclass_fields__]
    if not missing:
        return
    # Re-attach existing Field objects whose class attributes were removed by @dataclass
    # (e.g., fields with default_factory such as wrap_policy) before re-running dataclass().
    for fname, fobj in list(TorchtitanEngineConfig.__dataclass_fields__.items()):
        if fname in TorchtitanEngineConfig.__annotations__:
            setattr(TorchtitanEngineConfig, fname, fobj)
    for name, typ, default in missing:
        TorchtitanEngineConfig.__annotations__[name] = typ
        setattr(TorchtitanEngineConfig, name, default)
    for fn in ("__init__", "__repr__", "__eq__"):
        if fn in TorchtitanEngineConfig.__dict__:
            delattr(TorchtitanEngineConfig, fn)
    dataclass(TorchtitanEngineConfig)


extend_torchtitan_engine_config()


def _or_default(value: Optional[bool], default: bool) -> bool:
    return default if value is None else bool(value)


def resolve_tpu_torchtitan_options(engine_config: Any) -> tuple[bool, bool, Optional[str]]:
    """Resolve ``(use_splash_attention, use_simple_fsdp, tpu_eager_mode)`` for TPU TorchTitan.

    Each option comes from ``engine_kwargs.tpu`` (see ``get_tpu_engine_options``), else defaults from
    ``engine_config.use_torch_compile``: with ``use_torch_compile=True`` (splash also when
    ``context_parallel_size > 1``), enables torchtitan's TPU ``SplashAttention``, ``SimpleFSDP``
    (for FSDP/HSDP with optional TP and CP where PP/EP == 1) and ``DEFER_AND_FUSE``.
    """
    options = get_tpu_engine_options(engine_config)
    use_torch_compile = bool(getattr(engine_config, "use_torch_compile", False))
    cp_enabled = int(getattr(engine_config, "context_parallel_size", 1) or 1) > 1
    use_splash_attention = _or_default(options.use_splash_attention, use_torch_compile or cp_enabled)
    if use_torch_compile and not use_splash_attention:
        raise ValueError(
            "use_torch_compile=True on TPU requires use_splash_attention=True: torch_tpu's compiled "
            "F.scaled_dot_product_attention produces NaN q/k/v gradients, so every optimizer step would be "
            "skipped. Use splash attention + torch.compile (use_splash_attention=True), or set "
            "use_torch_compile=False."
        )

    simple_fsdp_compatible = all(
        getattr(engine_config, f"{dim}_parallel_size", 1) == 1 for dim in ("pipeline", "expert")
    )
    use_simple_fsdp = _or_default(options.use_simple_fsdp, use_torch_compile and simple_fsdp_compatible)
    if use_simple_fsdp and not simple_fsdp_compatible:
        raise ValueError("use_simple_fsdp supports FSDP/HSDP (+ optional TP/CP) only (pipeline/expert parallel size 1)")

    if options.tpu_eager_mode is None:
        tpu_eager_mode = "DEFER_AND_FUSE" if use_torch_compile else None
    else:
        tpu_eager_mode = options.tpu_eager_mode.strip().upper()
        tpu_eager_mode = None if tpu_eager_mode in ("", "NONE", "NULL") else tpu_eager_mode
    if tpu_eager_mode not in (None, "DEFER_AND_FUSE", "DEFER_NEVER"):
        raise ValueError(f"tpu_eager_mode {tpu_eager_mode!r} not supported")

    return use_splash_attention, use_simple_fsdp, tpu_eager_mode


def resolve_tpu_use_fused_linear_cross_entropy(engine_config: Any) -> bool:
    """Resolve whether to use the TPU Pallas fused linear + softmax cross-entropy kernel for logprobs.

    On by default with ``use_torch_compile`` when ``tensor_parallel_size == 1``. The kernel never writes the
    ``[tokens, vocab]`` logits to HBM and keeps them in fp32, so its log-probs are more precise than verl's GPU
    path, which rounds the logits to bf16 before the fp32 log-softmax. Disabled when ``tensor_parallel_size > 1``
    because ``FusedLossLinearWrapper`` returns a ``(hidden_states, weight_t)`` tuple that is incompatible with
    TorchTitan's ``ColwiseParallel`` output redistribution (TP uses :func:`compute_tpu_logprobs_chunked` instead).
    """
    if getattr(engine_config, "tensor_parallel_size", 1) > 1:
        return False
    return _or_default(
        get_tpu_engine_options(engine_config).use_fused_linear_cross_entropy,
        bool(getattr(engine_config, "use_torch_compile", False)),
    )


_PARALLEL_DIMS_CACHE: dict[tuple[Any, ...], Any] = {}
_PARALLEL_DIMS_PATCHED = False


def install_parallel_dims_cache_for_tpu() -> None:
    """Reuse ``ParallelDims`` and its ``DeviceMesh`` objects across engines in the same process.

    ``ActorRolloutRefWorker`` initializes both ``actor`` and ``ref`` ``TorchTitanTPUEngine`` instances
    in the same worker process. TorchTitan's ``ParallelDims`` caches ``DeviceMesh`` instances on ``self``
    because Dynamo traces ``DeviceMesh`` as a ``FakeScriptObject`` matched by object identity, whereas
    ``DTensor``'s ``ShardingPropagator`` LRU cache hashes ``DeviceMesh`` by value. If ``actor`` and
    ``ref`` construct separate ``ParallelDims`` objects with identical topology, ``ShardingPropagator``
    returns cached ``DTensorSpec``s holding ``ref``'s ``DeviceMesh`` when ``actor`` compiles the same
    sequence-length bucket, causing Dynamo to emit a non-placeholder ``_torchbind_obj0`` output that
    ``torch_tpu`` rejects. Caching ``ParallelDims.from_config`` and making ``ParallelDims.build_mesh``
    idempotent ensures ``actor`` and ``ref`` share the exact same ``DeviceMesh`` Python instances and
    process groups.
    """
    global _PARALLEL_DIMS_PATCHED
    if _PARALLEL_DIMS_PATCHED:
        return
    try:
        from torchtitan.distributed.parallel_dims import ParallelDims
    except ImportError:
        return

    orig_from_config = getattr(ParallelDims.from_config, "__func__", None)
    orig_build_mesh = getattr(ParallelDims, "build_mesh", None)
    if orig_from_config is None or orig_build_mesh is None:
        return

    @classmethod  # type: ignore[misc]
    def _cached_from_config(cls, parallelism_config: Any, world_size: int) -> Any:
        world_pg_id = (
            id(torch.distributed.group.WORLD)
            if torch.distributed.is_initialized() and torch.distributed.group.WORLD is not None
            else None
        )
        key = (
            world_pg_id,
            getattr(parallelism_config, "data_parallel_replicate_degree", 1),
            getattr(parallelism_config, "data_parallel_shard_degree", -1),
            getattr(parallelism_config, "context_parallel_degree", 1),
            getattr(parallelism_config, "tensor_parallel_degree", 1),
            getattr(parallelism_config, "pipeline_parallel_degree", 1),
            getattr(parallelism_config, "expert_parallel_degree", 1),
            world_size,
            getattr(parallelism_config, "spmd_backend", "partial_dtensor"),
            getattr(parallelism_config, "enable_tpu_topo_mesh", False),
        )
        cached = _PARALLEL_DIMS_CACHE.get(key)
        if cached is None:
            cached = orig_from_config(cls, parallelism_config, world_size)
            _PARALLEL_DIMS_CACHE[key] = cached
        return cached

    def _cached_build_mesh(self: Any) -> Any:
        if getattr(self, "_world_mesh", None) is not None and getattr(self, "_single_axis_meshes", None):
            return self._world_mesh
        return orig_build_mesh(self)

    ParallelDims.from_config = _cached_from_config
    ParallelDims.build_mesh = _cached_build_mesh
    _PARALLEL_DIMS_PATCHED = True


_SYNC_DTENSOR_REDISTRIBUTE_PATCHED = False


def install_sync_dtensor_redistribute_for_tpu() -> None:
    """Force ``redistribute_local_tensor(..., async_op=False)`` outside ``torch.compile`` tracing.

    TorchTitan's ``Module._redistribute_inputs`` / ``Module._redistribute_outputs`` (and PyTorch TP
    ``ParallelStyle``) call ``DTensor.redistribute(..., async_op=True)``, which sets ``ctx.async_op = True``
    on ``Redistribute`` for both forward and backward. Outside a ``torch.compile`` region (e.g. on
    uncompiled ``tok_embeddings``, ``norm``, and ``output`` at the boundaries of compiled
    ``TransformerBlock``s), returning an unwaited ``AsyncCollectiveTensor`` into a compiled graph input
    (forward ``tok_embeddings -> layers["0"]`` or backward ``output/norm -> layers["35"]``) causes Dynamo
    to unwrap ``AsyncCollectiveTensor.elem`` without dispatching ``wait_tensor(elem)``, permanently
    retaining ``elem`` in C++ ``c10d::WorkRegistry``. Forcing ``async_op=False`` when not compiling calls
    ``.wait()`` immediately in ``redistribute_local_tensor`` so ``WorkRegistry`` frees ``elem`` and
    synchronizes eager TPU redistributions across ranks.
    """
    global _SYNC_DTENSOR_REDISTRIBUTE_PATCHED
    if _SYNC_DTENSOR_REDISTRIBUTE_PATCHED:
        return
    try:
        import torch.distributed.tensor._api as _dt_api
        import torch.distributed.tensor._dispatch as _dt_disp
        import torch.distributed.tensor._redistribute as _redist
        from torch.distributed._functional_collectives import AsyncCollectiveTensor
    except ImportError:
        return

    orig_rlt = getattr(_redist, "redistribute_local_tensor", None)
    if orig_rlt is None:
        return

    def _sync_redistribute_local_tensor(
        local_tensor: torch.Tensor,
        current_spec: Any,
        target_spec: Any,
        *,
        async_op: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_eager = not torch.compiler.is_compiling() and not torch.compiler._is_non_strict_tracing()
        if is_eager:
            async_op = False
            if isinstance(local_tensor, torch.Tensor) and local_tensor.device.type == "tpu":
                tpu_synchronize(local_tensor, wait=False)
        out = orig_rlt(
            local_tensor,
            current_spec,
            target_spec,
            async_op=async_op,
            **kwargs,
        )
        if not async_op and isinstance(out, AsyncCollectiveTensor):
            out = out.wait()
        if is_eager and isinstance(out, torch.Tensor) and out.device.type == "tpu":
            tpu_synchronize(out, wait=False)
        return out

    _redist.redistribute_local_tensor = _sync_redistribute_local_tensor
    if hasattr(_dt_api, "redistribute_local_tensor"):
        _dt_api.redistribute_local_tensor = _sync_redistribute_local_tensor
    if hasattr(_dt_disp, "redistribute_local_tensor"):
        _dt_disp.redistribute_local_tensor = _sync_redistribute_local_tensor
    _SYNC_DTENSOR_REDISTRIBUTE_PATCHED = True


def _tpu_logprobs_chunk_step(
    logits_chunk: torch.Tensor,
    labels_chunk: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """All-gather one static sequence chunk of ``logits_chunk`` (if ``DTensor``) and compute log-probs."""
    from verl.utils.torch_functional import logprobs_from_logits_naive

    if isinstance(logits_chunk, DTensor):
        logits_chunk = logits_chunk.full_tensor()
    if logits_chunk.dim() > 2:
        logits_chunk = logits_chunk.squeeze(0)
    if temperature != 1.0:
        logits_chunk = logits_chunk / temperature
    return logprobs_from_logits_naive(logits_chunk.float(), labels_chunk)


def _eval_tpu_logprobs_chunks_no_grad(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    step: int,
    mesh: Any,
    placements: Any,
) -> torch.Tensor:
    total_tokens = int(labels.shape[0])
    seq_dim = 1 if local_logits.dim() == 3 else 0
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, total_tokens, step):
            end = min(start + step, total_tokens)
            l_slice = local_logits[:, start:end, :] if seq_dim == 1 else local_logits[start:end, :]
            l_chunk = DTensor.from_local(l_slice, mesh, placements, run_check=False) if mesh is not None else l_slice
            lp = _tpu_logprobs_chunk_step(l_chunk, labels[start:end], temperature)
            tpu_synchronize(lp, wait=False)
            chunks.append(lp)
        out = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        tpu_synchronize(out, wait=False)
        return out


class _TPULogProbsChunkedFn(torch.autograd.Function):
    """Chunked log-probabilities autograd function avoiding full-sequence ``SliceBackward0`` buffers."""

    @staticmethod
    def forward(
        ctx: Any,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        step: int,
        mesh: Any,
        placements: Any,
    ) -> torch.Tensor:
        out = _eval_tpu_logprobs_chunks_no_grad(local_logits, labels, temperature, step, mesh, placements)
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(local_logits, labels)
            ctx.temperature = temperature
            ctx.step = step
            ctx.mesh = mesh
            ctx.placements = placements
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Optional[torch.Tensor], None, None, None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None
        local_logits, labels = ctx.saved_tensors
        temperature = ctx.temperature
        step = ctx.step
        mesh = ctx.mesh
        placements = ctx.placements
        total_tokens = int(labels.shape[0])
        seq_dim = 1 if local_logits.dim() == 3 else 0
        grad_chunks: list[torch.Tensor] = []
        for start in range(0, total_tokens, step):
            end = min(start + step, total_tokens)
            l_slice = (
                (local_logits[:, start:end, :] if seq_dim == 1 else local_logits[start:end, :])
                .detach()
                .requires_grad_(True)
            )
            with torch.enable_grad():
                l_chunk = (
                    DTensor.from_local(l_slice, mesh, placements, run_check=False) if mesh is not None else l_slice
                )
                lp = _tpu_logprobs_chunk_step(l_chunk, labels[start:end], temperature)
                (g_slice,) = torch.autograd.grad(lp, l_slice, grad_outputs=grad_output[start:end])
            tpu_synchronize(g_slice, wait=False)
            grad_chunks.append(g_slice)
        grad_local_logits = grad_chunks[0] if len(grad_chunks) == 1 else torch.cat(grad_chunks, dim=seq_dim)
        tpu_synchronize(grad_local_logits, wait=False)
        return grad_local_logits, None, None, None, None, None


def compute_tpu_logprobs_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute per-token log-probabilities on TPU in static ``chunk_size`` token slices with checkpointing.

    All-gathering and running ``F.log_softmax`` over a full ``[T, V]`` vocabulary tensor at once (e.g.
    ``[3072, 151936]`` = ``0.93 GB`` in ``bfloat16``) materializes multiple ``~1 GB`` buffers and compiles
    a separate ``1.2–2.2 GB`` XLA executable for every sequence-length bucket ``T``. Evaluating static
    ``[chunk_size, V]`` chunks via :class:`_TPULogProbsChunkedFn` reuses a single static XLA executable across
    all sequence-length buckets, keeps only the TP-sharded ``DTensor`` logits saved for backward, and avoids
    chaining ``N`` full-sequence ``SliceBackward0`` + ``aten.add`` buffers before ``lm_head`` backward.
    """
    if labels.dim() > 1:
        labels = labels.squeeze(0)
    step = chunk_size if chunk_size is not None and chunk_size > 0 else get_tpu_seq_bucket_size()
    if isinstance(logits, DTensor):
        mesh = logits.device_mesh
        placements = logits.placements
        local_logits = logits.to_local()
    else:
        mesh = None
        placements = None
        local_logits = logits
    if torch.is_grad_enabled() and local_logits.requires_grad:
        return _TPULogProbsChunkedFn.apply(local_logits, labels, float(temperature), int(step), mesh, placements)
    return _eval_tpu_logprobs_chunks_no_grad(local_logits, labels, float(temperature), int(step), mesh, placements)


def configure_torch_compile_for_tpu(recompile_limit: int = 64) -> None:
    """Configure Dynamo for per-TransformerBlock ``torch.compile(backend="tpu")``.

    verl feeds packed sequences whose length is bucketed to a multiple of
    ``VERL_TPU_SEQ_BUCKET_SIZE``, so each block sees a small set of static shapes.
    - ``automatic_dynamic_shapes=False``: after the second distinct shape Dynamo
      would otherwise retrace with symbolic shapes, which the XLA backend handles
      badly (padding + recompiles, no fusion). Keep every bucket static.
    - ``recompile_limit``: one graph per (bucket, grad-mode) pair; the default
      limit (8) is hit quickly and then Dynamo fails under ``fullgraph=True``.

    Note: ``torch._dynamo.config`` stores overrides in a ``ContextVar``, so values set
    during ``init_model`` do not carry over to later Ray actor RPCs unless
    ``_config[name].default`` is updated too.
    """
    entries = getattr(torch._dynamo.config, "_config", {})

    def _set(name: str, value: Any) -> None:
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
        if name in entries:
            entries[name].default = value

    _set("automatic_dynamic_shapes", False)
    _set("capture_scalar_outputs", True)
    _set("skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    _set("recompile_limit", max(torch._dynamo.config.recompile_limit, recompile_limit))
    _set("accumulated_recompile_limit", max(torch._dynamo.config.accumulated_recompile_limit, recompile_limit * 16))


def get_tpu_seq_bucket_size(engine_config: Any = None) -> int:
    """Return the sequence length bucket multiple used on TPU to bound XLA compilations.

    Precedence:
    1. ``engine_kwargs.tpu.seq_bucket_size`` (``get_tpu_engine_options``; also reads the deprecated
       ``VERL_TPU_SEQ_BUCKET_SIZE``).
    2. ``engine_config.pad_to_length_bucket`` when ``engine_config.pad_to_length`` is enabled.
    3. ``_DEFAULT_TPU_SEQ_BUCKET_SIZE`` (256).

    With context parallelism the bucket is rounded up so each CP rank's shard stays aligned: a multiple of
    ``2 * cp`` (head-tail chunks) for Ulysses, and of ``128 * cp`` for KV all-gather, whose local query shard
    (``S / cp``) must be a multiple of the minimum splash block size (128).
    """
    options = get_tpu_engine_options(engine_config)
    if options.seq_bucket_size is not None:
        bucket = options.seq_bucket_size
    elif engine_config is not None and bool(getattr(engine_config, "pad_to_length", False)):
        cfg_bucket = getattr(engine_config, "pad_to_length_bucket", None)
        bucket = cfg_bucket if isinstance(cfg_bucket, int) and cfg_bucket > 0 else _DEFAULT_TPU_SEQ_BUCKET_SIZE
    else:
        bucket = _DEFAULT_TPU_SEQ_BUCKET_SIZE
    if engine_config is not None:
        cp_size = int(getattr(engine_config, "context_parallel_size", 1) or 1)
        if cp_size > 1:
            strategy = options.context_parallel_strategy or "all_gather"
            bucket = math.lcm(bucket, 128 * cp_size if strategy == "all_gather" else 2 * cp_size)
    return bucket


def bucket_length(length: int, bucket_size: Optional[int] = None) -> int:
    """Round ``length`` up to the nearest positive multiple of ``bucket_size``."""
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    length = max(int(length), 1)
    return ((length + step - 1) // step) * step


def unwrap_metadata(val: Any) -> Any:
    """Unwrap a singleton tensor or per-rank list/tuple metadata value to a Python scalar."""
    if isinstance(val, list | tuple):
        val = val[0] if len(val) > 0 else val
    if isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val
    return val


def tpu_synchronize(
    tensors: Optional[torch.Tensor] = None,
    *,
    wait: bool = True,
    strict: bool = False,
) -> None:
    """Materialize and synchronize pending TPU operations (for ``tensors`` or the whole device).

    Args:
        tensors: Optional tensor (e.g. ``loss``) to materialize; ``None`` synchronizes the whole TPU device.
        wait: Wait for device execution to complete when ``True``; dispatch asynchronously when ``False``.
        strict: Raise ``RuntimeError`` if ``torch_tpu`` cannot synchronize instead of logging a warning.
    """
    if isinstance(tensors, torch.Tensor) and tensors.device.type != "tpu":
        return
    try:
        from torch_tpu._internal import sync

        if tensors is None:
            sync.synchronize(wait=wait)
        else:
            sync.synchronize(tensors, wait=wait)
    except Exception as e:  # ImportError on hosts without torch_tpu (CPU tests) included
        if strict:
            raise RuntimeError(f"TPU synchronization failed: {e}") from e
        if not isinstance(e, ImportError):
            logger.warning("Could not synchronize via torch_tpu: %s", e)


def tpu_eager_mode_context(mode: Optional[str]):
    """Context manager running the enclosed ops under the ``torch_tpu`` eager ``mode`` (no-op for ``None``).

    Mirrors torchtitan's ``tpu_config.eager_mode`` (``torchtitan/experiments/tpu/gmain.py``), which TPU
    recipes set to ``DEFER_AND_FUSE``: ops outside the compiled blocks are deferred and compiled into
    fused XLA programs at the next materialization point instead of being launched one XLA program per op
    (the ``torch_tpu`` default, ``DEFER_NEVER``).

    Uses torch_tpu's public ``torch.tpu.set_eager_mode`` / ``torch.tpu.EagerMode`` (torch_tpu >= 0.2.0), the same
    context manager as the private ``torch_tpu._internal.execution_mode.set_eager_mode``.
    """
    if mode is None:
        return nullcontext()
    tpu = getattr(torch, "tpu", None)
    set_eager_mode = getattr(tpu, "set_eager_mode", None)
    eager_mode = getattr(tpu, "EagerMode", None)
    if set_eager_mode is None or eager_mode is None:  # no torch_tpu, e.g. the CPU tests
        return nullcontext()
    return set_eager_mode(getattr(eager_mode, mode))


@torch.no_grad()
def move_tpu_params_(
    module: nn.Module,
    device: torch.device | str,
    *,
    cache_cpu_copy: bool = False,
    include_grad: bool = True,
) -> None:
    """Move ``module``'s parameters (and optionally gradients) to ``device`` in place.

    ``DTensor`` parameters get a new ``_local_tensor`` instead of being replaced, so the wrapper, its
    ``DeviceMesh("tpu", ...)`` placements, the ``SimpleFSDP`` parametrization and ``torch.compile(backend="tpu")``
    guards all stay valid (no recompiles). With ``cache_cpu_copy`` (frozen ``forward_only`` reference models), the
    first host copy is kept on the parameter and reused, so later offloads skip the device-to-host transfer.
    """
    device = torch.device(device)
    to_cpu = device.type == "cpu"
    moved: dict[int, torch.Tensor] = {}  # tied parameters share one local tensor: move it once

    def move(t: torch.Tensor) -> torch.Tensor:
        if t.device.type == device.type:
            return t
        if id(t) not in moved:
            moved[id(t)] = t.to(device, non_blocking=not to_cpu)
        return moved[id(t)]

    for p in module.parameters():
        local = p._local_tensor if isinstance(p, DTensor) else p.data
        if to_cpu and cache_cpu_copy:
            if getattr(p, "_tpu_cpu_offload_tensor", None) is None:
                p._tpu_cpu_offload_tensor = move(local)
            new = p._tpu_cpu_offload_tensor
        else:
            new = move(local)
        if isinstance(p, DTensor):
            p._local_tensor = new
        else:
            p.data = new
        if include_grad and p.grad is not None:
            if isinstance(p.grad, DTensor):
                p.grad._local_tensor = move(p.grad._local_tensor)
            else:
                p.grad = move(p.grad)


def compute_global_batch_num_tokens(data: TensorDict, dp_group: Any, tp_size: int, cp_size: int = 1) -> Any:
    """Compute the global batch token count for loss normalization on TPU via CPU all-reduce.

    The all-reduce runs over every rank, and TP and CP ranks hold the same samples, so divide by ``tp * cp``.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / (tp_size * cp_size)
    return batch_num_tokens.item()


def align_micro_batch_shapes_across_ranks(micro_batches: list[TensorDict], bucket_size: Optional[int] = None) -> None:
    """Make every rank pad micro-batch ``i`` to the same packed length and response-length bucket.

    SimpleFSDP traces the all-gather/reduce-scatter into each compiled (``spmd_safe``) TransformerBlock, and
    collectives inside a program require every participant to run the same program: mismatched shapes halt
    the TPU (``sync_flag_public_access_error``). One CPU all-reduce(MAX) over all micro-batches' buckets gives
    each micro-batch a common ``tpu_padded_seq_len`` / ``tpu_max_response_len`` that
    :func:`pad_packed_inputs_for_tpu` pads up to.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    lens: list[int] = []
    for mb in micro_batches:
        ids = mb["input_ids"]
        n_tokens = ids.values().numel() if getattr(ids, "is_nested", False) else ids.numel()
        resp = mb["responses"] if "responses" in mb.keys() else None
        max_resp = (
            int(resp.offsets().diff().max().item()) if resp is not None and getattr(resp, "is_nested", False) else 0
        )
        lens += [bucket_length(n_tokens, step), bucket_length(max_resp, step) if max_resp else 0]
    lens_tensor = torch.tensor(lens, dtype=torch.int64)
    torch.distributed.all_reduce(lens_tensor, op=torch.distributed.ReduceOp.MAX)
    resolved_lens = lens_tensor.tolist()
    for i, mb in enumerate(micro_batches):
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", int(resolved_lens[2 * i]))
        tu.assign_non_tensor_data(mb, "tpu_max_response_len", int(resolved_lens[2 * i + 1]))


def get_tpu_simple_fsdp_parallelize_fn(torchtitan_name: str) -> Callable[..., nn.Module]:
    """Return torchtitan's TPU recipe ``parallelize_fn`` for ``torchtitan_name``; it shards with SimpleFSDP.

    With ``ParallelismConfig.use_simple_fsdp`` set, the recipe (``torchtitan/experiments/tpu/<model>/infra/
    parallelize.py``) wraps the TransformerBlocks with ``spmd_safe``, applies AC and per-block ``torch.compile``
    through the model's own ``parallelize_fn`` (``skip_dp=True``), picks shard dims that divide evenly, and
    shards every parameter with ``graph_trainer``'s SimpleFSDP. Under ``torch.compile`` the FSDP collectives
    live inside each compiled TransformerBlock.
    """
    try:
        module_name, fn_name = _TPU_SIMPLE_FSDP_RECIPES[torchtitan_name]
    except KeyError:
        raise ValueError(
            f"use_simple_fsdp=True needs a torchtitan TPU recipe, and there is none for {torchtitan_name!r} "
            f"(supported: {sorted(_TPU_SIMPLE_FSDP_RECIPES)}); set use_simple_fsdp=False"
        ) from None
    recipe_parallelize_fn = getattr(importlib.import_module(module_name), fn_name)

    def _parallelize(model: nn.Module, **kwargs: Any) -> nn.Module:
        model = recipe_parallelize_fn(model, **kwargs)
        init_states_without_simple_fsdp_parametrization(model)
        return model

    return _parallelize


def init_states_without_simple_fsdp_parametrization(model: nn.Module) -> None:
    """Make ``model.init_states`` run with SimpleFSDP's parametrization disabled.

    torchtitan's ``Trainer`` parallelizes first and initializes second. While the parametrization is active,
    reading ``.weight`` returns the all-gathered plain tensor, so ``Decoder.init_states`` re-tying
    ``tok_embeddings.weight = lm_head.weight`` (tied embeddings, e.g. Qwen3 0.6B-4B) raises ``TypeError: cannot
    assign ... as parameter 'weight'``. With it disabled, the re-tie points both modules at the same sharded
    parameter (SimpleFSDP shards them separately). torchtitan's graph trainer and HF backend models initialize
    this way; its TPU qwen3 / llama3 recipes do not yet.
    """
    from torchtitan.experiments.graph_trainer import simple_fsdp

    init_states = model.init_states

    def _init_states(*args: Any, **kwargs: Any) -> None:
        # disable_active_parametrization() is not reentrant, so only take it while the parametrization is active.
        active = simple_fsdp._active_parametrization
        with simple_fsdp.disable_active_parametrization() if active else nullcontext():
            init_states(*args, **kwargs)

    model.init_states = _init_states  # type: ignore[method-assign]


def _reject_out_transform(out_transform: Any) -> None:
    # TorchTitan applies ``out_transform`` as an epilogue over (out, lse) (torchtitan/models/common/attention.py).
    # Attention-sink models such as gpt-oss rely on it; these modules return no LSE, so ignoring it would be wrong.
    if out_transform is not None:
        raise NotImplementedError(
            "TPU attention does not support the out_transform epilogue used by attention-sink models (e.g. gpt-oss)"
        )


def _document_ids(positions: Optional[torch.Tensor], segment_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Per-token document ids: TorchTitan's ``Decoder`` passes ``segment_ids``, else documents restart at position 0."""
    if segment_ids is None and positions is not None:
        from torchtitan.models.common.attention import segment_ids_from_positions

        segment_ids = segment_ids_from_positions(positions)
    return segment_ids


class TPUVarlenAttention(torch.nn.Module):
    """Dense SDPA with causal + packed-document masking for TorchTitan's folded ``[T, H, D]`` (or ``[B, S, H, D]``).

    Used when splash attention is off and as the splash fallback (CPU, or lengths no splash block divides).
    TorchTitan's ``VarlenInnerAttention`` needs CUDA flash-attention and ``FlexInnerAttention`` a ``BlockMask``.
    Document boundaries come from ``segment_ids``, so bucket-padding tokens (position 0) only attend to themselves.
    """

    def __init__(self, local_window_size: int | None = None):
        super().__init__()
        self.local_window_size = local_window_size

    def forward(self, q, k, v, *, scale=None, positions=None, segment_ids=None, out_transform=None, **kwargs):
        _reject_out_transform(out_transform)
        segment_ids = _document_ids(positions, segment_ids)
        was_3d = q.dim() == 3
        if was_3d:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if q.shape[1] != k.shape[1]:
            # Q shorter than K/V only happens under KV all-gather CP, whose inner attention brings its own
            # reference attention (torchtitan's KVAllGatherCPInnerAttention.local_forward).
            raise ValueError(
                f"TPUVarlenAttention expects q and k of equal length, got {q.shape[1]} and {k.shape[1]}; "
                "KV all-gather CP must use its inner attention's local_forward."
            )
        seq_len = q.shape[1]
        idx = torch.arange(seq_len, device=q.device)
        mask = (idx[:, None] >= idx[None, :]).unsqueeze(0)  # [1, S, S] causal
        if self.local_window_size is not None:
            mask = mask & (idx[:, None] - idx[None, :] <= self.local_window_size)
        if segment_ids is not None:
            seg = segment_ids.reshape(-1, seq_len)
            mask = mask & (seg[:, :, None] == seg[:, None, :])
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if q.shape[1] != k.shape[1]:  # GQA
            k, v = (t.repeat_interleave(q.shape[1] // k.shape[1], dim=1) for t in (k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1), scale=scale).transpose(1, 2)
        return out.squeeze(0) if was_3d else out


def _is_kv_all_gather_cp(inner: torch.nn.Module) -> bool:
    """Whether ``inner`` is a torchtitan KV all-gather CP inner attention (Q shorter than K/V)."""
    try:
        from torchtitan.experiments.tpu.cp_attention import KVAllGatherCPInnerAttention
    except ImportError:
        return False
    return isinstance(inner, KVAllGatherCPInnerAttention)


def _splash_attention_cls() -> type:
    """torchtitan's TPU ``SplashAttention`` class, or a placeholder when torchtitan is unavailable."""
    try:
        from torchtitan.experiments.tpu.kernels.splash_attention import SplashAttention
    except ImportError:
        return type("_NoSplashAttention", (), {})
    return SplashAttention


def _tpu_attention(inner: torch.nn.Module, use_splash: bool, local_window_size: Optional[int]) -> Any:
    """The TPU attention to run in place of ``inner`` (or as its CP local attention).

    The reference attention is torchtitan's ``local_forward`` for KV all-gather CP (Q shorter than K/V) and
    ``TPUVarlenAttention`` otherwise, since torchtitan's ``VarlenInnerAttention`` needs CUDA flash-attention and
    ``FlexInnerAttention`` a ``BlockMask``. With ``use_splash`` it becomes the fallback of torchtitan's
    ``SplashAttention`` (CPU, or lengths no splash block divides); ``auto_block_size`` picks block sizes per
    call because verl's bucketed lengths (e.g. 768) do not always divide the default 512.
    """
    reference = inner.local_forward if _is_kv_all_gather_cp(inner) else TPUVarlenAttention(local_window_size)
    if not use_splash:
        return reference
    return _splash_attention_cls()(
        is_causal=True,
        local_window_size=local_window_size,
        original_module=reference,
        auto_block_size=True,
    )


def install_tpu_attention(modules: Any, use_splash: bool) -> int:
    """Run TPU attention for every ``inner_attention``: torchtitan's ``SplashAttention`` or ``TPUVarlenAttention``.

    CP inner attentions (``set_local_attention``) keep their communication and take the TPU attention as their
    local attention, as in torchtitan's ``use_splash_attention_patch``; KV all-gather CP without splash keeps its
    own ``local_forward``, which torchtitan runs only on CPU or with ``allow_reference_attention`` (the engine
    requires splash for it on TPU). Must run before the first forward: per-block ``torch.compile`` traces lazily,
    so the swapped module is what gets compiled. Returns the number of replaced modules.
    """
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    tpu_attention_types = (TPUVarlenAttention, _splash_attention_cls())
    replaced = installed = 0
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        for module in model.modules():
            inner = getattr(module, "inner_attention", None)
            if not isinstance(inner, torch.nn.Module):
                continue
            is_cp = callable(getattr(inner, "set_local_attention", None))
            current = getattr(inner, "_local_attn_override", None) if is_cp else inner
            if isinstance(current, tpu_attention_types) or (is_cp and not use_splash and _is_kv_all_gather_cp(inner)):
                installed += 1
                continue
            window = getattr(inner, "window_size", None)
            left = window[0] if isinstance(window, tuple) and len(window) == 2 and window[0] >= 0 else None
            attention = _tpu_attention(inner, use_splash, None if left is None else int(left))
            if is_cp:
                inner.set_local_attention(attention)
            else:
                module.inner_attention = attention
            replaced += 1
    if replaced + installed == 0:
        raise ValueError("no module with an `inner_attention` to replace with TPU attention was found")
    if replaced:
        logger.info(
            "Replaced %d inner attention modules with %s",
            replaced,
            "SplashAttention" if use_splash else "TPUVarlenAttention",
        )
    return replaced


def _enable_reduction_none_in_torchtitan_linear_loss() -> Any:
    """Import ``torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss`` and allow ``reduction='none'``.

    Tokamax's ``pallas_mosaic_tpu_kernel`` and reference kernel natively support ``reduction='none'`` in both
    forward (returning per-token NLL of shape ``(B,)``) and backward (accepting per-token cotangent ``dout``
    of shape ``(B,)``). TorchTitan's ``_my_bwd`` unconditionally calls ``jax.numpy.reshape(dout, ())`` for
    scalar reductions (``'sum'``/``'mean'``); rebinding ``jax.numpy.reshape`` in ``_my_bwd``'s globals to
    preserve 1D ``dout`` when ``dout.size != 1`` lets us call TorchTitan's ``linear_softmax_cross_entropy_loss``
    directly without duplicating any kernel or custom-op code.
    """
    from torchtitan.experiments.tpu.kernels import linear_softmax_cross_entropy_loss as lce_mod

    if getattr(lce_mod, "_verl_reduction_none_patched", False):
        return lce_mod

    import types

    orig_bwd = lce_mod._my_bwd
    real_jax = lce_mod.jax

    class _JaxNumpyProxy:
        @staticmethod
        def reshape(a: Any, shape: Any, *args: Any, **kwargs: Any) -> Any:
            if shape == () and getattr(a, "size", 1) != 1:
                return a
            return real_jax.numpy.reshape(a, shape, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(real_jax.numpy, name)

    class _JaxProxy:
        numpy = _JaxNumpyProxy()

        def __getattr__(self, name: str) -> Any:
            return getattr(real_jax, name)

    lce_mod._my_bwd = types.FunctionType(
        orig_bwd.__code__,
        {**orig_bwd.__globals__, "jax": _JaxProxy()},
        orig_bwd.__name__,
        orig_bwd.__defaults__,
        orig_bwd.__closure__,
    )
    lce_mod._verl_reduction_none_patched = True
    return lce_mod


def fused_linear_logprobs_tpu(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    temperature: float = 1.0,
    implementation: str = "mosaic_tpu",
    b_block_size: int = 1024,
    h_block_size: int = 512,
    v_block_size: int = 2048,
) -> torch.Tensor:
    """Compute per-token ``log_probs`` on TPU via the Pallas fused linear + softmax cross-entropy kernel.

    Avoids materializing the ``[T, V]`` logits and ``float32`` log-softmax tensors in HBM. Pads ``T`` up to
    a multiple of ``b_block_size`` (1024) as required by Mosaic TPU tiles and slices back to ``[:T]``, so
    padded rows receive zero upstream gradient in the backward pass.
    """
    if hidden_states.dim() > 2:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    if labels.dim() > 1:
        labels = labels.reshape(-1)

    b_dim = int(hidden_states.shape[0])
    x = hidden_states
    temp = float(temperature)
    if temp != 1.0:
        x = (x * (1.0 / temp)).to(weight_t.dtype)
    elif x.dtype != weight_t.dtype:
        x = x.to(weight_t.dtype)

    padded_b = max(b_block_size, ((b_dim + b_block_size - 1) // b_block_size) * b_block_size)
    pad_b = padded_b - b_dim
    if pad_b > 0:
        x = F.pad(x, (0, 0, 0, pad_b))
        labels = F.pad(labels, (0, pad_b), value=0)

    lce_mod = _enable_reduction_none_in_torchtitan_linear_loss()
    nll = lce_mod.linear_softmax_cross_entropy_loss(
        x.contiguous(),
        labels.contiguous(),
        weight_t,
        reduction="none",
        implementation=implementation,
        b_block_size=b_block_size,
        h_block_size=h_block_size,
        v_block_size=v_block_size,
    )
    if pad_b > 0:
        nll = nll[:b_dim]
    return -nll


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
    bucket_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Bucket-pad packed (nested) 1D sequences on CPU before the H2D copy.

    Pad tokens get position 0, so the attention treats each one as its own document (it only attends to itself).
    Returns ``(input_ids, position_ids, labels, orig_seq_len)``.
    """
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu()
    position_ids_cpu = position_ids.values().detach().cpu()
    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=0)

    orig_seq_len = int(input_ids_cpu.shape[-1])
    # tpu_padded_seq_len: common length across ranks from align_micro_batch_shapes_across_ranks (SimpleFSDP).
    padded_seq_len = max(
        bucket_length(orig_seq_len, step),
        int(tu.get_non_tensor_data(data=micro_batch, key="tpu_padded_seq_len", default=0) or 0),
    )
    pad_len = padded_seq_len - orig_seq_len
    if pad_len > 0:
        input_ids_cpu, labels_cpu, position_ids_cpu = (
            F.pad(t, (0, pad_len), value=0) for t in (input_ids_cpu, labels_cpu, position_ids_cpu)
        )

    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        raw_max_resp = int(micro_batch["responses"].offsets().diff().max().item())
        max_resp = max(
            bucket_length(raw_max_resp, step),
            int(tu.get_non_tensor_data(data=micro_batch, key="tpu_max_response_len", default=0) or 0),
        )
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_resp)

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        orig_seq_len,
    )
