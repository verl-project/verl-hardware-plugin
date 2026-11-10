# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU helper utilities for the TorchTitan training engine.

Provides sequence-length bucketing, host-side metadata unwrapping, packed-input padding,
``torch.compile(backend="tpu")`` Dynamo configuration, torchtitan's TPU ``SplashAttention``, the torchtitan TPU
recipe that shards with ``SimpleFSDP``, cross-rank micro-batch shape alignment, and ``DEFER_AND_FUSE``
eager-mode contexts for ``torch_tpu`` training.
"""

from __future__ import annotations

import functools
import importlib
import inspect
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
from verl_hardware_plugin.accelerators.tpu.device_api import tpu_synchronize
from verl_hardware_plugin.accelerators.tpu.engines.tpu_model_kernels import has_64mib_scoped_vmem
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
    try:
        model_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if module_name.startswith("torchtitan.experiments.tpu."):
            raise ImportError(
                f"Model {name!r} on TPU requires {module_name!r} from https://github.com/google-pytorch/torchtitan "
                "(upstream pytorch/torchtitan does not include torchtitan.experiments.tpu)."
            ) from exc
        raise
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
    (for FSDP/HSDP with optional TP, CP and EP where PP == 1) and ``DEFER_AND_FUSE``.
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

    # SimpleFSDP composes with TP, CP and EP (torchtitan's TPU Qwen3.5 wraps the routed experts on the expert-DP mesh
    # under EP, google-pytorch/torchtitan#6), but not with PP.
    simple_fsdp_compatible = getattr(engine_config, "pipeline_parallel_size", 1) == 1
    use_simple_fsdp = _or_default(options.use_simple_fsdp, use_torch_compile and simple_fsdp_compatible)
    if use_simple_fsdp and not simple_fsdp_compatible:
        raise ValueError("use_simple_fsdp supports FSDP/HSDP (+ optional TP/CP/EP) only (pipeline parallel size 1)")

    if options.tpu_eager_mode is None:
        tpu_eager_mode = "DEFER_AND_FUSE" if use_torch_compile else None
    else:
        tpu_eager_mode = options.tpu_eager_mode.strip().upper()
        tpu_eager_mode = None if tpu_eager_mode in ("", "NONE", "NULL") else tpu_eager_mode
    if tpu_eager_mode not in (None, "DEFER_AND_FUSE", "DEFER_NEVER"):
        raise ValueError(f"tpu_eager_mode {tpu_eager_mode!r} not supported")

    return use_splash_attention, use_simple_fsdp, tpu_eager_mode


def resolve_tpu_use_fused_linear_cross_entropy(engine_config: Any, *, required: bool = False) -> bool:
    """Whether to compute log-probs with the fused LM-head + cross-entropy kernel (``engines.fused_logprob``).

    The kernel never materializes the ``[tokens, vocab]`` logits, which saves memory and keeps them in fp32.
    It only supports ``tensor_parallel_size == 1``.

    Default: on with ``use_torch_compile``. With ``required`` (set by the engine for models that need it, e.g. MoE),
    it is on by default and TP > 1 raises instead of falling back. ``use_fused_linear_cross_entropy=False`` always
    turns it off.
    """
    user_choice = get_tpu_engine_options(engine_config).use_fused_linear_cross_entropy  # None (unset), True or False
    tp_size = int(getattr(engine_config, "tensor_parallel_size", 1) or 1)

    # 1. An explicit opt-out always wins.
    if user_choice is False:
        if required:
            logger.warning(
                "use_fused_linear_cross_entropy=False on a model that requires the fused log-prob: the "
                "[tokens, vocab] logits are materialized and large token budgets may run out of memory"
            )
        return False

    # 2. The kernel does not support tensor parallelism.
    if tp_size > 1:
        if required:
            raise ValueError(
                f"tensor_parallel_size={tp_size}: this model requires the fused LM-head log-prob "
                "(engine_kwargs.tpu.use_fused_linear_cross_entropy), which supports tensor_parallel_size=1 only. "
                "Use FSDP (data_parallel_shard_size) instead, or set use_fused_linear_cross_entropy=False."
            )
        return False

    # 3. On if the model requires it or the user asked for it; if unset, on with torch.compile.
    if required or user_choice:
        return True
    return bool(getattr(engine_config, "use_torch_compile", False))


_DEFAULT_SPLASH_MAX_BLOCK_SIZE = 512  # torchtitan's splash_block_size_for default
_MIN_SPLASH_BLOCK_SIZE = 128


def resolve_tpu_splash_max_block_size(engine_config: Any) -> int:
    """``engine_kwargs.tpu.splash_max_block_size`` (default 512); a power of two >= 128."""
    value = get_tpu_engine_options(engine_config).splash_max_block_size
    block = _DEFAULT_SPLASH_MAX_BLOCK_SIZE if value is None else int(value)
    if block < _MIN_SPLASH_BLOCK_SIZE or block & (block - 1):
        raise ValueError(f"engine_kwargs.tpu.splash_max_block_size must be a power of two >= 128, got {value!r}")
    return block


_MIXED_PRECISION_REDUCE_DTYPES = ("float32", "bfloat16")


def resolve_tpu_mixed_precision_reduce(engine_config: Any) -> Optional[str]:
    """``engine_kwargs.tpu.mixed_precision_reduce`` (``float32`` / ``bfloat16``), or ``None`` for torchtitan's."""
    value = get_tpu_engine_options(engine_config).mixed_precision_reduce
    if value is None:
        return None
    dtype = value.strip().lower()
    if dtype not in _MIXED_PRECISION_REDUCE_DTYPES:
        raise ValueError(
            f"engine_kwargs.tpu.mixed_precision_reduce must be one of {_MIXED_PRECISION_REDUCE_DTYPES}, got {value!r}"
        )
    return dtype


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


def align_micro_batch_shapes_across_ranks(
    micro_batches: list[TensorDict],
    bucket_size: Optional[int] = None,
    static_len: int = 0,
    static_response_len: int = 0,
) -> None:
    """Make every rank pad micro-batch ``i`` to the same packed length and response-length bucket.

    SimpleFSDP traces the all-gather/reduce-scatter into each compiled (``spmd_safe``) TransformerBlock, and
    collectives inside a program require every participant to run the same program: mismatched shapes halt
    the TPU (``sync_flag_public_access_error``). One CPU all-reduce(MAX) over all micro-batches' buckets gives
    each micro-batch a common ``tpu_padded_seq_len`` / ``tpu_max_response_len`` that
    :func:`pad_packed_inputs_for_tpu` pads up to.

    ``static_len`` / ``static_response_len`` (``engine_kwargs.tpu.static_micro_batch_len`` /
    ``static_max_response_len``) raise every packed / response length to at least that many tokens (rounded up to
    the bucket), so the trainer compiles one shape instead of one per bucket it meets.
    """
    step = bucket_size if bucket_size is not None and bucket_size > 0 else get_tpu_seq_bucket_size()
    static_padded = bucket_length(static_len, step) if static_len > 0 else 0
    static_resp_padded = bucket_length(static_response_len, step) if static_response_len > 0 else 0
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        for mb in micro_batches:
            if static_padded:
                tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", static_padded)
            if static_resp_padded:
                tu.assign_non_tensor_data(mb, "tpu_max_response_len", static_resp_padded)
        return
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
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", max(int(resolved_lens[2 * i]), static_padded))
        resp_len = int(resolved_lens[2 * i + 1])
        tu.assign_non_tensor_data(mb, "tpu_max_response_len", max(resp_len, static_resp_padded) if resp_len else 0)


def _wrap_routed_experts_on_edp_mesh(model: nn.Module, *, parallel_dims: Any, training: Any) -> None:
    """Wrap ``MoE.routed_experts.inner_experts`` on ``edp_mesh`` before ``apply_simple_fsdp`` when needed.

    Upstream ``apply_simple_fsdp`` checks ``isinstance(block, TransformerBlock)`` before wrapping ``inner_experts`` on
    ``edp_mesh``. That skips ``Qwen35TransformerBlock`` (which subclasses ``Module``) on torchtitan builds without
    ``apply_qwen3_5_simple_fsdp`` (google-pytorch/torchtitan#6), as well as blocks wrapped in ``CheckpointWrapper``
    under activation checkpointing; ``data_parallel(model, dp_mesh)`` would then fail trying to shard EP-sharded expert
    ``DTensor``s over ``dp_mesh``. ``data_parallel`` skips submodules it has already sharded.
    """
    if not getattr(parallel_dims, "ep_enabled", False):
        return
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.distributed.utils import get_spmd_backend
    from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
    from torchtitan.models.common.moe import MoE

    use_spmd_types = get_spmd_backend() == "spmd_types"
    if parallel_dims.dp_replicate_enabled:
        dp_mode = "hybrid_shard" if (parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled) else "replicate"
        edp_mesh_names = ["dp_replicate", "efsdp"]
    else:
        dp_mode = "fully_shard"
        edp_mesh_names = ["efsdp"]
    edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
    if edp_mesh is None:
        return
    efsdp_degree = edp_mesh["efsdp"].size()
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )
    non_dp_mesh = parallel_dims.get_optional_mesh("ep") if use_spmd_types else None

    for module_fqn, moe in model.named_modules():
        if not isinstance(moe, MoE):
            continue
        inner_experts = moe.routed_experts.inner_experts
        if hasattr(inner_experts, "parametrizations"):
            continue
        if inner_experts.num_experts % (efsdp_degree * parallel_dims.ep) == 0:
            experts_shard_dim = 0
        else:
            incompatible = [
                f"{name}{tuple(parameter.shape)}"
                for name, parameter in inner_experts.named_parameters(recurse=False)
                if parameter.ndim != 3 or parameter.shape[1] % efsdp_degree != 0
            ]
            if incompatible:
                raise ValueError(
                    f"Cannot FSDP-shard {module_fqn}.routed_experts.inner_experts on dimension 1 over "
                    f"{efsdp_degree} efsdp ranks; incompatible expert parameters: {incompatible}"
                )
            experts_shard_dim = 1
        moe.routed_experts.inner_experts = data_parallel(
            inner_experts,
            edp_mesh,
            dp_mode,
            mp_policy=mp_policy,
            shard_dim=experts_shard_dim,
            non_dp_mesh=non_dp_mesh,
        )


def _relax_non_static_ep_moe_compile(model: nn.Module, *, parallel_dims: Any, compile_config: Any) -> None:
    """Use ``fullgraph=False`` on compiled MoE blocks whose EP dispatcher performs host split syncs.

    Without ``TpuQwen35TokenDispatcher`` (google-pytorch/torchtitan#6), ``AllToAllTokenDispatcher`` /
    ``TpuAllToAllTokenDispatcher`` call ``.to("cpu")`` and ``.tolist()`` in ``_sync_token_count_exchange`` under
    ``ep_enabled``, which raises a Dynamo graph-break error when ``apply_compile`` compiles the block with
    ``fullgraph=True``.
    """
    if not getattr(parallel_dims, "ep_enabled", False) or not getattr(compile_config, "enable", False):
        return
    layers = getattr(model, "layers", None)
    if layers is None:
        return
    backend = getattr(compile_config, "backend", "tpu")
    _static_dispatchers = ("TpuQwen35TokenDispatcher", "TpuRaggedEPTokenDispatcher")
    for _, block in layers.named_children():
        if getattr(block, "_compiled_call_impl", None) is None:
            continue
        needs_graph_break = any(
            getattr(m, "token_dispatcher", None) is not None
            and not any(c.__name__ in _static_dispatchers for c in type(m.token_dispatcher).__mro__)
            for m in block.modules()
        )
        if needs_graph_break:
            block.compile(backend=backend, fullgraph=False)
            block._tpu_compile_relaxed = True  # type: ignore[attr-defined]


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
    recipe_module = importlib.import_module(module_name)
    recipe_parallelize_fn = getattr(recipe_module, fn_name)

    def _parallelize(model: nn.Module, **kwargs: Any) -> nn.Module:
        parallel_dims = kwargs.get("parallel_dims")
        ep_needs_edp_wrap = (
            bool(getattr(parallel_dims, "ep_enabled", False))
            and not hasattr(recipe_module, "apply_qwen3_5_simple_fsdp")
            and hasattr(recipe_module, "apply_simple_fsdp")
        )
        orig_apply_simple_fsdp = getattr(recipe_module, "apply_simple_fsdp", None)
        if ep_needs_edp_wrap and orig_apply_simple_fsdp is not None:

            def _apply_simple_fsdp_with_ep(m: nn.Module, *, parallel_dims: Any, training: Any, **fsdp_kw: Any):
                _wrap_routed_experts_on_edp_mesh(m, parallel_dims=parallel_dims, training=training)
                return orig_apply_simple_fsdp(m, parallel_dims=parallel_dims, training=training, **fsdp_kw)

            recipe_module.apply_simple_fsdp = _apply_simple_fsdp_with_ep  # type: ignore[attr-defined]
        try:
            model = recipe_parallelize_fn(model, **kwargs)
        finally:
            if ep_needs_edp_wrap and orig_apply_simple_fsdp is not None:
                recipe_module.apply_simple_fsdp = orig_apply_simple_fsdp  # type: ignore[attr-defined]
        _relax_non_static_ep_moe_compile(
            model,
            parallel_dims=parallel_dims,
            compile_config=kwargs.get("compile_config"),
        )
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


_SPLASH_64M_VMEM_BLOCK_KWARGS: dict[str, int] = {
    "block_q": 1024,
    "block_kv": 1024,
    "block_dkv": 1024,
    "block_kv_compute": 512,
    "block_q_dkv": 1024,
    "block_kv_dkv": 1024,
    "block_kv_dkv_compute": 512,
    "block_q_dq": 1024,
    "block_kv_dq": 1024,
}
_SPLASH_1024_SUBCLASSES: dict[type, type] = {}


def _splash_accepts_block_kwargs(splash_cls: type) -> bool:
    try:
        params = inspect.signature(splash_cls).parameters
    except (TypeError, ValueError):
        return False
    return "block_q" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _get_splash_1024_auto_cls(splash_cls: type) -> type:
    cached = _SPLASH_1024_SUBCLASSES.get(splash_cls)
    if cached is not None:
        return cached

    class _Splash1024AutoAttention(splash_cls):  # type: ignore[valid-type,misc]
        """Uses TorchTitan's 1024 Splash tiles on 1024-divisible TPU rows under >=60 MiB scoped VMEM, else auto-512."""

        def __init__(
            self,
            *,
            is_causal: bool = True,
            local_window_size: Optional[int] = None,
            original_module: Optional[nn.Module] = None,
        ) -> None:
            super().__init__(
                is_causal=is_causal,
                local_window_size=local_window_size,
                original_module=original_module,
                auto_block_size=True,
            )
            self._splash_1024 = splash_cls(
                is_causal=is_causal,
                local_window_size=local_window_size,
                original_module=original_module,
                auto_block_size=False,
                **_SPLASH_64M_VMEM_BLOCK_KWARGS,
            )

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            seq_len = q.shape[0] if q.dim() == 3 else q.shape[1]
            if q.device.type != "cpu" and seq_len >= 1024 and seq_len % 1024 == 0:
                return self._splash_1024(q, k, v, *args, **kwargs)
            return super().forward(q, k, v, *args, **kwargs)

    _SPLASH_1024_SUBCLASSES[splash_cls] = _Splash1024AutoAttention
    return _Splash1024AutoAttention


def _tpu_attention(inner: torch.nn.Module, use_splash: bool, local_window_size: Optional[int]) -> Any:
    """The TPU attention to run in place of ``inner`` (or as its CP local attention).

    The reference attention is torchtitan's ``local_forward`` for KV all-gather CP (Q shorter than K/V) and
    ``TPUVarlenAttention`` otherwise, since torchtitan's ``VarlenInnerAttention`` needs CUDA flash-attention and
    ``FlexInnerAttention`` a ``BlockMask``. With ``use_splash`` it becomes the fallback of torchtitan's
    ``SplashAttention`` (CPU, or lengths no splash block divides); ``auto_block_size`` picks block sizes per
    call because verl's bucketed lengths (e.g. 768) do not always divide the default 512. When >=60 MiB scoped
    VMEM is configured, 1024-divisible TPU rows dispatch to TorchTitan's 1024 Splash tiles (``block_kv_compute=512``,
    ``block_kv_dkv_compute=512``).
    """
    reference = inner.local_forward if _is_kv_all_gather_cp(inner) else TPUVarlenAttention(local_window_size)
    if not use_splash:
        return reference
    splash_cls = _splash_attention_cls()
    if has_64mib_scoped_vmem() and _splash_accepts_block_kwargs(splash_cls):
        return _get_splash_1024_auto_cls(splash_cls)(
            is_causal=True,
            local_window_size=local_window_size,
            original_module=reference,
        )
    return splash_cls(
        is_causal=True,
        local_window_size=local_window_size,
        original_module=reference,
        auto_block_size=True,
    )


def set_splash_max_block_size(max_block_size: int) -> None:
    """Sets the largest block size torchtitan's ``SplashAttention`` (``auto_block_size=True``) picks per call.

    ``SplashAttention`` takes its block size from the module-level ``splash_block_size_for(seq_len)``, whose cap is
    512; long packed micro-batches (8-24k tokens) run faster with larger blocks, since Splash's grid grows as
    ``(seq_len / block)^2``. Rebinds that function to one with ``max_block_size`` (the original for 512).
    Process-wide, like the engine option it implements; a no-op without torchtitan's TPU kernels.

    TODO(torchtitan): add a ``max_block_size`` parameter to ``SplashAttention`` upstream in torchtitan and pass it
    when constructing the attention module instead of rebinding ``splash_block_size_for``.
    """
    try:
        from torchtitan.experiments.tpu.kernels import splash_attention as splash_module
    except ImportError:
        return
    original = getattr(splash_module, "_verl_original_splash_block_size_for", None)
    if original is None:
        original = splash_module.splash_block_size_for
        splash_module._verl_original_splash_block_size_for = original
    if max_block_size == _DEFAULT_SPLASH_MAX_BLOCK_SIZE:
        splash_module.splash_block_size_for = original
    else:
        splash_module.splash_block_size_for = functools.partial(original, max_block_size=max_block_size)
    logger.info("Splash attention: largest block size %d", max_block_size)


def row_slot_permutation(cu_seqlens: torch.Tensor, padded_len: int, slot_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(slot_src, contig_src)``: packed -> one-sequence-per-``slot_len``-row layout and back.

    Sequence ``i`` moves to token ``i * slot_len``; the packed padding fills the rest. ``packed[slot_src]`` is the slot
    layout and ``slot_layout[contig_src]`` the packed one.
    """
    offsets = cu_seqlens.tolist()
    lengths = [b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    if padded_len % slot_len or len(lengths) > padded_len // slot_len or max(lengths, default=0) > slot_len:
        raise ValueError(
            f"gdn_row_slot_len={slot_len}: sequences {lengths} do not fit a {padded_len}-token micro-batch"
        )
    slot_src = torch.full((padded_len,), -1, dtype=torch.int64)
    for i, (start, length) in enumerate(zip(offsets[:-1], lengths, strict=True)):
        slot_src[i * slot_len : i * slot_len + length] = torch.arange(start, start + length)
    slot_src[slot_src < 0] = torch.arange(offsets[-1] if offsets else 0, padded_len)
    contig_src = torch.empty_like(slot_src)
    contig_src[slot_src] = torch.arange(padded_len)
    return slot_src, contig_src


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


FUSED_ENTROPY_MAX_CHUNK = 4096


def fused_entropy_chunk_size(num_tokens: int, max_chunk: int = FUSED_ENTROPY_MAX_CHUNK, align: int = 256) -> int:
    """Largest multiple of ``align`` that divides ``num_tokens`` and is <= ``max_chunk``; ``num_tokens`` otherwise.

    One static chunk shape keeps the chunk loop to a single compiled program per micro-batch shape.
    """
    if num_tokens <= max_chunk:
        return num_tokens
    chunk = (max_chunk // align) * align
    while chunk >= align:
        if num_tokens % chunk == 0:
            return chunk
        chunk -= align
    return num_tokens


@torch.no_grad()
def chunked_entropy_from_hidden_tpu(
    hidden_states: torch.Tensor, weight_t: torch.Tensor, temperature: float, chunk_size: Optional[int] = None
) -> torch.Tensor:
    """Per-token entropy of ``softmax(hidden_states @ weight_t / temperature)``, in token chunks, without gradient.

    For passes that report the entropy but do not differentiate it (old log-probs): the logits of one chunk of
    tokens at a time are materialized (bf16 matmul as the unfused LM head, fp32 entropy as on GPU) instead of the
    whole ``[T, V]`` logits.
    """
    from verl.utils.torch_functional import entropy_from_logits

    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    num_tokens = int(hidden.shape[0])
    chunk = chunk_size or fused_entropy_chunk_size(num_tokens)
    out = []
    for start in range(0, num_tokens, chunk):
        logits = (hidden[start : start + chunk] @ weight_t) / temperature
        out.append(entropy_from_logits(logits.float()))
    return torch.cat(out)


_PAD_VOCAB_STRIDE = 9973  # prime stride: spreads packed-padding token IDs across vocab instead of 1 or math experts


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
    bucket_size: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Bucket-pad packed (nested) 1D sequences on CPU before the H2D copy.

    Pad tokens get position 0, so the attention treats each one as its own document (it only attends to itself).
    Under expert parallelism (``TpuQwen35TokenDispatcher``), repeating token 0 sends every pad slot to the same 8
    experts, and cycling the micro-batch's own tokens concentrates pad traffic onto the same domain/math experts as
    real tokens; striding across ``vocab_size`` disperses pad slots across the full vocabulary and expert set.
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
        if vocab_size is not None and int(vocab_size) > 1:
            pad_ids = (torch.arange(pad_len, dtype=torch.int64) * _PAD_VOCAB_STRIDE) % int(vocab_size)
            input_ids_cpu = torch.cat((input_ids_cpu, pad_ids.to(dtype=input_ids_cpu.dtype)), dim=-1)
        elif orig_seq_len > 0:
            pad_ids = input_ids_cpu[torch.arange(pad_len, dtype=torch.int64) % orig_seq_len]
            input_ids_cpu = torch.cat((input_ids_cpu, pad_ids), dim=-1)
        else:
            input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)

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
