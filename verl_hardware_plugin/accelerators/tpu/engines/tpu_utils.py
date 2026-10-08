# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU helper utilities for the TorchTitan training engine.

Provides sequence-length bucketing, host-side metadata unwrapping, TorchTitan
config overrides, packed-input padding, ``torch.compile(backend="tpu")`` Dynamo
configuration, ``TPUSplashAttention``, ``SimpleFSDP`` wrapping, cross-rank
micro-batch shape alignment, and ``DEFER_AND_FUSE`` eager-mode contexts for
``torch_tpu`` training.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import TorchtitanEngineConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to attach the full bucket-padded device tensor to a CPU
# nested tensor returned from ``prepare_model_outputs``, so loss functions can
# consume the fixed-shape device tensor directly without XLA recompilation.
TPU_PADDED_VALUES_ATTR = "_tpu_padded_values"

_DEFAULT_TPU_SEQ_BUCKET_SIZE = 256


def extend_torchtitan_engine_config() -> None:
    """Register TPU-specific optional fields on ``TorchtitanEngineConfig`` for Hydra/OmegaConf."""
    extras: list[tuple[str, Any, Any]] = [
        ("use_splash_attention", Optional[bool], None),
        ("use_simple_fsdp", Optional[bool], None),
        ("tpu_eager_mode", Optional[str], None),
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


def _parse_optional_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    norm = raw.strip().lower()
    if norm in ("1", "true", "yes", "on"):
        return True
    if norm in ("0", "false", "no", "off"):
        return False
    return None


def resolve_tpu_torchtitan_options(engine_config: Any) -> tuple[bool, bool, Optional[str]]:
    """Resolve ``(use_splash_attention, use_simple_fsdp, tpu_eager_mode)`` for TPU TorchTitan.

    Precedence for each option:
    1. Explicit environment variable override (``VERL_TPU_USE_SPLASH_ATTENTION``,
       ``VERL_TPU_USE_SIMPLE_FSDP``, ``VERL_TPU_EAGER_MODE``).
    2. Explicit ``engine_config`` field value when non-``None``.
    3. Automatic TPU defaults based on ``engine_config.use_torch_compile``:
       when ``use_torch_compile=True``, enables ``TPUSplashAttention``, ``SimpleFSDP``
       (for pure FSDP/HSDP where TP/PP/CP/EP == 1), and ``DEFER_AND_FUSE``.
    """
    use_torch_compile = bool(getattr(engine_config, "use_torch_compile", False))

    splash_env = _parse_optional_bool_env("VERL_TPU_USE_SPLASH_ATTENTION")
    splash_cfg = getattr(engine_config, "use_splash_attention", None)
    if splash_env is not None:
        use_splash_attention = splash_env
    elif splash_cfg is not None:
        use_splash_attention = bool(splash_cfg)
    else:
        use_splash_attention = use_torch_compile

    if use_torch_compile and not use_splash_attention:
        raise ValueError(
            "use_torch_compile=True on TPU requires use_splash_attention=True: torch_tpu's compiled "
            "F.scaled_dot_product_attention produces NaN q/k/v gradients, so every optimizer step would be "
            "skipped. Use splash attention + torch.compile (use_splash_attention=True), or set "
            "use_torch_compile=False."
        )

    pure_fsdp = (
        getattr(engine_config, "tensor_parallel_size", 1) == 1
        and getattr(engine_config, "pipeline_parallel_size", 1) == 1
        and getattr(engine_config, "context_parallel_size", 1) == 1
        and getattr(engine_config, "expert_parallel_size", 1) == 1
    )
    simple_fsdp_env = _parse_optional_bool_env("VERL_TPU_USE_SIMPLE_FSDP")
    simple_fsdp_cfg = getattr(engine_config, "use_simple_fsdp", None)
    if simple_fsdp_env is not None:
        use_simple_fsdp = simple_fsdp_env
    elif simple_fsdp_cfg is not None:
        use_simple_fsdp = bool(simple_fsdp_cfg)
    else:
        use_simple_fsdp = bool(use_torch_compile and pure_fsdp)

    if use_simple_fsdp and not pure_fsdp:
        raise ValueError(
            "use_simple_fsdp supports pure FSDP/HSDP only (tensor/pipeline/context/expert parallel size 1)"
        )

    eager_env = os.environ.get("VERL_TPU_EAGER_MODE")
    eager_cfg = getattr(engine_config, "tpu_eager_mode", None)
    if eager_env is not None and eager_env.strip() != "":
        norm_eager = eager_env.strip().upper()
        tpu_eager_mode = None if norm_eager in ("NONE", "NULL") else norm_eager
    elif eager_cfg is not None:
        norm_cfg = str(eager_cfg).strip().upper()
        tpu_eager_mode = None if norm_cfg in ("", "NONE", "NULL") else norm_cfg
    else:
        tpu_eager_mode = "DEFER_AND_FUSE" if use_torch_compile else None

    if tpu_eager_mode not in (None, "DEFER_AND_FUSE", "DEFER_NEVER"):
        raise ValueError(f"tpu_eager_mode {tpu_eager_mode!r} not supported")

    return use_splash_attention, use_simple_fsdp, tpu_eager_mode


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
    import torch._dynamo

    entries = getattr(torch._dynamo.config, "_config", {})

    def _set(name: str, value: Any) -> None:
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
        if name in entries:
            entries[name].default = value

    _set("automatic_dynamic_shapes", False)
    _set("assume_static_by_default", True)
    _set("capture_scalar_outputs", True)
    _set("skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            _set(name, max(getattr(torch._dynamo.config, name), recompile_limit))
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        _set(
            "accumulated_recompile_limit",
            max(torch._dynamo.config.accumulated_recompile_limit, recompile_limit * 16),
        )


def splash_block_size_for(seq_len: int, max_block_size: int = 512, min_block_size: int = 128) -> int:
    """Largest power-of-two splash block size <= ``max_block_size`` that divides ``seq_len``.

    The splash kernel requires every block size to divide the sequence length. verl pads
    packed sequences to a multiple of ``VERL_TPU_SEQ_BUCKET_SIZE`` (256), so a fixed 512
    block (torchtitan's default) fails for e.g. 768 tokens. Returns 0 if no block
    >= ``min_block_size`` divides ``seq_len``.
    """
    block = max_block_size
    while block >= min_block_size:
        if seq_len % block == 0:
            return block
        block //= 2
    return 0


def get_tpu_seq_bucket_size() -> int:
    """Return the sequence length bucket multiple used on TPU to bound XLA compilations."""
    try:
        val = int(os.environ.get("VERL_TPU_SEQ_BUCKET_SIZE", _DEFAULT_TPU_SEQ_BUCKET_SIZE))
        return val if val > 0 else _DEFAULT_TPU_SEQ_BUCKET_SIZE
    except ValueError:
        return _DEFAULT_TPU_SEQ_BUCKET_SIZE


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


def synchronize_tpu_loss(loss: torch.Tensor) -> None:
    """Materialize the forward graph loss without blocking to split XLA forward and backward compilation passes."""
    if loss.device.type != "tpu":
        return
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(loss, wait=False)
        return
    except ImportError:
        pass
    tpu_mod = getattr(torch, "tpu", None)
    if tpu_mod is not None and hasattr(tpu_mod, "synchronize"):
        try:
            tpu_mod.synchronize()
        except Exception as e:
            logger.debug("torch.tpu.synchronize() failed during loss sync: %s", e)


def tpu_eager_mode_context(mode: Optional[str]):
    """Context manager running the enclosed ops under the ``torch_tpu`` eager ``mode`` (no-op for ``None``).

    Mirrors torchtitan's ``tpu_config.eager_mode`` (``torchtitan/experiments/tpu/gmain.py``), which TPU
    recipes set to ``DEFER_AND_FUSE``: ops outside the compiled blocks are deferred and compiled into
    fused XLA programs at the next materialization point instead of being launched one XLA program per op
    (the ``torch_tpu`` default, ``DEFER_NEVER``).
    """
    if mode is None:
        return nullcontext()
    try:
        from torch_tpu._internal import execution_mode

        return execution_mode.set_eager_mode(getattr(execution_mode.EagerMode, mode))
    except ImportError:
        return nullcontext()


def compute_global_batch_num_tokens(data: TensorDict, dp_group: Any, tp_size: int) -> Any:
    """Compute the global batch token count for loss normalization on TPU via CPU all-reduce."""
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def align_micro_batch_shapes_across_ranks(micro_batches: list[TensorDict]) -> None:
    """Make every rank pad micro-batch ``i`` to the same packed length and response-length bucket.

    SimpleFSDP traces the all-gather/reduce-scatter into each compiled (``spmd_safe``) TransformerBlock, and
    collectives inside a program require every participant to run the same program: mismatched shapes halt
    the TPU (``sync_flag_public_access_error``). One CPU all-reduce(MAX) over all micro-batches' buckets gives
    each micro-batch a common ``tpu_padded_seq_len`` / ``tpu_max_response_len`` that
    :func:`pad_packed_inputs_for_tpu` pads up to.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return
    bucket_size = get_tpu_seq_bucket_size()
    lens: list[int] = []
    for mb in micro_batches:
        ids = mb["input_ids"]
        n_tokens = ids.values().numel() if getattr(ids, "is_nested", False) else ids.numel()
        resp = mb["responses"] if "responses" in mb.keys() else None
        max_resp = (
            int(resp.offsets().diff().max().item()) if resp is not None and getattr(resp, "is_nested", False) else 0
        )
        lens += [bucket_length(n_tokens, bucket_size), bucket_length(max_resp, bucket_size) if max_resp else 0]
    lens_tensor = torch.tensor(lens, dtype=torch.int64)
    torch.distributed.all_reduce(lens_tensor, op=torch.distributed.ReduceOp.MAX)
    resolved_lens = lens_tensor.tolist()
    for i, mb in enumerate(micro_batches):
        tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", int(resolved_lens[2 * i]))
        tu.assign_non_tensor_data(mb, "tpu_max_response_len", int(resolved_lens[2 * i + 1]))


def make_simple_fsdp_parallelize_fn(parallelize_fn: Any, *, spmd_safe_blocks: bool = False):
    """Wrap a torchtitan ``parallelize_fn`` so it shards with SimpleFSDP instead of FSDP2.

    Mirrors torchtitan's TPU recipes (``experiments/tpu/qwen3/infra/parallelize.py`` with
    ``parallelism.use_simple_fsdp``): AC and per-block compile are applied by the model's own
    ``parallelize_fn`` with ``skip_dp=True``, then ``graph_trainer``'s ``apply_simple_fsdp`` turns every
    parameter into a ``Shard(0)`` DTensor behind a parametrization. Under ``torch.compile`` the FSDP
    collectives live inside each compiled TransformerBlock, and under activation checkpointing the
    all-gather is recomputed in backward.
    """
    from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
    from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization

    def _parallelize(model: nn.Module, *, parallel_dims: Any, training: Any, compile_config: Any, **kwargs: Any):
        if spmd_safe_blocks and compile_config.enable and hasattr(model, "layers"):
            from torchtitan.experiments.tpu.spmd_utils import apply_spmd_safe

            for block in model.layers.values():
                apply_spmd_safe(block)

        kwargs.pop("skip_dp", None)
        model = parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=training,
            compile_config=compile_config,
            skip_dp=True,
            **kwargs,
        )
        tied = bool(getattr(model, "enable_weight_tying", False))
        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
        if tied:
            model.tok_embeddings._parameters["weight"] = model.lm_head._parameters["weight"]

        init_weights = model.init_weights

        def _init_weights(*args: Any, **init_kwargs: Any):
            with disable_active_parametrization():
                return init_weights(*args, **init_kwargs)

        model.init_weights = _init_weights  # type: ignore[method-assign]
        logger.info("Applied SimpleFSDP (weight tying preserved: %s, spmd_safe blocks: %s)", tied, spmd_safe_blocks)
        return model

    return _parallelize


@contextmanager
def tpu_torchtitan_config_overrides(*, use_simple_fsdp: bool = False):
    """Force the TorchTitan config values that differ on TPU during ``TorchTitanEngine.__init__``."""
    from torchtitan.components.optimizer import OptimizersContainer

    import verl.workers.engine.torchtitan.transformer_impl as tt_impl

    forced: list[tuple[Any, str, Any]] = [
        (OptimizersContainer, "Config", {"implementation": "foreach"}),
        (tt_impl, "ParallelismConfig", {"spmd_backend": "default"}),
        (tt_impl, "CompileConfig", {"backend": "tpu"}),
        (tt_impl, "TrainingConfig", {"enable_cpu_offload": False}),
    ]

    saved: list[tuple[Any, str, Any]] = []
    try:
        for owner, name, overrides in forced:
            original = getattr(owner, name)
            saved.append((owner, name, original))

            def _make(original=original, overrides=overrides):
                def _construct(*args: Any, **kwargs: Any):
                    return original(*args, **{**kwargs, **overrides})

                return _construct

            setattr(owner, name, _make())

        if use_simple_fsdp and hasattr(tt_impl, "Trainer") and hasattr(tt_impl.Trainer, "Config"):
            orig_trainer_cfg_init = tt_impl.Trainer.Config.__init__
            saved.append((tt_impl.Trainer.Config, "__init__", orig_trainer_cfg_init))

            def _init_trainer_cfg(self_cfg: Any, *args: Any, **kwargs: Any) -> None:
                orig_trainer_cfg_init(self_cfg, *args, **kwargs)
                model_spec = getattr(self_cfg, "model_spec", None)
                if model_spec is not None and hasattr(model_spec, "parallelize_fn"):
                    model_spec.parallelize_fn = make_simple_fsdp_parallelize_fn(
                        model_spec.parallelize_fn, spmd_safe_blocks=True
                    )

            tt_impl.Trainer.Config.__init__ = _init_trainer_cfg  # type: ignore[method-assign]

        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


class TPUVarlenAttention(torch.nn.Module):
    """Packed-document causal attention module for eager TPU execution."""

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: Any = None,
        scale: float | None = None,
        out_transform: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if out_transform is not None:
            # TorchTitan applies out_transform as an epilogue over (out_BLNH, lse_BLN); see
            # torchtitan/models/common/attention.py. Attention-sink models such as gpt-oss rely
            # on it. This module does not produce an LSE, so silently ignoring out_transform
            # would return numerically wrong attention output.
            raise NotImplementedError(
                "TPUVarlenAttention does not support the out_transform epilogue used by "
                "attention-sink models (e.g. gpt-oss). Supported models must not set it."
            )

        xq, xk, xv = q_BLNH, k_BLNH, v_BLNH

        # Probe the value, not just the attribute name: a mask object can declare the attribute
        # and still leave it unset, in which case the packed path is not applicable.
        cu_seqs = None
        for attr in ("cu_seq_q", "cu_seqlens_q", "cu_seqlens"):
            candidate = getattr(attention_masks, attr, None)
            if candidate is not None:
                cu_seqs = candidate
                break

        if cu_seqs is not None:
            total_tokens = xq.shape[1]
            positions = torch.arange(total_tokens, device=xq.device)
            seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
            same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
            causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)
            mask = (same_seq_mask & causal_mask).unsqueeze(0).unsqueeze(0)
        elif isinstance(attention_masks, torch.Tensor):
            if attention_masks.dim() == 4:
                mask = attention_masks.to(torch.bool)
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
                padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                mask = causal_mask & padding_mask
        else:
            seq_len = xq.shape[1]
            positions = torch.arange(seq_len, device=xq.device)
            mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        if q.shape[1] != k.shape[1]:
            num_repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        return attn_out.transpose(1, 2)


class TPUSplashAttention(torch.nn.Module):
    """TorchTitan inner attention backed by the TPU splash attention Pallas kernel.

    Same contract as ``torchtitan.experiments.tpu.kernels.splash_attention.SplashAttention``
    ((B, S, H, D) in and out, causal + packed-document masking from ``segment_ids``), but the
    block sizes are chosen per call from the (static, bucketed) sequence length instead of
    being fixed at 512. Falls back to the original module if no block size fits.
    """

    def __init__(self, original_module: torch.nn.Module, local_window_size: int | None = None):
        super().__init__()
        self.original_module = original_module
        self.local_window_size = local_window_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        attention_masks: Any = None,
        positions: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        block = splash_block_size_for(int(q.shape[1]))
        if block == 0 or q.device.type == "cpu":
            return self.original_module(
                q,
                k,
                v,
                scale=scale,
                enable_gqa=enable_gqa,
                attention_masks=attention_masks,
                positions=positions,
                segment_ids=segment_ids,
                **kwargs,
            )

        from torch.distributed.tensor import DTensor
        from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
        from torchtitan.models.common.attention import segment_ids_from_positions

        q_mesh = q_placements = None
        if isinstance(q, DTensor):
            q_mesh, q_placements = q.device_mesh, q.placements
            q = q.to_local()
        k = k.to_local() if isinstance(k, DTensor) else k
        v = v.to_local() if isinstance(v, DTensor) else v
        positions = positions.to_local() if isinstance(positions, DTensor) else positions
        segment_ids = segment_ids.to_local() if isinstance(segment_ids, DTensor) else segment_ids
        if segment_ids is None and positions is not None:
            segment_ids = segment_ids_from_positions(positions)

        out = splash_sdpa(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            segment_ids=segment_ids,
            scale=scale,
            is_causal=True,
            local_window_size=self.local_window_size,
            enable_gqa=enable_gqa,
            block_q=block,
            block_kv=block,
            block_dkv=block,
            block_kv_compute=block,
            block_q_dkv=block,
            block_kv_dkv=block,
            block_kv_dkv_compute=block,
        ).transpose(1, 2)
        if q_mesh is not None:
            out = DTensor.from_local(out, q_mesh, q_placements)
        return out


def apply_splash_attention_tpu(modules: Any) -> int:
    """Swap every attention module's ``inner_attention`` for ``TPUSplashAttention``.

    The splash kernel takes the per-token ``segment_ids`` that ``Decoder.forward`` derives
    from ``positions`` (which restart at 0 for every packed sequence), so it applies the
    same causal + document-boundary mask as the dense [1, 1, S, S] mask built in
    ``pad_packed_inputs_for_tpu`` without materializing it. Bucket padding tokens have
    position 0, so each one becomes its own segment and only attends to itself; their
    outputs are discarded.

    Must run before the first forward: per-block ``torch.compile`` traces lazily, so the
    swapped module is what gets compiled. Returns the number of replaced modules.
    """
    if modules is None:
        return 0
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    replaced = 0
    already_splash = 0
    saw_nn_module = False
    for model in model_list:
        if not isinstance(model, torch.nn.Module):
            continue
        saw_nn_module = True
        for module in model.modules():
            inner = getattr(module, "inner_attention", None)
            if not isinstance(inner, torch.nn.Module):
                continue
            if isinstance(inner, TPUSplashAttention):
                already_splash += 1
                continue
            window = getattr(inner, "window_size", None)
            local_window_size = None
            if isinstance(window, tuple) and len(window) == 2 and window[0] >= 0:
                local_window_size = int(window[0])
            fallback = TPUVarlenAttention() if type(inner).__name__ == "VarlenAttention" else inner
            module.inner_attention = TPUSplashAttention(fallback, local_window_size=local_window_size)
            replaced += 1
    if saw_nn_module and replaced == 0 and already_splash == 0:
        raise ValueError("use_splash_attention=True but no module with an `inner_attention` was found")
    if replaced:
        logger.warning("Splash attention enabled on TPU: replaced %d inner attention modules", replaced)
    return replaced


def replace_varlen_attention_with_tpu_attention(modules: Any) -> int:
    """Replaces inner_attention submodules on model instances with TPUVarlenAttention."""
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    replaced = 0
    for root in model_list:
        if not isinstance(root, torch.nn.Module):
            continue
        for sub in root.modules():
            if hasattr(sub, "inner_attention") and not isinstance(
                sub.inner_attention, TPUVarlenAttention | TPUSplashAttention
            ):
                sub.inner_attention = TPUVarlenAttention()
                replaced += 1
    if replaced:
        logger.info("Replaced %d inner_attention submodule(s) with TPUVarlenAttention.", replaced)
    return replaced


def pad_packed_inputs_for_tpu(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    micro_batch: TensorDict,
    device: Any,
    build_attention_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Bucket-pad packed 1D sequences and optionally build a static 4D causal document mask on CPU before H2D transfer.

    With ``build_attention_mask=False`` (splash attention, which derives document boundaries
    from ``positions``) the dense mask is skipped and ``None`` is returned in its place.
    """
    bucket_size = get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu().unsqueeze(0)
    if position_ids.dim() == 3:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(1)
    else:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(0)

    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=1)

    orig_seq_len = int(input_ids_cpu.shape[1])
    # tpu_padded_seq_len: common length across ranks from align_micro_batch_shapes_across_ranks (SimpleFSDP).
    padded_seq_len = max(
        bucket_length(orig_seq_len, bucket_size),
        int(tu.get_non_tensor_data(data=micro_batch, key="tpu_padded_seq_len", default=0) or 0),
    )
    pad_len = padded_seq_len - orig_seq_len

    pos_2d_cpu = position_ids_cpu[0] if position_ids_cpu.dim() == 3 else position_ids_cpu
    if pad_len > 0:
        input_ids_cpu = F.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = F.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = F.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = F.pad(pos_2d_cpu, (0, pad_len), value=0)

    attention_mask_cpu = None
    if build_attention_mask:
        if getattr(input_ids, "is_nested", False):
            seq_lens = input_ids.offsets().diff().detach().cpu()
            seq_ids_1d = torch.repeat_interleave(torch.arange(1, len(seq_lens) + 1, dtype=torch.int64), seq_lens)
            if pad_len > 0:
                seq_ids_1d = F.pad(seq_ids_1d, (0, pad_len), value=0)
            seq_ids = seq_ids_1d.unsqueeze(0)
        else:
            first_dummy = pos_2d_cpu[:, :1] - 1
            boundary = torch.diff(pos_2d_cpu, prepend=first_dummy, dim=-1) != 1
            boundary[:, 0] = True
            seq_ids = boundary.cumsum(dim=-1)

        idx = torch.arange(padded_seq_len, dtype=seq_ids.dtype).unsqueeze(0)
        valid = idx < orig_seq_len
        seq_ids = torch.where(valid, seq_ids, -idx - 1)
        same_seq_mask = seq_ids.unsqueeze(2) == seq_ids.unsqueeze(1)
        causal_mask = idx.unsqueeze(2) >= idx.unsqueeze(1)
        attention_mask_cpu = (same_seq_mask & causal_mask).unsqueeze(1)

    if "responses" in micro_batch.keys() and getattr(micro_batch["responses"], "is_nested", False):
        resp_lens = micro_batch["responses"].offsets().diff().cpu()
        raw_max_resp = int(resp_lens.max().item())
        max_resp = max(
            bucket_length(raw_max_resp, bucket_size),
            int(tu.get_non_tensor_data(data=micro_batch, key="tpu_max_response_len", default=0) or 0),
        )
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_resp)

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous() if attention_mask_cpu is not None else None,
        orig_seq_len,
    )
