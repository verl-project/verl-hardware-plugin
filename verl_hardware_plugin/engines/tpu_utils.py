# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU helper utilities for the TorchTitan training engine.

Provides sequence-length bucketing, host-side metadata unwrapping, TorchTitan
config overrides, and packed-input padding for ``torch_tpu`` training.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to attach the full bucket-padded device tensor to a CPU
# nested tensor returned from ``prepare_model_outputs``, so loss functions can
# consume the fixed-shape device tensor directly without XLA recompilation.
TPU_PADDED_VALUES_ATTR = "_tpu_padded_values"

_DEFAULT_TPU_SEQ_BUCKET_SIZE = 256


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


def compute_global_batch_num_tokens(data: TensorDict, dp_group: Any, tp_size: int) -> Any:
    """Compute the global batch token count for loss normalization on TPU via CPU all-reduce."""
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


@contextmanager
def tpu_torchtitan_config_overrides():
    """Force the four TorchTitan config values that differ on TPU during ``TorchTitanEngine.__init__``."""
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
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


class TPUVarlenAttention(torch.nn.Module):
    """Packed-document causal attention module for TPU execution."""

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

        # TODO(tpu): Replace F.scaled_dot_product_attention with a standalone Pallas SplashAttention
        # kernel (via torch.tpu.pallas.jax_op) to avoid O(L^2) attention mask materialization.
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        return attn_out.transpose(1, 2)


def replace_varlen_attention_with_tpu_attention(modules: Any) -> int:
    """Replaces inner_attention submodules on model instances with TPUVarlenAttention."""
    model_list = modules if isinstance(modules, list | tuple | torch.nn.ModuleList) else [modules]
    replaced = 0
    for root in model_list:
        if not isinstance(root, torch.nn.Module):
            continue
        for sub in root.modules():
            if hasattr(sub, "inner_attention") and not isinstance(sub.inner_attention, TPUVarlenAttention):
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Bucket-pad packed 1D sequences and build a static 4D causal document mask on CPU before H2D transfer."""
    bucket_size = get_tpu_seq_bucket_size()
    input_ids_cpu = input_ids.values().detach().cpu().unsqueeze(0)
    if position_ids.dim() == 3:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(1)
    else:
        position_ids_cpu = position_ids.values().detach().cpu().unsqueeze(0)

    labels_cpu = torch.roll(input_ids_cpu, shifts=-1, dims=1)

    orig_seq_len = int(input_ids_cpu.shape[1])
    padded_seq_len = bucket_length(orig_seq_len, bucket_size)
    pad_len = padded_seq_len - orig_seq_len

    pos_2d_cpu = position_ids_cpu[0] if position_ids_cpu.dim() == 3 else position_ids_cpu
    if pad_len > 0:
        input_ids_cpu = torch.nn.functional.pad(input_ids_cpu, (0, pad_len), value=0)
        labels_cpu = torch.nn.functional.pad(labels_cpu, (0, pad_len), value=0)
        position_ids_cpu = torch.nn.functional.pad(position_ids_cpu, (0, pad_len), value=0)
        pos_2d_cpu = torch.nn.functional.pad(pos_2d_cpu, (0, pad_len), value=0)

    if getattr(input_ids, "is_nested", False):
        seq_lens = input_ids.offsets().diff().detach().cpu()
        seq_ids_1d = torch.repeat_interleave(torch.arange(1, len(seq_lens) + 1, dtype=torch.int64), seq_lens)
        if pad_len > 0:
            seq_ids_1d = torch.nn.functional.pad(seq_ids_1d, (0, pad_len), value=0)
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
        tu.assign_non_tensor_data(micro_batch, "max_response_len", bucket_length(raw_max_resp, bucket_size))

    return (
        input_ids_cpu.to(device=device).contiguous(),
        position_ids_cpu.to(device=device).contiguous(),
        labels_cpu.to(device=device).contiguous(),
        attention_mask_cpu.to(device=device).contiguous(),
        orig_seq_len,
    )
