# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Adapts the TPU engine's model outputs to verl's unmodified SFT and PPO/GRPO loss functions.

The TorchTitan TPU engine returns ``log_probs`` / ``entropy`` as CPU nested tensors and attaches the full
bucket-padded device tensor to each (``TPU_PADDED_VALUES_ATTR``). verl's losses (``sft_loss``, ``ppo_loss`` ->
``no_padding_2_padding``) know nothing about TPU buckets, so ``prepare_tpu_loss_inputs`` rewrites the loss inputs
into static, bucket-only shapes on device before the engine calls the loss. It is the only TPU code that knows the
RL loss fields (``old_log_probs``, ``advantages``, ...); the engine itself stays algorithm-agnostic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import TPU_PADDED_VALUES_ATTR, unwrap_metadata


class _TPUPaddedJaggedProxy:
    """Lightweight proxy exposing a 1D bucket-padded TPU tensor via ``.values()``."""

    def __init__(self, padded_values: torch.Tensor, base_nested: Any):
        self._padded_values = padded_values
        self._base_nested = base_nested
        self.is_nested = True

    def values(self) -> torch.Tensor:
        return self._padded_values

    def offsets(self) -> torch.Tensor:
        return self._base_nested.offsets()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_nested, name)


class _TPULossDataProxy:
    """Read-only view over a micro-batch ``TensorDict`` that overrides specific keys for loss calculation."""

    def __init__(self, base: TensorDict, overrides: dict[str, Any]):
        self._base = base
        self._overrides = overrides

    def __getitem__(self, key: str) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._overrides or key in self._base

    def keys(self):
        return self._base.keys()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _copy_tensordict_without_device(td: TensorDict) -> TensorDict:
    """Shallow-copy ``td`` (entries shared, host metadata kept) into a TensorDict with no device.

    ``TensorDict(td, batch_size=td.batch_size)`` cannot be used for this: on tensordict >= 0.11 it re-validates
    every ``NonTensorData`` entry (``dp_size``, ``batch_num_tokens``, ...) as a 0-d value and raises
    ``batch dimension mismatch``. Without a device, TPU tensors can be stored next to the CPU metadata.
    """
    out = td.copy()
    if out.device is not None:
        out.clear_device_()
    return out


def _response_gather_index(
    micro_batch: TensorDict, response_width: int, padded_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build, on CPU, the index/mask that gather every response's per-token outputs from a packed sequence.

    Mirrors verl's ``no_padding_2_padding`` convention: response token ``r`` of sequence ``i`` is predicted at packed
    position ``seq_end_i - resp_len_i - 1 + r``. Returns ``index`` (int64) and ``valid`` (bool), both shaped
    ``[batch_size, response_width]``; padded slots point at position 0 and are masked out.
    """
    prompt_lens = micro_batch["prompts"].offsets().diff().cpu()
    response_lens = micro_batch["responses"].offsets().diff().cpu()
    starts = ((prompt_lens + response_lens).cumsum(dim=0) - response_lens - 1).to(torch.int64)
    cols = torch.arange(response_width, dtype=torch.int64).unsqueeze(0)
    valid = cols < response_lens.unsqueeze(1)
    index = torch.where(valid, starts.unsqueeze(1) + cols, torch.zeros_like(cols)).clamp(0, max(0, padded_len - 1))
    return index, valid


def _gather_responses_static_layout(
    padded_values: torch.Tensor, index: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Gather responses out of the bucket-padded packed tensor into a static ``[bsz * (width + 1)]`` layout.

    Row ``i`` of the ``[bsz, width]`` gather is laid out as one segment of ``width + 1`` slots (responses first,
    one trailing pad slot), so a sequence with prompt length 1 and response length ``width`` selects exactly that
    row in ``no_padding_2_padding``. Every shape depends only on the bucketed lengths, never on the token counts.
    """
    device = padded_values.device
    rows = torch.gather(padded_values.unsqueeze(0).expand(index.shape[0], -1), 1, index.to(device=device))
    rows = rows * valid.to(device=device, dtype=padded_values.dtype)
    return F.pad(rows, (0, 1)).reshape(-1)


def prepare_tpu_loss_inputs(model_output: dict[str, Any], micro_batch: TensorDict) -> tuple[dict[str, Any], Any]:
    """Adapt differentiable TPU outputs for the unmodified SFT and PPO loss APIs.

    SFT consumes full bucket-padded buffers and a matching loss mask. PPO receives a real TensorDict with padded
    response fields on device. Its model outputs are gathered (static gather, CPU-built indices) from the full
    bucket-padded device tensors into a static layout, so ``ppo_loss`` -> ``no_padding_2_padding`` only ever sees
    shapes that depend on the sequence-length bucket, the micro-batch size and the response-length bucket. Trimming
    the outputs to the exact token count instead would make every micro-batch with new lengths compile new XLA
    programs (forward loss and the fused LM-head backward) for the whole run.
    """
    log_prob = model_output.get("log_probs")
    tpu_padded_log_prob = getattr(log_prob, TPU_PADDED_VALUES_ATTR, None)
    if tpu_padded_log_prob is None:
        return model_output, micro_batch

    if "old_log_probs" in micro_batch and "advantages" in micro_batch:
        # Preserve host sequence metadata while moving response fields independently
        # of the source TensorDict's device setting.
        loss_data = _copy_tensordict_without_device(micro_batch)
        responses = micro_batch["responses"]
        if micro_batch["prompts"].is_nested:
            # The bucketed width from pad_packed_inputs_for_tpu, the common width across ranks, or the real one.
            response_width = max(
                int(unwrap_metadata(tu.get_non_tensor_data(micro_batch, "max_response_len", 0)) or 0),
                int(unwrap_metadata(tu.get_non_tensor_data(micro_batch, "tpu_max_response_len", 0)) or 0),
                int(responses.offsets().diff().max().item()),
            )
        else:
            response_width = responses.shape[1]

        fields = ["response_mask", "old_log_probs", "advantages"]
        fields.extend(key for key in ("ref_log_prob", "rollout_is_weights") if key in micro_batch)
        padded_fields = micro_batch.select(*fields).to_padded_tensor()
        if micro_batch["prompts"].is_nested:
            # Legacy batches can retain response fields wider than the packed response bucket.
            response_width = max(response_width, *(tensor.shape[1] for tensor in padded_fields.values()))
            tu.assign_non_tensor_data(loss_data, "max_response_len", response_width)
        for key, tensor in padded_fields.items():
            if tensor.shape[1] > response_width:
                raise ValueError(f"{key} exceeds max_response_len ({tensor.shape[1]} > {response_width})")
            loss_data[key] = F.pad(tensor, (0, response_width - tensor.shape[1])).to(tpu_padded_log_prob.device)

        loss_model_output = dict(model_output)
        padded_keys = [
            key for key, value in model_output.items() if getattr(value, TPU_PADDED_VALUES_ATTR, None) is not None
        ]
        static_layout = (
            micro_batch["prompts"].is_nested
            and getattr(responses, "is_nested", False)
            and all(getattr(model_output[key], TPU_PADDED_VALUES_ATTR).dim() == 1 for key in padded_keys)
            # every packed output the loss slices must come with its bucket-padded device tensor
            and not any(
                getattr(value, "is_nested", False) and key not in padded_keys for key, value in model_output.items()
            )
        )
        if static_layout:
            batch_size = int(micro_batch.batch_size[0])
            segment = response_width + 1
            index_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            layout = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size * segment), torch.arange(batch_size + 1, dtype=torch.int64) * segment
            )
            for key in padded_keys:
                padded_values = getattr(model_output[key], TPU_PADDED_VALUES_ATTR)
                padded_len = int(padded_values.shape[0])
                if padded_len not in index_cache:
                    index_cache[padded_len] = _response_gather_index(micro_batch, response_width, padded_len)
                index, valid = index_cache[padded_len]
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    _gather_responses_static_layout(padded_values, index, valid), layout
                )
            # Matching sequence metadata for the static layout: prompt length 1 and response length
            # `response_width` for every row, so the per-row slices of `no_padding_2_padding` are static too.
            offsets = torch.arange(batch_size + 1, dtype=torch.int64)
            loss_data["prompts"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size, dtype=torch.int64), offsets
            )
            loss_data["responses"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(batch_size * response_width, dtype=torch.int64), offsets * response_width
            )
        else:
            for key in padded_keys:
                tensor = model_output[key]
                # Trim trailing bucket padding inside the plugin without detaching gradients.
                valid_count = int(tensor.offsets()[-1].item())
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    getattr(tensor, TPU_PADDED_VALUES_ATTR)[:valid_count], tensor
                )

        return loss_model_output, loss_data

    if "loss_mask" not in micro_batch.keys() or "responses" in micro_batch.keys():
        return model_output, micro_batch

    loss_mask = micro_batch["loss_mask"]
    loss_mask_vals = loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask
    rolled_mask = torch.roll(loss_mask_vals.detach().cpu(), shifts=-1, dims=0)
    target_len = int(tpu_padded_log_prob.shape[0])
    if rolled_mask.shape[0] < target_len:
        rolled_mask = F.pad(rolled_mask, (0, target_len - int(rolled_mask.shape[0])), value=0)
    prerolled_mask = torch.roll(rolled_mask, shifts=1, dims=0).to(device=tpu_padded_log_prob.device).contiguous()

    loss_model_output = dict(model_output)
    loss_model_output["log_probs"] = _TPUPaddedJaggedProxy(tpu_padded_log_prob, log_prob)
    loss_data = _TPULossDataProxy(
        micro_batch,
        {"loss_mask": _TPUPaddedJaggedProxy(prerolled_mask, loss_mask)},
    )
    return loss_model_output, loss_data
