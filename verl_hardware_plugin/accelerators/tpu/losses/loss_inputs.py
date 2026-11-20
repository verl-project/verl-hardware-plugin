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

import logging
import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss
from verl_hardware_plugin.accelerators.tpu.device_api import accelerator_is_tpu
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    can_pad_nested_tensor_rows,
    can_pad_tensordict_rows,
    first_unpaddable_tensordict_entry,
    unwrap_metadata,
)

__all__ = [
    "DEVICE_LOSS_SCALAR_KEYS",
    "ROW_PADDING_INVARIANT_POLICY_LOSSES",
    "ROW_PADDING_VARIANT_POLICY_LOSSES",
    "can_pad_tensordict_rows",
    "device_loss_scalars",
    "loss_rows_can_be_padded",
    "pad_tensordict_rows",
    "ppo_loss_config",
    "prepare_tpu_loss_inputs",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WARNED_UNPADDABLE: set[tuple[str, str]] = set()

# verl policy losses (``actor.policy_loss.loss_mode``) whose loss, gradients and metrics ignore fully masked rows: they
# reduce over rows only through ``agg_loss`` (which drops rows without valid tokens) and masked means, so rows padded
# up to ``tpu_padded_num_seqs`` change nothing. ``test_tpu_loss_inputs.py`` checks this for every mode listed here
# and every ``loss_agg_mode``, and that every mode verl registers is listed here or in
# ``ROW_PADDING_VARIANT_POLICY_LOSSES``.
ROW_PADDING_INVARIANT_POLICY_LOSSES = frozenset(
    {"vanilla", "gspo", "sapo", "gpg", "clip_cov", "kl_cov", "dro", "cispo", "dppo_tv", "dppo_kl"}
)
# Policy losses whose results count every row: ``geo_mean`` (``torch.mean`` of the per-sequence losses, so pad rows
# rescale the loss and its gradients) and ``bypass_mode`` (its ``rollout_corr/*`` metrics, e.g. ``training_ppl``,
# average over every row). They, and losses not listed above (e.g. custom ``register_policy_loss`` losses), keep the
# exact number of rows.
ROW_PADDING_VARIANT_POLICY_LOSSES = frozenset({"geo_mean", "bypass_mode"})
# ``agg_loss`` modes: masked sums, with the per-sequence modes dropping rows without valid tokens (``seq_mask``).
_ROW_PADDING_INVARIANT_AGG_MODES = frozenset(
    {"token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"}
)

# Global-batch loss metadata that changes every mini-batch. As a Python number, XLA bakes it into the loss programs as
# a constant and compiles them again for every new value. ``global_batch_size`` (the mini-batch size) and ``dp_size``
# are fixed for a run, so they stay host numbers.
DEVICE_LOSS_SCALAR_KEYS = ("batch_num_tokens",)


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

    def __len__(self) -> int:
        return len(self._base)

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str) and key in self._overrides:
            return self._overrides[key]
        return self._base[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._overrides:
            return self._overrides[key]
        return self._base.get(key, default)

    def __contains__(self, key: Any) -> bool:
        return (isinstance(key, str) and key in self._overrides) or key in self._base

    def keys(self, *args: Any, **kwargs: Any):
        base_keys = self._base.keys(*args, **kwargs)
        extra = [key for key in self._overrides if key not in self._base]
        if not extra:
            return base_keys
        return dict.fromkeys([*base_keys, *extra]).keys()

    def select(self, *keys: Any, **kwargs: Any) -> _TPULossDataProxy:
        if kwargs.get("inplace"):
            raise ValueError("_TPULossDataProxy is read-only; select(inplace=True) is not supported")
        base_keys = [key for key in keys if key in self._base or key not in self._overrides]
        selected_base = self._base.select(*base_keys, **kwargs)
        selected_overrides = {key: self._overrides[key] for key in keys if key in self._overrides}
        return _TPULossDataProxy(selected_base, selected_overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _with_overrides(model_output: dict[str, Any], data: Any, overrides: Optional[Mapping[str, Any]]):
    """``(model_output, data)``, with ``data`` wrapped in a ``_TPULossDataProxy`` when there are ``overrides``."""
    if not overrides:
        return model_output, data
    return model_output, _TPULossDataProxy(data, dict(overrides))


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


def _pad_rows(value: Any, extra: int) -> Any:
    """One per-row TensorDict entry (rows along dim 0) with ``extra`` empty / zero rows appended."""
    if isinstance(value, NonTensorStack):
        return [*value.tolist(), *([None] * extra)]
    if isinstance(value, TensorDict):
        if len(value.batch_size) == 0:
            raise TypeError("cannot pad the rows of a 0-D TensorDict")
        return pad_tensordict_rows(value, int(value.batch_size[0]) + extra)
    if isinstance(value, torch.Tensor) and value.is_nested:
        if not can_pad_nested_tensor_rows(value):
            raise TypeError("cannot pad the rows of a strided or non-contiguous nested tensor")
        offsets = value.offsets()
        return torch.nested.nested_tensor_from_jagged(
            value.values(),
            torch.cat([offsets, offsets[-1:].expand(extra)]),
            jagged_dim=value._ragged_idx,
        )
    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise TypeError("cannot pad the rows of a 0-D tensor")
        return torch.cat([value, value.new_zeros((extra, *value.shape[1:]))])
    raise TypeError(f"cannot pad the rows of a {type(value).__name__}")


def pad_tensordict_rows(td: TensorDict, num_rows: int) -> TensorDict:
    """Shallow copy of ``td`` (no device) with every per-row entry padded to ``num_rows`` rows.

    Dense tensors get zero rows, jagged nested tensors empty rows (including 3-D mRoPE ``position_ids`` with
    ``jagged_dim=2``), per-row non-tensor stacks ``None`` rows, and nested TensorDicts are padded recursively; scalar
    host metadata (``NonTensorData``) is kept as is. Every key of ``td`` survives, so a loss can read any field of the
    micro-batch whether or not its rows were padded.

    Raises:
        TypeError: for an entry whose rows it cannot pad.
    """
    batch_size = int(td.batch_size[0])
    if num_rows <= batch_size:
        return _copy_tensordict_without_device(td)
    extra = num_rows - batch_size
    out = TensorDict({}, batch_size=[num_rows, *td.batch_size[1:]])
    for key, value in td.items():
        if isinstance(value, NonTensorData):
            tu.assign_non_tensor_data(out, key, value.data)
        elif isinstance(value, NonTensorStack):
            tu.assign_non_tensor_stack(out, key, _pad_rows(value, extra))
        else:
            out[key] = _pad_rows(value, extra)
    return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping, dataclass/object, or ``None`` without raising ``AttributeError``."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def ppo_loss_config(loss_function: Any) -> Any:
    """The actor config of a ``partial(verl ppo_loss, config=...)`` loss function, else ``None``."""
    if not isinstance(loss_function, partial) or loss_function.func is not ppo_loss:
        return None
    keywords = loss_function.keywords or {}
    config = keywords.get("config")
    if config is None and loss_function.args:
        config = loss_function.args[0]
    return config


def loss_rows_can_be_padded(loss_function: Any) -> bool:
    """Whether padding the PPO loss inputs with fully masked rows leaves ``loss_function``'s results unchanged.

    True only for verl's ``ppo_loss`` with a policy loss in ``ROW_PADDING_INVARIANT_POLICY_LOSSES`` and an
    ``agg_loss`` mode. Any other loss (``geo_mean``, ``bypass_mode``, a custom ``register_policy_loss`` loss, a non-PPO
    loss) keeps the exact number of rows.
    """
    config = ppo_loss_config(loss_function)
    if config is None or _cfg_get(config, "loss_agg_mode") not in _ROW_PADDING_INVARIANT_AGG_MODES:
        return False
    policy_loss = _cfg_get(config, "policy_loss")
    return _cfg_get(policy_loss, "loss_mode", "vanilla") in ROW_PADDING_INVARIANT_POLICY_LOSSES


def device_loss_scalars(data: TensorDict, device: Any) -> dict[str, torch.Tensor]:
    """0-D float32 copies on ``device`` (or ``device()``) of the mini-batch's ``DEVICE_LOSS_SCALAR_KEYS``; empty when
    not on TPU.

    The engine calls this once per mini-batch, before the micro-batch loop and outside the deferred eager-mode region
    (with ``non_blocking=True`` so ``torch_tpu`` schedules an async H2D transfer instead of a blocking stream cut), and
    passes the result to :func:`prepare_tpu_loss_inputs` (which overrides the loss data's entries with it, for the PPO,
    SFT and critic losses) and to the ``graph_trainer`` step. verl's losses then divide by a float32 0-D tensor instead
    of a Python float: a bf16 masked sum is promoted to float32 instead of staying bf16; float32 sums are unchanged.

    When ``batch_num_tokens == 0`` globally (every DP rank has ``loss_mask == 0``, so every masked-sum numerator across
    the entire global batch is ``0.0``), ``batch_num_tokens`` is clamped to ``1.0`` (matching ``GraphPolicyLoss``'s
    ``clamp(min=1.0)``) so ``ppo_loss`` / ``sft_loss`` / ``value_loss`` produce finite ``0.0`` instead of ``0.0 / 0.0 =
    NaN``.
    """
    if not accelerator_is_tpu():
        return {}
    if callable(device):  # e.g. verl's get_device_id, resolved only on TPU
        device = device()
    out: dict[str, torch.Tensor] = {}
    for key in DEVICE_LOSS_SCALAR_KEYS:
        value = unwrap_metadata(tu.get_non_tensor_data(data, key, None))
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.device.type == "cpu":
                scalar = value.detach().to(dtype=torch.float32).reshape(())
                if key == "batch_num_tokens":
                    scalar = torch.clamp(scalar, min=1.0)
                out[key] = scalar.to(device=device, non_blocking=True)
            else:
                scalar = value.detach().to(device=device, dtype=torch.float32, non_blocking=True).reshape(())
                if key == "batch_num_tokens":
                    scalar = torch.clamp(scalar, min=1.0)
                out[key] = scalar
        else:
            scalar_val = float(value)
            if key == "batch_num_tokens":
                scalar_val = max(scalar_val, 1.0)
            out[key] = torch.tensor(scalar_val, dtype=torch.float32).to(device=device, non_blocking=True)
    return out


def _loss_num_rows(micro_batch: TensorDict) -> int:
    """Rows of the PPO loss inputs: the micro-batch's ``batch_size``, or ``tpu_padded_num_seqs`` when larger.

    The engine sets ``tpu_padded_num_seqs`` (``align_micro_batch_shapes_across_ranks``) only for losses that ignore
    fully masked rows (:func:`loss_rows_can_be_padded`) and only with ``engine_kwargs.tpu.loss_row_bucketing``.
    """
    batch_size = int(micro_batch.batch_size[0])
    num_rows = int(unwrap_metadata(tu.get_non_tensor_data(micro_batch, "tpu_padded_num_seqs", 0)) or 0)
    return max(batch_size, num_rows)


def _response_gather_index(
    micro_batch: TensorDict, response_width: int, padded_len: int, num_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build, on CPU, the index/mask that gather every response's per-token outputs from a packed sequence.

    Mirrors verl's ``no_padding_2_padding`` convention: response token ``r`` of sequence ``i`` is predicted at packed
    position ``seq_end_i - resp_len_i - 1 + r``. Returns ``index`` (int64) and ``valid`` (bool), both shaped
    ``[num_rows, response_width]``; padded slots, and the rows past the micro-batch's sequences, point at position 0
    and are masked out.
    """
    prompt_lens = micro_batch["prompts"].offsets().diff().cpu()
    response_lens = micro_batch["responses"].offsets().diff().cpu()
    starts = ((prompt_lens + response_lens).cumsum(dim=0) - response_lens - 1).to(torch.int64)
    if num_rows > response_lens.shape[0]:
        extra = num_rows - response_lens.shape[0]
        starts, response_lens = F.pad(starts, (0, extra)), F.pad(response_lens, (0, extra))
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
    The ``expand`` is a view (no ``[rows, padded_len]`` copy); the gathered output is ``[rows, width]``.
    """
    device = padded_values.device
    rows = torch.gather(padded_values.unsqueeze(0).expand(index.shape[0], -1), 1, index.to(device=device))
    rows = rows * valid.to(device=device, dtype=padded_values.dtype)
    return F.pad(rows, (0, 1)).reshape(-1)


def prepare_tpu_loss_inputs(
    model_output: dict[str, Any],
    micro_batch: TensorDict,
    loss_scalars: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[dict[str, Any], Any]:
    """Adapt differentiable TPU outputs for the unmodified SFT, PPO and critic loss APIs.

    SFT consumes full bucket-padded buffers and a matching loss mask. PPO receives a real TensorDict with padded
    response fields on device. Its model outputs are gathered (static gather, CPU-built indices) from the full
    bucket-padded device tensors into a static layout, so ``ppo_loss`` -> ``no_padding_2_padding`` only ever sees
    shapes that depend on the sequence-length bucket, the row count and the response-length bucket. Trimming the
    outputs to the exact token count instead would make every micro-batch with new lengths compile new XLA programs
    (forward loss and the fused LM-head backward) for the whole run. With ``tpu_padded_num_seqs`` (see
    :func:`_loss_num_rows`) every per-row field is padded with fully masked rows up to that row count.

    ``loss_scalars`` (:func:`device_loss_scalars`) override the matching entries of the loss data on every path.
    """
    log_prob = model_output.get("log_probs")
    tpu_padded_log_prob = getattr(log_prob, TPU_PADDED_VALUES_ATTR, None)
    if tpu_padded_log_prob is None:
        return _with_overrides(model_output, micro_batch, loss_scalars)

    if "old_log_probs" in micro_batch and "advantages" in micro_batch:
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
        batch_size = int(micro_batch.batch_size[0])
        num_rows = _loss_num_rows(micro_batch) if static_layout else batch_size
        loss_data = None
        if num_rows > batch_size:
            try:
                loss_data = pad_tensordict_rows(micro_batch, num_rows)
            except TypeError as e:
                # Defense-in-depth fallback when tpu_padded_num_seqs was set directly without align_micro_batch_*:
                # keep the exact row count and rate-limit warnings once per (field, type).
                warn_key = first_unpaddable_tensordict_entry(micro_batch) or ("<unknown>", type(e).__name__)
                if warn_key not in _WARNED_UNPADDABLE:
                    _WARNED_UNPADDABLE.add(warn_key)
                    logger.warning("Not padding the PPO loss rows to %d (field %r): %s", num_rows, warn_key[0], e)
                num_rows = batch_size
        if loss_data is None:
            loss_data = _copy_tensordict_without_device(micro_batch)
        if micro_batch["prompts"].is_nested:
            tu.assign_non_tensor_data(loss_data, "max_response_len", response_width)
        for key, tensor in padded_fields.items():
            if tensor.shape[1] > response_width:
                raise ValueError(f"{key} exceeds max_response_len ({tensor.shape[1]} > {response_width})")
            tensor = F.pad(tensor, (0, response_width - tensor.shape[1], 0, num_rows - batch_size))
            loss_data[key] = tensor.to(tpu_padded_log_prob.device)

        if static_layout:
            segment = response_width + 1
            index_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            layout = torch.nested.nested_tensor_from_jagged(
                torch.zeros(num_rows * segment), torch.arange(num_rows + 1, dtype=torch.int64) * segment
            )
            for key in padded_keys:
                padded_values = getattr(model_output[key], TPU_PADDED_VALUES_ATTR)
                padded_len = int(padded_values.shape[0])
                if padded_len not in index_cache:
                    index_cache[padded_len] = _response_gather_index(micro_batch, response_width, padded_len, num_rows)
                index, valid = index_cache[padded_len]
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    _gather_responses_static_layout(padded_values, index, valid), layout
                )
            # Matching sequence metadata for the static layout: prompt length 1 and response length
            # `response_width` for every row, so the per-row slices of `no_padding_2_padding` are static too.
            offsets = torch.arange(num_rows + 1, dtype=torch.int64)
            loss_data["prompts"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(num_rows, dtype=torch.int64), offsets
            )
            loss_data["responses"] = torch.nested.nested_tensor_from_jagged(
                torch.zeros(num_rows * response_width, dtype=torch.int64), offsets * response_width
            )
        else:
            for key in padded_keys:
                tensor = model_output[key]
                # Trim trailing bucket padding inside the plugin without detaching gradients.
                valid_count = int(tensor.offsets()[-1].item())
                loss_model_output[key] = _TPUPaddedJaggedProxy(
                    getattr(tensor, TPU_PADDED_VALUES_ATTR)[:valid_count], tensor
                )
        return _with_overrides(loss_model_output, loss_data, loss_scalars)

    if "loss_mask" not in micro_batch.keys() or "responses" in micro_batch.keys():
        return _with_overrides(model_output, micro_batch, loss_scalars)

    loss_mask = micro_batch["loss_mask"]
    loss_mask_vals = loss_mask.values() if getattr(loss_mask, "is_nested", False) else loss_mask
    rolled_mask = torch.roll(loss_mask_vals.detach().cpu(), shifts=-1, dims=0)
    target_len = int(tpu_padded_log_prob.shape[0])
    if rolled_mask.shape[0] < target_len:
        rolled_mask = F.pad(rolled_mask, (0, target_len - int(rolled_mask.shape[0])), value=0)
    prerolled_mask = torch.roll(rolled_mask, shifts=1, dims=0).to(device=tpu_padded_log_prob.device).contiguous()

    loss_model_output = dict(model_output)
    loss_model_output["log_probs"] = _TPUPaddedJaggedProxy(tpu_padded_log_prob, log_prob)
    overrides: dict[str, Any] = dict(loss_scalars or {})
    overrides["loss_mask"] = _TPUPaddedJaggedProxy(prerolled_mask, loss_mask)
    return loss_model_output, _TPULossDataProxy(micro_batch, overrides)
