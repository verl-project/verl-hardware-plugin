# Copyright (c) 2026 Google LLC. All rights reserved.
# Diagnostics adapted from verl: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0.

"""Standalone TorchTitan training engine for Google TPU (``torch_tpu``) devices.

Registers ``TorchTitanTPUEngineWithLMHead`` with verl's ``EngineRegistry`` under
``(model_type="language_model", backend="torchtitan", device="tpu", vendor="google")``.

This module implements the TPU TorchTitan engine directly on top of ``verl.workers.engine.base.BaseEngine``
without importing ``verl.workers.engine.torchtitan``, keeping the TPU TorchTitan backend completely
decoupled from core verl's GPU TorchTitan engine:
- ``spmd_backend="default"`` and ``CompileConfig(backend="tpu")`` with static-shape Dynamo settings
- torchtitan's TPU ``SplashAttention`` Pallas kernel for packed-document causal attention without dense 4D masks
- ``SimpleFSDP`` DTensor parameter sharding with ``spmd_safe`` compiled TransformerBlocks and cross-rank shape alignment
- ``DEFER_AND_FUSE`` eager execution mode for ops outside compiled blocks
- ``OptimizersContainer.Config(implementation="fused")`` (``engine_kwargs.tpu.optim_impl`` overrides) and
  ``foreach=False``
  grad norm clipping
- Host-side global token counting (avoiding eager device collectives inside the XLA SPMD trace)
- Graph-break loss synchronization between forward and backward
- Sequence-length bucketing for both packed (``use_remove_padding=True``) and padded inputs
- Plugin-local adapters preserving gradients for bucket-padded SFT and RL loss inputs
"""

from __future__ import annotations

import dataclasses
import functools
import gc
import importlib
import inspect
import io
import json
import logging
import math
import os
import re
import shutil
import sys
from collections.abc import Collection, Iterable, Mapping
from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional

import torch
import torch.distributed
from safetensors.torch import save_file
from tensordict import TensorDict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.experimental._context_parallel._load_balancer import _HeadTailLoadBalancer

from verl.trainer.config import CheckpointConfig
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from verl.utils.metric import AggregationType, Metric
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import (
    detach_tree,
    enable_full_determinism,
    postprocess_batch_func,
    prepare_micro_batches,
)
from verl.workers.utils.padding import no_padding_2_padding
from verl_hardware_plugin.accelerators.tpu.engines import (
    gdn_slot_schedule,
    graph_trainer_mode,
    tpu_model_kernels,
    tpu_ragged_ep,
)
from verl_hardware_plugin.accelerators.tpu.engines.cp_layer_transforms import (
    context_parallel_layer_transforms,
    module_available,
)
from verl_hardware_plugin.accelerators.tpu.engines.fused_logprob import fused_linear_logprobs_tpu
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    ContextParallelGather,
    align_micro_batch_num_seqs,
    align_micro_batch_shapes_across_ranks,
    apply_fp32_logits_tpu,
    apply_tpu_precision_patches,
    bucket_length,
    chunked_entropy_from_hidden_tpu,
    compute_global_batch_num_tokens,
    compute_tpu_logprobs_chunked,
    configure_torch_compile_for_tpu,
    create_noop_dataloader_config,
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    get_text_config,
    get_torchtitan_model_module_name,
    get_tpu_seq_bucket_size,
    get_tpu_simple_fsdp_parallelize_fn,
    install_sync_dtensor_redistribute_for_tpu,
    install_tpu_attention,
    move_tpu_params_,
    pad_packed_inputs_for_tpu,
    resolve_tpu_mixed_precision_reduce,
    resolve_tpu_splash_max_block_size,
    resolve_tpu_torchtitan_options,
    resolve_tpu_use_fused_linear_cross_entropy,
    row_slot_permutation,
    set_splash_max_block_size,
    spmd_no_typecheck_context,
    torchtitan_model_attn_backend,
    tpu_eager_mode_context,
    tpu_synchronize,
    unwrap_metadata,
)
from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights
from verl_hardware_plugin.accelerators.tpu.losses.loss_inputs import (
    _cfg_get,
    device_loss_scalars,
    loss_rows_can_be_padded,
    ppo_loss_config,
    prepare_tpu_loss_inputs,
)
from verl_hardware_plugin.accelerators.tpu.options import get_tpu_engine_options
from verl_hardware_plugin.accelerators.tpu.profilers import timeline

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _headtail_restore_indices_cpu(seq_len: int, cp_size: int) -> torch.Tensor:
    """The restore permutation of torch's ``_HeadTailLoadBalancer``, built on CPU (its ``argsort`` over a 64k-token
    sequence is slow on TPU). The engine keeps one device copy per shape (``_cp_restore_indices``)."""
    return _HeadTailLoadBalancer(seq_len, cp_size, "cpu")._generate_indices(restore=True)[0].to(torch.int64)


def _to_jagged(padded: torch.Tensor, cu_seqlens: torch.Tensor, orig_seq_len: Optional[int]) -> torch.Tensor:
    """Jagged view of a per-token output; for bucket-padded outputs a CPU view that carries the padded device tensor.

    The loss reads the fixed-shape device tensor (``TPU_PADDED_VALUES_ATTR``), so no new XLA shape per micro-batch.
    """
    if orig_seq_len is None:
        return torch.nested.nested_tensor_from_jagged(padded, cu_seqlens)
    jagged = torch.nested.nested_tensor_from_jagged(padded.detach().cpu()[:orig_seq_len], cu_seqlens.detach().cpu())
    setattr(jagged, TPU_PADDED_VALUES_ATTR, padded)
    return jagged


_LAYER_FQN_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# FQN prefix of one transformer layer (e.g. ``layers.3`` or ``vision_encoder.layers.3``).
_LAYER_PREFIX_RE = re.compile(r"^(.*?(?:^|\.)layers\.\d+)\.")
_OPTIMIZER_STEP_LAYERS_PER_CHUNK = 4
_OPTIMIZER_STEP_MAX_PARAMS_PER_CHUNK = 64

# ``checkpoint.save_contents`` entries stored in the TorchTitan DCP checkpoint (TorchTitan writes them together).
_DCP_CHECKPOINT_CONTENTS = frozenset({"model", "optimizer", "extra"})

# Size cap of one safetensors shard in the ``hf_model`` export, as in verl's FSDP ``hf_model`` export.
_HF_MAX_SHARD_BYTES = 5 * 1024**3


def write_hf_safetensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    out_dir: str,
    is_writer: bool,
    skip_names: Collection[str] = (),
    max_shard_bytes: int = _HF_MAX_SHARD_BYTES,
) -> Optional[dict[str, str]]:
    """Stream ``(hf_name, tensor)`` pairs into safetensors shards plus ``model.safetensors.index.json``.

    Every rank must drain ``named_tensors``: producing a tensor may run collectives (``TrainerWeights`` all-gathers
    each FSDP shard). Only ``is_writer`` copies the tensors to host memory and writes files, so host memory peaks at
    about one shard. ``skip_names`` drops tensors that HF re-creates on load, such as a tied ``lm_head.weight``.
    """
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = total_bytes = 0
    tmp_shards: list[tuple[str, list[str]]] = []

    def flush_shard() -> None:
        nonlocal shard, shard_bytes
        tmp_path = os.path.join(out_dir, f"model-{len(tmp_shards) + 1:05d}.safetensors.tmp")
        save_file(shard, tmp_path, metadata={"format": "pt"})
        tmp_shards.append((tmp_path, list(shard)))
        shard, shard_bytes = {}, 0

    if is_writer:
        os.makedirs(out_dir, exist_ok=True)
    for name, tensor in named_tensors:
        if not is_writer or name in skip_names:
            tpu_synchronize(tensor, wait=True)
            continue
        nbytes = tensor.numel() * tensor.element_size()
        if shard and shard_bytes + nbytes > max_shard_bytes:
            flush_shard()
        shard[name] = tensor.detach().contiguous().to("cpu", copy=True)
        shard_bytes += nbytes
        total_bytes += nbytes
    if not is_writer:
        return None
    if shard:
        flush_shard()

    weight_map: dict[str, str] = {}
    for i, (tmp_path, names) in enumerate(tmp_shards, start=1):
        file_name = f"model-{i:05d}-of-{len(tmp_shards):05d}.safetensors"
        os.replace(tmp_path, os.path.join(out_dir, file_name))
        weight_map.update(dict.fromkeys(names, file_name))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f, indent=2)
    return weight_map


# TODO: Upstream ``hf_range_reader`` into ``torchtitan`` (``StateDictAdapter.get_hf_storage_reader``) and remove
# ``_use_threaded_hf_reader`` once upstreamed.
# Safetensors files that the initial HF load reads in parallel (one thread per file; see engines/hf_range_reader.py).
_HF_LOAD_THREADS_ENV = "VERL_TPU_HF_LOAD_THREADS"


def _use_threaded_hf_reader(checkpointer: Any) -> None:
    """Makes the initial HF load pread byte ranges from ``VERL_TPU_HF_LOAD_THREADS`` (default 16) files at a time.

    torch's reader reads through mmap slices, ~6 MB/s per reader on a gcsfuse mount (hours for a 740 GB checkpoint).
    """
    sd_adapter = getattr(checkpointer, "sd_adapter", None)
    threads = int(os.environ.get(_HF_LOAD_THREADS_ENV, "16"))
    if sd_adapter is None or threads < 1 or not hasattr(sd_adapter, "get_hf_storage_reader"):
        return
    original = sd_adapter.get_hf_storage_reader

    def get_hf_storage_reader(path: str, from_quantized: bool = False, *args: Any, **kwargs: Any):
        reader = original(path, from_quantized, *args, **kwargs)
        if from_quantized or type(reader).__name__ != "HuggingFaceStorageReader":
            return reader
        from verl_hardware_plugin.accelerators.tpu.engines.hf_range_reader import make_range_read_hf_storage_reader

        return make_range_read_hf_storage_reader(type(reader), path, threads)

    sd_adapter.get_hf_storage_reader = get_hf_storage_reader


def _param_mesh_key(param: torch.Tensor) -> int | None:
    """Object identity of ``param``'s ``DeviceMesh`` when ``param`` is a ``DTensor``, else ``None``."""
    spec = getattr(param, "_spec", None)
    mesh = getattr(spec, "mesh", None) if spec is not None else None
    return None if mesh is None else id(mesh)


def _can_pin_memory() -> bool:
    """Whether ``torch.empty(..., pin_memory=True)`` has an active accelerator host allocator."""
    accel = getattr(torch, "accelerator", None)
    if accel is not None and callable(getattr(accel, "is_available", None)):
        try:
            return bool(accel.is_available())
        except Exception:
            return False
    return False


def _alloc_cpu_staging(shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a contiguous host staging buffer (pinned when an accelerator runtime is active)."""
    if _can_pin_memory():
        try:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(shape, dtype=dtype, device="cpu")


def _record_tpu_event(is_tpu: bool) -> Any:
    """Record and return a ``torch.tpu.Event`` on the active TPU stream, or ``None`` off-TPU."""
    if not is_tpu:
        return None
    tpu_mod = getattr(torch, "tpu", None)
    event_cls = getattr(tpu_mod, "Event", None)
    if event_cls is None:
        return None
    ev = event_cls()
    ev.record()
    return ev


def _wrap_local_like_param(local: torch.Tensor, like_param: torch.Tensor) -> torch.Tensor:
    """Wrap ``local`` in a ``DTensor`` view sharing ``local``'s storage when ``like_param`` is a ``DTensor``."""
    if isinstance(like_param, DTensor) and local.ndim > 0:
        spec = DTensorSpec(
            like_param._spec.mesh,
            like_param._spec.placements,
            tensor_meta=TensorMeta(like_param.shape, like_param.stride(), local.dtype),
        )
        return DTensor(local.view_as(local), spec, requires_grad=False)
    return local


def _direct_d2h_source(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure ``tensor`` satisfies ``torch_tpu`` ``CanCopyStorageDirect`` (zero offset, contiguous, tight storage)."""
    if not tensor.is_contiguous() or tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.nbytes:
        return tensor.contiguous()
    return tensor


@functools.lru_cache(maxsize=1)
def _cpu_fused_adamw_supported() -> bool:
    """Whether ``torch.optim.AdamW(..., fused=True)`` runs on CPU ``float32`` tensors."""
    try:
        probe = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device="cpu"))
        probe.grad = torch.zeros(1, dtype=torch.float32, device="cpu")
        torch.optim.AdamW([probe], lr=1e-3, fused=True).step()
        return True
    except Exception:
        return False


@torch.no_grad()
def _move_optimizer_state(optimizer: Any, params: list[torch.nn.Parameter], device: Any) -> None:
    """Moves the optimizer states of ``params`` to ``device``, reusing pinned host staging buffers on CPU."""
    target_device = torch.device(device)
    to_cpu = target_device.type == "cpu"
    host_cache: dict[tuple[int, str], torch.Tensor] = getattr(optimizer, "_tpu_pinned_host_state", None)  # type: ignore[assignment]
    if host_cache is None:
        host_cache = {}
        optimizer._tpu_pinned_host_state = host_cache

    any_tpu_d2h = False
    for p in params:
        state = optimizer.state.get(p)
        if not state:
            continue
        for key, value in list(state.items()):
            if not isinstance(value, torch.Tensor) or value.device == target_device:
                continue
            local_val = value._local_tensor if isinstance(value, DTensor) else value
            if local_val.ndim == 0:
                state[key] = value.to(target_device)
                continue
            cache_key = (id(p), str(key))
            if to_cpu:
                buf = host_cache.get(cache_key)
                if buf is None or buf.shape != local_val.shape or buf.dtype != local_val.dtype:
                    buf = _alloc_cpu_staging(local_val.shape, local_val.dtype)
                    host_cache[cache_key] = buf
                src = _direct_d2h_source(local_val)
                is_tpu_src = src.device.type == "tpu"
                buf.copy_(src, non_blocking=is_tpu_src)
                any_tpu_d2h = any_tpu_d2h or is_tpu_src
                state[key] = _wrap_local_like_param(buf, value)
            else:
                dev_local = torch.empty(local_val.shape, dtype=local_val.dtype, device=target_device)
                dev_local.copy_(local_val, non_blocking=target_device.type == "tpu")
                state[key] = _wrap_local_like_param(dev_local, value)
    if any_tpu_d2h:
        tpu_synchronize(wait=True)


def _state_to(state: dict, device: Any) -> dict:
    """``state`` (one parameter's optimizer state) with its tensors moved to ``device``."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) and v.device != torch.device(device) else v
        for k, v in state.items()
    }


def _scalar_float(val: Any) -> float:
    return float(val.item()) if isinstance(val, torch.Tensor) else float(val)


class _Fp32Masters:
    """FP32 master weights of bf16 parameters and their optimizer states (``engine_kwargs.tpu.fp32_master``).

    The model keeps bf16 parameters for forward, backward and the weight sync. The optimizer steps an FP32 master
    copy of each parameter instead (FP32 AdamW moments too) and the result is cast back into the bf16 parameter, so
    small updates (lr 1e-6 is ~3-4 orders of magnitude below a bf16 ulp of a weight) accumulate instead of rounding
    to zero.

    * ``host=False``: masters and states stay on the TPU and step through the inner optimizer one chunk at a time.
    * ``host=True`` (``optimizer_host_offload``): for ``AdamW``, masters and FP32 moments stay permanently resident on
      CPU and step via CPU ``fused=True`` ``AdamW``, while double-buffered pinned ``bfloat16`` staging buffers pipeline
      async gradient D2H (``2 B/param``) and stepped-weight H2D (``2 B/param``), cutting PCIe traffic ``6x`` vs moving
      ``(master, exp_avg, exp_avg_sq)`` (``24 B/param``) across PCIe every step. Non-``AdamW`` optimizers fall back to
      per-chunk device stepping.
    """

    def __init__(self, host: bool):
        self.home: Any = "cpu" if host else None
        self._entries: dict[int, tuple[torch.Tensor, dict]] = {}
        self._cpu_params: dict[int, torch.nn.Parameter] = {}
        self._cpu_opt: Optional[torch.optim.AdamW] = None
        self._staging_slots: tuple[
            dict[tuple[int, tuple[int, ...], torch.dtype], torch.Tensor],
            dict[tuple[int, tuple[int, ...], torch.dtype], torch.Tensor],
        ] = ({}, {})

    def clear(self) -> None:
        """Drop all master weights and cached CPU optimizer state."""
        self._entries.clear()
        self._cpu_params.clear()
        self._staging_slots[0].clear()
        self._staging_slots[1].clear()
        if self._cpu_opt is not None:
            self._cpu_opt.state.clear()

    def can_step_on_cpu(self, opt: Any, group: Optional[Mapping[str, Any]] = None) -> bool:
        if self.home != "cpu" or type(opt) is not torch.optim.AdamW:
            return False
        if group is not None and bool(group.get("differentiable", False)):
            return False
        return True

    def _ensure_cpu_adamw(self, group: Mapping[str, Any]) -> torch.optim.AdamW:
        betas = group.get("betas", (0.9, 0.999))
        lr = _scalar_float(group.get("lr", 1e-3))
        eps = _scalar_float(group.get("eps", 1e-8))
        weight_decay = _scalar_float(group.get("weight_decay", 0.0))
        amsgrad = bool(group.get("amsgrad", False))
        maximize = bool(group.get("maximize", False))
        b0, b1 = _scalar_float(betas[0]), _scalar_float(betas[1])
        if self._cpu_opt is None:
            dummy = torch.nn.Parameter(torch.empty(0, dtype=torch.float32, device="cpu"))
            use_fused = _cpu_fused_adamw_supported()
            self._cpu_opt = torch.optim.AdamW(
                [dummy],
                lr=lr,
                betas=(b0, b1),
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                maximize=maximize,
                fused=use_fused,
                foreach=False if use_fused else True,
            )
            self._cpu_opt.param_groups[0]["params"] = []
        else:
            cg = self._cpu_opt.param_groups[0]
            cg["lr"] = lr
            cg["betas"] = (b0, b1)
            cg["eps"] = eps
            cg["weight_decay"] = weight_decay
            cg["amsgrad"] = amsgrad
            cg["maximize"] = maximize
        return self._cpu_opt

    def _get_slot_buffer(self, slot: int, idx: int, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        pool = self._staging_slots[slot]
        key = (idx, tuple(shape), dtype)
        buf = pool.get(key)
        if buf is None:
            buf = _alloc_cpu_staging(shape, dtype)
            pool[key] = buf
        return buf

    @torch.no_grad()
    def _get_or_bind_cpu_master(self, p: torch.nn.Parameter, cpu_opt: torch.optim.AdamW) -> torch.nn.Parameter:
        pid = id(p)
        entry = self._entries.get(pid)
        if entry is None:
            old_mp = self._cpu_params.pop(pid, None)
            if old_mp is not None:
                cpu_opt.state.pop(old_mp, None)
            local_p = p._local_tensor if isinstance(p, DTensor) else p.data
            local_master = local_p.detach().to(device="cpu", dtype=torch.float32).contiguous()
            mp = torch.nn.Parameter(local_master, requires_grad=True)
            self._cpu_params[pid] = mp
            self._entries[pid] = (_wrap_local_like_param(local_master, p), {})
            return mp

        master_view, state_view = entry
        local_master = master_view._local_tensor if isinstance(master_view, DTensor) else master_view
        if local_master.device.type != "cpu" or local_master.dtype != torch.float32 or not local_master.is_contiguous():
            local_master = local_master.detach().to(device="cpu", dtype=torch.float32).contiguous()
            master_view = _wrap_local_like_param(local_master, p)

        mp = self._cpu_params.get(pid)
        if (
            mp is None
            or mp.data.data_ptr() != local_master.data_ptr()
            or (mp not in cpu_opt.state and bool(state_view))
        ):
            if mp is not None:
                cpu_opt.state.pop(mp, None)
            mp = torch.nn.Parameter(local_master, requires_grad=True)
            self._cpu_params[pid] = mp
            if state_view:
                cpu_state: dict[str, Any] = {}
                bound_state_view: dict[str, Any] = {}
                for k, v in state_view.items():
                    if isinstance(v, torch.Tensor):
                        lv = v._local_tensor if isinstance(v, DTensor) else v
                        if lv.device.type != "cpu" or lv.dtype != torch.float32 or not lv.is_contiguous():
                            lv = lv.detach().to(device="cpu", dtype=torch.float32).contiguous()
                        cpu_state[k] = lv
                        bound_state_view[k] = _wrap_local_like_param(lv, p)
                    else:
                        cpu_state[k] = v
                        bound_state_view[k] = v
                cpu_opt.state[mp] = cpu_state
                self._entries[pid] = (master_view, bound_state_view)
            else:
                self._entries[pid] = (master_view, {})
        return mp

    @torch.no_grad()
    def step_group_chunks_on_cpu(
        self,
        group: Mapping[str, Any],
        chunks: list[list[torch.nn.Parameter]],
        *,
        max_norm: float,
        grad_norm: torch.Tensor,
    ) -> None:
        """Pipeline TPU gradient clip + pinned bf16 D2H with CPU fused FP32 AdamW + pinned bf16 H2D."""
        active_chunks = [[p for p in chunk if p.grad is not None] for chunk in chunks]
        active_chunks = [c for c in active_chunks if c]
        if not active_chunks:
            return

        is_tpu = grad_norm.device.type == "tpu"
        cpu_opt = self._ensure_cpu_adamw(group)
        d2h_events: list[Any] = [None, None]
        h2d_events: list[Any] = [None, None]
        slot_bufs: list[list[torch.Tensor]] = [[], []]

        def _enqueue_d2h(chunk_params: list[torch.nn.Parameter], slot: int) -> None:
            if h2d_events[slot] is not None:
                h2d_events[slot].synchronize()
                h2d_events[slot] = None
            for p in chunk_params:
                if isinstance(p.grad, DTensor) and any(pl.is_partial() for pl in p.grad.placements):
                    p.grad = p.grad.redistribute(placements=p.placements)
            torch.nn.utils.clip_grads_with_norm_(chunk_params, max_norm, grad_norm, foreach=False)
            if is_tpu:
                # Fuse all gradient clip scalings of this chunk into one async XLA program before D2H DMAs.
                tpu_synchronize(wait=False)
            cur_bufs: list[torch.Tensor] = []
            for idx, p in enumerate(chunk_params):
                assert p.grad is not None
                local_g = p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
                src_g = _direct_d2h_source(local_g)
                buf = self._get_slot_buffer(slot, idx, src_g.shape, src_g.dtype)
                buf.copy_(src_g, non_blocking=is_tpu)
                cur_bufs.append(buf)
                p.grad = None
            slot_bufs[slot] = cur_bufs
            d2h_events[slot] = _record_tpu_event(is_tpu)

        def _step_cpu_and_enqueue_h2d(chunk_params: list[torch.nn.Parameter], slot: int) -> None:
            if d2h_events[slot] is not None:
                d2h_events[slot].synchronize()
                d2h_events[slot] = None
            elif is_tpu:
                tpu_synchronize(wait=True)

            mps: list[torch.nn.Parameter] = []
            bufs = slot_bufs[slot]
            for p, buf in zip(chunk_params, bufs, strict=True):
                mp = self._get_or_bind_cpu_master(p, cpu_opt)
                mp.grad = buf.to(dtype=torch.float32)
                mps.append(mp)

            cpu_opt.param_groups[0]["params"] = mps
            try:
                cpu_opt.step()
            finally:
                cpu_opt.param_groups[0]["params"] = []

            for p, mp, buf in zip(chunk_params, mps, bufs, strict=True):
                pid = id(p)
                master_view, state_view = self._entries[pid]
                if not state_view:
                    cpu_state = cpu_opt.state.get(mp, {})
                    bound_state = {
                        k: _wrap_local_like_param(v, p) if isinstance(v, torch.Tensor) else v
                        for k, v in cpu_state.items()
                    }
                    self._entries[pid] = (master_view, bound_state)
                mp.grad = None
                buf.copy_(mp.detach())
                local_p = p._local_tensor if isinstance(p, DTensor) else p.data
                local_p.copy_(buf, non_blocking=is_tpu)
            h2d_events[slot] = _record_tpu_event(is_tpu)

        _enqueue_d2h(active_chunks[0], 0)
        for i, chunk_params in enumerate(active_chunks):
            if i + 1 < len(active_chunks):
                _enqueue_d2h(active_chunks[i + 1], (i + 1) % 2)
            _step_cpu_and_enqueue_h2d(chunk_params, i % 2)

        for slot in (0, 1):
            if h2d_events[slot] is not None:
                h2d_events[slot].synchronize()
                h2d_events[slot] = None
        if is_tpu:
            tpu_synchronize(wait=True)

    @torch.no_grad()
    def begin(self, opt: Any, chunk: list[torch.nn.Parameter]) -> list[tuple[torch.nn.Parameter, torch.nn.Parameter]]:
        """FP32 parameters (with FP32 gradients and their saved optimizer states) for ``chunk``'s stepped params."""
        pairs = []
        for p in chunk:
            if p.grad is None:
                continue
            entry = self._entries.get(id(p))
            if entry is None:
                master, state = p.detach().float(), {}
            else:
                master, state = entry[0].to(p.device), _state_to(entry[1], p.device)
            master_param = torch.nn.Parameter(master, requires_grad=True)
            master_param.grad = p.grad.float()
            if state:
                opt.state[master_param] = state
            pairs.append((p, master_param))
        return pairs

    @torch.no_grad()
    def write_back(self, pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]]) -> None:
        """Casts the stepped FP32 masters into their bf16 parameters (on the device, before the chunk's sync)."""
        for p, master_param in pairs:
            p.copy_(master_param.detach())
            p.grad = None
            master_param.grad = None

    def end(self, opt: Any, pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]]) -> None:
        """Saves the masters and their optimizer states (to host memory with ``host``) after the chunk's sync."""
        for p, master_param in pairs:
            state = opt.state.pop(master_param, {})
            master = master_param.detach()
            if self.home is not None:
                master, state = master.to(self.home), _state_to(state, self.home)
            self._entries[id(p)] = (master, state)


def _fp32_masters_of(engine: Any) -> Optional[_Fp32Masters]:
    """``engine``'s FP32 master weights of ``engine_kwargs.tpu.fp32_master`` (None when it is off), created lazily."""
    tpu_options = getattr(engine, "_tpu_options", None)
    if not (tpu_options and tpu_options.fp32_master):
        return None
    masters = getattr(engine, "_fp32_master_store", None)
    if masters is None:
        masters = _Fp32Masters(host=bool(tpu_options.optimizer_host_offload))
        engine._fp32_master_store = masters
    return masters


class _NonSeekableWriter(io.RawIOBase):
    """Append-only binary stream wrapper so ``torch.save`` never seeks backwards on GCSFuse streaming mounts."""

    def __init__(self, raw: Any):
        self._raw = raw

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def write(self, b: Any) -> int:
        return self._raw.write(b)

    def flush(self) -> None:
        self._raw.flush()


def _fp32_master_ckpt_filename(rank: int, world_size: int) -> str:
    return f"fp32_master_rank_{rank:05d}_of_{world_size:05d}.pt"


def _pack_ckpt_tensor(value: Any) -> Any:
    if isinstance(value, DTensor):
        return {"__dtensor_local__": value._local_tensor.detach().cpu()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _unpack_ckpt_tensor_like(saved: Any, like_param: torch.Tensor, target_device: torch.device) -> Any:
    if isinstance(saved, dict) and "__dtensor_local__" in saved:
        local = saved["__dtensor_local__"].to(device=target_device)
        return _wrap_local_like_param(local, like_param)
    if isinstance(saved, torch.Tensor):
        return saved.to(device=target_device)
    return saved


@torch.no_grad()
def _copy_local_into_(dst: torch.Tensor, saved: Any) -> None:
    src = saved["__dtensor_local__"] if isinstance(saved, dict) and "__dtensor_local__" in saved else saved
    target = dst._local_tensor if isinstance(dst, DTensor) else dst.data
    target.copy_(src.to(device=target.device, dtype=target.dtype))


def _gcs_ckpt_blob_target(path: str) -> Optional[tuple[str, str]]:
    """Resolve a direct ``gs://`` URI or a ``VERL_GCS_CKPT_BUCKET``-backed mount path to ``(bucket, blob)``.

    Streaming multi-terabyte checkpoints directly via ``google.cloud.storage`` avoids staging 35-70 GiB/rank shards
    onto node-local root disks through GCSFuse's ``/gcsfuse-tmp`` cache.
    """
    if path.startswith("gs://"):
        without_scheme = path[len("gs://") :].strip("/")
        bucket_name, _, blob_name = without_scheme.partition("/")
        if bucket_name:
            return bucket_name, blob_name
        return None
    bucket_name = os.environ.get("VERL_GCS_CKPT_BUCKET", "").strip()
    if not bucket_name:
        return None
    mount_root = os.path.normpath(os.environ.get("VERL_GCS_CKPT_MOUNT", "/gcs_ckpt"))
    norm = os.path.normpath(path)
    mount_prefix = f"{mount_root.rstrip('/')}/"
    if norm == mount_root or norm.startswith(mount_prefix):
        rel = norm[len(mount_prefix) :].lstrip("/") if norm != mount_root else ""
        prefix = os.environ.get("VERL_GCS_CKPT_PREFIX", "").strip("/")
        blob_name = f"{prefix}/{rel}".strip("/") if prefix and rel else (prefix or rel)
        return bucket_name, blob_name
    return None


def _prune_old_fp32_master_checkpoints(local_path: str, max_ckpt_to_keep: Optional[int]) -> None:
    if max_ckpt_to_keep is None or max_ckpt_to_keep <= 0:
        return
    norm_path = os.path.normpath(local_path)
    step_dir = os.path.dirname(norm_path)
    if os.path.basename(norm_path) != "actor" or not os.path.basename(step_dir).startswith("global_step_"):
        return
    ckpt_root = os.path.dirname(step_dir)
    gcs_root = _gcs_ckpt_blob_target(ckpt_root)
    if gcs_root is not None:
        from google.cloud import storage

        bucket_name, root_prefix = gcs_root
        prefix_with_slash = f"{root_prefix.rstrip('/')}/"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        it = bucket.list_blobs(prefix=prefix_with_slash, delimiter="/")
        list(it)  # populate it.prefixes
        found_gcs: list[tuple[int, str]] = []
        for sub_prefix in it.prefixes:
            rel_name = sub_prefix[len(prefix_with_slash) :].rstrip("/")
            m = re.fullmatch(r"global_step_(\d+)", rel_name)
            if m:
                found_gcs.append((int(m.group(1)), sub_prefix))
        found_gcs.sort(key=lambda item: item[0])
        if len(found_gcs) > max_ckpt_to_keep:
            for _, old_prefix in found_gcs[:-max_ckpt_to_keep]:
                logger.info(
                    "Removing old GCS checkpoint prefix gs://%s/%s (max_ckpt_to_keep=%d)",
                    bucket_name,
                    old_prefix,
                    max_ckpt_to_keep,
                )
                blobs = list(bucket.list_blobs(prefix=old_prefix))
                for i in range(0, len(blobs), 100):
                    bucket.delete_blobs(blobs[i : i + 100])
        return

    if not os.path.isdir(ckpt_root):
        return
    found: list[tuple[int, str]] = []
    for entry in os.listdir(ckpt_root):
        m = re.fullmatch(r"global_step_(\d+)", entry)
        if m:
            full = os.path.join(ckpt_root, entry)
            if os.path.isdir(full):
                found.append((int(m.group(1)), full))
    found.sort(key=lambda item: item[0])
    if len(found) > max_ckpt_to_keep:
        for _, old_dir in found[:-max_ckpt_to_keep]:
            logger.info("Removing old checkpoint directory %s (max_ckpt_to_keep=%d)", old_dir, max_ckpt_to_keep)
            shutil.rmtree(old_dir, ignore_errors=True)


def _partition_params_into_chunks(
    params: list[torch.nn.Parameter],
    param_names: list[str] | None = None,
    *,
    layers_per_chunk: int = _OPTIMIZER_STEP_LAYERS_PER_CHUNK,
    max_params_per_chunk: int = _OPTIMIZER_STEP_MAX_PARAMS_PER_CHUNK,
) -> list[list[torch.nn.Parameter]]:
    """Group parameters into bounded single-mesh chunks (by 4-layer block when FQNs are available).

    Under ``DEFER_AND_FUSE``, clipping and stepping all parameters of a large MoE model in a single
    un-synchronized graph fuses 500+ parameter updates into one giant HLO module whose live input/output
    temporaries exceed TPU HBM (``CompileTimeHbmOom``). Grouping by 4-layer cycle keeps each HLO module
    small and gives identical parameter shapes across repeating hybrid layer blocks for XLA cache hits.
    Splitting by ``DeviceMesh`` (in deterministic first-seen parameter order across ranks) also keeps
    ``(fsdp,)`` dense and ``(efsdp, ep)`` routed-expert ``DTensor``s in separate ``foreach``/``fused`` AdamW
    calls on torchtitan builds that do not split optimizer param groups by mesh.
    """
    if not params:
        return []
    mesh_order: dict[int | None, int] = {}

    def _mesh_slot(p: torch.nn.Parameter) -> int:
        mk = _param_mesh_key(p)
        return mesh_order.setdefault(mk, len(mesh_order))

    if param_names is None or len(param_names) != len(params):
        by_mesh: dict[int, list[torch.nn.Parameter]] = {}
        for p in params:
            by_mesh.setdefault(_mesh_slot(p), []).append(p)
        return [
            bucket[i : i + max_params_per_chunk]
            for slot in sorted(by_mesh)
            for bucket in (by_mesh[slot],)
            for i in range(0, len(bucket), max_params_per_chunk)
        ]

    buckets: dict[tuple[int, int, int], list[torch.nn.Parameter]] = {}
    for p, name in zip(params, param_names, strict=True):
        m = _LAYER_FQN_RE.search(name)
        layer_key = (0, int(m.group(1)) // layers_per_chunk) if m is not None else (1, 0)
        key = (*layer_key, _mesh_slot(p))
        buckets.setdefault(key, []).append(p)

    chunks: list[list[torch.nn.Parameter]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        for i in range(0, len(bucket), max_params_per_chunk):
            chunks.append(bucket[i : i + max_params_per_chunk])
    return chunks


class TPUEngineEvalModeCtx(BaseEngineCtx):
    """Eval-mode context manager that skips ``reshard()`` on non-``FSDPModule`` modules (e.g. ``SimpleFSDP``)."""

    def __init__(self, engine: TorchTitanTPUEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanTPUEngine)

        # Reshard the FSDP2 modules. SimpleFSDP modules are not FSDPModule and keep no unsharded copy.
        if (
            self.engine.engine_config.data_parallel_shard_size > 1
            or self.engine.engine_config.context_parallel_size > 1
        ):
            for module in self.engine.module:
                for submodule in module.modules():
                    if isinstance(submodule, FSDPModule):
                        submodule.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class TPUEngineTrainModeCtx(BaseEngineCtx):
    """Train-mode context manager for ``TorchTitanTPUEngine``."""

    def __init__(self, engine: TorchTitanTPUEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


_SPLASH_ROUTING_SELECTIVE_AC_CLS: Any = None
_PALLAS_SPLASH_FWD_OP_MARKER = "splash_attn_fwd"


def _op_qualified_name(op: Any) -> str:
    name_fn = getattr(op, "name", None)
    if callable(name_fn):
        try:
            return str(name_fn())
        except Exception:
            pass
    return str(op)


class _SplashAndRoutingSaveOps(set):
    """Selective-AC save set keeping only Pallas ``SplashAttention`` forward outputs and MoE router ``aten.topk``."""

    def __contains__(self, op: Any) -> bool:
        try:
            if super().__contains__(op):
                return True
        except TypeError:
            pass
        return _PALLAS_SPLASH_FWD_OP_MARKER in _op_qualified_name(op)


def _splash_and_routing_selective_ac_config() -> Any:
    """Builds a ``SelectiveAC.Config`` saving only ``SplashAttention`` forward ``(out, lse)`` and ``aten.topk``."""
    global _SPLASH_ROUTING_SELECTIVE_AC_CLS
    if _SPLASH_ROUTING_SELECTIVE_AC_CLS is None:
        from torchtitan.distributed.activation_checkpoint import SelectiveAC

        class _SplashAndRoutingSelectiveAC(SelectiveAC):
            @dataclasses.dataclass(kw_only=True, slots=True)
            class Config(SelectiveAC.Config):
                preserve_rng_state: bool = False
                force_recompute_mm_shapes_by_fqns: list[str] = dataclasses.field(default_factory=list)

            def get_save_ops(self) -> set:
                return _SplashAndRoutingSaveOps({torch.ops.aten.topk.default})

        _SPLASH_ROUTING_SELECTIVE_AC_CLS = _SplashAndRoutingSelectiveAC
    return _SPLASH_ROUTING_SELECTIVE_AC_CLS.Config()


def _import_tpu_selective_ac() -> Any:
    """Imports ``TPUSelectiveAC`` (``components/`` on HEAD, top-level on older branches, else ``SelectiveAC``)."""
    for mod_name in (
        "torchtitan.experiments.tpu.components.activation_checkpoint",
        "torchtitan.experiments.tpu.activation_checkpoint",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "TPUSelectiveAC"):
            return mod.TPUSelectiveAC
    try:
        from torchtitan.experiments.tpu.components.activation_checkpoint import TPUSelectiveAC
    except ImportError:
        try:
            from torchtitan.experiments.tpu.activation_checkpoint import TPUSelectiveAC
        except ImportError:
            from torchtitan.distributed.activation_checkpoint import SelectiveAC as TPUSelectiveAC
    return TPUSelectiveAC


def _import_tpu_cp_attention() -> Any:
    """Imports torchtitan's TPU ``cp_attention`` module (``distributed/`` on HEAD, top-level on older branches)."""
    for mod_name in (
        "torchtitan.experiments.tpu.distributed.cp_attention",
        "torchtitan.experiments.tpu.cp_attention",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            return mod
    for mod_name in (
        "torchtitan.experiments.tpu.distributed.cp_attention",
        "torchtitan.experiments.tpu.cp_attention",
    ):
        if module_available(mod_name):
            return importlib.import_module(mod_name)
    raise RuntimeError(
        "Context parallelism on TPU requires a TPU torchtitan build with cp_attention "
        "(distributed/cp_attention.py); turn context parallelism off on upstream torchtitan HEAD."
    )


def _import_set_unflatten_seq_len() -> Any:
    """Imports ``set_unflatten_seq_len`` (``components/unflatten_shim`` on HEAD, top-level on older branches)."""
    for mod_name in (
        "torchtitan.experiments.tpu.components.unflatten_shim",
        "torchtitan.experiments.tpu.unflatten_shim",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "set_unflatten_seq_len"):
            return mod.set_unflatten_seq_len
    last_exc: Optional[ImportError] = None
    for mod_name in (
        "torchtitan.experiments.tpu.components.unflatten_shim",
        "torchtitan.experiments.tpu.unflatten_shim",
    ):
        try:
            return importlib.import_module(mod_name).set_unflatten_seq_len
        except ImportError as e:
            last_exc = e
            continue
    raise RuntimeError(
        "unflatten_shim (components/unflatten_shim.py) requires a TPU torchtitan build; "
        "turn gdn_row_slot_len off on upstream torchtitan HEAD."
    ) from last_exc


def _validate_cp_gdn_options_for_degree(context_parallel_size: int, tpu_options: Any) -> None:
    """Rejects ``context_parallel_gdn_*`` options when ``context_parallel_size <= 1``."""
    if int(context_parallel_size or 1) > 1:
        return
    set_opts = [
        f"engine_kwargs.tpu.{name}={getattr(tpu_options, name)!r}"
        for name in ("context_parallel_gdn_strategy", "context_parallel_gdn_reduce_dtype")
        if getattr(tpu_options, name, None) is not None
    ]
    if set_opts:
        raise ValueError(
            f"{', '.join(set_opts)} requires context_parallel_size > 1 "
            f"(got context_parallel_size={context_parallel_size}); "
            "unset GatedDeltaNet CP options when context parallelism is disabled."
        )


def _add_rollout_consistency_metrics(
    loss_function: Callable, loss_model_output: dict[str, Any], loss_data: TensorDict, metrics: dict[str, Any]
) -> None:
    """Adds the trainer-vs-rollout log-prob gap metrics (KL, k3, |gap|) of verl's PPO loss to ``metrics``.

    Shared by the eager and ``trainer_mode=graph_trainer`` step paths so both log the same metrics.
    """
    names = (
        "actor/ppo_kl_token_mean",
        "actor/ppo_k3_token_mean",
        "actor/logprob_gap_abs_token_mean",
    )
    config = ppo_loss_config(loss_function)
    global_batch_info = _cfg_get(config, "global_batch_info")
    if isinstance(global_batch_info, Mapping) and any(name not in metrics for name in names):
        if global_batch_info.get("batch_num_tokens") is not None:
            policy_loss = _cfg_get(config, "policy_loss")
            mode = _cfg_get(policy_loss, "loss_mode", "vanilla")
            correction = _cfg_get(policy_loss, "rollout_correction")
            # Rejection sampling and other losses use a different token denominator.
            if mode == "vanilla" or (
                mode == "bypass_mode"
                and _cfg_get(correction, "loss_type", "ppo_clip") == "ppo_clip"
                and _cfg_get(correction, "rollout_rs") is None
            ):
                with torch.no_grad():
                    log_prob = no_padding_2_padding(loss_model_output["log_probs"], loss_data)
                    fields = loss_data.select("old_log_probs", "response_mask").to_padded_tensor()
                    mask = fields["response_mask"]
                    gap = torch.where(mask.bool(), log_prob.float() - fields["old_log_probs"].float(), 0.0)
                    ratio = gap.clamp(-20.0, 20.0)
                    values = (-ratio, (ratio.exp() - ratio - 1.0).clamp_min(0.0), gap.abs())
                    # SUM micro-batches, then MEAN DP ranks: global token means.
                    for name, value in zip(names, values, strict=True):
                        metrics.setdefault(
                            name,
                            Metric(
                                AggregationType.SUM,
                                agg_loss(value, mask, "token-mean", **global_batch_info),
                            ),
                        )


class TorchTitanTPUEngine(BaseEngine):
    """Standalone TorchTitan engine for Google TPU with ``tpu_dist`` / PJRT SPMD execution."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanTPUEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2 or SimpleFSDP).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        tpu_options = get_tpu_engine_options(engine_config)
        _validate_cp_gdn_options_for_degree(getattr(engine_config, "context_parallel_size", 1), tpu_options)

        from torchtitan.components.checkpointer import CheckpointManager
        from torchtitan.components.loss import CrossEntropyLoss
        from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer, ParamGroupConfig
        from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
        from torchtitan.distributed.activation_checkpoint import FullAC
        from torchtitan.trainer import Trainer

        TPUSelectiveAC = _import_tpu_selective_ac()

        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        # Device copies of the head-tail CP restore permutation, per (seq_len, cp, device): see _cp_restore_indices.
        self._cp_restore_idx_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

        (
            self._use_splash_attention,
            self._use_simple_fsdp,
            self._tpu_eager_mode,
        ) = resolve_tpu_torchtitan_options(engine_config)

        if engine_config.use_torch_compile:
            configure_torch_compile_for_tpu()
            tpu_model_kernels.patch_torchtitan_tpu_dynamo_caches()

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)
        self._torchtitan_name = torchtitan_name

        self._tpu_options = tpu_options

        # Get ModelSpec from model registry
        model_module = importlib.import_module(get_torchtitan_model_module_name(torchtitan_name))
        # Optional TPU kernels (engine_kwargs.tpu.model_kernels, see engines/tpu_model_kernels.py). Only torchtitan
        # builds with the Qwen3.5 optimization PRs (google-pytorch/torchtitan#73-#80) provide any; on torchtitan
        # main this is empty and every kernel hook below is a no-op.
        self._tpu_kernels = tpu_model_kernels.resolve_tpu_model_kernels(torchtitan_name, tpu_options.model_kernels)
        if self._tpu_kernels:
            logger.info("Optional torchtitan TPU kernels of %s: %s", torchtitan_name, sorted(self._tpu_kernels))
        # engine_kwargs.tpu.trainer_mode=graph_trainer: TorchTitan's TpuGraphTrainer traces and compiles the whole
        # training step (engines/graph_trainer_mode.py).
        self._graph_trainer_mode = graph_trainer_mode.is_graph_trainer_mode(tpu_options)
        if self._graph_trainer_mode:
            graph_trainer_mode.validate_graph_trainer_engine(
                engine_config, tpu_options, use_simple_fsdp=self._use_simple_fsdp, tpu_kernels=self._tpu_kernels
            )
        self._graph_policy_loss_spec: Optional[graph_trainer_mode.GraphPolicyLossSpec] = None
        self._graph_trainer_traced_len: Optional[int] = None
        registry_params = inspect.signature(model_module.model_registry).parameters
        fuse_qkv_kwargs = (
            {"fuse_qkv": False}
            if tpu_options.fuse_qkv is False
            and (
                "fuse_qkv" in registry_params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in registry_params.values())
            )
            else {}
        )
        # Mandatory per-model TPU setup (engines/tpu_model_kernels.py, _MODEL_SETUPS), e.g. Qwen3 MoE.
        setup = tpu_model_kernels.model_setup(torchtitan_name, self.model_config.hf_config)
        # MoE models require the fused LM-head log-prob: their per-chip token budgets only fit without the
        # [tokens, vocab] logits, so the engine raises rather than falling back to them.
        self._fused_logprob_required = setup is not None and setup.requires_fused_logprob
        if self._graph_trainer_mode:
            self._use_fused_linear_cross_entropy = True
        else:
            self._use_fused_linear_cross_entropy = resolve_tpu_use_fused_linear_cross_entropy(
                engine_config, required=self._fused_logprob_required
            )
        model_spec = model_module.model_registry(
            torchtitan_flavor,
            attn_backend=torchtitan_model_attn_backend(torchtitan_name, self.engine_config.attn_type),
            seq_len=self.engine_config.max_seq_len,
            **fuse_qkv_kwargs,
            **tpu_model_kernels.model_registry_kwargs(self._tpu_kernels),
            **(setup.registry_kwargs() if setup else {}),
        )
        if self.model_config.hf_config.model_type == "qwen3_moe" and torchtitan_flavor in (
            "30B-A3B",
            "30B-A3B-global-aux",
        ):
            from torchtitan.models.common.moe import MoE

            if tpu_options.qwen3_pallas_experts is False:
                raise ValueError("qwen3_pallas_experts=False is unsupported: Qwen3 MoE requires Pallas on TPU")
            if tpu_options.moe_paired_grouped_mm is None and tpu_options.moe_paired_gemm is not None:
                tpu_options = self._tpu_options = dataclasses.replace(
                    tpu_options, moe_paired_grouped_mm=tpu_options.moe_paired_gemm
                )
            # Upstream configures the experts and balancing; only disable training-only auxiliary losses here.
            moe_configs = list(model_spec.model.traverse(MoE.Config))
            if not moe_configs:
                raise ValueError("Qwen3-30B config has no MoE layers")
            for _, moe, _, _ in moe_configs:
                if hasattr(moe, "aux_loss"):
                    moe.aux_loss = None
                if hasattr(moe.router, "aux_loss"):
                    moe.router.aux_loss = None
        tpu_model_kernels.configure_model_spec(
            model_spec,
            self._tpu_kernels,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            ep_token_chunks=tpu_options.ep_token_chunks,
            ep_paired_grouped_mm=bool(tpu_options.ep_paired_grouped_mm),
        )
        if setup is not None and setup.configure is not None:
            setup.configure(
                model_spec,
                tpu_model_kernels.ModelSetupContext(
                    expert_parallel_size=self.engine_config.expert_parallel_size,
                    seq_bucket_size=get_tpu_seq_bucket_size(self.engine_config),
                    moe_paired_grouped_mm=bool(tpu_options.moe_paired_grouped_mm),
                    moe_native_weight_layout=bool(tpu_options.moe_native_weight_layout),
                ),
            )
        if self._use_simple_fsdp:
            model_spec = dataclasses.replace(
                model_spec,
                parallelize_fn=get_tpu_simple_fsdp_parallelize_fn(torchtitan_name),
            )

        # torch_tpu's aten::_fused_adamw_ updates params, exp_avg and exp_avg_sq in place (buffer donation): no
        # param-sized temporary as in the foreach path (+4.2 GB/chip for Qwen3-4B at FSDP=4) and ~2x faster.
        # Under 2D (dp_shard, tp) SimpleFSDP, parameters have heterogeneous 2D DTensor placements
        # ((_StridedShard(0), Shard(0)), (Shard(0), Shard(1)), (Replicate(), Replicate())), so default to "for-loop".
        default_optim_impl = "for-loop" if self.engine_config.tensor_parallel_size > 1 else "fused"
        optim_impl = tpu_options.optim_impl or default_optim_impl
        optimizer = OptimizersContainer.Config(
            implementation=optim_impl,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=self.optimizer_config.name,
                    optimizer_kwargs={
                        "lr": self.optimizer_config.lr,
                        "eps": self.optimizer_config.eps,
                        "betas": (self.optimizer_config.betas[0], self.optimizer_config.betas[1]),
                        "weight_decay": self.optimizer_config.weight_decay,
                    },
                )
            ],
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism_kwargs: dict[str, Any] = {
            "data_parallel_replicate_degree": self.engine_config.data_parallel_replicate_size,
            "data_parallel_shard_degree": self.engine_config.data_parallel_shard_size,
            "fsdp_reshard_after_forward": self.engine_config.reshard_after_forward,
            "tensor_parallel_degree": self.engine_config.tensor_parallel_size,
            "pipeline_parallel_degree": self.engine_config.pipeline_parallel_size,
            "context_parallel_degree": self.engine_config.context_parallel_size,
            "expert_parallel_degree": self.engine_config.expert_parallel_size,
            "spmd_backend": self.engine_config.spmd_backend,
            "use_simple_fsdp": self._use_simple_fsdp,
        }
        topo_mesh = tpu_ragged_ep.should_enable_tpu_topo_mesh(tpu_options.enable_tpu_topo_mesh, ParallelismConfig)
        if topo_mesh is not None:
            parallelism_kwargs["enable_tpu_topo_mesh"] = topo_mesh
        parallelism = ParallelismConfig(**parallelism_kwargs)
        if parallelism.context_parallel_degree > 1:
            self._enable_context_parallel(parallelism, model_spec.model)
        async_save = not self.engine_config.forward_only and (
            bool(getattr(self.checkpoint_config, "async_save", False))
            or os.environ.get("VERL_TORCHTITAN_ASYNC_CHECKPOINT", "0") == "1"
        )
        # interval=1: save whenever verl calls save_checkpoint (TorchTitan's default 500 skips the other steps).
        checkpoint = CheckpointManager.Config(
            enable=True,
            interval=1,
            initial_load_in_hf=True,
            initial_load_path=model_config.path,
            async_mode="async" if async_save else "disabled",
        )
        compile_config = CompileConfig(enable=self.engine_config.use_torch_compile, backend="tpu")

        # On TPU, FSDP2 CPUOffloadPolicy is incompatible with torch_tpu SPMD / PJRT sharding,
        # and CUDA graphs are not applicable.
        training_kwargs: dict[str, Any] = {
            "enable_cpu_offload": False,
            "disable_cuda_graphs": True,
            "max_norm": self.optimizer_config.clip_grad,
        }
        if tpu_options.mixed_precision_param is not None:
            training_kwargs["mixed_precision_param"] = tpu_options.mixed_precision_param
        if self.engine_config.forward_only:
            # dtype controls parameter storage; mixed_precision_param controls
            # FSDP's compute/all-gather dtype. Preserve both independent choices.
            engine_dtype = getattr(self.engine_config, "dtype", "bfloat16")
            if engine_dtype in ("bfloat16", "float32"):
                training_kwargs["dtype"] = engine_dtype
        elif tpu_options.param_dtype == "bfloat16":
            # Pure bf16 training (engine_kwargs.tpu.param_dtype): bf16 parameters, gradients and optimizer states.
            training_kwargs["dtype"] = "bfloat16"
            training_kwargs["mixed_precision_reduce"] = "bfloat16"
        if self.engine_config.max_seq_len is not None:
            training_kwargs["max_context_length"] = self.engine_config.max_seq_len
        mixed_precision_reduce = resolve_tpu_mixed_precision_reduce(engine_config)
        if mixed_precision_reduce is not None:
            training_kwargs["mixed_precision_reduce"] = mixed_precision_reduce
        training = TrainingConfig(**training_kwargs)

        # Pad positions run 0..pad_size-1 (< bucket) and index the RoPE cache sized by max_context_length.
        if self.engine_config.pad_to_length and self.engine_config.pad_to_length_bucket > training.max_context_length:
            raise ValueError(
                f"pad_to_length_bucket ({self.engine_config.pad_to_length_bucket}) must not exceed the RoPE "
                f"cache length ({training.max_context_length}); raise engine.max_seq_len or lower the bucket"
            )

        # Activation checkpointing mode. Under spmd_backend="spmd_types" with eager execution
        # (use_torch_compile=False), selective/full AC recompute runs on the autograd backward thread
        # where the thread-local SPMD mesh is inactive, so spmd.assert_type() raises "no current mesh".
        ac_mode = self.engine_config.activation_checkpoint
        if parallelism.spmd_backend == "spmd_types" and not self.engine_config.use_torch_compile and ac_mode != "none":
            logger.warning(
                "Disabling activation_checkpoint=%r because spmd_backend='spmd_types' with "
                "use_torch_compile=False does not support eager AC recomputation.",
                ac_mode,
            )
            ac_mode = "none"
        if self.engine_config.forward_only:
            activation_checkpoint = None
        elif tpu_options.ac_save_splash_and_routing and ac_mode != "none":
            activation_checkpoint = _splash_and_routing_selective_ac_config()
        else:
            activation_checkpoint = {
                # torchtitan's TPUSelectiveAC drops collectives from the selective-AC save set so backward
                # recomputes them with their wait_tensor, avoiding torch-tpu's wait-on-placeholder error.
                "selective": TPUSelectiveAC.Config,
                "full": FullAC.Config,
                "none": lambda: None,
            }[ac_mode]()

        # Construct Torchtitan's Trainer.Config
        trainer_config_kwargs: dict[str, Any] = dict(
            model_spec=model_spec,
            hf_assets_path=self.model_config.path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile_config,
            training=training,
            activation_checkpoint=activation_checkpoint,
            # Use a no-op dataloader since verl has its own data loading
            dataloader=create_noop_dataloader_config(),
            # Provide a concrete loss so Trainer.__init__ can build it;
            # verl uses its own loss function and ignores this one.
            loss=CrossEntropyLoss.Config(),
        )
        if self._graph_trainer_mode:
            # TpuGraphTrainer.Config: the same fields, a GraphTrainerCompileConfig (aot_fx_trace), the TPU graph-trainer
            # settings and no eager activation checkpointing (the graph's memory policy replaces it).
            self.config = graph_trainer_mode.build_graph_trainer_config(
                trainer_config_kwargs, tpu_options, activation_checkpoint=ac_mode
            )
        else:
            self.config = Trainer.Config(**trainer_config_kwargs)
        if self._use_fused_linear_cross_entropy:
            self._enable_fused_logprob_head()
        logger.warning(
            "TPU log-prob path: %s",
            "fused Tokamax linear + log-softmax (Pallas forward, chunked backward)"
            if self._use_fused_linear_cross_entropy
            else "LM-head logits + chunked log-softmax",
        )
        self._gdn_row_slot_len = int(tpu_options.gdn_row_slot_len or 0)
        if self._gdn_row_slot_len:
            if not self._use_fused_linear_cross_entropy:
                raise ValueError("gdn_row_slot_len needs the fused log-prob path (use_fused_linear_cross_entropy)")
            if parallelism.context_parallel_degree > 1:
                raise ValueError(
                    "gdn_row_slot_len is not supported with context parallelism; "
                    "use gdn_auto_row_slot_len with context_parallel_strategy='ulysses'"
                )
            # Each gdn_row_slot_len-token row holds one sequence, so the GatedDeltaNet needs no document resets.
            self.config.override.imports.append(
                ("torchtitan.experiments.tpu.qwen3_5.gated_delta.gated_delta_document_resets", {"enabled": False})
            )
        self._cp_full_positions = bool(tpu_options.cp_full_positions)
        self._gdn_auto_row_slot_len = int(tpu_options.gdn_auto_row_slot_len or 0)
        self._sort_micro_batches_by_slot = bool(tpu_options.sort_micro_batches_by_slot)
        if self._gdn_auto_row_slot_len:
            if not isinstance(self, TorchTitanTPUEngineWithLMHead) or not self._use_fused_linear_cross_entropy:
                raise ValueError(
                    "gdn_auto_row_slot_len requires TorchTitanTPUEngineWithLMHead with use_fused_linear_cross_entropy"
                )
            if self._graph_trainer_mode:
                raise ValueError("gdn_auto_row_slot_len is not supported with trainer_mode='graph_trainer'")
            if parallelism.tensor_parallel_degree > 1:
                raise ValueError("gdn_auto_row_slot_len does not support tensor parallelism")
            if parallelism.context_parallel_degree > 1 and parallelism.context_parallel_load_balancer is not None:
                raise ValueError(
                    "gdn_auto_row_slot_len under context parallelism requires contiguous token sharding "
                    "(context_parallel_strategy='ulysses', load_balancer=None)"
                )
            seq_bucket_sz = get_tpu_seq_bucket_size(self.engine_config)
            if not tpu_options.static_micro_batch_len and seq_bucket_sz % self._gdn_auto_row_slot_len != 0:
                logger.warning(
                    "gdn_auto_row_slot_len=%d is set without static_micro_batch_len and seq_bucket_size=%d is not "
                    "divisible by gdn_auto_row_slot_len; micro-batches whose bucketed length is not a multiple of "
                    "%d will fall back to single-doc or segmented execution.",
                    self._gdn_auto_row_slot_len,
                    seq_bucket_sz,
                    self._gdn_auto_row_slot_len,
                )
        if tpu_options.fuse_gdn_projections:
            if parallelism.tensor_parallel_degree > 1:
                raise ValueError("fuse_gdn_projections does not support tensor parallelism")
            gdn_slot_schedule.enable_fused_gdn_projections(self.config.model_spec.model)
        for override_import in tpu_model_kernels.override_imports(self._tpu_kernels):
            if override_import not in self.config.override.imports:
                self.config.override.imports.append(override_import)
        if self._graph_trainer_mode:
            self.trainer = graph_trainer_mode.build_rl_graph_trainer(self.config)
        else:
            self.trainer = Trainer(self.config)
        if engine_config.use_torch_compile:
            tpu_model_kernels.patch_torchtitan_tpu_dynamo_caches()
        self.parallel_dims = self.trainer.parallel_dims
        self._dp_group = self._build_dp_gloo_group()
        # Like torchtitan's init_distributed (which verl bypasses): no autograd multithreading, so backward-thread
        # activation-checkpoint recompute can access the thread-local SPMD mesh / process groups.
        torch.autograd.set_multithreading_enabled(False)

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        model_parts = self.trainer.model_parts
        gdn_slot_schedule.validate_model_cp_slot_support(
            model_parts,
            need_cp_full_positions=self._cp_full_positions and self.parallel_dims.cp_enabled,
            need_document_resets_override=bool(self._gdn_auto_row_slot_len),
        )
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        apply_tpu_precision_patches(model_parts)
        if tpu_options.fp32_logits and not self._use_fused_linear_cross_entropy:
            apply_fp32_logits_tpu(model_parts)
        if self._use_splash_attention:
            set_splash_max_block_size(resolve_tpu_splash_max_block_size(engine_config))
        install_tpu_attention(model_parts, use_splash=self._use_splash_attention)
        if self.parallel_dims.ep_enabled:
            ep_cap = (
                tpu_options.ep_capacity_factor
                if tpu_options.ep_capacity_factor is not None
                else tpu_model_kernels.DEFAULT_EP_CAPACITY_FACTOR
            )
            ep_mode, n_ep = tpu_ragged_ep.install_tpu_ep_token_dispatcher(
                model_parts,
                ep_dispatcher=tpu_options.ep_dispatcher,
                capacity_factor=ep_cap,
                spmd_backend=parallelism.spmd_backend,
                compile_config=compile_config,
            )
            if n_ep:
                logger.warning(
                    "Configured %s TPU EP token dispatcher (capacity_factor=%.2f) on %d modules",
                    ep_mode,
                    ep_cap,
                    n_ep,
                )
            if engine_config.use_torch_compile and ep_mode == "static":
                n_barriers = tpu_model_kernels.install_ep_chunk_barriers(model_parts)
                if n_barriers:
                    logger.info("Installed EP chunk window barriers on %d routed-experts modules", n_barriers)

        if self._graph_trainer_mode:
            # After every module swap above: annotate the final module tree for the graph passes.
            graph_trainer_mode.prepare_model_for_graph_trainer(model_parts)

        if engine_config.use_torch_compile:
            n_compiled = sum(
                getattr(m, "_compiled_call_impl", None) is not None
                for part in model_parts
                if isinstance(part, torch.nn.Module)
                for m in part.modules()
            )
            logger.warning("torch.compile(backend='tpu') enabled on %d modules", n_compiled)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params. ``ref.torchtitan.param_offload=True`` keeps the frozen reference model in host
        # memory outside compute_ref_log_prob: frees its bf16 shard of HBM, costs one host-to-TPU copy per step.
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._simple_fsdp_loss_scale = 1.0 / self.get_data_parallel_size() if self._use_simple_fsdp else None

        entropy_fn = (
            verl_F.entropy_from_logits_with_chunking
            if self.engine_config.entropy_from_logits_with_chunking
            else verl_F.entropy_from_logits
        )
        self.compute_entropy_from_logits = (
            torch.compile(entropy_fn, backend="tpu", dynamic=False)
            if self.engine_config.use_torch_compile
            else entropy_fn
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if timeline.trainer_lane() is not None:
            timeline.start_hbm_sampler(group=timeline.TRAINER_WORKERS, lane=timeline.HBM_LANE)

        logger.info(
            "TorchTitanTPUEngine initialized (compile=%s, splash=%s, simple_fsdp=%s, eager_mode=%s)",
            engine_config.use_torch_compile,
            self._use_splash_attention,
            self._use_simple_fsdp,
            self._tpu_eager_mode,
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self) -> bool:
        """Whether the current rank is the first rank in the model-parallel group that holds model outputs."""
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def _enable_context_parallel(self, parallelism: Any, model: Any) -> None:
        """Turn on TPU context parallelism the way a torchtitan CP recipe does.

        Like ``torchtitan_recipes/muse_glimmer.py`` and the TPU ``qwen3_*_cp*`` recipes: set the SPMD backend and
        the load balancer, swap in the TPU CP inner attention with core ``ContextParallelTransform`` (plus the CP
        transforms of the model's non-attention sequence mixers, see ``cp_layer_transforms``), then check the
        TPU-specific rules. Core ``Trainer.Config`` validates the rest when it is built. The strategy comes from
        ``engine_kwargs.tpu.context_parallel_strategy`` so verl keeps CP as a config option.
        """
        from torchtitan.config.transform import ContextParallelTransform, transform_model_config_

        cp_attention = _import_tpu_cp_attention()
        install_tpu_cp_patches = cp_attention.install_tpu_cp_patches
        tpu_cp_inner_attention = cp_attention.tpu_cp_inner_attention
        tpu_cp_load_balancer = cp_attention.tpu_cp_load_balancer
        validate_tpu_context_parallel = cp_attention.validate_tpu_context_parallel

        tpu_options = get_tpu_engine_options(self.engine_config)
        strategy = tpu_options.context_parallel_strategy or "all_gather"
        if strategy == "all_gather" and not self._use_splash_attention:
            # torchtitan refuses to run its plain-PyTorch reference attention for KV all-gather CP on TPU: it builds
            # the full (S / cp, S) score matrix. Fail here, before the model is built.
            raise ValueError(
                "context_parallel_strategy='all_gather' on TPU needs use_splash_attention=True: torchtitan's KV "
                "all-gather CP has no other TPU attention kernel. Turn splash on, or use "
                "context_parallel_strategy='ulysses'."
            )
        if tpu_options.gdn_row_slot_len:
            raise ValueError(
                f"gdn_row_slot_len is not supported with context_parallel_size > 1 "
                f"(got gdn_row_slot_len={tpu_options.gdn_row_slot_len}, context_parallel_strategy={strategy!r}): "
                "static gdn_row_slot_len permutes input_ids and positions into row slots without permuting labels "
                "before cp_shard, and does not post-gather-unpermute outputs across CP shards. "
                "Unset gdn_row_slot_len or turn context parallelism off."
            )
        parallelism.spmd_backend = "spmd_types"
        load_balancer = tpu_cp_load_balancer(strategy)
        parallelism.context_parallel_load_balancer = load_balancer
        inner_attention = model.first_full_attention_backend
        cp_inner_attention = tpu_cp_inner_attention(inner_attention._owner, strategy)
        # The non-attention sequence mixers the model contains (e.g. Qwen3.5's GatedDeltaNet) run over the same CP
        # token layout through their own transforms, as torchtitan's CP recipes add them (cp_layer_transforms).
        # Raises if the model needs one that the installed torchtitan does not ship.
        layer_transforms = context_parallel_layer_transforms(model, load_balancer, tpu_options)
        transform_model_config_(
            model, [ContextParallelTransform(inner_attention=cp_inner_attention), *layer_transforms]
        )
        validate_tpu_context_parallel(model, parallelism)
        # Runtime FSDP workaround for torch without spmd_types FSDP (see install_tpu_cp_patches).
        install_tpu_cp_patches()

    def _cp_attention_owner(self) -> Any:
        """Return the CP inner-attention class whose ``cp_shard`` shards model inputs.

        Mirrors torchtitan's ``Decoder._cp_shard_inputs``: ``_enable_context_parallel`` swaps the inner attention
        config for a CP variant when ``cp > 1``, and that variant's owner class knows how its strategy shards inputs.
        """
        inner_attention = self.config.model_spec.model.first_full_attention_backend
        owner = getattr(inner_attention, "_owner", None)
        if owner is None or not callable(getattr(owner, "cp_shard", None)):
            raise RuntimeError(
                f"Context parallelism is enabled but the inner attention {type(inner_attention).__name__} "
                "has no cp_shard; _enable_context_parallel must run before the model is built."
            )
        return owner

    def _gather_context_parallel(self, tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        """All-gather a per-token tensor sharded along dim 0 by CP and restore the original token order."""
        load_balancer_type = self.trainer.config.parallelism.context_parallel_load_balancer
        if load_balancer_type not in (None, "headtail"):
            raise NotImplementedError(f"context_parallel_load_balancer={load_balancer_type!r} is not supported")
        gathered = ContextParallelGather.apply(tensor, self.get_context_parallel_group())
        if load_balancer_type == "headtail":
            gathered = gathered.index_select(0, self._cp_restore_indices(seq_len, gathered.device))
        return gathered

    def _cp_restore_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """The head-tail restore permutation for ``seq_len`` tokens on ``device``, built once per shape.

        Built on CPU and copied to the device with ``non_blocking=True`` like the other CPU-built index tensors of a
        micro-batch (e.g. the labels in ``prepare_model_inputs``), avoiding a blocking H2D stream cut after
        ``ContextParallelGather`` on cache miss so the first micro-batch of a shape fuses like the others.
        """
        key = (int(seq_len), int(self.parallel_dims.cp), device)
        restore_idx = self._cp_restore_idx_cache.get(key)
        if restore_idx is None:
            restore_idx = _headtail_restore_indices_cpu(int(seq_len), int(self.parallel_dims.cp)).to(
                device=device, non_blocking=True
            )
            self._cp_restore_idx_cache[key] = restore_idx
        return restore_idx

    def _build_dp_gloo_group(self):
        """Build a host-side Gloo ProcessGroup per data-parallel slice when model parallelism is active.

        Subgroups created by ``init_device_mesh("tpu", ...)`` register only ``tpu_dist``, which compiles
        subgroup collectives (e.g. ``replica_groups={{0,2},{1,3}}``) into XLA SPMD programs with a hardware
        barrier across all TPU chips in the world. ``TrainingWorker._postprocess_output`` executes
        ``all_reduce(loss, op=AVG, group=dp_group)`` and ``all_gather_object(..., group=dp_group)`` only on
        ``is_mp_src_rank_with_outputs()`` ranks (``tp_rank == 0``), so a ``tpu_dist`` subgroup collective
        stalls waiting for ``tp_rank > 0`` chips. Using a host-side ``gloo`` group for ``get_data_parallel_group()``
        when ``dp_size < world_size`` lets ``tp_rank == 0`` ranks aggregate metrics over TCP independently.
        """
        pd_dict = getattr(self.parallel_dims, "__dict__", {})
        if "_verl_dp_gloo_group" in pd_dict:
            return pd_dict["_verl_dp_gloo_group"]
        if not torch.distributed.is_initialized():
            return None
        dp_size = self.get_data_parallel_size()
        world_size = torch.distributed.get_world_size()
        if dp_size >= world_size:
            return None
        try:
            default_backend = str(torch.distributed.get_backend(None))
        except Exception:
            default_backend = ""
        if "gloo" not in default_backend:
            return None

        pp = self.parallel_dims.pp
        cp = self.parallel_dims.cp
        tp = self.parallel_dims.tp
        if pp * dp_size * cp * tp != world_size:
            return None

        dataloading_mesh = getattr(self.parallel_dims, "_global_meshes", {}).get("dataloading")
        if dataloading_mesh is not None and hasattr(dataloading_mesh, "mesh"):
            grid = dataloading_mesh.mesh.detach().cpu().reshape(pp, dp_size, cp, tp)
        else:
            grid = torch.arange(world_size, dtype=torch.int64).reshape(pp, dp_size, cp, tp)

        rank = torch.distributed.get_rank()
        my_group = None
        for pp_idx in range(pp):
            for cp_idx in range(cp):
                for tp_idx in range(tp):
                    ranks = grid[pp_idx, :, cp_idx, tp_idx].tolist()
                    pg = torch.distributed.new_group(ranks=ranks, backend="gloo")
                    if rank in ranks:
                        my_group = pg
        self.parallel_dims._verl_dp_gloo_group = my_group
        return my_group

    def _drop_checkpoint_state_dict_cache(self) -> None:
        """Free torchtitan ``ModelWrapper.cached_state_dict`` after checkpoint load or save."""
        for wrapper in self.checkpointer.states.values():
            wrapper.cached_state_dict = {}

    def initialize(self):
        """Initialize model/checkpointer and release unused optimizer state on forward-only reference workers."""
        self.module = self.trainer.model_parts
        self.checkpointer = self.trainer.checkpointer
        _use_threaded_hf_reader(self.checkpointer)
        # load initial HF weights
        self.checkpointer.load()
        tpu_synchronize(wait=True)

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
            if self._tpu_options.frozen_param_regex:
                pattern = re.compile(self._tpu_options.frozen_param_regex)
                self._frozen_params = [p for m in self.module for n, p in m.named_parameters() if pattern.search(n)]
                logger.info(
                    "frozen_param_regex=%r freezes %d parameters",
                    self._tpu_options.frozen_param_regex,
                    len(self._frozen_params),
                )
        else:
            self.optimizer = None
            self.lr_scheduler = None
            for part in self.module:
                for p in part.parameters():
                    p.requires_grad_(False)

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        self._drop_checkpoint_state_dict_cache()
        if getattr(self.engine_config, "use_torch_compile", False) and self.parallel_dims.tp_enabled:
            install_sync_dtensor_redistribute_for_tpu()

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return TPUEngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return TPUEngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        if getattr(self, "_dp_group", None) is not None:
            return self._dp_group
        mesh = self._get_data_parallel_mesh()
        if mesh is not None:
            return mesh.get_group()
        # If world_size == dp_size (e.g. single GPU, or all ranks are DP),
        # return WORLD so that collective ops in _postprocess_output
        # (allgather_dict_into_dict, all_reduce) still run and produce the
        # correct metric aggregation format.
        if torch.distributed.get_world_size() == self.get_data_parallel_size():
            return torch.distributed.group.WORLD
        return None

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        mesh = self.parallel_dims.get_optional_mesh("cp")
        return None if mesh is None else mesh.get_group()

    def _get_data_parallel_mesh(self):
        """Data-parallel mesh (dp_replicate x dp_shard). Excludes cp: CP ranks share the same samples."""
        return self.parallel_dims.get_optional_mesh("batch")

    def train_batch(self, data: TensorDict, loss_function: Callable) -> dict[str, Any]:
        with timeline.span("update_actor", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()):
            return super().train_batch(data, loss_function)

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> dict[str, Any]:
        span_name = "compute_ref_log_prob" if self.engine_config.forward_only else "compute_log_prob"
        with timeline.span(span_name, group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()):
            return super().infer_batch(data, loss_function)

    def _allow_auto_row_slot(self) -> bool:
        """Whether dynamic per-micro-batch GDN row-slot scheduling (``gdn_auto_row_slot_len``) is active."""
        auto_slot_len = int(getattr(self, "_gdn_auto_row_slot_len", 0) or 0)
        slot_len = int(getattr(self, "_gdn_row_slot_len", 0) or 0)
        if (
            auto_slot_len <= 0
            or slot_len > 0
            or bool(getattr(self, "_graph_trainer_mode", False))
            or not isinstance(self, TorchTitanTPUEngineWithLMHead)
            or not bool(getattr(self, "_use_fused_linear_cross_entropy", False))
            or bool(getattr(self.parallel_dims, "tp_enabled", False))
        ):
            return False
        if bool(getattr(self.parallel_dims, "cp_enabled", False)):
            parallelism = getattr(getattr(getattr(self, "trainer", None), "config", None), "parallelism", None)
            if getattr(parallelism, "context_parallel_load_balancer", None) is not None:
                return False
        return True

    def _effective_mb_slot_len(
        self,
        micro_batch: TensorDict,
        *,
        allow_auto_row_slot: bool,
        forward_only: bool,
        padded_seq_len: Optional[int] = None,
    ) -> int:
        """Resolved per-micro-batch GDN unflatten slot length, gated on ``allow_auto_row_slot`` and entropy mode."""
        if not allow_auto_row_slot:
            return 0
        mb_slot_len = int(
            tu.get_non_tensor_data(data=micro_batch, key=gdn_slot_schedule.MB_SLOT_LEN_KEY, default=0) or 0
        )
        if mb_slot_len <= 0:
            return 0
        use_rmpad = bool(
            unwrap_metadata(tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True))
        )
        if not use_rmpad:
            return 0
        if padded_seq_len is None:
            ids = micro_batch["input_ids"]
            orig_len = int(ids.values().numel()) if getattr(ids, "is_nested", False) else int(ids.numel())
            bucket_sz = get_tpu_seq_bucket_size(self.engine_config)
            padded_seq_len = max(
                bucket_length(orig_len, bucket_sz),
                int(tu.get_non_tensor_data(data=micro_batch, key="tpu_padded_seq_len", default=0) or 0),
            )
        if padded_seq_len < mb_slot_len or padded_seq_len % mb_slot_len != 0:
            return 0
        ids = micro_batch["input_ids"]
        if getattr(ids, "is_nested", False):
            offs = ids.offsets()
            n_seqs = int(offs.numel() - 1)
            max_seq = int(offs.diff().max().item()) if n_seqs > 0 else 0
            if n_seqs > padded_seq_len // mb_slot_len or max_seq > mb_slot_len:
                return 0
        calculate_entropy = bool(
            unwrap_metadata(tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False))
        )
        if mb_slot_len < padded_seq_len and calculate_entropy and not forward_only:
            return 0
        return mb_slot_len

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False):
        """Perform forward and optionally backward pass with host-side token counting and XLA graph break."""
        graph_mode = getattr(self, "_graph_trainer_mode", False)
        if graph_mode and forward_only:
            # Phase 1 runs in bypass mode with the KL loss off, so the actor never needs a forward-only pass. Without
            # per-block compile there is no fast eager forward to fall back to (TODO-3).
            raise graph_trainer_mode.forward_only_error()
        tu.assign_non_tensor(
            data, sp_size=self.engine_config.tensor_parallel_size * self.engine_config.context_parallel_size
        )

        dp_group = self.get_data_parallel_group()
        batch_num_tokens = compute_global_batch_num_tokens(
            data, dp_group, self.engine_config.tensor_parallel_size, self.engine_config.context_parallel_size
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens)
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())
        # Once per mini-batch, before the micro-batch loop (and its deferred eager-mode region): the per-step loss
        # scalars as 0-D device tensors, so the loss programs do not bake them in as constants.
        loss_scalars = device_loss_scalars(data, get_device_id) if loss_function is not None else {}

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=dp_group,
            same_micro_num_in_dp=True,
        )
        tpu_options = get_tpu_engine_options(self.engine_config)
        static_len = int(tpu_options.static_micro_batch_len or 0)
        if forward_only and tpu_options.static_log_prob_micro_batch_len:
            static_len = int(tpu_options.static_log_prob_micro_batch_len)
        static_response_len = int(tpu_options.static_max_response_len or 0)
        slot_len = int(getattr(self, "_gdn_row_slot_len", 0) or 0)
        # Pad the PPO loss rows to a common bucket only for losses that provably ignore fully masked rows.
        bucket_rows = tpu_options.loss_row_bucketing is not False and loss_rows_can_be_padded(loss_function)
        static_num_seqs = int(tpu_options.static_micro_batch_num_seqs or 0)
        auto_slot_len = int(getattr(self, "_gdn_auto_row_slot_len", 0) or 0)
        bucket_sz = get_tpu_seq_bucket_size(self.engine_config)
        allow_auto_row_slot = self._allow_auto_row_slot()
        effective_forward_only = forward_only or not torch.is_grad_enabled()
        if (
            getattr(self, "_sort_micro_batches_by_slot", False)
            and indices is not None
            and len(micro_batches) > 1
            and not slot_len
        ):
            micro_batches, indices = gdn_slot_schedule.sort_micro_batches_for_slots(
                micro_batches,
                indices,
                auto_slot_len=auto_slot_len,
                static_len=static_len,
                bucket_size=bucket_sz,
                forward_only=effective_forward_only,
                allow_row_slot=allow_auto_row_slot,
            )
        if (
            self._use_simple_fsdp
            or self.parallel_dims.cp_enabled
            or static_len
            or static_response_len
            or slot_len
            or allow_auto_row_slot
        ):
            align_micro_batch_shapes_across_ranks(
                micro_batches,
                bucket_size=bucket_sz,
                static_len=static_len,
                static_response_len=static_response_len,
                gdn_row_slot_len=slot_len,
                num_seqs_bucketing=bucket_rows,
                static_num_seqs=static_num_seqs,
            )
        elif bucket_rows:
            align_micro_batch_num_seqs(micro_batches, static_num_seqs=static_num_seqs)
        if allow_auto_row_slot:
            gdn_slot_schedule.plan_micro_batch_gdn_slots(
                micro_batches,
                auto_slot_len=auto_slot_len,
                forward_only=effective_forward_only,
                allow_row_slot=True,
                allow_single_doc=True,
            )

        if "kda" in getattr(self, "_tpu_kernels", ()) and not getattr(self, "_gdn_row_slot_len", 0):
            # Only with Tokamax's GatedDeltaNet kernels (google-pytorch/torchtitan#74): count only micro-batches that
            # still run segmented, and synchronize the max document budget across all SPMD ranks.
            segmented_mbs = [
                mb
                for mb in micro_batches
                if self._effective_mb_slot_len(
                    mb,
                    allow_auto_row_slot=allow_auto_row_slot,
                    forward_only=effective_forward_only,
                )
                == 0
            ]
            local_max_docs = max((gdn_slot_schedule.segmented_document_count(mb) for mb in segmented_mbs), default=0)
            global_max_docs = gdn_slot_schedule.sync_max_segmented_documents(local_max_docs)
            if global_max_docs > 0:
                budget = tpu_model_kernels.packed_document_budget(global_max_docs)
                updated = tpu_model_kernels.set_packed_document_budget(self.module, budget)
                if updated:
                    logger.info("Tokamax GatedDeltaNet: max_num_segments=%d in %d kernels", budget, updated)

        if graph_mode:
            return self._graph_trainer_forward_backward(
                data, micro_batches, indices, loss_function, batch_num_tokens, loss_scalars
            )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()
        timeline_ctx = (
            timeline.span(
                "fwd_bwd",
                group=timeline.TRAINER_WORKERS,
                lane=timeline.trainer_lane(),
                micro_batches=len(micro_batches),
                tokens=batch_num_tokens,
            )
            if not forward_only
            else nullcontext()
        )

        row_ctx = (
            gdn_slot_schedule.unflatten_seq_len_context(self._gdn_row_slot_len)
            if getattr(self, "_gdn_row_slot_len", 0)
            else nullcontext()
        )
        with timeline_ctx, tpu_eager_mode_context(self._tpu_eager_mode), row_ctx:
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                mb_slot_len = self._effective_mb_slot_len(
                    micro_batch,
                    allow_auto_row_slot=allow_auto_row_slot,
                    forward_only=effective_forward_only,
                )
                mb_row_ctx: Any = (
                    gdn_slot_schedule.unflatten_seq_len_context(mb_slot_len) if mb_slot_len > 0 else nullcontext()
                )
                with (
                    self.trainer.train_context(),
                    ctx,
                    mb_row_ctx,
                    torch.profiler.record_function(f"micro_batch{micro_batch_idx}"),
                ):
                    loss, output = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only, loss_scalars=loss_scalars
                    )
                    if not forward_only:
                        with spmd_no_typecheck_context():
                            if self._simple_fsdp_loss_scale is not None:
                                loss = loss * self._simple_fsdp_loss_scale
                            tpu_synchronize(loss, wait=False)
                            loss.backward()
                            del loss
                            tpu_synchronize(wait=True)
                    else:
                        del loss
                output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def _graph_trainer_forward_backward(
        self,
        data: TensorDict,
        micro_batches: list[TensorDict],
        indices: Any,
        loss_function: Callable,
        batch_num_tokens: Any,
        loss_scalars: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        """``trainer_mode=graph_trainer``: the mini-batch's micro-batch through TorchTitan's traced training step.

        The micro-batch is prepared as in the eager engine (packed, bucket-padded to ``static_micro_batch_len``, same
        shape on every rank), so the one traced, compiled step serves every step.
        ``TpuGraphTrainer.forward_backward_step`` runs the model, the fused LM-head log-prob, the PPO policy loss and
        the backward as one executable and writes the gradients to ``param.grad`` for the unchanged
        ``optimizer_step``. It assigns rather than accumulates ``param.grad``, so exactly one micro-batch per mini-batch
        per rank is supported (TODO-1). The loss is normalized by the mini-batch's GLOBAL token count, as in the eager
        engine. verl's ``ppo_loss`` then runs eagerly under ``no_grad`` on the per-token log-probs the graph returns,
        so the reported loss and metrics are the eager engine's.
        """
        if len(micro_batches) != 1:
            raise graph_trainer_mode.multi_micro_batch_error(len(micro_batches))
        self._ensure_graph_policy_loss(loss_function, micro_batches[0])
        timeline_ctx = timeline.span(
            "fwd_bwd",
            group=timeline.TRAINER_WORKERS,
            lane=timeline.trainer_lane(),
            micro_batches=len(micro_batches),
            tokens=batch_num_tokens,
        )
        output_lst = []
        with timeline_ctx, tpu_eager_mode_context(self._tpu_eager_mode):
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                output_lst.append(
                    self._graph_trainer_micro_batch(
                        micro_batch, micro_batch_idx, loss_function, batch_num_tokens, loss_scalars
                    )
                )
        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def _graph_trainer_micro_batch(
        self,
        micro_batch: TensorDict,
        micro_batch_idx: int,
        loss_function: Callable,
        batch_num_tokens: Any,
        loss_scalars: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        """One micro-batch of ``_graph_trainer_forward_backward``: the traced step, then verl's loss and metrics."""
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)
        if extra_kwargs or set(extra_inputs) != {"positions"} or "orig_seq_len" not in output_args:
            raise NotImplementedError(
                "trainer_mode=graph_trainer supports packed text-only inputs (use_remove_padding=True, no "
                f"multi-modal or CP inputs), got {sorted(extra_inputs)} / {sorted(extra_kwargs)}"
            )
        padded_len = int(input_ids.shape[-1])
        if self._graph_trainer_traced_len is None:
            self._graph_trainer_traced_len = padded_len
        elif padded_len != self._graph_trainer_traced_len:
            raise graph_trainer_mode.shape_change_error(self._graph_trainer_traced_len, padded_len)

        # Fixed pytree with fixed shapes every step: the packed [T] RL loss inputs and a 0-d tensor token count (the
        # mini-batch's global count, the same for every micro-batch).
        old_log_probs, advantages, loss_mask = graph_trainer_mode.pack_rl_loss_targets(micro_batch, padded_len)
        device = input_ids.device
        labels = (
            output_args["labels"],
            old_log_probs.to(device),
            advantages.to(device),
            loss_mask.to(device),
        )
        # The same 0-D tensor as the eager loss (device_loss_scalars), else built here (not on TPU, e.g. CPU tests),
        # clamped to >= 1.0 when batch_num_tokens == 0 globally to match GraphPolicyLoss and avoid 0.0 / 0.0 = NaN.
        global_valid_tokens = (loss_scalars or {}).get("batch_num_tokens")
        if global_valid_tokens is None or global_valid_tokens.device != device:
            global_valid_tokens = torch.tensor(max(float(batch_num_tokens), 1.0), dtype=torch.float32, device=device)
        eager_scalars = (
            loss_scalars
            if loss_scalars and "batch_num_tokens" in loss_scalars
            else {**(loss_scalars or {}), "batch_num_tokens": global_valid_tokens}
        )
        input_dict = {"input": input_ids, "labels": labels, "positions": extra_inputs["positions"]}
        # bf16 autocast as in the eager engine; it must be active while the step is traced (step 1), which records the
        # casts into the graph.
        with (
            torch.profiler.record_function(f"micro_batch{micro_batch_idx}"),
            torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16),
        ):
            graph_loss = self.trainer.forward_backward_step(
                input_dict=input_dict, global_valid_tokens=global_valid_tokens
            )
        (log_probs,) = self.trainer.last_loss_aux

        # verl's loss and metrics, eagerly on the graph's log-probs (same code path as the eager engine's outputs).
        with (
            torch.no_grad(),
            torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16),
            spmd_no_typecheck_context(),
        ):
            model_output = {
                "log_probs": _to_jagged(log_probs, micro_batch["input_ids"].offsets(), output_args["orig_seq_len"])
            }
            loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch, eager_scalars)
            loss, metrics = loss_function(
                model_output=loss_model_output, data=loss_data, dp_group=self.get_data_parallel_group()
            )
            _add_rollout_consistency_metrics(loss_function, loss_model_output, loss_data, metrics)
            for value in model_output.values():
                if hasattr(value, TPU_PADDED_VALUES_ATTR):
                    delattr(value, TPU_PADDED_VALUES_ATTR)
        verl_loss = loss.detach().item()
        graph_trainer_mode.check_loss_consistency(
            graph_loss.detach().float().item(),
            verl_loss,
            self.get_data_parallel_size(),
            raise_on_mismatch=os.getenv("VERL_TPU_LOSS_CONSISTENCY_RAISE", "1") != "0",
        )
        return {"model_output": detach_tree(model_output), "loss": verl_loss, "metrics": metrics}

    def _ensure_graph_policy_loss(self, loss_function: Callable, micro_batch: TensorDict) -> None:
        """Installs the traced PPO loss as ``TpuGraphTrainer.loss_fn`` on the first step; later steps must match it.

        Its constants (clip ratios, temperature) are baked into the traced graph, so a change would be silently ignored.
        """
        spec = graph_trainer_mode.GraphPolicyLossSpec.from_loss_function(
            loss_function,
            temperature=float(unwrap_metadata(micro_batch["temperature"])),
            calculate_entropy=bool(
                unwrap_metadata(tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False))
            ),
            has_rollout_is_weights="rollout_is_weights" in micro_batch.keys(),
        )
        if self._graph_policy_loss_spec is None:
            self.trainer.loss_fn = graph_trainer_mode.GraphPolicyLoss(spec, self._tpu_kernels)
            self._graph_policy_loss_spec = spec
        elif spec != self._graph_policy_loss_spec:
            raise RuntimeError(
                f"trainer_mode=graph_trainer traced the policy loss with {self._graph_policy_loss_spec}, got {spec}"
            )

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
        return_dtensor: bool = False,
    ) -> torch.Tensor:
        """Perform a forward pass through the trainer model without backward."""
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )
        # Non-PP forward. train_context (SPMD mesh) is set by the caller.
        assert len(model_parts) == 1
        pred = model_parts[0](inputs, **(extra_inputs or {}), **(extra_kwargs or {}))
        if isinstance(pred, tuple):
            return tuple(p.full_tensor() if isinstance(p, DTensor) else p for p in pred)
        if isinstance(pred, DTensor) and not return_dtensor:
            pred = pred.full_tensor()
        return pred

    def forward_step(
        self,
        micro_batch: TensorDict,
        loss_function: Callable,
        forward_only: bool,
        loss_scalars: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    @torch.no_grad()
    def optimizer_step(self):
        """Clip gradients and step the optimizer, a few transformer layers at a time.

        Same math as torchtitan's ``clip_grad_norm_`` (global L2 norm, EP and PP aware) followed by
        ``OptimizersContainer.step()``, but split into many small XLA programs instead of one. Under
        ``DEFER_AND_FUSE`` torch_tpu records ops lazily and compiles everything up to the next sync into one program,
        so a single clip + ``optimizer.step()`` becomes one program over every parameter. XLA keeps all of its inputs
        (param, grad, exp_avg, exp_avg_sq) alive until all of its outputs (new param, exp_avg, exp_avg_sq) exist:
        for Qwen3.5-35B-A3B on 8 TensorCores (16.1 GiB of fp32 parameters per core) with ``optim_impl=foreach`` that
        is 7 x 16.1 = 112.5 GiB of temporaries, more than the 94.7 GiB HBM (``CompileTimeHbmOom``), on torchtitan main
        as well as on the optimized builds. torch_tpu's in-place ``fused`` AdamW fits in one program, but that program
        takes ~23 s per step there, against < 1 s for the chunks. Programs of 4 layers each stay small, and the
        repeating layer blocks give the same shapes, so XLA compiles each chunk program once. This is plugin-only;
        torchtitan's own trainer keeps bf16 parameters and AdamW state for this model and has no equivalent.

        1. ``_chunked_grad_norm``: the norm of each chunk's gradients, each its own program, combined into the global
           norm (one host sync).
        2. A non-finite norm skips the step, as before.
        3. ``_chunked_clip_and_step``: per chunk, clip with the global norm and step only that chunk's parameters,
           then sync and free its gradients.
        """
        with (
            timeline.span("optimizer_step", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()),
            tpu_eager_mode_context(self._tpu_eager_mode),
        ):
            for p in getattr(self, "_frozen_params", ()):
                p.grad = None  # frozen: out of the gradient norm, and AdamW skips parameters without a gradient
            param_chunks = self._optimizer_param_chunks()
            grad_norm = self._chunked_grad_norm(param_chunks)
            tpu_synchronize(grad_norm, wait=True)
            grad_norm_val = grad_norm.item()
            if not math.isfinite(grad_norm_val):
                logger.warning("grad_norm is not finite (%s); skipping this optimizer step", grad_norm_val)
                self.optimizer.zero_grad()
                if grad_norm.device.type == "tpu":
                    tpu_synchronize(wait=True)
                return grad_norm_val
            self._chunked_clip_and_step(param_chunks, grad_norm)
            return grad_norm_val

    def _optimizer_param_chunks(self) -> list[tuple[Any, list[tuple[dict, list, list[list]]]]]:
        """``[(optimizer, [(param_group, its params, chunks of its params)])]`` for every optimizer of the container.

        Chunks group parameters by their 4-layer block (from the FQNs), see ``_partition_params_into_chunks``.
        """
        name_by_id = {id(p): name for m in self.module for name, p in m.named_parameters()}
        result = []
        for opt in self.optimizer.optimizers:
            group_entries = []
            for group in opt.param_groups:
                params = list(group["params"])
                names = group.get("param_names")
                if names is None or len(names) != len(params):
                    names = [name_by_id.get(id(p), "") for p in params]
                group_entries.append((group, params, _partition_params_into_chunks(params, names)))
            result.append((opt, group_entries))
        return result

    def _chunked_grad_norm(self, param_chunks) -> torch.Tensor:
        """Global gradient L2 norm, computed one chunk at a time; like torchtitan's ``clip_grad_norm_``.

        With EP, expert (``ep`` mesh) and non-expert gradients live on different meshes and are reduced separately
        before combining, as torchtitan does. With PP, the squared norms are summed over the pipeline stages.
        """
        chunk_norms: list[torch.Tensor] = []
        for _, group_entries in param_chunks:
            for _, _, chunks in group_entries:
                for chunk in chunks:
                    grads = [p.grad for p in chunk if p.grad is not None]
                    if not grads:
                        continue
                    if self.parallel_dims.ep_enabled:
                        is_ep = [isinstance(g, DTensor) and "ep" in (g.device_mesh.mesh_dim_names or ()) for g in grads]
                        ep = [g for g, e in zip(grads, is_ep, strict=True) if e]
                        non_ep = [g for g, e in zip(grads, is_ep, strict=True) if not e]
                        grad_lists = [g_list for g_list in (ep, non_ep) if g_list]
                    else:
                        grad_lists = [grads]
                    for g_list in grad_lists:
                        norm = torch.nn.utils.get_total_norm(
                            g_list, norm_type=2.0, error_if_nonfinite=False, foreach=False
                        )
                        if isinstance(norm, DTensor):
                            norm = norm.full_tensor()
                        tpu_synchronize(norm, wait=False)  # dispatch now: one small program per chunk
                        chunk_norms.append(norm)
        if not chunk_norms:
            return torch.tensor(0.0, device=get_device_name())
        grad_norm = torch.nn.utils.get_total_norm(chunk_norms, norm_type=2.0, error_if_nonfinite=False, foreach=False)
        if (pp_mesh := self.parallel_dims.get_optional_mesh("pp")) is not None:
            grad_norm = grad_norm**2.0
            torch.distributed.all_reduce(grad_norm, op=torch.distributed.ReduceOp.SUM, group=pp_mesh.get_group())
            grad_norm = grad_norm**0.5
        return grad_norm

    def _chunked_clip_and_step(self, param_chunks, grad_norm: torch.Tensor) -> None:
        """Clips every chunk with the global ``grad_norm`` and steps the optimizers one chunk at a time.

        Each inner optimizer steps only the current chunk: its param groups are narrowed to the chunk's parameters
        for the call (and restored afterwards), so the optimizer state and hyperparameters are untouched. Stepping
        the inner optimizers bypasses ``OptimizersContainer.step()``, so its pre/post step hooks run here, e.g. the
        MoE load-balancing expert-bias update that torchtitan registers as a pre-step hook. It is registered only for
        MoE models that keep load balancing on; a ``ModelSetup`` may turn it off (e.g. Qwen3 MoE).
        """
        for hook in getattr(self.optimizer, "_optimizer_step_pre_hooks", {}).values():
            hook(self.optimizer, (), {})
        max_norm = self.config.training.max_norm
        tpu_options = getattr(self, "_tpu_options", None)
        host_offload = bool(tpu_options and tpu_options.optimizer_host_offload)
        masters = _fp32_masters_of(self)
        for opt, group_entries in param_chunks:
            for group, _, _ in group_entries:
                group["params"] = []
            try:
                for group, _, chunks in group_entries:
                    if masters is not None and masters.can_step_on_cpu(opt, group):
                        masters.step_group_chunks_on_cpu(group, chunks, max_norm=max_norm, grad_norm=grad_norm)
                        continue
                    for chunk in chunks:
                        if not any(p.grad is not None for p in chunk):
                            continue
                        torch.nn.utils.clip_grads_with_norm_(chunk, max_norm, grad_norm, foreach=False)
                        if masters is not None:
                            pairs = masters.begin(opt, chunk)
                            group["params"] = [master_param for _, master_param in pairs]
                            opt.step()
                            masters.write_back(pairs)
                            if grad_norm.device.type == "tpu":
                                tpu_synchronize(wait=True)
                            masters.end(opt, pairs)
                            group["params"] = []
                            continue
                        group["params"] = chunk
                        if host_offload:
                            _move_optimizer_state(opt, chunk, chunk[0].device)
                        opt.step()
                        if grad_norm.device.type == "tpu":
                            tpu_synchronize(wait=True)  # end this chunk's program before tracing the next one
                        if host_offload:
                            # The chunk's optimizer states live in host memory between steps (engine_kwargs.tpu.
                            # optimizer_host_offload), so only one chunk's states are ever on the device.
                            _move_optimizer_state(opt, chunk, "cpu")
                        for p in chunk:
                            p.grad = None
                        group["params"] = []
            finally:
                for group, params, _ in group_entries:
                    group["params"] = params
        for hook in getattr(self.optimizer, "_optimizer_step_post_hooks", {}).values():
            hook(self.optimizer, (), {})

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or accelerator."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        with tpu_eager_mode_context(self._tpu_eager_mode):
            if device == device_name:
                target_device = torch.device(f"{device_name}:{get_device_id()}")
                if model:
                    for module in self.module:
                        move_tpu_params_(module, target_device, include_grad=grad)
                if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                    load_fsdp_optimizer(self.optimizer, target_device)
            elif device == "cpu":
                if model:
                    for module in self.module:
                        move_tpu_params_(
                            module, "cpu", cache_cpu_copy=bool(self.engine_config.forward_only), include_grad=grad
                        )
                if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                    offload_fsdp_optimizer(self.optimizer)
                if model or optimizer:
                    gc.collect()
                    tpu_synchronize(wait=True)
            else:
                raise ValueError(f"Invalid device type: {device}")

    def _save_fp32_master_checkpoint(
        self,
        masters: _Fp32Masters,
        local_path: str,
        global_step: int,
        max_ckpt_to_keep: Optional[int],
    ) -> None:
        is_dist = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_dist else 0
        world_size = torch.distributed.get_world_size() if is_dist else 1
        if rank == 0 and _gcs_ckpt_blob_target(local_path) is None:
            os.makedirs(local_path, exist_ok=True)
        if is_dist:
            torch.distributed.barrier()

        saved_masters: dict[str, dict[str, Any]] = {}
        saved_params: dict[str, Any] = {}
        for idx, module in enumerate(self.module):
            for name, p in module.named_parameters():
                fqn = f"{idx}.{name}"
                entry = masters._entries.get(id(p))
                if entry is not None:
                    master, state = entry
                    saved_masters[fqn] = {
                        "master": _pack_ckpt_tensor(master),
                        "state": {k: _pack_ckpt_tensor(v) for k, v in state.items()},
                    }
                else:
                    saved_params[fqn] = _pack_ckpt_tensor(p)

        saved_buffers = {
            f"{idx}.{bname}": _pack_ckpt_tensor(b)
            for idx, module in enumerate(self.module)
            for bname, b in module.named_buffers()
        }
        optimizer_param_groups = (
            [
                [{k: v for k, v in g.items() if k not in ("params", "param_names")} for g in opt.param_groups]
                for opt in getattr(self.optimizer, "optimizers", ())
            ]
            if getattr(self, "optimizer", None) is not None
            else None
        )
        payload = {
            "format_version": 1,
            "global_step": int(global_step),
            "rank": rank,
            "world_size": world_size,
            "masters": saved_masters,
            "params": saved_params,
            "buffers": saved_buffers,
            "optimizer_param_groups": optimizer_param_groups,
            "lr_scheduler": self.lr_scheduler.state_dict() if getattr(self, "lr_scheduler", None) is not None else None,
        }
        final_file = os.path.join(local_path, _fp32_master_ckpt_filename(rank, world_size))
        gcs_target = _gcs_ckpt_blob_target(final_file)
        if gcs_target is not None:
            import time as _time

            from google.cloud import storage

            bucket_name, blob_name = gcs_target
            for attempt_idx in range(3):
                try:
                    client = storage.Client()
                    blob = client.bucket(bucket_name).blob(blob_name)
                    with blob.open("wb", chunk_size=64 * 1024 * 1024, ignore_flush=True) as raw_f:
                        with io.BufferedWriter(_NonSeekableWriter(raw_f), buffer_size=16 * 1024 * 1024) as stream:
                            torch.save(payload, stream)
                    break
                except Exception as exc:
                    if attempt_idx == 2:
                        raise
                    logger.warning(
                        "Rank %d/%d retrying direct GCS checkpoint upload to gs://%s/%s after error: %s",
                        rank,
                        world_size,
                        bucket_name,
                        blob_name,
                        exc,
                    )
                    _time.sleep(2.0 * (attempt_idx + 1))
        else:
            tmp_file = f"{final_file}.tmp"
            with open(tmp_file, "wb") as raw_f:
                with io.BufferedWriter(_NonSeekableWriter(raw_f), buffer_size=16 * 1024 * 1024) as stream:
                    torch.save(payload, stream)
            os.replace(tmp_file, final_file)
        if is_dist:
            torch.distributed.barrier()

        if rank == 0:
            _prune_old_fp32_master_checkpoints(local_path, max_ckpt_to_keep)
        if is_dist:
            torch.distributed.barrier()

    def _load_fp32_master_checkpoint(self, masters: _Fp32Masters, local_path: str) -> None:
        is_dist = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_dist else 0
        world_size = torch.distributed.get_world_size() if is_dist else 1
        ckpt_file = os.path.join(local_path, _fp32_master_ckpt_filename(rank, world_size))
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"FP32 master checkpoint shard not found for rank {rank}/{world_size}: {ckpt_file}")

        masters.clear()
        if getattr(self, "optimizer", None) is not None:
            for opt in getattr(self.optimizer, "optimizers", ()):
                opt.state.clear()
        gc.collect()

        payload: dict[str, Any] = {}
        num_slots = min(2, max(1, (world_size + 3) // 4)) if is_dist else 1
        for slot in range(num_slots):
            if not is_dist or ((rank % 8) // 4) % num_slots == slot:
                payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if is_dist:
                torch.distributed.barrier()

        if payload.get("world_size") != world_size:
            raise ValueError(
                f"Checkpoint world_size={payload.get('world_size')} does not match current world_size={world_size}"
            )

        saved_masters: dict[str, dict[str, Any]] = payload.get("masters", {})
        saved_params: dict[str, Any] = payload.get("params", {})
        saved_buffers: dict[str, Any] = payload.get("buffers", {})

        for idx, module in enumerate(self.module):
            for name, p in module.named_parameters():
                fqn = f"{idx}.{name}"
                if fqn in saved_masters:
                    entry_data = saved_masters.pop(fqn)
                    master_saved = entry_data["master"]
                    _copy_local_into_(p, master_saved)
                    param_dev = p._local_tensor.device if isinstance(p, DTensor) else p.device
                    home_dev = torch.device(masters.home) if masters.home is not None else param_dev
                    master = _unpack_ckpt_tensor_like(master_saved, p, home_dev)
                    state = {k: _unpack_ckpt_tensor_like(v, p, home_dev) for k, v in entry_data["state"].items()}
                    masters._entries[id(p)] = (master, state)
                elif fqn in saved_params:
                    _copy_local_into_(p, saved_params.pop(fqn))
                else:
                    raise KeyError(f"Parameter {fqn!r} missing from FP32 master checkpoint {ckpt_file}")

            for bname, b in module.named_buffers():
                bfqn = f"{idx}.{bname}"
                if bfqn in saved_buffers:
                    _copy_local_into_(b, saved_buffers.pop(bfqn))

        saved_opt_groups = payload.get("optimizer_param_groups")
        if saved_opt_groups is not None and getattr(self, "optimizer", None) is not None:
            for opt, groups in zip(self.optimizer.optimizers, saved_opt_groups, strict=True):
                for g, sg in zip(opt.param_groups, groups, strict=True):
                    g.update(sg)

        lr_sched_state = payload.get("lr_scheduler")
        if lr_sched_state is not None and getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.load_state_dict(lr_sched_state)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint according to ``checkpoint.save_contents``."""
        with timeline.span(
            "save_checkpoint", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane(), step=global_step
        ):
            save_contents = self._checkpoint_save_contents()
            if self._is_offload_param:
                self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

            if "hf_model" in save_contents:
                self._save_hf_model(os.path.join(local_path, "huggingface"))

            if save_contents & _DCP_CHECKPOINT_CONTENTS:
                self._save_dcp_checkpoint(local_path, global_step, max_ckpt_to_keep)

            torch.distributed.barrier()
            if self._is_offload_param:
                self.to(device="cpu", model=True, optimizer=False, grad=False)

    def _checkpoint_save_contents(self) -> set[str]:
        """``checkpoint.save_contents`` as a set; defaults to the DCP contents when unset."""
        checkpoint_config = getattr(self, "checkpoint_config", None)
        contents = getattr(checkpoint_config, "save_contents", None)
        return set(contents) if contents is not None else set(_DCP_CHECKPOINT_CONTENTS)

    def _save_dcp_checkpoint(self, local_path: str, global_step: int, max_ckpt_to_keep: Optional[int]) -> None:
        """Save the TorchTitan DCP checkpoint (or FP32 master shards when ``fp32_master=True``)."""
        masters = _fp32_masters_of(self)
        if masters is not None:
            self._save_fp32_master_checkpoint(masters, local_path, global_step, max_ckpt_to_keep)
            return

        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir
        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)
        self._drop_checkpoint_state_dict_cache()

        save_future = getattr(self.checkpointer, "save_future", None)
        if save_future is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            tracker_file = os.path.join(os.path.dirname(parent_dir), "latest_checkpointed_iteration.txt")

            def _write_latest_tracker(fut, path=tracker_file, step=global_step):
                if fut.exception() is None:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(step))

            save_future.add_done_callback(_write_latest_tracker)

    def _save_hf_model(self, hf_dir: str) -> None:
        """Write the full model in HF format (bf16 safetensors shards, config, tokenizer) to ``hf_dir``."""
        is_writer = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        hf_config = self.model_config.hf_config
        skip_names = {"lm_head.weight"} if getattr(hf_config, "tie_word_embeddings", False) else set()
        if is_writer:
            os.makedirs(hf_dir, exist_ok=True)
            hf_config.save_pretrained(hf_dir)
            if self.model_config.generation_config is not None:
                self.model_config.generation_config.save_pretrained(hf_dir)
            processing_class = self.model_config.processor or self.model_config.tokenizer
            if processing_class is not None:
                processing_class.save_pretrained(hf_dir)

        weights, _ = self.get_per_tensor_param()
        weight_map = write_hf_safetensors(weights, hf_dir, is_writer=is_writer, skip_names=skip_names)
        if weight_map is not None:
            logger.warning(
                "Saved hf_model (%d tensors in %d shards) to %s",
                len(weight_map),
                len(set(weight_map.values())),
                hf_dir,
            )

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        with timeline.span("load_checkpoint", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()):
            masters = _fp32_masters_of(self)
            if masters is not None:
                if self._is_offload_param:
                    self.to(device=get_device_name(), model=True, optimizer=False, grad=False)
                self._load_fp32_master_checkpoint(masters, local_path)
                for module in self.module:
                    for p in module.parameters():
                        if hasattr(p, "_tpu_cpu_offload_tensor"):
                            delattr(p, "_tpu_cpu_offload_tensor")
                tpu_synchronize(wait=True)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if self._is_offload_param:
                    self.to(device="cpu", model=True, optimizer=False, grad=False)
                return

            if self._is_offload_param:
                self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

            # Override TorchTitan's folder to use verl's path
            parent_dir = os.path.dirname(local_path)
            self.checkpointer.folder = parent_dir

            # Extract step number from path (verl uses global_step_N format)
            match = re.search(r"global_step_(\d+)", local_path)
            if match:
                step = int(match.group(1))
                self.checkpointer.load(step=step)
            else:
                # Fallback to latest
                self.checkpointer.load(step=-1)
            for module in self.module:
                for p in module.parameters():
                    if hasattr(p, "_tpu_cpu_offload_tensor"):
                        delattr(p, "_tpu_cpu_offload_tensor")
            tpu_synchronize(wait=True)

            torch.distributed.barrier()
            if self._is_offload_param:
                self.to(device="cpu", model=True, optimizer=False, grad=False)

            if self._is_offload_optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)

    def _to_hf_named_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Re-key a torchtitan state dict to HuggingFace names, values untouched."""
        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
        sd_adapter = self.checkpointer.sd_adapter
        if sd_adapter is not None:
            params = sd_adapter.to_hf(params)

        # When weight tying is enabled, the sd_adapter skips lm_head.weight during
        # to_hf() conversion (since it's the same tensor as embed_tokens.weight in
        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
        # add it back as a reference to embed_tokens.weight.
        if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
            params["lm_head.weight"] = params["model.embed_tokens.weight"]
        return params

    def _to_hf_named_param_chunks(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """``_to_hf_named_params`` of ``params``, lazily, one transformer layer (and the non-layer rest) at a time.

        State-dict adapters fuse tensors within a layer, e.g. Qwen3.5's MoE ``w1`` / ``w3`` into ``gate_up_proj`` and
        its GatedDeltaNet q/k/v into ``in_proj_qkv``, which materializes new tensors. Converting each layer only when
        the export reaches it keeps one layer's fused copies alive instead of the whole model's.
        """
        groups: dict[str, dict[str, Any]] = {}
        for name, param in params.items():
            m = _LAYER_PREFIX_RE.match(name)
            groups.setdefault(m.group(1) if m else "", {})[name] = param
        params.clear()
        return (self._to_hf_named_params(groups.pop(key)) for key in list(groups))

    def _expert_stack_slots(self, name: str, param: Any) -> Optional[list[tuple[str, tuple]]]:
        """If ``name`` is a fused expert stack, enumerate ALL its experts' HF tensors, else ``None``."""
        sd_adapter = self.checkpointer.sd_adapter
        from_hf_map = getattr(sd_adapter, "from_hf_map", None) if sd_adapter is not None else None
        if not from_hf_map or param.ndim < 2:
            return None
        abstract = re.sub(r"(\d+)", "{}", name, count=1)
        hf_template = next((hf for hf, titan in from_hf_map.items() if titan == abstract), None)
        if hf_template is None or hf_template.count("{}") != 2:
            return None
        match = re.search(r"\d+", name)
        if match is None:
            return None
        layer_id = match.group(0)
        slot_shape = tuple(int(d) for d in param.shape[1:])
        return [(hf_template.format(layer_id, e), slot_shape) for e in range(int(param.shape[0]))]

    def get_per_tensor_param(self, **kwargs):
        """HF-named parameters cast to ``bfloat16``, as a single-use ``TrainerWeights``.

        Iterating it all-gathers every DTensor (``DTensor.full_tensor()``) after the cast; ``local_shards()`` exports
        this rank's FSDP shards without the gather where the receiver can place them.
        """
        if self._is_offload_param:
            self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

        params = {}
        for module in self.module:
            params.update(module.state_dict())

        if self._is_offload_param:
            self.to(device="cpu", model=True, optimizer=False, grad=False)

        stacks = {}
        for name, param in params.items():
            slots = self._expert_stack_slots(name, param)
            if slots is not None:
                stacks[name] = slots
        expert_stacks = [(params[name], slots) for name, slots in stacks.items()]
        dense = self._to_hf_named_param_chunks({k: v for k, v in params.items() if k not in stacks})
        params.clear()

        # q/k/v_proj arrive here replicated, but not for free: torchtitan's FusedQKVLinear (fuse_qkv=True) keeps
        # wqkv FSDP sharded, and its state_dict() hook (_split_qkv_on_save) all-gathers it to Replicate() before
        # splitting it into q/k/v; see weight_export.fsdp_dim0_shard.
        # TODO(tpu): split wqkv locally on save (or train with fuse_qkv=False) so q/k/v export as local shards.
        # tpu-sync installs Qwen3.5 straight into vLLM's TP-sharded parameters and receives the tensors that vLLM
        # shards in several pieces (in_proj_qkv, conv1d, the stacked experts' gate_up_proj) as those pieces
        # (rollout/qwen3_5_layout.py); plain iteration keeps the HF layout.
        local_transform = None
        expert_stack_wire = None
        if getattr(self, "_torchtitan_name", None) == "qwen3_5":
            from verl_hardware_plugin.accelerators.tpu.rollout.qwen3_5_layout import make_tpu_sync_local_layout

            hf_config = self.model_config.hf_config
            local_transform = make_tpu_sync_local_layout(getattr(hf_config, "text_config", None) or hf_config)
        elif getattr(self, "_torchtitan_name", None) == "qwen3":
            # Qwen3 MoE: tpu-sync sends each expert stack as one 3-D tensor, so that FSDP shards of whole experts are
            # sent without a gather and the rollout installs its slices straight into vLLM's w13 / w2
            # (rollout/qwen3_moe_layout.py). Dense Qwen3 has no expert stacks.
            from verl_hardware_plugin.accelerators.tpu.rollout import qwen3_moe_layout

            expert_stack_wire = qwen3_moe_layout.trainer_expert_stack_wire(
                getattr(self.model_config.hf_config, "model_type", None), self.engine_config.expert_parallel_size
            )
        return (
            TrainerWeights(
                dense,
                expert_stacks,
                get_device_id(),
                local_transform=local_transform,
                expert_stack_wire=expert_stack_wire,
                rollout_expert_parallel=bool(self._tpu_options.rollout_expert_parallel),
                rollout_fp8_moe=self._tpu_options.rollout_fp8_moe,
            ),
            None,
        )


@EngineRegistry.register(
    model_type="language_model",
    backend="torchtitan",
    device="tpu",
    vendor="google",
)
class TorchTitanTPUEngineWithLMHead(TorchTitanTPUEngine):
    """Standalone TorchTitan engine for language models with LM head on Google TPU."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Prepare bucket-padded, contiguous TPU inputs for TorchTitan forward execution."""
        use_remove_padding = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )
        pad_mode = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        bucket_size = get_tpu_seq_bucket_size(self.engine_config)
        if use_remove_padding:
            vocab_size = getattr(get_text_config(self.model_config.hf_config), "vocab_size", None)
            cu_seqlens = input_ids.offsets()
            input_ids, position_ids, labels, orig_seq_len = pad_packed_inputs_for_tpu(
                input_ids=input_ids,
                position_ids=position_ids,
                micro_batch=micro_batch,
                device=get_device_id(),
                bucket_size=bucket_size,
                vocab_size=vocab_size,
            )
            output_args["orig_seq_len"] = orig_seq_len
            if getattr(self, "_gdn_row_slot_len", 0):
                if self.parallel_dims.cp_enabled:
                    raise ValueError(
                        "gdn_row_slot_len is not supported with context parallelism; "
                        "use gdn_auto_row_slot_len with context_parallel_strategy='ulysses'"
                    )
                # One sequence per GatedDeltaNet row; prepare_model_outputs restores the packed order.
                slot_src, contig_src = row_slot_permutation(cu_seqlens, input_ids.shape[-1], self._gdn_row_slot_len)
                slot_src = slot_src.to(input_ids.device)
                input_ids, position_ids = input_ids[..., slot_src], position_ids[..., slot_src]
                output_args["gdn_contig_src"] = contig_src.to(input_ids.device)
        else:
            if self.parallel_dims.cp_enabled and getattr(self, "_cp_full_positions", False):
                if multi_modal_inputs or position_ids.dim() > 2:
                    raise ValueError(
                        "cp_full_positions supports 1-D text position_ids only "
                        "(got multimodal inputs or multi-dimensional MRoPE position_ids)"
                    )
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = bucket_length(int(max(input_ids.offsets().diff())), bucket_size=bucket_size)

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0).to(get_device_id())
            input_ids = (
                torch.nested.to_padded_tensor(input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len))
                .reshape(-1)
                .to(get_device_id())
            )

            if position_ids.dim() == 3:
                position_ids = (
                    torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, 4, max_seq_len))
                    .transpose(0, 1)
                    .reshape(4, -1)
                    .to(get_device_id())
                )
            else:
                position_ids = (
                    torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, max_seq_len))
                    .reshape(-1)
                    .to(get_device_id())
                )
            output_args["batch_size"] = batch_size
            output_args["max_seq_len"] = max_seq_len

        padded_seq_len = int(input_ids.shape[-1])
        output_args["padded_seq_len"] = padded_seq_len
        extra_kwargs: dict[str, Any] = {}
        slot_src_cpu: Optional[torch.Tensor] = None
        mb_slot_len = self._effective_mb_slot_len(
            micro_batch,
            allow_auto_row_slot=self._allow_auto_row_slot(),
            forward_only=not torch.is_grad_enabled(),
            padded_seq_len=padded_seq_len,
        )
        if mb_slot_len > 0 and use_remove_padding:
            if mb_slot_len < padded_seq_len:
                input_ids, position_ids, labels, slot_src_cpu, contig_src_dev = (
                    gdn_slot_schedule.permute_packed_for_row_slots(
                        cu_seqlens, padded_seq_len, mb_slot_len, input_ids, position_ids, labels
                    )
                )
                output_args[gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY] = contig_src_dev
            extra_kwargs["gated_delta_document_resets"] = False

        cp_full_pos_dev: Optional[torch.Tensor] = None
        if self.parallel_dims.cp_enabled:
            if getattr(self, "_gdn_row_slot_len", 0):
                raise ValueError(
                    "gdn_row_slot_len is not supported with context parallelism; "
                    "use gdn_auto_row_slot_len with context_parallel_strategy='ulysses'"
                )
            if getattr(self, "_cp_full_positions", False):
                if use_remove_padding:
                    full_pos_cpu = gdn_slot_schedule.build_cp_full_positions(
                        micro_batch, padded_seq_len, unsegmented=mb_slot_len > 0, slot_src_cpu=slot_src_cpu
                    )
                    if position_ids.ndim <= 2 and int(position_ids.numel()) == int(full_pos_cpu.numel()):
                        cp_full_pos_dev = position_ids.reshape(-1).contiguous()
                    else:
                        cp_full_pos_dev = full_pos_cpu.to(device=input_ids.device, non_blocking=True).contiguous()
                else:
                    if multi_modal_inputs or position_ids.ndim != 1 or int(position_ids.numel()) != padded_seq_len:
                        raise ValueError(
                            "cp_full_positions supports 1-D text position_ids only "
                            "(got multimodal inputs or multi-dimensional MRoPE position_ids)"
                        )
                    cp_full_pos_dev = position_ids.contiguous()
            cp_batch: dict[str, Any] = {"input": input_ids, "positions": position_ids}
            if use_remove_padding:
                cp_batch["labels"] = labels
            # Shard through the CP inner attention's own ``cp_shard``, as torchtitan's ``Decoder._cp_shard_inputs``
            # does, so strategy-specific input handling (load balancing, the all_gather CP-rank tensor) applies.
            cp_batch = self._cp_attention_owner().cp_shard(
                cp_batch,
                None,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.config.parallelism.context_parallel_load_balancer,
                self.trainer.config.parallelism.context_parallel_ptrr_mask_key,
            )
            input_ids = cp_batch.pop("input")
            position_ids = cp_batch.pop("positions")
            if use_remove_padding:
                labels = cp_batch.pop("labels")
            extra_kwargs.update(cp_batch)

        # Rows restart at position 0, so attention derives document (and row) boundaries from the positions.
        extra_inputs = {"positions": position_ids, **multi_modal_inputs}
        if cp_full_pos_dev is not None:
            extra_inputs["cp_full_positions"] = cp_full_pos_dev
        extra_inputs = {
            k: v if not isinstance(v, torch.Tensor) or v.is_contiguous() else v.contiguous()
            for k, v in extra_inputs.items()
        }
        output_args["labels"] = labels
        input_ids = input_ids.contiguous()

        if getattr(self.trainer.config.parallelism, "spmd_backend", None) == "spmd_types":
            import spmd_types as spmd
            from torchtitan.distributed.spmd_types import annotate_input_spmd_types
            from torchtitan.models.common.decoder_sharding import CP, DP, TP, SpmdType, decoder_input_sharding

            sharding = dict(decoder_input_sharding())
            spmd_inputs: dict[str, Any] = {"input": input_ids, "positions": extra_inputs["positions"]}
            if use_remove_padding:
                spmd_inputs["labels"] = labels
            if "cp_full_positions" in extra_inputs:
                spmd_inputs["cp_full_positions"] = extra_inputs["cp_full_positions"]
                sharding["cp_full_positions"] = SpmdType(
                    {DP: spmd.V, CP: spmd.R, TP: spmd.R},
                    partition_spec=spmd.PartitionSpec(DP),
                )
            annotate_input_spmd_types(self.parallel_dims, spmd_inputs, sharding)

        return input_ids, extra_inputs, extra_kwargs, output_args

    def _enable_fused_logprob_head(self) -> None:
        """Makes the LM head return ``(hidden_states, weight_t)`` (torchtitan's fused-loss override) for the fused
        log-prob, and checks that the Tokamax kernel can be imported.

        Without them the fused log-prob is turned off with a warning, or, if the model requires it, the engine
        raises.
        """
        from verl_hardware_plugin.accelerators.tpu.engines.fused_logprob import require_tokamax

        try:
            from torchtitan.experiments.tpu.qwen3.config_registry import _FUSED_LOSS_OVERRIDE

            require_tokamax()
        except (ImportError, RuntimeError) as e:
            if self._fused_logprob_required or self._graph_trainer_mode:
                msg = (
                    "trainer_mode=graph_trainer requires the fused LM-head log-prob"
                    if self._graph_trainer_mode
                    else "This model requires the fused LM-head log-prob "
                    "(engine_kwargs.tpu.use_fused_linear_cross_entropy)"
                )
                raise RuntimeError(f"{msg}, but it is unavailable: {e!r}.") from e
            logger.warning("Disabling use_fused_linear_cross_entropy: %r", e)
            self._use_fused_linear_cross_entropy = False
            return
        if _FUSED_LOSS_OVERRIDE not in self.config.override.imports:
            self.config.override.imports.append(_FUSED_LOSS_OVERRIDE)

    def _maybe_gather_context_parallel(self, tensor: torch.Tensor, output_args: dict) -> torch.Tensor:
        """Under CP, gather this rank's token shard into the full bucket-padded sequence (original token order)."""
        if not self.parallel_dims.cp_enabled:
            return tensor
        return self._gather_context_parallel(tensor, output_args["padded_seq_len"])

    def prepare_model_outputs(self, logits: torch.Tensor, output_args: dict, micro_batch: TensorDict):
        """Convert TPU model logits into jagged log_probs/entropy while retaining bucket-padded device buffers."""
        use_remove_padding = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )
        pad_mode = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = float(unwrap_metadata(micro_batch["temperature"]))
        calculate_entropy = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        )
        labels = output_args["labels"]
        model_output: dict[str, torch.Tensor] = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        fused_log_probs = None
        if isinstance(logits, tuple):
            hidden_states, lm_head_weight_t = logits
            if "gdn_contig_src" in output_args:
                # gdn_row_slot_len: back from one sequence per row to the packed token order.
                hidden_states = hidden_states.index_select(-2, output_args["gdn_contig_src"])
            if use_remove_padding and not self.parallel_dims.tp_enabled:
                if labels.dim() > 1:
                    labels = labels.squeeze(0)
                hidden_rmpad = hidden_states.squeeze(0) if hidden_states.dim() > 2 else hidden_states
                fused_log_probs = fused_linear_logprobs_tpu(
                    hidden_rmpad,
                    labels,
                    lm_head_weight_t,
                    temperature=temperature,
                    # Tuned tiles with the optional kernels (google-pytorch/torchtitan#78), else the defaults.
                    **tpu_model_kernels.lce_block_sizes(
                        getattr(self, "_tpu_kernels", frozenset()),
                        hidden_size=int(lm_head_weight_t.shape[0]),
                        vocab_size=int(lm_head_weight_t.shape[1]),
                    ),
                )
                post_gather_src = output_args.get(gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY)
                if calculate_entropy and not torch.is_grad_enabled():
                    # Entropy without gradient (old log-probs): one chunk of logits at a time, never all of them.
                    entropy_rmpad = chunked_entropy_from_hidden_tpu(hidden_rmpad, lm_head_weight_t, temperature)
                    gathered_entropy = self._maybe_gather_context_parallel(entropy_rmpad, output_args)
                    if post_gather_src is not None:
                        gathered_entropy = gathered_entropy.index_select(0, post_gather_src)
                    model_output["entropy"] = _to_jagged(
                        gathered_entropy,
                        cu_seqlens,
                        output_args.get("orig_seq_len"),
                    )
                if not calculate_entropy or not torch.is_grad_enabled():
                    fused_log_probs = self._maybe_gather_context_parallel(fused_log_probs, output_args)
                    if post_gather_src is not None:
                        fused_log_probs = fused_log_probs.index_select(0, post_gather_src)
                    model_output["log_probs"] = _to_jagged(fused_log_probs, cu_seqlens, output_args.get("orig_seq_len"))
                    return model_output
            assert gdn_slot_schedule.POST_GATHER_CONTIG_SRC_KEY not in output_args, (
                "CP row-slot permutation must not enter the differentiated-entropy logits path"
            )
            # A differentiated entropy (entropy loss) needs the logits. The log-probs still come from the fused kernel,
            # so the reference, old-log-prob and update passes all compute them the same way (an on-policy PPO ratio
            # is exactly 1).
            logits = hidden_states @ lm_head_weight_t

        if use_remove_padding:
            if labels.dim() > 1:
                labels = labels.squeeze(0)
            if (
                not calculate_entropy
                and fused_log_probs is None
                and (isinstance(logits, DTensor) or logits.device.type == "tpu")
            ):
                log_probs = compute_tpu_logprobs_chunked(
                    logits,
                    labels,
                    temperature=temperature,
                    chunk_size=get_tpu_seq_bucket_size(self.engine_config),
                )
                del logits
            else:
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                logits_rmpad = (logits.squeeze(0) if logits.dim() > 2 else logits) / temperature
                del logits

                # Like verl's GPU path (flash-attn cross entropy, CUDA autocast), take the log-softmax and entropy of
                # the bf16 logits in fp32; torch_tpu's autocast would leave them in bf16.
                logits_rmpad = logits_rmpad.float()
                if fused_log_probs is not None:
                    log_probs = fused_log_probs
                else:
                    log_probs = logprobs_from_logits_naive(logits_rmpad, labels)

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        if self.engine_config.entropy_from_logits_with_chunking:
                            entropy_rmpad = self.compute_entropy_from_logits(
                                logits_rmpad,
                                chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                            )
                        else:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )
                del logits_rmpad

            orig_seq_len = output_args.get("orig_seq_len")
            log_probs = self._maybe_gather_context_parallel(
                log_probs.squeeze(0) if log_probs.dim() > 1 else log_probs, output_args
            )
            log_probs = _to_jagged(log_probs, cu_seqlens, orig_seq_len)
            if calculate_entropy:
                entropy_rmpad = self._maybe_gather_context_parallel(entropy_rmpad, output_args)
                entropy = _to_jagged(entropy_rmpad, cu_seqlens, orig_seq_len)
        else:
            if isinstance(logits, DTensor):
                logits = logits.full_tensor()
            if self.parallel_dims.cp_enabled:
                logits = self._maybe_gather_context_parallel(
                    logits.squeeze(0) if logits.dim() > 2 else logits, output_args
                )
            if logits.dim() == 2 and "batch_size" in output_args and "max_seq_len" in output_args:
                logits = logits.view(output_args["batch_size"], output_args["max_seq_len"], -1)
            logits = (logits / temperature).float()  # fp32 log-softmax and entropy, as above
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            del logits
            log_probs = logprobs_from_logits_naive(logits_rmpad, output_args["labels"])
            del logits_rmpad
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(
        self,
        micro_batch: TensorDict,
        loss_function: Callable,
        forward_only: bool,
        loss_scalars: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        """Run one micro-batch forward step without eagerly transferring the full CPU TensorDict to TPU.

        ``loss_scalars``: the mini-batch's ``device_loss_scalars``, passed on to ``prepare_tpu_loss_inputs``.
        """
        device_name = get_device_name()
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(
                inputs=input_ids,
                extra_inputs=extra_inputs,
                extra_kwargs=extra_kwargs,
                return_dtensor=True,
            )
            # verl's output and loss ops carry no spmd_types annotations (CP runs under spmd_types).
            with spmd_no_typecheck_context():
                model_output = self.prepare_model_outputs(
                    logits=logits, output_args=output_args, micro_batch=micro_batch
                )

                if loss_function is not None:
                    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch, loss_scalars)
                    loss, metrics = loss_function(
                        model_output=loss_model_output,
                        data=loss_data,
                        dp_group=self.get_data_parallel_group(),
                    )
                    _add_rollout_consistency_metrics(loss_function, loss_model_output, loss_data, metrics)
                else:
                    assert forward_only, "loss_function must be provided when not forward_only"
                    loss = torch.tensor(1.0, device=device_name)
                    metrics = {}

                for value in model_output.values():
                    if hasattr(value, TPU_PADDED_VALUES_ATTR):
                        delattr(value, TPU_PADDED_VALUES_ATTR)

                output = {
                    "model_output": detach_tree(model_output),
                    "loss": loss.detach().item(),
                    "metrics": metrics,
                }

        return loss, output
