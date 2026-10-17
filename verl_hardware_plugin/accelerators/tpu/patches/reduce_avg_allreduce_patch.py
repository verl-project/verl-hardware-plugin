# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatches for Google TPU (``tpu_dist``) distributed collectives and runtime helpers.

Why this is needed:
1. ``torch_tpu``'s ``ProcessGroupTPU`` (``tpu_dist``) implements ``ReduceOp.SUM``,
   ``ReduceOp.MAX``, ``ReduceOp.MIN``, and ``ReduceOp.PRODUCT``, but does not
   implement ``ReduceOp.AVG``. Call sites in verl-core (``verl/workers/engine_workers.py``,
   ``verl/trainer/sft_trainer.py``, and ``verl/utils/profiler/performance.py::reduce_timing``)
   call ``torch.distributed.all_reduce(x, op=ReduceOp.AVG, ...)`` directly.
2. Subgroups created by ``init_device_mesh("tpu", ...)`` register only the ``tpu`` device
   (``tpu_dist``), not ``cpu:gloo``. Calling ``torch.distributed.all_gather_object(..., group=dp_group)``
   (via ``allgather_dict_into_dict`` in ``TrainingWorker._postprocess_output``) routes object
   serialization to TPU tensors and invokes ``tensor.resize_()``, which PJRT/XLA rejects.
3. ``verl.utils.distributed.set_numa_affinity`` queries ``pynvml`` (NVIDIA NVML) when
   ``libnuma.so`` is present, and ``verl.utils.torch_functional.logprobs_from_logits``
   falls back to row-wise Python loops (``logprobs_from_logits_v2``) instead of vectorized
   ``logprobs_from_logits_naive`` (run in fp32, as on GPU) on TPU.

All patches are applied lazily when ``PlatformTPU()`` is constructed (i.e. only when the
``tpu`` platform is selected for the process).
"""

import logging
import os
import pickle
from typing import Any

import torch
import torch.distributed as dist

import verl.utils.distributed as verl_dist
import verl.utils.torch_functional as verl_tf

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False
_runtime_helpers_applied = False


def _tpu_available() -> bool:
    if not hasattr(torch, "tpu"):
        try:
            import torch_tpu  # noqa: F401
        except Exception:
            return False
    tpu = getattr(torch, "tpu", None)
    is_avail = getattr(tpu, "is_available", None)
    return bool(callable(is_avail) and is_avail())


def _is_tpu_all_reduce(tensor: Any, group: Any) -> bool:
    backend = str(dist.get_backend(group))
    if backend == "tpu_dist":
        return True
    if "tpu:tpu_dist" in backend:
        device = getattr(tensor, "device", None)
        return getattr(device, "type", None) == "tpu"
    return False


def _detach_tpu_tensors_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "tpu":
            return obj.detach().cpu()
        return obj
    if isinstance(obj, dict):
        return {k: _detach_tpu_tensors_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_tpu_tensors_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_tpu_tensors_to_cpu(v) for v in obj)
    return obj


def _noop_numa_affinity() -> None:
    return None


def _apply_verl_runtime_helpers() -> None:
    global _runtime_helpers_applied
    if _runtime_helpers_applied:
        return

    try:
        if hasattr(verl_dist, "set_numa_affinity"):
            try:
                verl_dist.set_numa_affinity.__code__ = _noop_numa_affinity.__code__
            except Exception:
                pass
            verl_dist.set_numa_affinity = _noop_numa_affinity
    except Exception as e:
        logger.debug("Skipping set_numa_affinity TPU patch: %s", e)

    try:
        orig_logprobs = getattr(verl_tf, "logprobs_from_logits", None)
        naive_logprobs = getattr(verl_tf, "logprobs_from_logits_naive", None)
        if callable(orig_logprobs) and callable(naive_logprobs) and not getattr(orig_logprobs, "_tpu_patched", False):

            def _patched_logprobs_from_logits(logits, labels, inplace_backward=True):
                device = getattr(logits, "device", None)
                if getattr(device, "type", None) == "tpu":
                    # fp32 log-softmax, as flash-attn's cross entropy computes it on GPU.
                    return naive_logprobs(logits.float(), labels)
                return orig_logprobs(logits, labels, inplace_backward=inplace_backward)

            _patched_logprobs_from_logits._tpu_patched = True  # type: ignore[attr-defined]
            verl_tf.logprobs_from_logits = _patched_logprobs_from_logits
    except Exception as e:
        logger.debug("Skipping logprobs_from_logits TPU patch: %s", e)

    _runtime_helpers_applied = True


def apply() -> None:
    global _applied
    _apply_verl_runtime_helpers()
    if _applied:
        return
    if not _tpu_available():
        return

    original_all_reduce = dist.all_reduce
    original_all_gather_object = getattr(dist, "all_gather_object", None)

    def _patched_all_reduce(tensor, op=None, group=None, async_op=False):
        # Forward `op` only when the caller passed one. Under torch.compile, dynamo rewrites the original
        # `dist.all_reduce` into a functional collective, but it only converts a ReduceOp to its string form when
        # the traced function is the current `dist.all_reduce` attribute (i.e. this wrapper). A forwarded
        # `ReduceOp.SUM` default would reach `_functional_collectives.all_reduce` unconverted and fail with
        # "'RedOpType' object has no attribute 'lower'" (e.g. vLLM's compiled tensor-parallel all-reduce).
        if op is None:
            return original_all_reduce(tensor, group=group, async_op=async_op)
        if op != dist.ReduceOp.AVG or async_op or not _is_tpu_all_reduce(tensor, group):
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

        world_size = dist.get_world_size(group=group)
        result = original_all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
        tensor.div_(world_size)
        return result

    dist.all_reduce = _patched_all_reduce

    if callable(original_all_gather_object):

        def _patched_all_gather_object(object_list, obj, group=None):
            cpu_obj = _detach_tpu_tensors_to_cpu(obj)
            backend = str(dist.get_backend(group))
            if backend != "tpu_dist":
                return original_all_gather_object(object_list, cpu_obj, group=group)

            group_size = dist.get_world_size(group=group)
            if group is not None and group_size == dist.get_world_size():
                try:
                    default_backend = str(dist.get_backend(None))
                except Exception:
                    default_backend = ""
                if "gloo" in default_backend:
                    return original_all_gather_object(object_list, cpu_obj, group=None)

            raw_bytes = pickle.dumps(cpu_obj)
            local_size = len(raw_bytes)
            size_tensor = torch.tensor([local_size], dtype=torch.int32, device="tpu")
            size_list = [torch.zeros(1, dtype=torch.int32, device="tpu") for _ in range(group_size)]
            dist.all_gather(size_list, size_tensor, group=group)
            sizes = [int(s.cpu().item()) for s in size_list]
            max_size = max(max(sizes), 1)

            padded = torch.zeros(max_size, dtype=torch.int32)
            if local_size > 0:
                padded[:local_size] = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).to(torch.int32)
            send_tensor = padded.to(device="tpu")
            recv_list = [torch.zeros(max_size, dtype=torch.int32, device="tpu") for _ in range(group_size)]
            dist.all_gather(recv_list, send_tensor, group=group)
            for idx, (recv_tensor, sz) in enumerate(zip(recv_list, sizes, strict=True)):
                data_bytes = bytes(recv_tensor.cpu()[:sz].to(torch.uint8).tolist())
                object_list[idx] = pickle.loads(data_bytes)

        dist.all_gather_object = _patched_all_gather_object

    _applied = True
    logger.info(
        "[verl_hardware_plugin] Patched torch.distributed.all_reduce(op=AVG) and all_gather_object for tpu_dist"
    )
