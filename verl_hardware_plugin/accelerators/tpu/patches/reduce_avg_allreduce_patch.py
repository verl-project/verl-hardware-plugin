# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Monkeypatch of ``torch.distributed.all_reduce`` for TPU tensors sent to Gloo-only process groups.

Why this is needed: with model parallelism (``dp_size < world_size``) the TorchTitan TPU engine returns a
host-side Gloo data-parallel group, and verl's ``TrainingWorker._postprocess_output`` all-reduces a TPU tensor with
``ReduceOp.AVG`` on it. Gloo cannot take TPU tensors and has no AVG, so the tensor is staged through CPU and AVG
becomes SUM + divide. Every other call passes through unchanged (torch-tpu >= 0.2.0 implements AVG on ``tpu_dist``).

Applied lazily when ``PlatformTPU()`` is constructed (i.e. only when the ``tpu`` platform is selected).
"""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def _tpu_available() -> bool:
    if not hasattr(torch, "tpu"):
        try:
            import torch_tpu  # noqa: F401
        except Exception:
            return False
    from verl_hardware_plugin.accelerators.tpu import device_api

    return device_api.is_available()


def _is_gloo_only_group_with_tpu_tensor(tensor: Any, group: Any) -> bool:
    if group is None or torch.compiler.is_compiling():
        return False
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "tpu":
        return False
    try:
        backend = str(dist.get_backend(group))
    except Exception:
        return False
    return "gloo" in backend and "tpu_dist" not in backend


def apply() -> None:
    global _applied
    if _applied:
        return
    if not _tpu_available():
        return

    original_all_reduce = dist.all_reduce
    original_all_gather_into_tensor = getattr(dist, "all_gather_into_tensor", None)

    def _patched_all_reduce(tensor, op=None, group=None, async_op=False):
        # Forward `op` only when the caller passed one. Under torch.compile, dynamo rewrites the original
        # `dist.all_reduce` into a functional collective, but it only converts a ReduceOp to its string form when
        # the traced function is the current `dist.all_reduce` attribute (i.e. this wrapper). A forwarded
        # `ReduceOp.SUM` default would reach `_functional_collectives.all_reduce` unconverted and fail with
        # "'RedOpType' object has no attribute 'lower'" (e.g. vLLM's compiled tensor-parallel all-reduce).
        if not async_op and _is_gloo_only_group_with_tpu_tensor(tensor, group):
            cpu_tensor = tensor.detach().cpu()
            if op is None:
                result = original_all_reduce(cpu_tensor, group=group, async_op=False)
            elif op == dist.ReduceOp.AVG:
                world_size = dist.get_world_size(group=group)
                result = original_all_reduce(cpu_tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
                cpu_tensor.div_(world_size)
            else:
                result = original_all_reduce(cpu_tensor, op=op, group=group, async_op=False)
            tensor.copy_(cpu_tensor)
            return result
        if op is None:
            return original_all_reduce(tensor, group=group, async_op=async_op)
        # torch-tpu >= 0.2.0 implements ReduceOp.AVG on tpu_dist natively (SUM + divide inside the HLO).
        return original_all_reduce(tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = _patched_all_reduce

    if callable(original_all_gather_into_tensor):

        def _patched_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
            # Same staging as all_reduce: a Gloo-only group (e.g. the CP-aware DP group) cannot take TPU tensors.
            if not async_op and _is_gloo_only_group_with_tpu_tensor(input_tensor, group):
                cpu_out = output_tensor.detach().cpu()
                result = original_all_gather_into_tensor(
                    cpu_out, input_tensor.detach().cpu(), group=group, async_op=False
                )
                output_tensor.copy_(cpu_out)
                return result
            return original_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)

        dist.all_gather_into_tensor = _patched_all_gather_into_tensor

    _applied = True
    logger.info("[verl_hardware_plugin] Patched torch.distributed.all_reduce for TPU tensors on Gloo-only groups")
