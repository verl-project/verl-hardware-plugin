# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Deferring the per-micro-batch host syncs of ``forward_backward_batch`` (``VERL_TPU_DEFER_METRIC_SYNC=1``).

Every micro-batch of the eager engine reads several 0-d device tensors back to the host before its backward runs:
``loss.detach().item()`` in ``forward_step`` and verl's ``Metric.append``, which calls ``.item()`` on each scalar
tensor metric of ``ppo_loss`` (``actor/pg_loss``, ``actor/pg_clipfrac``, ``actor/ppo_kl``, ...). The first of them
blocks the host until the whole micro-batch forward has run on the device, so the device then idles while the host
dispatches the backward. With the flag on, those scalars stay on the device during the micro-batch loop and are read
back once afterwards, grouped by dtype, in one stacked transfer per dtype (:func:`materialize_host_scalars`).

The reported loss and metrics are unchanged: the same tensors, read later. Off by default.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import torch

DEFER_METRIC_SYNC_ENV = "VERL_TPU_DEFER_METRIC_SYNC"
MB_SYNC_WAIT_ENV = "VERL_TPU_MB_SYNC_WAIT"


def defer_metric_sync_enabled() -> bool:
    """Whether ``VERL_TPU_DEFER_METRIC_SYNC=1`` keeps micro-batch scalars on the device until the loop ends."""
    return os.environ.get(DEFER_METRIC_SYNC_ENV, "0") == "1"


def micro_batch_sync_wait() -> bool:
    """Whether each micro-batch's backward is waited for (``VERL_TPU_MB_SYNC_WAIT``, default ``1``).

    ``0`` only dispatches the backward (``tpu_synchronize(wait=False)``), so the host prepares the next micro-batch
    while the device still runs this one.
    """
    return os.environ.get(MB_SYNC_WAIT_ENV, "1") != "0"


@contextlib.contextmanager
def defer_metric_items(enabled: bool = True) -> Iterator[None]:
    """Makes ``verl.utils.metric.Metric.append`` keep scalar tensors instead of calling ``.item()`` on them.

    Only for the duration of the context, and only in this process; :func:`materialize_host_scalars` must convert
    the kept tensors before the metrics leave the engine (they are pickled for ``all_gather_object``).
    """
    if not enabled:
        yield
        return
    from verl.utils.metric import Metric

    original_append = Metric.append

    def _append(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            self.values.append(value.detach().reshape(()))
            return
        return original_append(self, value)

    Metric.append = _append
    try:
        yield
    finally:
        Metric.append = original_append


def _collect(obj: Any, found: list[torch.Tensor]) -> None:
    from verl.utils.metric import Metric

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            found.append(obj)
    elif isinstance(obj, Metric):
        for v in obj.values:
            _collect(v, found)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, found)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect(v, found)


def _replace(obj: Any, values: dict[int, Any]) -> Any:
    from verl.utils.metric import Metric

    if isinstance(obj, torch.Tensor):
        return values.get(id(obj), obj)
    if isinstance(obj, Metric):
        obj.values = [_replace(v, values) for v in obj.values]
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _replace(obj[k], values)
        return obj
    if isinstance(obj, list):
        obj[:] = [_replace(v, values) for v in obj]
        return obj
    if isinstance(obj, tuple):
        return tuple(_replace(v, values) for v in obj)
    return obj


@torch.no_grad()
def materialize_host_scalars(output_lst: list[dict[str, Any]]) -> None:
    """Replaces every scalar tensor of the outputs' ``loss`` and ``metrics`` with a Python number, in place.

    The tensors are stacked per (device, dtype) and copied to the host once per group, i.e. one D2H transfer (and one
    small device program) per group instead of one blocking ``.item()`` per scalar. ``model_output`` is untouched.
    """
    found: list[torch.Tensor] = []
    for o in output_lst:
        _collect(o.get("loss"), found)
        _collect(o.get("metrics"), found)
    if not found:
        return
    groups: dict[tuple[str, torch.dtype], list[torch.Tensor]] = {}
    seen: set[int] = set()
    for t in found:
        if id(t) in seen:
            continue
        seen.add(id(t))
        groups.setdefault((str(t.device), t.dtype), []).append(t)
    values: dict[int, Any] = {}
    for tensors in groups.values():
        host = torch.stack([t.detach().reshape(()) for t in tensors]).cpu().tolist()
        for t, v in zip(tensors, host, strict=True):
            values[id(t)] = v
    for o in output_lst:
        if "loss" in o:
            o["loss"] = _replace(o["loss"], values)
        if "metrics" in o:
            o["metrics"] = _replace(o["metrics"], values)
