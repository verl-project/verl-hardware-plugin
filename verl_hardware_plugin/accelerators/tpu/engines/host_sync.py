# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Deferred host reads of each micro-batch's loss and metrics (``engine_kwargs.tpu.defer_metric_sync``).

By default ``forward_step`` reads every micro-batch's loss back with ``loss.detach().item()``. That blocks the host
until the micro-batch forward has finished on the device, and only then is the backward dispatched, so the device
idles while the host traces and dispatches it.

With ``defer_metric_sync=True`` the loss stays a 0-d device tensor during the micro-batch loop, and
:func:`materialize_step_scalars` reads all deferred scalars back after the loop, in one batched transfer per
(device, dtype). The reported values are unchanged: the same tensors, read later.

The scalar *metrics* of verl's PPO loss are a separate matter: they are deferred by verl itself, whose ``Metric``
keeps device tensors and reads them lazily, and whose policy losses return tensor metrics (verl-project/verl#8776).
With a verl release from before that change, ``Metric`` and the policy losses still call ``.item()`` inside the loss
function, so only the loss read is deferred and the host still blocks once per micro-batch (a one-time warning
says so). This module does not patch verl.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from verl.utils.metric import Metric

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_warned_legacy_verl = False


def verl_defers_metric_reads() -> bool:
    """Whether the installed verl's ``Metric`` keeps scalar tensors on the device until they are read."""
    return hasattr(Metric, "materialize_all")


def warn_if_metrics_not_deferred() -> None:
    """Warns once when ``defer_metric_sync`` can only defer the loss read (verl before verl-project/verl#8776)."""
    global _warned_legacy_verl
    if not _warned_legacy_verl and not verl_defers_metric_reads():
        _warned_legacy_verl = True
        logger.warning(
            "engine_kwargs.tpu.defer_metric_sync: the installed verl reads PPO loss metrics with .item() inside the "
            "loss function, so the host still blocks once per micro-batch; only the loss read is deferred. A verl "
            "with lazy Metric values (verl-project/verl#8776) removes that sync too."
        )


def _scalars_to_host_fallback(tensors: Sequence[torch.Tensor]) -> list[Any]:
    """``[t.item() for t in tensors]`` with one stacked transfer per (device, dtype), for verl without it."""
    host: list[Any] = [None] * len(tensors)
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for i, t in enumerate(tensors):
        if type(t) is torch.Tensor and t.numel() == 1:
            groups.setdefault((t.device, t.dtype), []).append(i)
        else:  # DTensor and other subclasses: read one by one, exactly as .item() did
            host[i] = t.detach().item()
    with torch.no_grad():
        for idx in groups.values():
            group = [tensors[i].detach().reshape(()) for i in idx]
            # Pad to a power of two (repeating a tensor) so a varying micro-batch count compiles few stack shapes.
            group += group[:1] * ((1 << (len(group) - 1).bit_length()) - len(group))
            for i, v in zip(idx, torch.stack(group).cpu().tolist(), strict=False):
                host[i] = v
    return host


def _materialize(scalars: list[Any], metrics: Iterable[Metric]) -> None:
    try:
        from verl.utils.metric import materialize_scalars
    except ImportError:  # verl before verl-project/verl#8776: Metric values are already Python numbers
        slots = [i for i, v in enumerate(scalars) if isinstance(v, torch.Tensor)]
        for i, v in zip(slots, _scalars_to_host_fallback([scalars[i] for i in slots]), strict=True):
            scalars[i] = v
        return
    materialize_scalars(scalars, metrics=metrics)


@torch.no_grad()
def materialize_step_scalars(output_lst: list[dict[str, Any]]) -> None:
    """Reads the deferred ``loss`` and ``Metric`` values of ``forward_step`` outputs back to the host, in place.

    Only the ``loss`` entries and the values held by ``Metric`` objects in ``metrics`` are converted, i.e. exactly
    what ``.item()`` would have converted with the option off; anything else (``model_output``, raw tensors in
    ``metrics``) is left as is.
    """
    with_loss = [o for o in output_lst if "loss" in o]
    losses = [o["loss"] for o in with_loss]
    metrics: dict[int, Metric] = {}
    for o in output_lst:
        for value in (o.get("metrics") or {}).values():
            if isinstance(value, Metric):
                metrics.setdefault(id(value), value)
    _materialize(losses, metrics.values())
    for o, loss in zip(with_loss, losses, strict=True):
        o["loss"] = loss
