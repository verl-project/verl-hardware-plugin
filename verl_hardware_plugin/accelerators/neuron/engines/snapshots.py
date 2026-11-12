# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-parameter sharded policy snapshots for decoupled PPO with FSDP and TP."""

from functools import wraps

import torch
from torch.distributed.tensor import DTensor


def save_model_parts(model_parts):
    snapshots = []
    for module in model_parts:
        snapshot = {}
        for name, parameter in module.named_parameters():
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            spec = parameter._spec if isinstance(parameter, DTensor) else None
            snapshot[name] = (local.detach().to("cpu", copy=True), spec)
        snapshots.append(snapshot)
    return snapshots


@torch.no_grad()
def restore_model_parts(model_parts, snapshots):
    for module, snapshot in zip(model_parts, snapshots, strict=True):
        parameters = dict(module.named_parameters())
        if parameters.keys() != snapshot.keys():
            raise ValueError("Policy snapshot parameter names changed")
        for name, parameter in parameters.items():
            saved, spec = snapshot[name]
            current_spec = parameter._spec if isinstance(parameter, DTensor) else None
            if current_spec != spec:
                raise ValueError(f"Policy snapshot sharding changed for {name}")
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            if local.shape != saved.shape or local.dtype != saved.dtype:
                raise ValueError(f"Policy snapshot shape or dtype changed for {name}")
            local.copy_(saved)


def install_snapshot_handlers():
    from verl.experimental.separation.engine_workers import DetachActorWorker

    original = DetachActorWorker._get_strategy_handlers
    if getattr(original, "_neuron_snapshots", False):
        return

    @wraps(original)
    def get_handlers(worker):
        if worker.config.actor.strategy == "torchtitan" and worker.actor.engine.trainer.device.type == "neuron":
            return save_model_parts, restore_model_parts
        return original(worker)

    get_handlers._neuron_snapshots = True
    DetachActorWorker._get_strategy_handlers = get_handlers
