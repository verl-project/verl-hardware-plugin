# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lightweight CPU-only Ray actor registry for tracking model weight checkpoint ObjectRefs.

On TPU, trainer and rollout slices hold exclusive ``libtpu`` locks and cannot share a collective
communicator, so trainer rank 0 stores weights in Ray Plasma and registers the ``ObjectRef`` here
for rollout ``TPUWorker`` processes to fetch by training step.

The ``raiden`` checkpoint engine uses the same actor as a rendezvous point instead: the weight-sync
orchestrator publishes the address of its Raiden controller, the trainer publishes the full shape of
every tensor it sends (the rollout workers size their receive buffers from it), and both sides post
weight statistics for the optional parity check.
"""

from typing import Any

import ray

RAY_WEIGHT_REGISTRY_ACTOR_NAME = "RayWeightRegistry"
RAY_WEIGHT_REGISTRY_NAMESPACE = "verl"


class RayWeightRegistryState:
    """In-memory state container holding Ray ObjectRefs to synchronized model weights across steps."""

    # Weight statistics are only read by the parity check of the step that wrote them.
    MAX_STATS_STEPS = 5

    def __init__(self) -> None:
        self.weights: dict[int, Any] = {}
        self.controller_address: str | None = None
        self.global_shapes: dict[str, list[int]] = {}
        self.stats: dict[int, dict[str, Any]] = {}

    def set_weights(self, step: int, ref: Any) -> None:
        """Stores the weight reference for ``step`` and evicts any prior step entries."""
        self.weights = {step: ref}

    def get_weights(self, step: int) -> Any | None:
        """Returns the weight reference registered for ``step``, or ``None`` if absent."""
        return self.weights.get(step)

    def set_controller_address(self, address: str | None) -> None:
        """Records the ``host:port`` of the Raiden controller started by the weight-sync orchestrator."""
        self.controller_address = address

    def get_controller_address(self) -> str | None:
        """Returns the Raiden controller address, or ``None`` before the orchestrator published it."""
        return self.controller_address

    def set_global_shapes(self, shapes: dict[str, list[int]]) -> None:
        """Records the full (unsharded) shape of every tensor the trainer sends, keyed by HF name."""
        self.global_shapes = dict(shapes)

    def get_global_shapes(self) -> dict[str, list[int]]:
        """Returns the shapes recorded by ``set_global_shapes`` (empty before the first sync)."""
        return self.global_shapes

    def set_stats(self, step: int, stats: dict[str, Any]) -> None:
        """Records trainer rank 0's weight statistics for ``step``, keeping only the latest steps."""
        self.stats[step] = {"master": stats}
        for old_step in sorted(self.stats)[: -self.MAX_STATS_STEPS]:
            del self.stats[old_step]

    def get_stats(self, step: int) -> dict[str, Any] | None:
        """Returns ``{"master": stats}`` for ``step``, or ``None``."""
        return self.stats.get(step)

    def clear(self) -> None:
        """Drops all cached entries to reset state left by a previous job."""
        self.weights.clear()
        self.controller_address = None
        self.global_shapes.clear()
        self.stats.clear()


RayWeightRegistry = ray.remote(num_cpus=0)(RayWeightRegistryState)


def get_ray_weight_registry() -> Any:
    """Returns the detached ``RayWeightRegistry`` actor handle, creating the actor on first use."""
    try:
        return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
    except ValueError:
        pass
    try:
        return RayWeightRegistry.options(
            name=RAY_WEIGHT_REGISTRY_ACTOR_NAME,
            namespace=RAY_WEIGHT_REGISTRY_NAMESPACE,
            lifetime="detached",
        ).remote()
    except Exception:
        # Another process created the actor between the lookup and the creation.
        return ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
