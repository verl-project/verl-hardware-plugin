# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lightweight CPU-only Ray actor registry for tracking model weight checkpoint ObjectRefs.

On TPU, trainer and rollout slices hold exclusive ``libtpu`` locks and cannot share a collective
communicator, so trainer rank 0 stores weights in Ray Plasma and registers the ``ObjectRef`` here
for rollout ``TPUWorker`` processes to fetch by training step.
"""

from typing import Any

import ray


class RayWeightRegistryState:
    """In-memory state container holding Ray ObjectRefs to synchronized model weights across steps."""

    def __init__(self) -> None:
        self.weights: dict[int, Any] = {}

    def set_weights(self, step: int, ref: Any) -> None:
        """Stores the weight reference for ``step`` and evicts any prior step entries."""
        self.weights = {step: ref}

    def get_weights(self, step: int) -> Any | None:
        """Returns the weight reference registered for ``step``, or ``None`` if absent."""
        return self.weights.get(step)

    def clear(self) -> None:
        """Drops all cached entries to reset state left by a previous job."""
        self.weights.clear()


RayWeightRegistry = ray.remote(num_cpus=0)(RayWeightRegistryState)
