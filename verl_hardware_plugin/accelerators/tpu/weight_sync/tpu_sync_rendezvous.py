# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Per-job rendezvous for the ``tpu-sync`` checkpoint engine.

The weight-sync orchestrator (on the driver) publishes the address of its tpu-sync controller here, the trainer
publishes the full shape of every tensor it sends (the rollout workers size their receive buffers from it), and
both sides post weight statistics for the optional parity check. No weights pass through this actor.

The actor belongs to one job: its name carries the Ray job id of the process that creates it (the driver), and it
is not detached, so it goes away with the job. Jobs that share a Ray cluster therefore never see, reuse or kill
each other's rendezvous, and an actor left by an older plugin version cannot be picked up by a newer job.
``RayWeightRegistry`` stays the ``ray-plasma-store`` engine's registry of weight ``ObjectRef``s.
"""

from typing import Any, Optional

import ray

TPU_SYNC_RENDEZVOUS_NAMESPACE = "verl"
_NAME_PREFIX = "TpuSyncRendezvous"


class TpuSyncRendezvousState:
    """The controller address, the trainer's tensor shapes and the parity statistics of one job."""

    # Weight statistics are only read by the parity check of the step that wrote them.
    MAX_STATS_STEPS = 5

    def __init__(self) -> None:
        self.controller_address: Optional[str] = None
        self.global_shapes: dict[str, list[int]] = {}
        self.stats: dict[int, dict[str, Any]] = {}

    def set_controller_address(self, address: Optional[str]) -> None:
        """Records the ``host:port`` of the tpu-sync controller started by the weight-sync orchestrator."""
        self.controller_address = address

    def get_controller_address(self) -> Optional[str]:
        """Returns the tpu-sync controller address, or ``None`` before the orchestrator published it."""
        return self.controller_address

    def set_global_shapes(self, shapes: dict[str, list[int]]) -> None:
        """Records the full (unsharded) shape of every tensor the trainer sends, keyed by HF name."""
        self.global_shapes = dict(shapes)

    def get_global_shapes(self) -> dict[str, list[int]]:
        """Returns the shapes recorded by ``set_global_shapes`` (empty before the first sync)."""
        return self.global_shapes

    def set_stats(self, step: int, stats: dict[str, Any]) -> None:
        """Records trainer rank 0's weight statistics for ``step`` (every rank sent the full weights)."""
        self.stats[step] = {"master": stats}
        self._prune_stats()

    def set_rank_stats(self, step: int, rank: int, stats: dict[str, Any]) -> None:
        """Records one trainer rank's statistics of the weights it sent for ``step``.

        Used when every rank sends only its own shards: the parity check sums the ``"ranks"`` entries once every
        rank has posted.
        """
        self.stats.setdefault(step, {}).setdefault("ranks", {})[rank] = stats
        self._prune_stats()

    def get_stats(self, step: int) -> Optional[dict[str, Any]]:
        """Returns ``{"master": stats}`` or ``{"ranks": {rank: stats}}`` for ``step``, or ``None``."""
        return self.stats.get(step)

    def _prune_stats(self) -> None:
        for old_step in sorted(self.stats)[: -self.MAX_STATS_STEPS]:
            del self.stats[old_step]


TpuSyncRendezvous = ray.remote(num_cpus=0)(TpuSyncRendezvousState)


def tpu_sync_rendezvous_name(job_id: Optional[str] = None) -> str:
    """The rendezvous actor name of ``job_id`` (by default, the Ray job of the calling process)."""
    if job_id is None:
        job_id = ray.get_runtime_context().get_job_id()
    return f"{_NAME_PREFIX}-{job_id}"


def create_tpu_sync_rendezvous() -> tuple[Any, str]:
    """Creates this job's rendezvous actor (or returns it if it exists) and returns ``(handle, name)``.

    Called once per job by the weight-sync orchestrator on the driver. The actor is owned by the calling process
    (not detached), so it lives as long as the job's driver.
    """
    name = tpu_sync_rendezvous_name()
    try:
        return ray.get_actor(name, namespace=TPU_SYNC_RENDEZVOUS_NAMESPACE), name
    except ValueError:
        pass
    try:
        # The default scheduling strategy: created from inside a placement-group task, the actor would otherwise
        # be captured by that group.
        handle = TpuSyncRendezvous.options(
            name=name, namespace=TPU_SYNC_RENDEZVOUS_NAMESPACE, scheduling_strategy="DEFAULT"
        ).remote()
    except ValueError:
        # Created concurrently by another caller of the same job.
        handle = ray.get_actor(name, namespace=TPU_SYNC_RENDEZVOUS_NAMESPACE)
    return handle, name


def get_tpu_sync_rendezvous(name: Optional[str] = None) -> Any:
    """Looks up a rendezvous actor by ``name`` (by default, the one of the calling process's Ray job).

    Raises:
        ValueError: if the actor does not exist (yet).
    """
    return ray.get_actor(name or tpu_sync_rendezvous_name(), namespace=TPU_SYNC_RENDEZVOUS_NAMESPACE)
