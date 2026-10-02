# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Route ``RayResourcePool`` slice affinity and bundle shaping through ``PlatformTPU``.

verl-core (``verl/single_controller/ray/base.py``, ``RayResourcePool``) builds
one placement-group bundle per worker as ``{"CPU": n, <resource>: 1,
<accelerator_type>: 1e-4}``. On a multi-slice GKE TPU cluster that is wrong
in two ways that ``PlatformTPU`` already knows how to fix but is never asked
about:

- ``accelerator_type`` is ``None`` unless the caller sets it, so a pool can
  straddle two slices, which the TPU mesh cannot span.
  ``PlatformTPU.auto_assign_accelerator_type`` pins the trainer pool to the
  first slice and rollout pools to the second.
- Rollout pools must not reserve ``TPU`` chips: the vLLM Ray executor places
  its own workers and requests the chips itself, so a bundle that also holds
  them leaves vLLM waiting forever.
  ``PlatformTPU.configure_placement_group_bundle`` drops the reservation.

This module wraps ``RayResourcePool.__init__`` to fill in ``accelerator_type``
and replaces ``RayResourcePool.get_placement_groups`` with the verl-core body
plus the single hook call. Delete it once verl-core calls these two hooks
itself.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False


def _patched_get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
    """``RayResourcePool.get_placement_groups`` from verl main with the bundle hook inserted."""
    import ray
    from ray.util.placement_group import placement_group

    from verl.plugin.platform import get_platform
    from verl.single_controller.ray.base import sort_placement_group_by_node_ip

    if self.pgs is not None:
        return self.pgs

    pg_name_prefix = (
        name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
    )
    current_platform = get_platform()
    if device_name != current_platform.device_name:
        logger.warning(
            f"Requested device {device_name} does not match current platform device {current_platform.device_name}"
        )
    device_name = current_platform.ray_resource_name()

    bundle = {"CPU": self.max_colocate_count}
    # verl-core: ``if self.use_gpu: bundle[device_name] = 1`` + accelerator label.
    current_platform.configure_placement_group_bundle(
        bundle, self.use_gpu, device_name, self.name_prefix, self.accelerator_type
    )
    pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

    lifetime = "detached" if self.detached else None

    pgs = [
        placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
        for idx, bundles in enumerate(pg_scheme)
    ]

    ray.get([pg.ready() for pg in pgs])

    self.pgs = sort_placement_group_by_node_ip(pgs)
    return pgs


_original_pool_init = None


def _patched_pool_init(self, *args, **kwargs):
    """``RayResourcePool.__init__`` followed by slice assignment via ``PlatformTPU``.

    Module-level (not a closure over the platform) so that it pickles by reference if Ray ever
    serializes the pool class; see ``worker_local_rank_patch`` for the full reasoning.
    """
    from verl.plugin.platform import get_platform
    from verl_hardware_plugin.patches.tpu import apply_all

    platform = get_platform()
    # Pools are first built by the trainer, i.e. in the TaskRunner after the trainer modules
    # are imported: the point at which the non-eager (trainer-side) patches can be installed.
    apply_all(platform, import_targets=False)

    _original_pool_init(self, *args, **kwargs)
    if self.accelerator_type is None:
        self.accelerator_type = platform.auto_assign_accelerator_type(
            self.name_prefix, None, required_tpus=self.world_size
        )


def apply(platform) -> bool:
    """Install the patch once. Returns True once installed.

    ``platform`` is accepted for interface uniformity with the other patches but deliberately
    not captured, so nothing here holds a reference to the platform instance.
    """
    del platform
    global _applied, _original_pool_init
    if _applied:
        return True

    from verl.single_controller.ray import base as ray_base

    pool_cls = ray_base.RayResourcePool
    if pool_cls.__init__ is not _patched_pool_init:  # never wrap our own wrapper
        _original_pool_init = pool_cls.__init__
        pool_cls.__init__ = _patched_pool_init
    pool_cls.get_placement_groups = _patched_get_placement_groups
    _applied = True
    logger.info("Applied TPU RayResourcePool patch (slice affinity + rollout bundle shaping)")
    return True
