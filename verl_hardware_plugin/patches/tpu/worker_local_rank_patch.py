# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Derive ``LOCAL_RANK`` from ``TPU_VISIBLE_CHIPS`` in ``Worker.__init__``.

verl-core (``verl/single_controller/base/worker.py``,
``Worker._setup_env_cuda_visible_devices``) ends with::

    if is_ray_noset_visible_devices:
        local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
        os.environ["LOCAL_RANK"] = local_rank
        get_torch_device().set_device(int(local_rank))

Ray does not enumerate TPU chips per actor, so on TPU that lookup is empty
(rollout pools do not even reserve the ``TPU`` resource) and raises. The chip
index is instead carried in ``TPU_VISIBLE_CHIPS``, which
``PlatformTPU.get_worker_env_vars`` sets when the actor is created and
``PlatformTPU.ray_local_rank_override`` reads back. In addition,
``CheckpointEngineWorker`` must not call ``set_device`` eagerly: it runs next
to a training worker that already owns the chip, and initializing the PJRT
client a second time locks it.

The CUDA/HIP/ROCR environment reconciliation that makes up the rest of the
method has no TPU equivalent, so this patch replaces the method rather than
wrapping it. Delete once verl-core calls ``ray_local_rank_override`` itself.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False

# Workers that share a chip with a training worker and must not touch the device in __init__.
_NO_EAGER_SET_DEVICE = ("CheckpointEngineWorker",)


def _tpu_setup_env_visible_devices(self):
    """Replacement for ``Worker._setup_env_cuda_visible_devices`` on TPU.

    Module-level on purpose: Ray cloudpickles worker classes, and a function defined at module
    scope is pickled by reference. A closure over the platform instance would be pickled by
    value together with the platform (and its ``torch.tpu`` proxy), which cannot be unpickled.
    The platform is therefore looked up when the method runs, by which time it is initialized.
    """
    from verl.plugin.platform import get_platform
    from verl.utils.ray_utils import ray_noset_visible_devices

    # Heavier verl modules (``verl.workers.engine_workers``) are imported by the time any
    # Worker is constructed, so this is the last reliable point to install their patches
    # when the Ray worker_process_setup_hook was not configured.
    from verl_hardware_plugin.patches.tpu import apply_all

    platform = get_platform()
    apply_all(platform, import_targets=False)

    if not ray_noset_visible_devices():
        return
    local_rank = str(platform.ray_local_rank_override())
    os.environ["LOCAL_RANK"] = local_rank
    if type(self).__name__ not in _NO_EAGER_SET_DEVICE:
        platform.set_device(int(local_rank))


def apply(platform) -> bool:
    """Install the patch once. Returns True once installed.

    ``platform`` is accepted for interface uniformity with the other patches but deliberately
    not captured (see ``_tpu_setup_env_visible_devices``).
    """
    del platform
    global _applied
    if _applied:
        return True

    from verl.single_controller.base import worker as worker_module

    worker_module.Worker._setup_env_cuda_visible_devices = _tpu_setup_env_visible_devices
    _applied = True
    logger.info("Applied TPU Worker LOCAL_RANK patch")
    return True
