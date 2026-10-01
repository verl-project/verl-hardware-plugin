# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU monkeypatches for verl-core call sites that do not yet consult the platform.

``PlatformTPU`` already implements every hook a TPU run needs
(``auto_assign_accelerator_type``, ``configure_placement_group_bundle``,
``ray_local_rank_override`` ...), but verl main only calls a subset of them
(``supports_colocated_worker_groups``, ``ray_resource_name``,
``get_worker_env_vars``; see verl-project/verl#7849). Each module in this
package bridges one remaining call site by routing it through the existing
hook, so that once verl-core calls the hook itself the corresponding patch
can simply be deleted. The patch modules are:

- ``ray_resource_pool_patch``: slice affinity + rollout bundles without a
  ``TPU`` reservation (``RayResourcePool``).
- ``worker_local_rank_patch``: ``LOCAL_RANK`` from ``TPU_VISIBLE_CHIPS`` and
  no eager ``set_device`` in ``CheckpointEngineWorker`` (``Worker``).

Every patch is idempotent and only installs itself when its target module is
fully imported, because ``PlatformTPU()`` may be constructed from a
module-level ``get_device_name()`` call while heavier verl modules are still
importing. ``apply_all(import_targets=True)`` -- used by the Ray
``worker_process_setup_hook`` ``patch_ray_worker``, which runs in a fresh
worker process before any verl import -- imports the targets explicitly and is
the authoritative installation point; the other call sites are best-effort
re-tries.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType
from typing import Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# (target module, patch module, import the target eagerly in apply_all(import_targets=True)).
# Order matters only for readability.
_PATCHES: tuple[tuple[str, str, bool], ...] = (
    ("verl.single_controller.ray.base", "verl_hardware_plugin.patches.tpu.ray_resource_pool_patch", True),
    ("verl.single_controller.base.worker", "verl_hardware_plugin.patches.tpu.worker_local_rank_patch", True),
)


def _fully_imported(module_name: str) -> Optional[ModuleType]:
    """Return the module if it is in ``sys.modules`` and finished executing, else None."""
    module = sys.modules.get(module_name)
    if module is None:
        return None
    spec = getattr(module, "__spec__", None)
    # ``_initializing`` is set by importlib while the module body is executing.
    if spec is not None and getattr(spec, "_initializing", False):
        return None
    return module


def apply_all(platform, *, import_targets: bool) -> list[str]:
    """Install every TPU core patch whose target module is available.

    Args:
        platform: The ``PlatformTPU`` instance the patches route through. Passed in
            explicitly because this may run from ``PlatformTPU.__init__``, before
            ``get_platform()`` has a singleton to return.
        import_targets: Import the (eager) target modules first. Only safe in a process
            that is not in the middle of importing verl (e.g. the Ray
            ``worker_process_setup_hook``); elsewhere pass ``False`` and the
            patches for modules that are not fully imported yet are skipped and
            picked up by a later call.

    Returns:
        Names of the patch modules that are installed after this call.
    """
    installed: list[str] = []
    for target_name, patch_name, eager in _PATCHES:
        if import_targets and eager:
            try:
                importlib.import_module(target_name)
            except Exception as e:  # pragma: no cover - depends on the verl install
                logger.warning("TPU patch for %s skipped: import failed: %s", target_name, e)
                continue
        if _fully_imported(target_name) is None:
            logger.debug("TPU patch for %s deferred: target not fully imported", target_name)
            continue
        patch_module = importlib.import_module(patch_name)
        apply: Callable[[object], bool] = patch_module.apply
        if apply(platform):
            installed.append(patch_name.rsplit(".", 1)[-1])
    return installed


def is_applied(patch_name: str) -> bool:
    """Whether the named patch module (e.g. ``"ray_resource_pool_patch"``) is installed."""
    module = sys.modules.get(f"verl_hardware_plugin.patches.tpu.{patch_name}")
    return bool(module is not None and getattr(module, "_applied", False))
