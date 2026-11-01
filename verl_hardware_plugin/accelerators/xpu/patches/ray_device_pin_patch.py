# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Pin each Ray-launched XPU worker to the physical device Ray assigned it.

``Worker._setup_env_cuda_visible_devices()`` only calls ``set_device()`` when
a ``RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`` var is set. By default verl
relies on Ray's per-actor visibility mask, but torch-xpu discards
``ZE_AFFINITY_MASK`` after lazy init, so every actor enumerates all physical
devices and ``xpu:N`` resolves to physical device N regardless of the mask --
every rank lands on ``xpu:0``. The collision does not raise; it surfaces
later as OOM or as collectives silently sharing a device.

Patches ``Worker`` from the plugin side; the core-side alternative is
``PlatformBase.ray_device_index()`` (verl-project/verl#8145).

Which target ``apply()`` patches depends on timing relative to
``Worker.__init__``:

- Before it (the usual case: ``PlatformXPU.__init__`` runs during
  ``import verl``, ahead of any ``Worker``): wraps
  ``Worker._setup_env_cuda_visible_devices``.
- From inside ``Worker.__init__``, after that method already ran: wraps
  ``Worker._configure_with_store`` instead, writing ``store["_local_rank"]``
  directly.
- From inside an in-flight ``_configure_with_store`` call (its own
  ``env_keys()`` can trigger platform construction): neither wrap can still
  take effect for this worker, so it is patched directly from the call
  frame. The platform singleton is itself under construction on the stack
  above at this point, so this path can't call anything that resolves it
  (``_xpu_selected()``, ``get_resource_name()``) without recursing; it
  checks ``torch.xpu`` directly and only pins when ``VERL_PLATFORM=intel``
  is set, leaving the worker unpinned otherwise rather than guess on a
  mixed-hardware host.

Both core targets are private API. ``_reset()`` is for tests only.
"""

import logging
import os
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_applied = False
_apply_lock = threading.Lock()

# Saved so _reset() can restore them. apply() can run from a platform
# detection probe on a host where XPU is not the selected platform, so
# "installed" and "active" are tracked separately via _xpu_selected().
_originals: dict[str, Callable] = {}


def _xpu_selected() -> bool:
    """Whether XPU is the platform verl selected for this process."""
    from verl.plugin.platform import get_platform

    try:
        return get_platform().device_name == "xpu"
    except Exception:  # pragma: no cover - platform not resolvable yet
        return False


def _device_module():
    import torch

    return torch.xpu


def _xpu_hardware_present() -> bool:
    """Direct ``torch.xpu`` check for the in-flight fallback, which cannot
    call ``_xpu_selected()`` (would recurse into platform construction)."""
    try:
        import torch

        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:  # pragma: no cover - torch not importable
        return False


def _xpu_confirmed_selected_without_detection() -> bool:
    """Whether ``VERL_PLATFORM=intel`` is set -- the one selection outcome
    core resolves with a plain env read, before any platform is probed or
    constructed. False does not mean XPU loses, only that the in-flight
    fallback can't tell without recursing into the detection still running
    above it on the stack, and so leaves the worker unpinned rather than
    guess on a mixed-hardware host."""
    return os.environ.get("VERL_PLATFORM", "").strip().lower() == "intel"


def _ray_physical_index():
    """Physical device index Ray assigned to this actor, or None.

    None covers in-process ``Worker`` construction with no Ray (e.g.
    ``SFTTrainer._build_engine()`` under torchrun), CPU-only actors, and any
    unexpected error talking to Ray -- failing open (unpinned) rather than
    raising out of ``Worker.__init__``, consistent with every other gating
    helper in this module.
    """
    try:
        import ray

        if not ray.is_initialized():
            return None

        from verl.utils.device import get_resource_name

        accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get(get_resource_name())
        if not accelerator_ids:
            return None
        return int(accelerator_ids[0])
    except Exception:
        logger.warning("[verl_hardware_plugin] Could not read Ray's accelerator assignment", exc_info=True)
        return None


def _ray_physical_index_no_platform_lookup():
    """Same as ``_ray_physical_index()``, for the in-flight fallback: reads
    Ray's accelerator ids under the literal ``"GPU"`` key instead of calling
    ``get_resource_name()`` (= ``get_platform().ray_resource_name()``,
    recursive here). Not a guess -- ``PlatformXPU.ray_resource_name()`` is
    hardcoded to ``"GPU"``."""
    try:
        import ray

        if not ray.is_initialized():
            return None

        accelerator_ids = ray.get_runtime_context().get_accelerator_ids().get("GPU")
        if not accelerator_ids:
            return None
        return int(accelerator_ids[0])
    except Exception:
        logger.warning("[verl_hardware_plugin] Could not read Ray's accelerator assignment", exc_info=True)
        return None


def _worker_init_running() -> bool:
    """Whether a ``Worker.__init__`` frame is already on the call stack."""
    from verl.single_controller.base.worker import Worker

    target = Worker.__init__.__code__
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None:
        if frame.f_code is target:
            return True
        frame = frame.f_back
    return False


def _configure_with_store_frame() -> Optional[FrameType]:
    """The in-flight ``Worker._configure_with_store`` frame, if any -- its
    ``env_keys() -> get_platform()`` can trigger platform construction mid-call,
    too late for reinstalling the method on the class to affect this call."""
    from verl.single_controller.base.worker import Worker

    target = Worker._configure_with_store.__code__
    frame: Optional[FrameType] = sys._getframe(1)
    while frame is not None:
        if frame.f_code is target:
            return frame
        frame = frame.f_back
    return None


def _pin_in_flight_configure_call(frame: FrameType) -> None:
    """Fix up a ``_configure_with_store`` call that is already running.

    Mutates ``store`` before the caller's dict comprehension reads
    ``store["_local_rank"]``, since ``env_keys()`` -- the only thing that can
    have triggered this -- runs before that comprehension's body does.
    """
    if not _xpu_confirmed_selected_without_detection():
        logger.warning(
            "[verl_hardware_plugin] apply() triggered from inside an in-flight "
            "Worker._configure_with_store call, with no VERL_PLATFORM override: cannot confirm "
            "XPU will be selected without recursing into the auto-detection still running on the "
            "stack above this call. Leaving this worker unpinned; later workers in this process "
            "use the class-level patch installed below."
        )
        return
    if not _xpu_hardware_present():
        logger.warning(
            "[verl_hardware_plugin] apply() triggered from inside an in-flight "
            "Worker._configure_with_store call, with VERL_PLATFORM=intel set, but torch.xpu is not "
            "available on this host. Leaving this worker unpinned; later workers in this process "
            "use the class-level patch installed below."
        )
        return
    store = frame.f_locals.get("store")
    if store is None:
        return
    index = _ray_physical_index_no_platform_lookup()
    if index is None:
        return
    store["_local_rank"] = index
    _device_module().set_device(index)
    logger.warning(
        "[verl_hardware_plugin] Pinned xpu:%d in-flight: apply() was triggered from inside an "
        "already-running Worker._configure_with_store call (via env_keys() -> get_platform()), after "
        "_setup_env_cuda_visible_devices had already run without pinning. Patched this call's store "
        "directly; later workers in this process use the class-level patch installed below.",
        index,
    )


def apply() -> None:
    global _applied

    # Guards the check-then-act below: two threads racing into apply() before
    # either sets _applied could otherwise both read the same "original" and
    # both install a wrapper, double-wrapping Worker._configure_with_store /
    # _setup_env_cuda_visible_devices and corrupting _originals for _reset().
    with _apply_lock:
        if _applied:
            return

        from verl.single_controller.base.worker import Worker

        configure_frame = _configure_with_store_frame()
        if configure_frame is not None:
            _pin_in_flight_configure_call(configure_frame)

        late = _worker_init_running()

        if not late:
            original_setup = Worker._setup_env_cuda_visible_devices

            def _patched_setup(self):
                original_setup(self)
                if not _xpu_selected():
                    return
                index = _ray_physical_index()
                if index is None:
                    return
                os.environ["LOCAL_RANK"] = str(index)
                _device_module().set_device(index)
                logger.info("[verl_hardware_plugin] Pinned xpu:%d from Ray's accelerator assignment", index)

            Worker._setup_env_cuda_visible_devices = _patched_setup
            _originals["_setup_env_cuda_visible_devices"] = original_setup
        else:
            # Preferred target already ran for this instance. Fix up at the
            # next core entry point in the same __init__: _configure_with_store
            # re-exports store["_local_rank"] to the LOCAL_RANK env var.
            original_configure = Worker._configure_with_store

            def _patched_configure(self, store: dict):
                if _xpu_selected():
                    index = _ray_physical_index()
                    if index is not None:
                        store["_local_rank"] = index
                        _device_module().set_device(index)
                        logger.info(
                            "[verl_hardware_plugin] Late-pinned xpu:%d from Ray's accelerator assignment "
                            "(patch was installed after _setup_env_cuda_visible_devices had run)",
                            index,
                        )
                return original_configure(self, store=store)

            Worker._configure_with_store = _patched_configure
            _originals["_configure_with_store"] = original_configure
            logger.warning(
                "[verl_hardware_plugin] Ray device pinning patched at _configure_with_store: this plugin was "
                "loaded after Worker.__init__ had already started, so the preferred patch target had already "
                "run. Device pinning now depends on core's private store keys."
            )

        _applied = True


def _reset() -> None:
    """Restore the patched methods. Test-only; a real run never un-patches."""
    global _applied

    from verl.single_controller.base.worker import Worker

    for name, original in _originals.items():
        setattr(Worker, name, original)
    _originals.clear()
    _applied = False
