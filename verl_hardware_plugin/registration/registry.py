# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Discover backend declarations and run registration in dependency order."""

import importlib
import logging
import os

from verl_hardware_plugin.registration.backend import BackendRegistration

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Add a backend here once; its registration module owns every supported stage.
BACKEND_MODULES = (
    "verl_hardware_plugin.integrations.flagos.registration",
    "verl_hardware_plugin.accelerators.xpu.registration",
    "verl_hardware_plugin.accelerators.mlu.registration",
    "verl_hardware_plugin.accelerators.metax.registration",
    "verl_hardware_plugin.accelerators.enflame.registration",
    "verl_hardware_plugin.accelerators.iluvatar.registration",
    "verl_hardware_plugin.accelerators.musa.registration",
    "verl_hardware_plugin.accelerators.tpu.registration",
    "verl_hardware_plugin.accelerators.supa.registration",
)
STAGES = ("platform", "engines", "profiler", "rollout")


def _load_backends() -> list[BackendRegistration]:
    backends = []
    for module in BACKEND_MODULES:
        try:
            backends.append(importlib.import_module(module).BACKEND)
        except Exception as e:
            logger.debug("Backend declaration %s not loaded: %s", module, e)
    return backends


def _register_stage(stage: str, backends: list[BackendRegistration]) -> None:
    if stage == "engines":
        backends = sorted(backends, key=lambda backend: backend.engine_order)
    for backend in backends:
        if stage in ("platform", "engines"):
            groups: tuple[tuple[str, ...], ...] = (
                ((backend.platform,),) if stage == "platform" and backend.platform else ()
            )
            if stage == "engines":
                groups = backend.engines
            for group in groups:
                try:
                    for module in group:
                        importlib.import_module(module, backend.package)
                    logger.info("Registered %s: %s %s", stage, backend.package, group)
                except Exception as e:
                    logger.debug("%s %s %s not registered: %s", backend.package, stage, group, e)
        else:
            hook = getattr(backend, stage)
            if hook is not None:
                try:
                    hook()
                    logger.info("Registered %s: %s", stage, backend.package)
                except Exception as e:
                    logger.debug("%s %s not registered: %s", backend.package, stage, e)


def register_stage(stage: str) -> None:
    """Run one stage, including through the legacy ``register_all_*`` helpers."""
    if stage not in STAGES:
        raise ValueError(f"Unknown registration stage: {stage}")
    _register_stage(stage, _load_backends())


def register_all() -> None:
    """Register platforms, engines, profilers, then rollout loaders."""
    backends = _load_backends()
    for stage in STAGES:
        _register_stage(stage, backends)
