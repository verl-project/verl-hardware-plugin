# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl hardware plugin - Multi-chip platform and engine support.

This package registers hardware platforms (MetaX, XPU, MLU, Enflame GCU,
Biren SUPA) and their corresponding training engines with verl's
plugin system.

Discovered automatically via setuptools entry_points (verl.plugins group).
"""

import importlib
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BACKEND_MODULES = (
    "verl_hardware_plugin.integrations.flagos",
    "verl_hardware_plugin.accelerators.xpu",
    "verl_hardware_plugin.accelerators.mlu",
    "verl_hardware_plugin.accelerators.metax",
    "verl_hardware_plugin.accelerators.enflame",
    "verl_hardware_plugin.accelerators.iluvatar",
    "verl_hardware_plugin.accelerators.musa",
    "verl_hardware_plugin.accelerators.tpu",
    "verl_hardware_plugin.accelerators.supa",
)


def load_backends() -> None:
    """Import each backend's self-contained registration entry point."""
    # Upstream engines cache get_platform() during import, so register platforms first.
    for module_name in BACKEND_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.debug("Failed to load backend %s: %s", module_name, exc)
    for module_name in BACKEND_MODULES:
        try:
            importlib.import_module(f"{module_name}.registration")
        except Exception as exc:
            logger.debug("Failed to register backend %s: %s", module_name, exc)


load_backends()

logger.info("verl-hardware-plugin loaded successfully")
