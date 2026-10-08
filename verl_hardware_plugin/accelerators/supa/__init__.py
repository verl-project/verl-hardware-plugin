# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SUPA accelerator modules."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.supa import platform_supa  # noqa: F401
except Exception as exc:
    logger.debug("SUPA platform registration failed: %s", exc)
