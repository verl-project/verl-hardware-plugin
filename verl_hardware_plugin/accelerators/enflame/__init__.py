# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""ENFLAME accelerator modules."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.enflame import platform_enflame  # noqa: F401
except Exception as exc:
    logger.debug("Enflame platform registration failed: %s", exc)
