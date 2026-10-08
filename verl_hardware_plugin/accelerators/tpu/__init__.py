# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU accelerator modules."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.tpu import platform_tpu  # noqa: F401
except Exception as exc:
    logger.debug("TPU platform registration failed: %s", exc)
