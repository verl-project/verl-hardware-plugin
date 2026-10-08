# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""ILUVATAR accelerator modules."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.iluvatar import platform_cuda_iluvatar  # noqa: F401
except Exception as exc:
    logger.debug("Iluvatar platform registration failed: %s", exc)
