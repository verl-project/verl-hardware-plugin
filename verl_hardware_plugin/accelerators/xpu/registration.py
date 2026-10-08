# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Intel XPU components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.xpu.engines import fsdp_xpu  # noqa: F401
except Exception as exc:
    logger.debug("XPU FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.xpu.engines import megatron_xpu  # noqa: F401
except Exception as exc:
    logger.debug("XPU Megatron engine registration failed: %s", exc)
