# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Moore Threads MUSA components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.musa.engines import fsdp_musa  # noqa: F401
except Exception as exc:
    logger.debug("MUSA FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.musa.engines import megatron_musa  # noqa: F401
except Exception as exc:
    logger.debug("MUSA Megatron engine registration failed: %s", exc)
