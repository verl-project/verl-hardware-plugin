# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Biren SUPA components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.supa.engines import fsdp_supa  # noqa: F401
except Exception as exc:
    logger.debug("SUPA FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.supa.engines import megatron_supa  # noqa: F401
except Exception as exc:
    logger.debug("SUPA Megatron engine registration failed: %s", exc)
