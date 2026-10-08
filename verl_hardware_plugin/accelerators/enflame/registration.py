# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Enflame GCU components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.enflame.engines import fsdp_enflame  # noqa: F401
except Exception as exc:
    logger.debug("Enflame FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.enflame.engines import megatron_enflame  # noqa: F401
except Exception as exc:
    logger.debug("Enflame Megatron engine registration failed: %s", exc)
