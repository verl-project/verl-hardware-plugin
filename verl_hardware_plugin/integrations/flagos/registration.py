# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register cross-accelerator FlagOS components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.integrations.flagos.engines import fsdp_flagos  # noqa: F401
except Exception as exc:
    logger.debug("FlagOS FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.integrations.flagos.engines import megatron_flagos  # noqa: F401
except Exception as exc:
    logger.debug("FlagOS Megatron engine registration failed: %s", exc)
