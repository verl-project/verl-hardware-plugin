# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Iluvatar components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.iluvatar.engines import fsdp_iluvatar  # noqa: F401
except Exception as exc:
    logger.debug("Iluvatar FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.iluvatar.engines import megatron_iluvatar  # noqa: F401
except Exception as exc:
    logger.debug("Iluvatar Megatron engine registration failed: %s", exc)
