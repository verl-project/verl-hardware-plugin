# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Cambricon MLU components and profiler patches when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.mlu.engines import fsdp_mlu  # noqa: F401
except Exception as exc:
    logger.debug("MLU FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.mlu.engines import megatron_mlu  # noqa: F401
except Exception as exc:
    logger.debug("MLU Megatron engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.mlu.engines import (
        cncl_checkpoint_engine,  # noqa: F401
        cnixl_checkpoint_engine,  # noqa: F401
    )
except Exception as exc:
    logger.debug("MLU checkpoint engine registration failed: %s", exc)


def apply_mlu_profiler_patches():
    """Apply all MLU profiler monkey-patches (torch + cnprec). Idempotent.

    Delegates to the profilers package so there is a single implementation of
    the patch list; this module is imported by the plugin's backend loading,
    so the patches are applied in production without any extra call.
    """
    from verl_hardware_plugin.accelerators.mlu.profilers import register_all_profiles

    register_all_profiles()


try:
    apply_mlu_profiler_patches()
except Exception as exc:
    logger.debug("MLU profiler registration failed: %s", exc)
