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
    """Apply all MLU profiler monkey-patches. Idempotent."""
    from verl_hardware_plugin.accelerators.mlu.profilers.torch_profile_mlu import (
        _patch_get_torch_profiler,
        _patch_tool_config,
    )

    _patch_tool_config()
    _patch_get_torch_profiler()


try:
    apply_mlu_profiler_patches()
except Exception as exc:
    logger.debug("MLU profiler registration failed: %s", exc)
