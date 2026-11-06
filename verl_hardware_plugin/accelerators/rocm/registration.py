# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register AMD ROCm components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.rocm.engines import mori_checkpoint_engine  # noqa: F401
except Exception as exc:
    logger.debug("ROCm MoRI checkpoint engine registration failed: %s", exc)
    try:  # say why when backend "mori" is requested, e.g. MoRI is not installed
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        CheckpointEngineRegistry.record_import_error("mori_checkpoint_engine", exc)
    except Exception:
        pass
