# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Enflame GCU components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _gcu_expected() -> bool:
    """Whether this host looks like it should be able to run on GCU.

    Used only to pick the log level: a failure on a real GCU means something is
    broken and deserves a warning, while the same ImportError on a CUDA host is
    expected because the engine intentionally refuses to load there.
    """
    try:
        from verl_hardware_plugin.accelerators.enflame.platform_enflame import gcu_available

        return gcu_available()
    except Exception:
        return False


try:
    from verl_hardware_plugin.accelerators.enflame.engines import fsdp_enflame  # noqa: F401
except Exception as exc:
    logger.debug("Enflame FSDP engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.enflame.engines import megatron_enflame  # noqa: F401
except Exception as exc:
    logger.debug("Enflame Megatron engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.enflame.engines import eccl_checkpoint_engine  # noqa: F401
except Exception as exc:
    try:  # say why when backend "eccl"/"nccl" is requested on a GCU host
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        CheckpointEngineRegistry.record_import_error("eccl_checkpoint_engine", exc)
    except Exception:
        pass
    if _gcu_expected():
        logger.warning("Enflame ECCL checkpoint engine registration failed on GCU: %s", exc)
    else:
        logger.debug("Enflame ECCL checkpoint engine registration failed: %s", exc)
