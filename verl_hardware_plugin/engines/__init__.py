# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Engine registration for all supported hardware backends."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def register_all_engines():
    """Import all engine modules to trigger their @register decorators."""

    # FlagOS engines
    try:
        from verl_hardware_plugin.engines import fsdp_flagos  # noqa: F401

        logger.info("Registered engines: fsdp_flagos")
    except Exception as e:
        logger.debug("FlagOS FSDP engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import megatron_flagos  # noqa: F401

        logger.info("Registered engines: megatron_flagos")
    except Exception as e:
        logger.debug("FlagOS Megatron engines not registered: %s", e)

    # XPU engines
    try:
        from verl_hardware_plugin.engines import fsdp_xpu  # noqa: F401

        logger.info("Registered engines: fsdp_xpu")
    except Exception as e:
        logger.debug("XPU FSDP engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import megatron_xpu  # noqa: F401

        logger.info("Registered engines: megatron_xpu")
    except Exception as e:
        logger.debug("XPU Megatron engines not registered: %s", e)

    # MLU engines
    try:
        from verl_hardware_plugin.engines import fsdp_mlu  # noqa: F401

        logger.info("Registered engines: fsdp_mlu")
    except Exception as e:
        logger.debug("MLU FSDP engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import megatron_mlu  # noqa: F401

        logger.info("Registered engines: megatron_mlu")
    except Exception as e:
        logger.debug("MLU Megatron engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import cncl_checkpoint_engine  # noqa: F401

        logger.info("Registered engines: cncl_checkpoint_engine")
    except Exception as e:
        logger.debug("CNCL Checkpoint engine not registered: %s", e)

    # MetaX engines
    try:
        from verl_hardware_plugin.engines import fsdp_metax  # noqa: F401

        logger.info("Registered engines: fsdp_metax")
    except Exception as e:
        logger.debug("MetaX FSDP engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import megatron_metax  # noqa: F401

        logger.info("Registered engines: megatron_metax")
    except Exception as e:
        logger.debug("MetaX Megatron engines not registered: %s", e)
