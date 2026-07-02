# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Engine registration for all supported hardware backends.

Engines extend verl's base FSDP/Megatron training engines with hardware-specific
logic. Each engine module uses @EngineRegistry.register() at import time.

The registration decorator takes a compound key:
    @EngineRegistry.register(
        model_type="language_model",  # "language_model" or "value_model"
        backend=["fsdp", "fsdp2"],    # which training backend(s) this engine supports
        device="xpu",                  # torch device type
        vendor="intel",                # vendor identifier (matches platform.vendor_name)
    )

Engine lookup priority (in EngineRegistry.get_engine_cls):
    1. Exact match: (device, vendor) — e.g. ("cuda", "metax") → MetaX-specific engine
    2. Device-only fallback: device — e.g. "npu" → base NPU engine
    3. NVIDIA fallback: ("cuda", "nvidia") → base CUDA engine (for unknown CUDA vendors)

This means:
- If you register an engine for ("cuda", "metax"), MetaX users get that engine
- If no MetaX engine exists, they fall back to the base CUDA engine
- Non-CUDA devices (xpu, mlu, npu) must have explicit engine registrations
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def register_all_engines():
    """Import all engine modules to trigger their @register decorators.

    Called from verl_hardware_plugin/__init__.py during plugin discovery.
    Follows the same conditional-import pattern as platform registration.
    """

    # FlagOS engines (CUDA-compatible with FlagCX communication)
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

    # Intel XPU engines (xccl communication, sum-based reduction)
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

    # Cambricon MLU engines (CNCL communication)
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

    # MetaX engines (CUDA-compatible with vendor-specific optimizations)
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

    # Enflame engines (ECCL communication)
    try:
        from verl_hardware_plugin.engines import fsdp_enflame  # noqa: F401

        logger.info("Registered engines: fsdp_enflame")
    except Exception as e:
        logger.debug("Enflame FSDP engines not registered: %s", e)

    try:
        from verl_hardware_plugin.engines import megatron_enflame  # noqa: F401

        logger.info("Registered engines: megatron_enflame")
    except Exception as e:
        logger.debug("Enflame Megatron engines not registered: %s", e)
