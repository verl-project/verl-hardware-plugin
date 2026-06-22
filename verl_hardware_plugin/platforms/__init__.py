# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Platform registration for all supported hardware backends.

Each platform module uses @PlatformRegistry.register() at import time.
We import them conditionally to avoid hard failures when the corresponding
hardware SDK is not installed.

The import pattern:
    try:
        from verl_hardware_plugin.platforms import platform_xxx
    except Exception:
        pass  # SDK not installed — skip this platform

This ensures that:
1. A user who only has Intel hardware can install the plugin without
   Cambricon's torch_mlu being present.
2. Each platform's @register decorator fires only if the import succeeds.
3. Errors are logged at DEBUG level for troubleshooting.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def register_all_platforms():
    """Import all platform modules to trigger their @register decorators.

    Called from verl_hardware_plugin/__init__.py during plugin discovery.
    Each platform module contains a class decorated with:
        @PlatformRegistry.register(platform="vendor_name")

    The @register decorator fires at class definition time (i.e., when the
    module is imported), so simply importing the module is sufficient to
    register the platform.
    """

    # Intel XPU — requires intel-extension-for-pytorch
    try:
        from verl_hardware_plugin.platforms import platform_xpu  # noqa: F401

        logger.info("Registered platform: intel (xpu)")
    except Exception as e:
        logger.debug("XPU platform not registered: %s", e)

    # Cambricon MLU — requires torch_mlu
    try:
        from verl_hardware_plugin.platforms import platform_mlu  # noqa: F401

        logger.info("Registered platform: cambricon (mlu)")
    except Exception as e:
        logger.debug("MLU platform not registered: %s", e)

    # MetaX — CUDA-compatible, no extra extension needed
    try:
        from verl_hardware_plugin.platforms import platform_cuda_metax  # noqa: F401

        logger.info("Registered platform: metax (cuda)")
    except Exception as e:
        logger.debug("MetaX platform not registered: %s", e)

    # Enflame GCU — requires torch_gcu
    try:
        from verl_hardware_plugin.platforms import platform_enflame  # noqa: F401

        logger.info("Registered platform: enflame (gcu)")
    except Exception as e:
        logger.debug("ENFLAME platform not registered: %s", e)

