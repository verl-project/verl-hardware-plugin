# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl hardware plugin - Multi-chip platform and engine support.

This package registers hardware platforms (MetaX, XPU, MLU, Enflame GCU,
Biren SUPA) and their corresponding training engines with verl's
plugin system.

Discovered automatically via setuptools entry_points (verl.plugins group).
"""

import logging
import os

from verl_hardware_plugin.registration import register_all
from verl_hardware_plugin.registration.engines import register_all_engines  # noqa: F401
from verl_hardware_plugin.registration.platforms import register_all_platforms  # noqa: F401
from verl_hardware_plugin.registration.profilers import register_all_profiles  # noqa: F401
from verl_hardware_plugin.registration.rollout import register_all_rollouts  # noqa: F401

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

register_all()

logger.info("verl-hardware-plugin loaded successfully")
