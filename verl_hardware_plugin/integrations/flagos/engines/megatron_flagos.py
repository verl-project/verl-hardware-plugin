# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine for FlagOS multi-chip devices.

Extends the base Megatron engine with FlagGems operator acceleration.
"""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
from verl_hardware_plugin.utils import FLEnvManager, may_enable_flag_gems

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="cuda", vendor="flagos")
class MegatronFlagOSEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron Engine with FlagGems operator acceleration for FlagOS devices."""

    def initialize(self):
        logger.info("Initializing MegatronFlagOSEngineWithLMHead - FL Status: %s", FLEnvManager.get_summary())
        may_enable_flag_gems(phase="training")
        super().initialize()
