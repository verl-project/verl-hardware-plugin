# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine for Intel XPU devices."""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="xpu", vendor="intel")
class MegatronXPUEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron Engine for Intel XPU with xccl communication backend."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronXPUEngineWithLMHead initialized for XPU")
