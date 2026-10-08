# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine for Cambricon MLU devices."""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead, MegatronEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="mlu", vendor="cambricon")
class MegatronMLUEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron Engine for Cambricon MLU with CNCL communication backend."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronMLUEngineWithLMHead initialized for MLU")


@EngineRegistry.register(model_type="value_model", backend="megatron", device="mlu", vendor="cambricon")
class MegatronMLUEngineWithValueHead(MegatronEngineWithValueHead):
    """Megatron Engine for Cambricon MLU value model training."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronMLUEngineWithValueHead initialized for MLU")
