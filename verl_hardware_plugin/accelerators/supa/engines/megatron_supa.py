# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine registration for Biren SUPA."""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead, MegatronEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",
    backend="megatron",
    device="supa",
    vendor="biren",
)
class MegatronSupaEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron engine registration for Biren SUPA."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronSupaEngineWithLMHead initialized for Biren SUPA")


@EngineRegistry.register(
    model_type="value_model",
    backend="megatron",
    device="supa",
    vendor="biren",
)
class MegatronSupaEngineWithValueHead(MegatronEngineWithValueHead):
    """Megatron value-model engine registration for Biren SUPA."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronSupaEngineWithValueHead initialized for Biren SUPA")
