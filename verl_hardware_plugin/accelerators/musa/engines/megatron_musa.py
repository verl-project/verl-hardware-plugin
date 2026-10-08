# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine registration for Moore Threads MUSA."""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead, MegatronEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",
    backend="megatron",
    device="musa",
    vendor="moore_threads",
)
class MegatronMUSAEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron engine registration for Moore Threads MUSA.

    MUSA-specific lifecycle hooks are installed by ``verl_musa_patch`` before
    this registry module is imported.
    """

    def initialize(self):
        super().initialize()
        logger.info("MegatronMUSAEngineWithLMHead initialized for MUSA")


@EngineRegistry.register(
    model_type="value_model",
    backend="megatron",
    device="musa",
    vendor="moore_threads",
)
class MegatronMUSAEngineWithValueHead(MegatronEngineWithValueHead):
    """Megatron value-model engine registration for Moore Threads MUSA."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronMUSAEngineWithValueHead initialized for MUSA")
