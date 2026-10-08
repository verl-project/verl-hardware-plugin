# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine wrappers for Biren devices."""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="supa", vendor="biren")
class FSDPSupaEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for Biren SUPA devices."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPSupaEngineWithLMHead initialized")

    def initialize(self):
        super().initialize()
        logger.info("FSDPSupaEngineWithLMHead initialized for Biren device")


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="supa", vendor="biren")
class FSDPSupaEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine for SUPA value model training on Biren devices."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPSupaEngineWithValueHead initialized")

    def initialize(self):
        super().initialize()
