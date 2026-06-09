# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for FlagOS multi-chip devices.

Extends the base FSDP engine with FlagGems operator acceleration.
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead
from verl_hardware_plugin.utils import FLEnvManager, may_enable_flag_gems

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="cuda", vendor="flagos")
class FSDPFlagOSEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine with FlagGems operator acceleration for FlagOS devices."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPFlagOSEngineWithLMHead initialized")

    def initialize(self):
        logger.info("Initializing FSDPFlagOSEngineWithLMHead - FL Status: %s", FLEnvManager.get_summary())
        may_enable_flag_gems(phase="training")
        super().initialize()


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="cuda", vendor="flagos")
class FSDPFlagOSEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine with FlagGems for FlagOS value model training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPFlagOSEngineWithValueHead initialized")

    def initialize(self):
        logger.info("Initializing FSDPFlagOSEngineWithValueHead - FL Status: %s", FLEnvManager.get_summary())
        may_enable_flag_gems(phase="training")
        super().initialize()
