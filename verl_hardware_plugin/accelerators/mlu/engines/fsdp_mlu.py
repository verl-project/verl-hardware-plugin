# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for Cambricon MLU devices."""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="mlu", vendor="cambricon")
class FSDPMLUEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for Cambricon MLU with CNCL communication backend."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPMLUEngineWithLMHead initialized")

    def initialize(self):
        super().initialize()
        from verl_hardware_plugin.accelerators.mlu.utils import linear_cross_entropy  # noqa: F401

        logger.info("FSDPMLUEngineWithLMHead initialized for MLU")


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="mlu", vendor="cambricon")
class FSDPMLUEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine for Cambricon MLU value model training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPMLUEngineWithValueHead initialized")

    def initialize(self):
        super().initialize()
        from verl_hardware_plugin.accelerators.mlu.utils import linear_cross_entropy  # noqa: F401
