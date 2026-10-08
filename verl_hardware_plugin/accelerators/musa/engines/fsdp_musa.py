# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engines for Moore Threads MUSA devices."""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",
    backend=["fsdp", "fsdp2"],
    device="musa",
    vendor="moore_threads",
)
class FSDPMUSAEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP language-model engine for MUSA with MCCL communication."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPMUSAEngineWithLMHead initialized")


@EngineRegistry.register(
    model_type="value_model",
    backend=["fsdp", "fsdp2"],
    device="musa",
    vendor="moore_threads",
)
class FSDPMUSAEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP value-model engine for MUSA with MCCL communication."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPMUSAEngineWithValueHead initialized")
