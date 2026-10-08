# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Megatron engine for MetaX devices."""

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="megatron", device="cuda", vendor="metax")
class MegatronMetaXEngineWithLMHead(MegatronEngineWithLMHead):
    """Megatron Engine for MetaX GPUs with NCCL communication backend."""

    def initialize(self):
        super().initialize()
        logger.info("MegatronMetaXEngineWithLMHead initialized for MetaX")
