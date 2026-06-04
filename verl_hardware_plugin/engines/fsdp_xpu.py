# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for Intel XPU devices.

Extends the base FSDP engine with XPU-specific workarounds
(e.g., force sum reduction for xccl backend).

Why is this engine needed?
    The Intel xccl collective communication backend does NOT support
    ReduceOp.AVG for allreduce operations. FSDP's gradient synchronization
    normally uses AVG for efficiency. This engine forces sum-based reduction
    followed by manual division, which is functionally equivalent but
    compatible with xccl.

Registration:
    @EngineRegistry.register(device="xpu", vendor="intel")
    This means verl will automatically select this engine when:
    - The detected platform is "intel" (device_name="xpu", vendor_name="intel")
    - The user is training with FSDP backend

Example:
    # This happens automatically when platform is Intel XPU:
    export VERL_PLATFORM=intel
    python -m verl.trainer.main --config config.yaml --trainer.backend=fsdp
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="xpu", vendor="intel")
class FSDPXPUEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for Intel XPU with xccl communication backend.

    Inherits all behavior from FSDPEngineWithLMHead, then applies the
    force_sum_reduction workaround after model initialization.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPXPUEngineWithLMHead initialized")

    def initialize(self):
        """Initialize the FSDP model, then apply XPU-specific workarounds.

        The key workaround: force sum-based gradient reduction.
        This is needed because xccl does not support ReduceOp.AVG.
        The FSDP wrapper will use SUM + manual division instead.
        """
        super().initialize()
        # xccl does not support ReduceOp.AVG; force sum-based reduction
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
            logger.info("Enabled force_sum_reduction_for_comms for XPU")


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="xpu", vendor="intel")
class FSDPXPUEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine for Intel XPU value model training.

    Same xccl workaround as the language model engine above.
    Value models have an additional linear head on top of the base model.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPXPUEngineWithValueHead initialized")

    def initialize(self):
        """Initialize the FSDP value model, then apply xccl workaround."""
        super().initialize()
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
