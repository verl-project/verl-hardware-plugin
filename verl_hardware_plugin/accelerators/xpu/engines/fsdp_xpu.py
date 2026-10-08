# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for Intel XPU devices.

Extends the base FSDP engine with XPU-specific workarounds
(e.g., force sum reduction for xccl backend).

Why is this engine needed?
    Intel's oneCCL (xccl) collective library implements ReduceOp.AVG on its
    SYCL-kernel execution path, but not on its scheduler path -- and which
    path a given collective takes is selected internally by oneCCL, not
    capability-aware, so an AVG request can abort mid-collective
    (`average operation is not supported for the scheduler path`) rather
    than reliably succeed or reliably fail. FSDP's gradient synchronization
    normally uses AVG for efficiency. This engine forces sum-based reduction
    followed by manual division, which is functionally equivalent and,
    unlike AVG, not subject to that path-selection abort. (Fix tracked for
    oneCCL 2022.2 / torch 2.15; not yet available on the stack this plugin
    targets. A separate, narrower double-division bug on very small
    messages, intel/torch-xpu-ops#3020, is fixed on that same stack and is
    not what this workaround is for.)

Why not patch verl-core's apply_fsdp2() instead?
    verl/workers/engine/fsdp/transformer_impl.py does
    `from verl.utils.fsdp_utils import apply_fsdp2` at its own import time, so a
    plugin that reassigns `verl.utils.fsdp_utils.apply_fsdp2` after that module
    has already imported the name would have no effect on this call site --
    the same import-order fragility that motivates verl core's own
    profiler-marker dispatch design. This engine sidesteps the problem
    entirely: it does not patch anything. It calls
    set_force_sum_reduction_for_comms() on the already-constructed model
    *after* super().initialize() (which internally calls the real
    apply_fsdp2()) returns, from inside the plugin's own EngineRegistry
    subclass. That is ordinary method-override behavior, not a monkeypatch,
    so it is robust regardless of when/how anything else imported
    apply_fsdp2.

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


def _force_sum_reduction_on_all_fsdp_modules(root) -> None:
    """Apply set_force_sum_reduction_for_comms(True) to root and every nested
    FSDP2-wrapped submodule, not just the root.

    fully_shard() wraps the transformer layers individually as well as the
    root model (verl/utils/fsdp_utils.py::apply_fsdp2), so each of those
    per-layer wrappers has its own communication group that also defaults to
    ReduceOp.AVG. Only fixing the root leaves per-layer gradient sync using
    AVG, which is subject to oneCCL's unreliable AVG path selection (see
    module docstring) instead of the guaranteed-correct SUM+divide path.
    """
    if hasattr(root, "set_force_sum_reduction_for_comms"):
        root.set_force_sum_reduction_for_comms(True)
    count = 1
    for submodule in root.modules():
        if submodule is root:
            continue
        if hasattr(submodule, "set_force_sum_reduction_for_comms"):
            submodule.set_force_sum_reduction_for_comms(True)
            count += 1
    logger.info("Enabled force_sum_reduction_for_comms on %d FSDP module(s) for XPU", count)


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

        The key workaround: force sum-based gradient reduction, root module
        and every per-layer FSDP2 wrapper. This is needed because oneCCL's
        AVG path selection isn't capability-aware and can abort mid-collective
        (see module docstring). The FSDP wrapper will use SUM + manual
        division instead.
        """
        super().initialize()
        # oneCCL's AVG path selection isn't capability-aware and can abort;
        # force sum-based reduction instead (see module docstring)
        _force_sum_reduction_on_all_fsdp_modules(self.module)


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
        _force_sum_reduction_on_all_fsdp_modules(self.module)
