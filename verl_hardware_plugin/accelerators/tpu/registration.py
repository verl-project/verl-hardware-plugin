# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Google TPU components and a lazy rollout hook when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.tpu.engines import (
        torchtitan_tpu,  # noqa: F401
        tpu_checkpoint_engine,  # noqa: F401
    )
except Exception as exc:
    logger.debug("TPU TorchTitan engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.tpu.engines import (
        raiden_checkpoint_engine,  # noqa: F401
        tpu_checkpoint_engine,  # noqa: F401
    )
except Exception as exc:
    logger.debug("TPU Raiden checkpoint engine registration failed: %s", exc)


def register_rollout() -> None:
    """Keep the current vLLM loader on non-TPU devices without importing vLLM."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    wrapped_loader = RolloutReplicaRegistry._registry["vllm"]

    def _load_vllm() -> type:
        from verl.utils.device import get_resource_name

        if get_resource_name() == "TPU":
            from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm import TPUvLLMReplica

            return TPUvLLMReplica
        return wrapped_loader()

    RolloutReplicaRegistry.register("vllm", _load_vllm)


try:
    register_rollout()
except Exception as exc:
    logger.debug("TPU rollout registration failed: %s", exc)
