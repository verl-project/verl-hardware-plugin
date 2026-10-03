# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rollout replica registrations for hardware platforms.

The loaders registered here are resolved lazily by ``RolloutReplicaRegistry.get``, so importing
this package does not import vLLM.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def register_all_rollouts() -> None:
    """Register rollout replica loaders with verl's ``RolloutReplicaRegistry``."""
    try:
        from verl.workers.rollout.replica import RolloutReplicaRegistry
    except Exception as e:
        logger.debug("Rollout replicas not registered: %s", e)
        return

    # Wrap the current ``vllm`` loader (upstream's, or another plugin's): TPU gets the TPU replica,
    # every other device keeps the wrapped loader.
    wrapped_loader = RolloutReplicaRegistry._registry["vllm"]

    def _load_vllm() -> type:
        from verl.utils.device import get_resource_name

        if get_resource_name() == "TPU":
            from verl_hardware_plugin.rollout.tpu_vllm import TPUvLLMReplica

            return TPUvLLMReplica
        return wrapped_loader()

    RolloutReplicaRegistry.register("vllm", _load_vllm)
    logger.info("Registered rollout replica loader: vllm (TPU-aware)")
