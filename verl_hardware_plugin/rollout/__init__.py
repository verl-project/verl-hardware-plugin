# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rollout replica registrations for hardware platforms.

The loaders registered here are resolved lazily by ``RolloutReplicaRegistry.get``, so importing
this package does not import vLLM.
"""

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _make_vllm_loader(fallback: Optional[Callable[[], type]]) -> Callable[[], type]:
    """Return a ``vllm`` loader that picks the TPU replica on TPU and defers elsewhere."""

    def _load_vllm() -> type:
        from verl.utils.device import get_resource_name

        if get_resource_name() == "TPU":
            from verl_hardware_plugin.rollout.tpu_vllm import TPUvLLMReplica

            return TPUvLLMReplica
        if fallback is not None:
            return fallback()
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

        return vLLMReplica

    return _load_vllm


def register_all_rollouts() -> None:
    """Register rollout replica loaders with verl's ``RolloutReplicaRegistry``."""
    try:
        from verl.workers.rollout.replica import RolloutReplicaRegistry
    except Exception as e:
        logger.debug("Rollout replicas not registered: %s", e)
        return

    # Chain to whatever ``vllm`` loader is registered now (upstream's, or another plugin's).
    previous = RolloutReplicaRegistry._registry.get("vllm")
    if getattr(previous, "_verl_hardware_plugin", False):
        return
    loader = _make_vllm_loader(previous)
    loader._verl_hardware_plugin = True  # type: ignore[attr-defined]
    RolloutReplicaRegistry.register("vllm", loader)
    logger.info("Registered rollout replica loader: vllm (TPU-aware)")
