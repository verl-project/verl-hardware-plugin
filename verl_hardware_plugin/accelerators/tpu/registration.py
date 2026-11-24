# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register Google TPU components and lazy rollout hooks when imported."""

import logging
import os

from verl_hardware_plugin.accelerators.tpu.options import inproc_rollout_enabled

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

try:
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu  # noqa: F401
except Exception as exc:
    logger.debug("TPU TorchTitan engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.tpu.weight_sync import ray_checkpoint_engine  # noqa: F401
except Exception as exc:
    logger.debug("TPU ray-plasma-store checkpoint engine registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine  # noqa: F401
except Exception as exc:
    logger.debug("TPU tpu-sync checkpoint engine registration failed: %s", exc)


def register_rollout() -> None:
    """Keep upstream vLLM and SGLang loaders on non-TPU devices without importing them eagerly."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    if "vllm" in RolloutReplicaRegistry._registry:
        wrapped_vllm = RolloutReplicaRegistry._registry["vllm"]

        def _load_vllm() -> type:
            from verl.utils.device import get_resource_name

            if inproc_rollout_enabled():
                # in-process colocation: the replica's server is the trainer worker itself (rollout/inproc)
                from verl_hardware_plugin.accelerators.tpu.rollout import inproc
                from verl_hardware_plugin.accelerators.tpu.rollout.inproc import engine as inproc_engine

                inproc.check_switches()
                return inproc_engine.TPUInProcReplica
            if get_resource_name() == "TPU":
                from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm import TPUvLLMReplica

                return TPUvLLMReplica
            return wrapped_vllm()

        try:
            RolloutReplicaRegistry.register("vllm", _load_vllm)
        except Exception as exc:
            logger.debug("TPU vLLM rollout registration failed: %s", exc)

    if "sglang" in RolloutReplicaRegistry._registry:
        wrapped_sglang = RolloutReplicaRegistry._registry["sglang"]

        def _load_sglang() -> type:
            from verl.utils.device import get_resource_name

            if get_resource_name() == "TPU":
                # Run upstream's loader first so its mock_vllm shims are installed when vLLM is absent.
                wrapped_sglang()
                from verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang import TPUSGLangReplica

                return TPUSGLangReplica
            return wrapped_sglang()

        try:
            RolloutReplicaRegistry.register("sglang", _load_sglang)
        except Exception as exc:
            logger.debug("TPU SGLang rollout registration failed: %s", exc)


try:
    register_rollout()
except Exception as exc:
    logger.debug("TPU rollout registration failed: %s", exc)

# In-process colocation (rollout/inproc), in every process that imports verl: the ("vllm", "async") rollout class and
# the trainer worker's direct rollout methods. The plugin's backend loader swallows a raise here, so the failure is
# logged as a warning and the later fail-closed points refuse instead: the replica loader above re-checks the switches
# when verl creates the replicas, and the rollout re-checks them in init_model.
if inproc_rollout_enabled():
    try:
        from verl_hardware_plugin.accelerators.tpu.rollout import inproc

        inproc.register()
    except Exception as exc:
        logger.warning("inproc REGISTRATION FAILED in pid %s: %s: %s", os.getpid(), type(exc).__name__, exc)
        raise
