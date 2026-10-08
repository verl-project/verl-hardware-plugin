# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Google TPU registration declaration and lazy rollout hook."""

from verl_hardware_plugin.registration.backend import BackendRegistration


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


BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_tpu",
    engines=(
        (".engines.torchtitan_tpu", ".engines.tpu_checkpoint_engine"),
        (".engines.raiden_checkpoint_engine", ".engines.tpu_checkpoint_engine"),
    ),
    rollout=register_rollout,
    # Preserve the existing order: TPU platform before SUPA, TPU engines after it.
    engine_order=1,
)
