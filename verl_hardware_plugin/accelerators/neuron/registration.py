# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Lazy, platform-specific training and rollout registration."""

import importlib


def load_rollout():
    from .rollout.vllm_neuron import NeuronVLLMReplica

    return NeuronVLLMReplica


def register():
    from verl.plugin.platform import get_platform

    if get_platform().vendor_name != "aws":
        return
    from .compat_verl import install_ray_resource_compatibility, install_separate_async_logging_compatibility
    from .runtime import ensure_neuron, install_worker_setup, process_role

    install_ray_resource_compatibility()
    install_separate_async_logging_compatibility()
    install_worker_setup()
    if process_role() == "trainer":
        ensure_neuron()
        importlib.import_module(__package__ + ".engines.torchtitan_neuron")
    importlib.import_module(__package__ + ".weight_sync.checkpoint_engine")
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    RolloutReplicaRegistry.register("vllm", load_rollout)


register()
