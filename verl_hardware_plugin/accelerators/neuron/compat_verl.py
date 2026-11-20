# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Neuron-only adapters for the released verl 0.9.1 controller."""

import functools
import inspect
import logging


def install_ray_resource_compatibility():
    """Use exclusive NeuronCore allocations without changing other platforms."""
    import ray

    from verl.plugin.platform import get_platform
    from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager

    if getattr(RayResourcePool.__init__, "_neuron_compat", False):
        return
    original_init = RayResourcePool.__init__
    signature = inspect.signature(original_init)
    original_create = ResourcePoolManager.create_resource_pool
    original_check = ResourcePoolManager._check_resource_available

    @functools.wraps(original_init)
    def initialize(*args, **kwargs):
        if get_platform().vendor_name == "aws":
            bound = signature.bind(*args, **kwargs)
            bound.arguments["max_colocate_count"] = 1
            return original_init(*bound.args, **bound.kwargs)
        return original_init(*args, **kwargs)

    @functools.wraps(original_create)
    def create(manager):
        if get_platform().vendor_name == "aws":
            manager.max_colocate_count = 1
        return original_create(manager)

    @functools.wraps(original_check)
    def check(manager):
        if get_platform().vendor_name != "aws":
            return original_check(manager)
        resources = ray._private.state.available_resources_per_node()
        available = sum(node.get("neuron_cores", 0) for node in resources.values())
        required = sum(sum(counts) for counts in manager.resource_pool_spec.values())
        if available < required:
            raise ValueError(
                f"Total available neuron_cores {available} is less than total desired neuron_cores {required}"
            )

    initialize._neuron_compat = True
    RayResourcePool.__init__ = initialize
    ResourcePoolManager.create_resource_pool = create
    ResourcePoolManager._check_resource_available = check


def install_separate_async_logging_compatibility():
    """Discard 0.9.1's print-only ``flush`` argument on this trainer's logger."""
    logger = logging.getLogger("verl.trainer.ppo.v1.trainer_separate_async")
    original = logger.info
    if getattr(original, "_neuron_compat", False) is True:
        return

    @functools.wraps(original)
    def info(*args, **kwargs):
        kwargs.pop("flush", None)
        return original(*args, **kwargs)

    info._neuron_compat = True
    logger.info = info
