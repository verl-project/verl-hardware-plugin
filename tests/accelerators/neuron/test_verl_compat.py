# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Exercise stable verl resource contracts without allocating Ray workers."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from verl_hardware_plugin.accelerators.neuron.compat_verl import (
    install_ray_resource_compatibility,
    install_separate_async_logging_compatibility,
)


def test_exclusive_pools_and_native_resource_accounting(monkeypatch):
    import ray

    from verl.plugin import platform
    from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager

    for owner, name in (
        (RayResourcePool, "__init__"),
        (ResourcePoolManager, "create_resource_pool"),
        (ResourcePoolManager, "_check_resource_available"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    current = SimpleNamespace(vendor_name="aws")
    monkeypatch.setattr(platform, "get_platform", lambda: current)
    monkeypatch.setattr(ray._private.state, "available_resources_per_node", lambda: {"host": {"neuron_cores": 64}})
    install_ray_resource_compatibility()
    wrapped = RayResourcePool.__init__
    install_ray_resource_compatibility()
    assert RayResourcePool.__init__ is wrapped
    assert RayResourcePool([2], True, "actor", 8).max_colocate_count == 1
    manager = ResourcePoolManager({"actor": [32], "rollout": [32]}, {})
    manager.create_resource_pool()
    assert manager.max_colocate_count == 1
    assert all(pool.max_colocate_count == 1 for pool in manager.resource_pool_dict.values())
    manager.resource_pool_spec["extra"] = [1]
    with pytest.raises(ValueError, match="neuron_cores 64"):
        manager._check_resource_available()
    current.vendor_name = "nvidia"
    assert RayResourcePool([2], max_colocate_count=8).max_colocate_count == 8
    with pytest.raises(ValueError, match="GPUs"):
        manager._check_resource_available()


def test_stable_trainer_logging_is_idempotent(monkeypatch):
    logger = logging.getLogger("verl.trainer.ppo.v1.trainer_separate_async")
    original = Mock()
    monkeypatch.setattr(logger, "info", original)
    install_separate_async_logging_compatibility()
    wrapped = logger.info
    install_separate_async_logging_compatibility()
    assert logger.info is wrapped
    logger.info("standalone %s", 4, flush=True, extra={"step": 0})
    original.assert_called_once_with("standalone %s", 4, extra={"step": 0})
