# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the TPU patches of verl-core call sites (``verl_hardware_plugin/patches/tpu``).

The patches rewrite class attributes of verl-core modules process-wide, so every test that
installs one saves the original attribute, resets the module's ``_applied`` flag and restores
both afterwards -- otherwise the state would leak into unrelated tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def platform():
    from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

    return PlatformTPU()


@pytest.fixture
def _restore_module_attrs():
    """Yield a recorder; attributes recorded via ``save(obj, name)`` are restored on teardown."""
    saved: list[tuple[object, str, object]] = []

    def save(obj, name):
        saved.append((obj, name, getattr(obj, name)))

    yield save
    for obj, name, value in reversed(saved):
        setattr(obj, name, value)


@pytest.fixture
def _reset_patch(request):
    """Reset ``_applied`` on the named patch module before and after the test."""

    def reset(patch_name):
        module = importlib.import_module(f"verl_hardware_plugin.patches.tpu.{patch_name}")
        module._applied = False
        request.addfinalizer(lambda: setattr(module, "_applied", False))
        return module

    return reset


# ---------------------------------------------------------------------------
# apply_all / import-order guard
# ---------------------------------------------------------------------------


def test_fully_imported_rejects_modules_still_initializing():
    from verl_hardware_plugin.patches.tpu import _fully_imported

    assert _fully_imported("definitely.not.imported.module") is None

    name = "_tpu_patch_test_initializing"
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__spec__._initializing = True
    sys.modules[name] = module
    try:
        assert _fully_imported(name) is None
        module.__spec__._initializing = False
        assert _fully_imported(name) is module
    finally:
        del sys.modules[name]


def test_apply_all_skips_targets_that_are_not_imported(platform):
    import verl_hardware_plugin.patches.tpu as patches

    eager = ("not.imported.target", "verl_hardware_plugin.patches.tpu.x", True)
    lazy = ("not.imported.trainer", "verl_hardware_plugin.patches.tpu.y", False)
    with (
        mock.patch.object(patches, "_PATCHES", (eager, lazy)),
        mock.patch.object(importlib, "import_module", side_effect=ImportError("nope")) as fake_import,
    ):
        assert patches.apply_all(platform, import_targets=False) == []
        fake_import.assert_not_called()

        # With import_targets=True only the eager target is imported; the non-eager one is
        # never pulled in just to be patched.
        assert patches.apply_all(platform, import_targets=True) == []
        assert [c.args[0] for c in fake_import.call_args_list] == ["not.imported.target"]


def test_platform_tpu_init_applies_patches():
    from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

    with mock.patch("verl_hardware_plugin.patches.tpu.apply_all") as fake_apply:
        PlatformTPU()

    fake_apply.assert_called_once()
    assert fake_apply.call_args.kwargs == {"import_targets": False}


# ---------------------------------------------------------------------------
# RayResourcePool: slice affinity + rollout bundle shaping
# ---------------------------------------------------------------------------


def _two_slice_nodes():
    return [
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
    ]


def test_ray_resource_pool_patch_shapes_bundles(platform, _restore_module_attrs, _reset_patch):
    import ray

    import verl.plugin.platform.platform_manager as pm
    from verl.single_controller.ray import base as ray_base

    ray_pg = sys.modules["ray.util.placement_group"]  # ``ray.util.placement_group`` is the function

    patch_mod = _reset_patch("ray_resource_pool_patch")
    _restore_module_attrs(ray_base.RayResourcePool, "__init__")
    _restore_module_attrs(ray_base.RayResourcePool, "get_placement_groups")
    _restore_module_attrs(pm, "_current_platform")
    pm._current_platform = platform

    assert patch_mod.apply(platform) is True
    assert patch_mod.apply(platform) is True  # idempotent

    created: dict[str, list] = {}

    def fake_placement_group(bundles, strategy, name, lifetime):
        created[name] = bundles
        return mock.MagicMock(name=name)

    with (
        mock.patch.object(ray, "is_initialized", return_value=True),
        mock.patch.object(ray, "nodes", return_value=_two_slice_nodes()),
        mock.patch.object(ray, "get"),
        mock.patch.object(ray_pg, "placement_group", side_effect=fake_placement_group),
        mock.patch.object(ray_base, "sort_placement_group_by_node_ip", side_effect=lambda pgs: pgs),
    ):
        trainer = ray_base.RayResourcePool(process_on_nodes=[2], use_gpu=True, name_prefix="global_pool")
        rollout = ray_base.RayResourcePool(process_on_nodes=[2], use_gpu=True, name_prefix="rollout_pool")
        pinned = ray_base.RayResourcePool(
            process_on_nodes=[1], use_gpu=True, name_prefix="rollout_pool_x", accelerator_type="tpu-group-7"
        )
        assert trainer.accelerator_type == "tpu-group-0"
        assert rollout.accelerator_type == "tpu-group-1"
        assert pinned.accelerator_type == "tpu-group-7"  # caller's choice wins

        trainer.get_placement_groups(device_name="tpu")
        rollout.get_placement_groups(device_name="tpu")

    trainer_bundles = next(b for n, b in created.items() if n.startswith("global_pool"))
    rollout_bundles = next(b for n, b in created.items() if n.startswith("rollout_pool"))
    assert trainer_bundles == [{"CPU": 1, "TPU": 1, "tpu-group-0": 1e-4}] * 2
    # vLLM's Ray executor reserves the chips itself; the pool only carries the slice label.
    assert rollout_bundles == [{"CPU": 1, "tpu-group-1": 1e-4}] * 2


# ---------------------------------------------------------------------------
# Worker: LOCAL_RANK from TPU_VISIBLE_CHIPS
# ---------------------------------------------------------------------------


def test_worker_local_rank_patch(platform, _restore_module_attrs, _reset_patch):
    import verl.plugin.platform.platform_manager as pm
    from verl.single_controller.base import worker as worker_module

    patch_mod = _reset_patch("worker_local_rank_patch")
    _restore_module_attrs(worker_module.Worker, "_setup_env_cuda_visible_devices")
    _restore_module_attrs(pm, "_current_platform")
    pm._current_platform = platform
    assert patch_mod.apply(platform) is True

    class FakeTrainingWorker(worker_module.Worker):
        def __init__(self):  # bypass Worker.__init__
            pass

    class CheckpointEngineWorker(worker_module.Worker):
        def __init__(self):
            pass

    env = {"TPU_VISIBLE_CHIPS": "3", "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1"}
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch("verl.utils.ray_utils.ray_noset_visible_devices", return_value=True),
        mock.patch.object(platform, "set_device") as set_device,
        mock.patch("verl_hardware_plugin.patches.tpu.apply_all") as fake_apply_all,
    ):
        os.environ.pop("LOCAL_RANK", None)
        FakeTrainingWorker()._setup_env_cuda_visible_devices()
        assert os.environ["LOCAL_RANK"] == "3"
        set_device.assert_called_once_with(3)
        # the Worker patch re-triggers apply_all for targets imported after the platform was built
        fake_apply_all.assert_called_with(platform, import_targets=False)

        set_device.reset_mock()
        CheckpointEngineWorker()._setup_env_cuda_visible_devices()
        set_device.assert_not_called()

    with (
        mock.patch("verl.utils.ray_utils.ray_noset_visible_devices", return_value=False),
        mock.patch.object(platform, "set_device") as set_device,
        mock.patch("verl_hardware_plugin.patches.tpu.apply_all"),
    ):
        FakeTrainingWorker()._setup_env_cuda_visible_devices()
        set_device.assert_not_called()


# ---------------------------------------------------------------------------
# Pickling: Ray cloudpickles worker classes; the patches must not drag the platform along
# ---------------------------------------------------------------------------


def test_patched_classes_cloudpickle_by_reference(platform, _restore_module_attrs, _reset_patch):
    """Regression: a closure over the platform made Ray's actor import recurse forever.

    ``WorkerDict`` is created dynamically, so cloudpickle serializes it by value. If a patched
    method is a closure, its cells (the platform, the ``torch.tpu`` proxy) go with it and the
    unpickled proxy has no ``_original_module``. Module-level functions are pickled by reference.
    """
    import cloudpickle

    from verl.single_controller.base import worker as worker_module
    from verl.single_controller.ray import base as ray_base

    worker_patch = _reset_patch("worker_local_rank_patch")
    pool_patch = _reset_patch("ray_resource_pool_patch")
    _restore_module_attrs(worker_module.Worker, "_setup_env_cuda_visible_devices")
    _restore_module_attrs(ray_base.RayResourcePool, "__init__")
    _restore_module_attrs(ray_base.RayResourcePool, "get_placement_groups")
    assert worker_patch.apply(platform) and pool_patch.apply(platform)

    def make_worker_dict():  # mimics create_colocated_worker_cls: class defined inside a function
        class WorkerDict(worker_module.Worker):
            def __init__(self):
                pass

        return WorkerDict

    for obj in (
        make_worker_dict(),
        worker_module.Worker._setup_env_cuda_visible_devices,
        ray_base.RayResourcePool.__init__,
        ray_base.RayResourcePool.get_placement_groups,
    ):
        payload = cloudpickle.dumps(obj)
        assert b"PlatformTPU" not in payload and b"TPUDeviceModuleProxy" not in payload
        cloudpickle.loads(payload)


def test_tpu_device_module_proxy_survives_pickling():
    import pickle

    from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU, TPUDeviceModuleProxy

    proxy = TPUDeviceModuleProxy(types.SimpleNamespace(device_count=lambda: 4))
    restored = pickle.loads(pickle.dumps(proxy))
    assert isinstance(restored, TPUDeviceModuleProxy)
    assert "_original_module" in restored.__dict__  # rebuilt from the local torch.tpu, not left empty
    assert restored.memory_allocated() == 0  # fallback path works after unpickling

    # An object created without __init__ must not recurse in __getattr__.
    bare = TPUDeviceModuleProxy.__new__(TPUDeviceModuleProxy)
    with pytest.raises(AttributeError):
        bare._original_module  # noqa: B018

    # The whole platform also round-trips (pickle bypasses __init__, so this exercises the state path).
    restored_platform = pickle.loads(pickle.dumps(PlatformTPU()))
    assert restored_platform.device_name == "tpu"
    assert isinstance(restored_platform.device_module.device_count(), int)


def test_ray_resource_pool_patch_retriggers_apply_all(platform, _restore_module_attrs, _reset_patch):
    """The trainer builds the first pool after importing the trainer modules: the late trigger."""
    import verl.plugin.platform.platform_manager as pm
    from verl.single_controller.ray import base as ray_base

    patch_mod = _reset_patch("ray_resource_pool_patch")
    _restore_module_attrs(ray_base.RayResourcePool, "__init__")
    _restore_module_attrs(ray_base.RayResourcePool, "get_placement_groups")
    _restore_module_attrs(pm, "_current_platform")
    pm._current_platform = platform
    assert patch_mod.apply(platform) is True

    with mock.patch("verl_hardware_plugin.patches.tpu.apply_all") as fake_apply_all:
        ray_base.RayResourcePool(process_on_nodes=[1], use_gpu=True, name_prefix="global_pool")
    fake_apply_all.assert_called_once_with(platform, import_targets=False)
