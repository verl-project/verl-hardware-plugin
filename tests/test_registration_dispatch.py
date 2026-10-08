# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SDK-free tests for the unified, backend-owned registration declarations."""

import builtins
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def registry():
    """Load registration code without running the plugin's hardware entry point."""
    package = ModuleType("verl_hardware_plugin")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "verl_hardware_plugin")]
    with mock.patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "verl_hardware_plugin" or name.startswith("verl_hardware_plugin."):
                del sys.modules[name]
        sys.modules[package.__name__] = package
        yield importlib.import_module("verl_hardware_plugin.registration.registry")


def test_new_backend_has_one_entry_for_all_stages(registry):
    events = []
    backend = registry.BackendRegistration(
        package="new_backend",
        platform=".platform",
        engines=((".fsdp",), (".megatron",)),
        profiler=lambda: events.append("profiler"),
        rollout=lambda: events.append("rollout"),
    )

    def load(name, package=None):
        if name == "new_backend.registration":
            return SimpleNamespace(BACKEND=backend)
        assert package == "new_backend"
        events.append(name)

    with (
        mock.patch.object(registry, "BACKEND_MODULES", ("new_backend.registration",)),
        mock.patch.object(registry, "importlib", SimpleNamespace(import_module=load)),
    ):
        registry.register_all()

    assert events == [".platform", ".fsdp", ".megatron", "profiler", "rollout"]


def test_optional_failures_are_isolated_by_group_backend_and_stage(registry):
    events = []

    def fail_hook():
        events.append("bad.profiler")
        raise RuntimeError("optional profiler SDK is unavailable")

    backends = {
        "bad.registration": registry.BackendRegistration(
            package="bad",
            platform=".platform",
            engines=((".missing", ".dependent"), (".independent",)),
            profiler=fail_hook,
            rollout=lambda: events.append("bad.rollout"),
        ),
        "good.registration": registry.BackendRegistration(
            package="good",
            platform=".platform",
            engines=((".engine",),),
            profiler=lambda: events.append("good.profiler"),
            rollout=lambda: events.append("good.rollout"),
        ),
    }

    def load(name, package=None):
        if name == "unavailable.registration":
            raise ImportError("optional declaration dependency is unavailable")
        if name in backends:
            return SimpleNamespace(BACKEND=backends[name])
        events.append(package + name)
        if package == "bad" and name in (".platform", ".missing"):
            raise ImportError("optional SDK is unavailable")

    with (
        mock.patch.object(registry, "BACKEND_MODULES", ("unavailable.registration", *backends)),
        mock.patch.object(registry, "importlib", SimpleNamespace(import_module=load)),
    ):
        registry.register_all()

    assert events == [
        "bad.platform",
        "good.platform",
        "bad.missing",
        "bad.independent",
        "good.engine",
        "bad.profiler",
        "good.profiler",
        "bad.rollout",
        "good.rollout",
    ]


def test_declarations_do_not_import_sdks_or_implementations(registry):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("torch", "verl.", "vllm", "ray", "flag_gems")), name
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
        backends = registry._load_backends()

    assert len(backends) == 9
    assert all("trainium" not in backend.package for backend in backends)
    implementation_modules = [
        name
        for name in sys.modules
        if name.startswith(("verl_hardware_plugin.accelerators.", "verl_hardware_plugin.integrations."))
        and name.count(".") > 2
        and not name.endswith(".registration")
    ]
    assert implementation_modules == []


def test_existing_registration_order_and_groups_are_preserved(registry):
    backends = registry._load_backends()
    assert [backend.platform for backend in backends if backend.platform] == [
        ".platform_xpu",
        ".platform_mlu",
        ".platform_cuda_metax",
        ".platform_enflame",
        ".platform_cuda_iluvatar",
        ".platform_musa",
        ".platform_tpu",
        ".platform_supa",
    ]
    groups = [group for backend in sorted(backends, key=lambda b: b.engine_order) for group in backend.engines]
    assert groups == [
        (".engines.fsdp_flagos",),
        (".engines.megatron_flagos",),
        (".engines.fsdp_xpu",),
        (".engines.megatron_xpu",),
        (".engines.fsdp_mlu",),
        (".engines.megatron_mlu",),
        (".engines.cncl_checkpoint_engine", ".engines.cnixl_checkpoint_engine"),
        (".engines.fsdp_metax",),
        (".engines.megatron_metax",),
        (".engines.fsdp_enflame",),
        (".engines.megatron_enflame",),
        (".engines.fsdp_iluvatar",),
        (".engines.megatron_iluvatar",),
        (".engines.fsdp_musa",),
        (".engines.megatron_musa",),
        (".engines.fsdp_supa",),
        (".engines.megatron_supa",),
        (".engines.torchtitan_tpu", ".engines.tpu_checkpoint_engine"),
        (".engines.raiden_checkpoint_engine", ".engines.tpu_checkpoint_engine"),
    ]
    events = []
    backends = [
        replace(
            backend,
            profiler=(lambda: events.append("profiler")) if backend.profiler else None,
            rollout=(lambda: events.append("rollout")) if backend.rollout else None,
        )
        for backend in backends
    ]
    with (
        mock.patch.object(registry, "_load_backends", return_value=backends),
        mock.patch.object(
            registry, "importlib", SimpleNamespace(import_module=lambda name, package: events.append(name))
        ),
    ):
        registry.register_all()
    assert events == (
        [backend.platform for backend in backends if backend.platform]
        + [module for group in groups for module in group]
        + ["profiler", "rollout"]
    )


@pytest.mark.parametrize(
    ("module", "function", "stage"),
    [
        ("platforms", "register_all_platforms", "platform"),
        ("engines", "register_all_engines", "engines"),
        ("profilers", "register_all_profiles", "profiler"),
        ("rollout", "register_all_rollouts", "rollout"),
    ],
)
def test_legacy_entry_points_forward_to_unified_dispatcher(registry, module, function, stage):
    wrapper = importlib.import_module(f"verl_hardware_plugin.registration.{module}")
    with mock.patch.object(wrapper, "register_stage") as register_stage:
        getattr(wrapper, function)()
    register_stage.assert_called_once_with(stage)


def test_legacy_mlu_profiler_hook_forwards_to_accelerator(registry):
    backend = importlib.import_module("verl_hardware_plugin.accelerators.mlu.registration")
    legacy = importlib.import_module("verl_hardware_plugin.registration.profilers")
    with mock.patch.object(backend, "apply_mlu_profiler_patches") as apply:
        legacy.apply_mlu_profiler_patches()
    apply.assert_called_once_with()


def test_mlu_profiler_stage_remains_idempotent(registry):
    class ToolConfig:
        def __post_init__(self):
            self.original_contents = tuple(self.contents)

    modules = {}
    for name in ("verl", "verl.utils", "verl.utils.profiler", "verl.utils.profiler.config"):
        module = modules[name] = ModuleType(name)
        module.__path__ = []
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules["verl.utils.profiler.config"].TorchProfilerToolConfig = ToolConfig
    profile = ModuleType("verl.utils.profiler.torch_profile")
    original = profile.get_torch_profiler = mock.Mock()
    modules[profile.__name__] = profile
    modules["verl.utils.profiler"].torch_profile = profile
    modules["torch"] = ModuleType("torch")

    with (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(registry, "BACKEND_MODULES", ("verl_hardware_plugin.accelerators.mlu.registration",)),
    ):
        registry.register_stage("profiler")
        patched_config = ToolConfig.__post_init__
        patched_profiler = profile.get_torch_profiler
        assert patched_profiler is not original
        registry.register_stage("profiler")
        assert ToolConfig.__post_init__ is patched_config
        assert profile.get_torch_profiler is patched_profiler
        config = ToolConfig()
        config.contents = ["cpu", "mlu"]
        config.__post_init__()
        assert config.contents == ["cpu", "mlu"]
        assert config.original_contents == ("cpu",)
        assert profile.get_torch_profiler(["cpu"], "unused") is original.return_value
        original.assert_called_once()


def test_rollout_hook_is_lazy_and_preserves_non_tpu_loader(registry):
    backend = importlib.import_module("verl_hardware_plugin.accelerators.tpu.registration")
    original = mock.Mock(return_value=type("OriginalReplica", (), {}))

    class RolloutRegistry:
        _registry = {"vllm": original}

        @classmethod
        def register(cls, name, loader):
            cls._registry[name] = loader

    fake_replica = ModuleType("verl.workers.rollout.replica")
    fake_replica.RolloutReplicaRegistry = RolloutRegistry
    fake_device = ModuleType("verl.utils.device")
    fake_device.get_resource_name = mock.Mock(return_value="GPU")
    fake_tpu = ModuleType("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
    fake_tpu.TPUvLLMReplica = type("TPUReplica", (), {})
    real_import = builtins.__import__

    def reject_vllm(name, *args, **kwargs):
        assert "vllm" not in name, "rollout registration eagerly imported vLLM"
        return real_import(name, *args, **kwargs)

    with mock.patch.dict(sys.modules, {fake_replica.__name__: fake_replica, fake_device.__name__: fake_device}):
        with mock.patch.object(builtins, "__import__", side_effect=reject_vllm):
            backend.register_rollout()
            original.assert_not_called()
            assert RolloutRegistry._registry["vllm"]() is original.return_value
        original.assert_called_once_with()

        fake_device.get_resource_name.return_value = "TPU"
        with mock.patch.dict(sys.modules, {fake_tpu.__name__: fake_tpu}):
            assert RolloutRegistry._registry["vllm"]() is fake_tpu.TPUvLLMReplica
        original.assert_called_once_with()


def test_unknown_stage_is_rejected(registry):
    with pytest.raises(ValueError, match="Unknown registration stage"):
        registry.register_stage("unknown")
