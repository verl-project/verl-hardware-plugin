# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SDK-free tests executing the real backend package and registration imports."""

import builtins
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def plugin():
    """Load the actual root entry point, initially suppressing backend imports."""
    path = Path(__file__).resolve().parents[1] / "verl_hardware_plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location("verl_hardware_plugin", path)
    package = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "verl_hardware_plugin" or name.startswith("verl_hardware_plugin."):
                del sys.modules[name]
        sys.modules[package.__name__] = package
        with mock.patch.object(importlib, "import_module") as load:
            spec.loader.exec_module(package)
        assert load.call_args_list == [mock.call(name) for name in package.BACKEND_MODULES] + [
            mock.call(f"{name}.registration") for name in package.BACKEND_MODULES
        ]
        yield package


PLATFORMS = {
    "xpu": "platform_xpu",
    "mlu": "platform_mlu",
    "metax": "platform_cuda_metax",
    "enflame": "platform_enflame",
    "iluvatar": "platform_cuda_iluvatar",
    "musa": "platform_musa",
    "tpu": "platform_tpu",
    "supa": "platform_supa",
}
ENGINES = {
    "flagos": ("fsdp_flagos", "megatron_flagos"),
    "xpu": ("fsdp_xpu", "megatron_xpu"),
    "mlu": ("fsdp_mlu", "megatron_mlu", "cncl_checkpoint_engine", "cnixl_checkpoint_engine"),
    "metax": ("fsdp_metax", "megatron_metax"),
    "enflame": ("fsdp_enflame", "megatron_enflame"),
    "iluvatar": ("fsdp_iluvatar", "megatron_iluvatar"),
    "musa": ("fsdp_musa", "megatron_musa"),
    "tpu": ("torchtitan_tpu", "tpu_checkpoint_engine", "raiden_checkpoint_engine"),
    "supa": ("fsdp_supa", "megatron_supa"),
}


def backend_package(backend):
    namespace = "integrations" if backend == "flagos" else "accelerators"
    return f"verl_hardware_plugin.{namespace}.{backend}"


def fake_packages(names):
    modules = {}
    for name in names:
        module = modules[name] = ModuleType(name)
        module.__path__ = []
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    return modules


@contextmanager
def implementation_stubs(*, failures=(), real_profiler=False):
    """Replace implementation leaves only; execute real package initializers."""
    events = []
    leaves = {f"{backend_package(backend)}.{module}" for backend, module in PLATFORMS.items()}
    leaves.update(
        f"{backend_package(backend)}.engines.{module}" for backend, engines in ENGINES.items() for module in engines
    )
    profiler_name = f"{backend_package('mlu')}.profilers.torch_profile_mlu"
    if not real_profiler:
        leaves.add(profiler_name)

    def record(name):
        events.append(name)
        if name in failures:
            raise ImportError(f"Optional dependency unavailable: {name}")

    class RolloutRegistry:
        _registry = {"vllm": mock.Mock(return_value=type("OriginalReplica", (), {}))}

        @classmethod
        def register(cls, name, loader):
            record("rollout:vllm")
            cls._registry[name] = loader

    modules = fake_packages(
        (
            "verl",
            "verl.workers",
            "verl.workers.rollout",
            "verl.workers.rollout.replica",
            "verl.utils",
            "verl.utils.device",
        )
    )
    modules["verl.workers.rollout.replica"].RolloutReplicaRegistry = RolloutRegistry
    device = modules["verl.utils.device"]
    device.get_resource_name = mock.Mock(return_value="GPU")
    real_import = builtins.__import__

    def load_leaf(name):
        if name in sys.modules:
            return
        record(name)
        module = ModuleType(name)
        if name == profiler_name:
            module._patch_tool_config = lambda: record("profiler:config")
            module._patch_get_torch_profiler = lambda: record("profiler:torch")
        sys.modules[name] = module

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
        if resolved in leaves:
            load_leaf(resolved)
        for child in fromlist or ():
            leaf = f"{resolved}.{child}"
            if leaf in leaves:
                load_leaf(leaf)
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch.dict(sys.modules, modules), mock.patch.object(builtins, "__import__", side_effect=guarded_import):
        yield SimpleNamespace(events=events, registry=RolloutRegistry, device=device)


def test_one_backend_entry_loads_package_then_registration(plugin):
    with (
        mock.patch.object(plugin, "BACKEND_MODULES", ("new_backend",)),
        mock.patch.object(importlib, "import_module") as load,
    ):
        plugin.load_backends()
    assert load.call_args_list == [mock.call("new_backend"), mock.call("new_backend.registration")]


def test_all_platforms_are_registered_before_any_engine(plugin):
    expected_platforms = [f"{backend_package(name)}.{module}" for name, module in PLATFORMS.items()]
    with implementation_stubs() as state:
        plugin.load_backends()
        assert state.events[: len(expected_platforms)] == expected_platforms
        actual_engines = [name for name in state.events if ".engines." in name]
        assert actual_engines == [
            f"{backend_package(backend)}.engines.{engine}" for backend, engines in ENGINES.items() for engine in engines
        ]
        assert state.events.count("profiler:config") == 1
        assert state.events.count("profiler:torch") == 1
        assert state.events.count("rollout:vllm") == 1
        assert all(f"{package}.registration" in sys.modules for package in plugin.BACKEND_MODULES)
        assert not any("trainium" in name for name in plugin.BACKEND_MODULES)
        assert not any(name.startswith("verl_hardware_plugin.accelerators.trainium") for name in sys.modules)


@pytest.mark.parametrize("backend", ENGINES)
def test_importing_backend_registration_loads_every_component(plugin, backend):
    with implementation_stubs() as state:
        importlib.import_module(f"{backend_package(backend)}.registration")
        expected = []
        if backend in PLATFORMS:
            expected.append(f"{backend_package(backend)}.{PLATFORMS[backend]}")
        expected.extend(f"{backend_package(backend)}.engines.{engine}" for engine in ENGINES[backend])
        if backend == "mlu":
            expected.extend(
                [f"{backend_package(backend)}.profilers.torch_profile_mlu", "profiler:config", "profiler:torch"]
            )
        if backend == "tpu":
            expected.append("rollout:vllm")
        assert state.events == expected


@pytest.mark.parametrize(
    ("failure", "skipped"),
    [
        ("accelerators.mlu.platform_mlu", ()),
        ("accelerators.mlu.engines.fsdp_mlu", ()),
        ("accelerators.mlu.engines.megatron_mlu", ()),
        ("accelerators.mlu.engines.cncl_checkpoint_engine", ("accelerators.mlu.engines.cnixl_checkpoint_engine",)),
        ("accelerators.mlu.profilers.torch_profile_mlu", ("profiler:config", "profiler:torch")),
        ("profiler:config", ("profiler:torch",)),
        ("profiler:torch", ()),
        ("accelerators.tpu.engines.torchtitan_tpu", ()),
        ("accelerators.tpu.engines.tpu_checkpoint_engine", ()),
        ("accelerators.tpu.engines.raiden_checkpoint_engine", ()),
        ("rollout:vllm", ()),
    ],
)
def test_optional_component_failure_does_not_stop_other_registration(plugin, failure, skipped):
    failure = f"verl_hardware_plugin.{failure}" if "." in failure else failure
    skipped = {f"verl_hardware_plugin.{name}" if "." in name else name for name in skipped}
    expected = {f"{backend_package(name)}.{module}" for name, module in PLATFORMS.items()}
    expected.update(
        f"{backend_package(backend)}.engines.{engine}" for backend, engines in ENGINES.items() for engine in engines
    )
    expected.update(
        (f"{backend_package('mlu')}.profilers.torch_profile_mlu", "profiler:config", "profiler:torch", "rollout:vllm")
    )
    with implementation_stubs(failures=(failure,)) as state:
        plugin.load_backends()
        assert failure in state.events
        assert set(state.events) == expected - skipped
        assert all(f"{package}.registration" in sys.modules for package in plugin.BACKEND_MODULES)


def test_missing_backend_does_not_stop_following_backends(plugin):
    def load(name):
        if name.startswith("missing"):
            raise RuntimeError("Broken optional backend")

    with (
        mock.patch.object(plugin, "BACKEND_MODULES", ("missing", "available")),
        mock.patch.object(importlib, "import_module", side_effect=load) as imported,
    ):
        plugin.load_backends()
    assert imported.call_args_list == [
        mock.call("missing"),
        mock.call("available"),
        mock.call("missing.registration"),
        mock.call("available.registration"),
    ]


def test_repeated_loading_uses_import_cache_without_rewrapping_rollout(plugin):
    with implementation_stubs() as state:
        plugin.load_backends()
        first_events = state.events.copy()
        first_loader = state.registry._registry["vllm"]
        plugin.load_backends()
        assert state.events == first_events
        assert state.registry._registry["vllm"] is first_loader


def test_mlu_profiler_patches_remain_idempotent(plugin):
    class ToolConfig:
        def __post_init__(self):
            self.original_contents = tuple(self.contents)

    modules = fake_packages(("verl", "verl.utils", "verl.utils.profiler", "verl.utils.profiler.config"))
    modules["verl.utils.profiler.config"].TorchProfilerToolConfig = ToolConfig
    profile = ModuleType("verl.utils.profiler.torch_profile")
    original = profile.get_torch_profiler = mock.Mock()
    modules[profile.__name__] = profile
    modules["verl.utils.profiler"].torch_profile = profile
    modules["torch"] = ModuleType("torch")

    with implementation_stubs(real_profiler=True), mock.patch.dict(sys.modules, modules):
        backend = importlib.import_module(f"{backend_package('mlu')}.registration")
        patched_config = ToolConfig.__post_init__
        patched_profiler = profile.get_torch_profiler
        assert patched_profiler is not original
        backend.apply_mlu_profiler_patches()
        assert ToolConfig.__post_init__ is patched_config
        assert profile.get_torch_profiler is patched_profiler
        config = ToolConfig()
        config.contents = ["cpu", "mlu"]
        config.__post_init__()
        assert config.contents == ["cpu", "mlu"]
        assert config.original_contents == ("cpu",)
        assert profile.get_torch_profiler(["cpu"], "unused") is original.return_value
        original.assert_called_once()


def test_rollout_hook_is_lazy_and_preserves_non_tpu_loader(plugin):
    fake_tpu = ModuleType("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
    fake_tpu.TPUvLLMReplica = type("TPUReplica", (), {})
    with implementation_stubs() as state:
        original = state.registry._registry["vllm"]
        guarded_import = builtins.__import__

        def reject_vllm(name, *args, **kwargs):
            assert "vllm" not in name, "rollout registration eagerly imported vLLM"
            return guarded_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", side_effect=reject_vllm):
            importlib.import_module(f"{backend_package('tpu')}.registration")
            original.assert_not_called()
            assert state.registry._registry["vllm"]() is original.return_value
        original.assert_called_once_with()

        state.device.get_resource_name.return_value = "TPU"
        with mock.patch.dict(sys.modules, {fake_tpu.__name__: fake_tpu}):
            assert state.registry._registry["vllm"]() is fake_tpu.TPUvLLMReplica
        original.assert_called_once_with()
