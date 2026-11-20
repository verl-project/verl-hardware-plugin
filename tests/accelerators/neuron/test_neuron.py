# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for native Trainium adapters; no accelerator allocation is required."""

import ast
from contextvars import Context
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.neuron.engines.batch import host_jagged, padded_inputs
from verl_hardware_plugin.accelerators.neuron.engines.compat import configure_torch_compile_for_neuron
from verl_hardware_plugin.accelerators.neuron.engines.config import NeuronTorchtitanEngineConfig
from verl_hardware_plugin.accelerators.neuron.platform_neuron import PlatformNeuron
from verl_hardware_plugin.accelerators.neuron.runtime import (
    _CUDAOnlyCheckpointFinder,
    install_local_device_count,
    isolate_checkpoint_imports,
    physical_core,
    prepare_worker_environment,
)
from verl_hardware_plugin.accelerators.neuron.weight_sync.streaming import iter_weight_buckets, validate_box_coverage


def test_compile_settings_survive_a_new_rpc_context(monkeypatch):
    config = torch._dynamo.config
    for name in ("automatic_dynamic_shapes", "recompile_limit"):
        entry = config._config[name]
        monkeypatch.setattr(entry, "default", entry.default)
    limit = max(config.recompile_limit, 64)
    Context().run(configure_torch_compile_for_neuron, limit)
    assert Context().run(lambda: (config.automatic_dynamic_shapes, config.recompile_limit)) == (False, limit)


def test_native_backend_has_no_engine_library_dependency():
    root = Path(__file__).resolve().parents[3]
    for path in (root / "verl_hardware_plugin/accelerators/neuron").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(module.startswith("neuron_rl_core") for module in modules), path
    assert "neuron-rl-core" not in (root / "pyproject.toml").read_text()


def test_platform_resources():
    platform = PlatformNeuron()
    assert (platform.device_name, platform.vendor_name) == ("neuron", "aws")
    assert platform.ray_resource_name() == "neuron_cores"
    assert platform.ray_resource_options(2) == {"resources": {"neuron_cores": 2}}
    assert platform.ray_resource_options(0) == {}
    assert not platform.supports_colocated_worker_groups()
    assert not platform.is_ipc_supported()
    with pytest.raises(ValueError):
        platform.ray_resource_options(0.5)


@pytest.mark.parametrize("role,visible", [("controller", "0"), ("trainer", ""), ("rollout", "")])
def test_cpu_controller_does_not_query_neuron_runtime(monkeypatch, role, visible):
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", role)
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", visible)
    platform = PlatformNeuron()
    platform._device_module = None
    assert not platform.is_available()


@pytest.mark.parametrize("available", [False, True])
def test_rollout_availability_does_not_initialize_training_runtime(monkeypatch, available):
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "rollout")
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0")
    platform = PlatformNeuron()
    platform._device_module = None
    monkeypatch.setattr(platform, "is_platform_available", lambda: available)
    assert platform.is_available() is available


def test_ray_device_count_uses_the_process_view(monkeypatch):
    module = SimpleNamespace(device_count=lambda: 32)
    monkeypatch.setattr(torch, "neuron", module, raising=False)
    install_local_device_count()
    assert module.device_count() == 1
    assert 1 % module.device_count() == 0


def test_worker_keeps_physical_core(monkeypatch):
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "7")
    prepare_worker_environment()
    import os

    assert os.environ["NEURON_RT_VISIBLE_CORES"] == "7"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "2"


def test_worker_refuses_unassigned_cores(monkeypatch):
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "2")
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    with pytest.raises(RuntimeError, match="did not assign"):
        prepare_worker_environment()


@pytest.mark.parametrize("visible,rank,expected", [("7", 0, 7), ("4-7", 2, 6), ("2,6", 1, 6)])
def test_physical_core(visible, rank, expected):
    assert physical_core(visible, rank) == expected


def test_physical_core_rejects_out_of_range():
    with pytest.raises(ValueError):
        physical_core("4-5", 2)


def test_only_cuda_checkpoint_import_is_blocked(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    isolate_checkpoint_imports()
    isolate_checkpoint_imports()
    finders = [finder for finder in sys.meta_path if isinstance(finder, _CUDAOnlyCheckpointFinder)]
    assert len(finders) == 1
    with pytest.raises(ImportError, match="use neuron_nixl"):
        finders[0].find_spec("verl.checkpoint_engine.nixl_checkpoint_engine")
    for module in (
        "nixl._api",
        "verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport",
        "verl.checkpoint_engine.base",
    ):
        assert finders[0].find_spec(module) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"tensor_parallel_size": 0},
        {"tensor_parallel_size": -1},
        {"tensor_parallel_size": 1.5},
        {"attn_type": "varlen"},
        {"spmd_backend": "spmd_types"},
        {"use_remove_padding": True},
        {"param_offload": True},
        {"optimizer_offload": True},
        {"max_seq_len": 513},
        {"dtype": "float16"},
    ],
)
def test_config_rejects_unsupported_settings(overrides):
    with pytest.raises(ValueError):
        NeuronTorchtitanEngineConfig(**overrides)


def test_config_defaults():
    config = NeuronTorchtitanEngineConfig()
    assert config.max_seq_len == 1024
    assert not config.use_remove_padding


def test_static_rows_do_not_cross_sample_boundaries():
    inputs = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    padded, positions, labels, offsets = padded_inputs(inputs, 4, 0)
    assert padded.tolist() == [[1, 2, 3, 0], [4, 5, 0, 0]]
    assert labels.tolist() == [[2, 3, 0, 0], [5, 0, 0, 0]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    jagged = host_jagged(values, offsets)
    assert jagged.values().tolist() == [0, 1, 2, 4, 5]
    jagged.values().sum().backward()
    assert values.grad.tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    with pytest.raises(ValueError):
        padded_inputs(inputs, 2, 0)


def entry(name, shape, numel, offset=0, param_offset=0):
    return SimpleNamespace(name=name, shape=shape, numel=numel, offset=offset, param_offset=param_offset)


def test_streaming_buckets_split_parameters():
    buckets = [
        SimpleNamespace(buf_numel=4, entries=[entry("first", (6,), 4)]),
        SimpleNamespace(buf_numel=4, entries=[entry("first", (6,), 2, param_offset=4), entry("second", (1,), 1, 2)]),
    ]
    weights = [("first", torch.arange(6, dtype=torch.bfloat16)), ("second", torch.tensor([9], dtype=torch.bfloat16))]
    actual = [buffer.tolist() for buffer in iter_weight_buckets(weights, buckets, "cpu")]
    assert actual == [[0, 1, 2, 3], [4, 5, 9, 0]]
    with pytest.raises(ValueError, match="Missing"):
        list(iter_weight_buckets(weights[:1], buckets, "cpu"))
    with pytest.raises(ValueError, match="schedule"):
        list(iter_weight_buckets(list(reversed(weights)), buckets, "cpu"))


def test_coverage_is_fail_closed():
    piece = SimpleNamespace(name="weight", numel=4, dst_box=((0, 2), (0, 2)))
    plan = SimpleNamespace(pieces=[piece], skipped=[])
    validate_box_coverage([plan], {"weight": (2, 2)})
    with pytest.raises(ValueError, match="coverage"):
        validate_box_coverage([plan], {"weight": (2, 3)})
    with pytest.raises(ValueError, match="coverage"):
        validate_box_coverage([plan, plan], {"weight": (2, 2)})
    with pytest.raises(ValueError, match="cannot apply"):
        validate_box_coverage([SimpleNamespace(skipped=["unknown"], pieces=[])], {})


def test_stream_rejects_extra_parameters_and_wrong_dtype():
    bucket = SimpleNamespace(buf_numel=2, entries=[entry("first", (2,), 2)])
    weights = [("first", torch.ones(2, dtype=torch.bfloat16)), ("extra", torch.ones(1, dtype=torch.bfloat16))]
    with pytest.raises(ValueError, match="absent"):
        list(iter_weight_buckets(weights, [bucket], "cpu"))
    with pytest.raises(ValueError, match="bfloat16"):
        list(iter_weight_buckets([("first", torch.ones(2))], [bucket], "cpu"))


def test_coverage_rejects_same_size_overlaps():
    piece = SimpleNamespace(name="weight", numel=2, dst_box=((0, 1), (0, 2)))
    plan = SimpleNamespace(pieces=[piece, piece], skipped=[])
    with pytest.raises(ValueError, match="Overlapping"):
        validate_box_coverage([plan], {"weight": (2, 2)})
