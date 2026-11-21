# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU-side tests for PlatformENFLAME and the torch_gcu probes behind it.

No GCU is required: the probes and ``torch.gcu`` are faked. These cover the
contract verl core relies on (platform key, backend string, Ray resource,
device env var, IPC answer) and the torch_gcu shims applied at init.
"""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.enflame import platform_enflame
from verl_hardware_plugin.accelerators.enflame.platform_enflame import PlatformENFLAME


@pytest.fixture
def platform():
    return PlatformENFLAME()


def test_platform_is_registered_under_enflame():
    from verl.plugin.platform.platform_manager import PlatformRegistry

    assert "enflame" in PlatformRegistry.registered_names()
    assert PlatformRegistry.get("enflame") is PlatformENFLAME


def test_platform_identity(platform):
    assert platform.device_name == "gcu"
    assert platform.vendor_name == "enflame"
    assert platform.visible_devices_envvar() == "TOPS_VISIBLE_DEVICES"
    assert platform.ray_resource_name() == "GPU"
    assert platform.ray_resource_options(4.0) == {"num_gpus": 4.0}
    assert platform.ray_noset_envvars() == ["RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES"]


@pytest.mark.parametrize(
    ("use_flagcx", "expected"),
    [(None, "eccl"), ("", "eccl"), ("0", "eccl"), ("1", "flagcx"), ("true", "flagcx"), ("TRUE", "flagcx")],
)
def test_communication_backend_follows_use_flagcx(platform, monkeypatch, use_flagcx, expected):
    if use_flagcx is None:
        monkeypatch.delenv("USE_FLAGCX", raising=False)
    else:
        monkeypatch.setenv("USE_FLAGCX", use_flagcx)

    assert platform.communication_backend_name() == expected


@pytest.mark.parametrize(("is_available", "expected"), [(lambda: True, True), (lambda: False, False)])
def test_gcu_available_requires_an_available_device(monkeypatch, is_available, expected):
    monkeypatch.setattr(platform_enflame, "_ensure_torch_gcu", lambda: True)
    monkeypatch.setattr(torch, "gcu", SimpleNamespace(is_available=is_available), raising=False)

    assert platform_enflame.gcu_available() is expected


def test_gcu_available_is_false_without_torch_gcu(monkeypatch):
    monkeypatch.setattr(platform_enflame, "_ensure_torch_gcu", lambda: False)

    assert platform_enflame.gcu_available() is False


def test_gcu_available_survives_a_failing_probe(monkeypatch):
    def explode():
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(platform_enflame, "_ensure_torch_gcu", lambda: True)
    monkeypatch.setattr(torch, "gcu", SimpleNamespace(is_available=explode), raising=False)

    assert platform_enflame.gcu_available() is False


@pytest.mark.parametrize("available", [True, False])
def test_platform_availability_delegates_to_the_gcu_probe(platform, monkeypatch, available):
    monkeypatch.setattr(platform_enflame, "gcu_available", lambda: available)

    assert platform.is_available() is available
    assert platform.is_platform_available() is available


def test_ipc_is_reported_supported_to_skip_the_shm_fallback(platform):
    # verl computes use_shm = not is_support_ipc(); torch_gcu has no Python
    # multiprocessing SHM path, so this must stay True.
    assert platform.is_ipc_supported() is True


def test_rollout_env_vars_include_the_gcu_stability_settings(platform):
    assert platform.rollout_env_vars() == {
        "NCCL_CUMEM_ENABLE": "0",
        "TORCH_ECCL_AVOID_RECORD_STREAMS": "1",
        "TORCHGCU_INDUCTOR_ENABLE": "0",
        "TORCHDYNAMO_DISABLE": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }


def test_rollout_env_vars_returns_a_fresh_copy(platform):
    env_vars = platform.rollout_env_vars()
    env_vars["TORCHDYNAMO_DISABLE"] = "0"

    assert platform.rollout_env_vars()["TORCHDYNAMO_DISABLE"] == "1"


def test_platform_exposes_no_collective_shim_or_cudart(platform):
    assert platform.get_collective_module() is None
    assert platform.cudart() is None


@pytest.fixture
def fresh_patch_state(monkeypatch):
    monkeypatch.setattr(platform_enflame, "_gcu_runtime_patched", False)


def test_patch_gcu_runtime_adds_the_missing_shims(fresh_patch_state):
    # torch.gcu.Stream exposes gcu_stream on the class (a C-level descriptor),
    # which is what the shim looks for.
    class Stream:
        gcu_stream = 1234

    gcu = ModuleType("fake_gcu")
    gcu.Stream = Stream

    platform_enflame._patch_gcu_runtime(gcu)

    assert gcu.ipc_collect() is None
    assert Stream().cuda_stream == 1234


def test_patch_gcu_runtime_keeps_native_attributes(fresh_patch_state):
    class Stream:
        gcu_stream = 1234
        cuda_stream = "native"

    gcu = ModuleType("fake_gcu")
    gcu.Stream = Stream
    gcu.ipc_collect = lambda: "native"

    platform_enflame._patch_gcu_runtime(gcu)

    assert gcu.ipc_collect() == "native"
    assert Stream().cuda_stream == "native"


def test_patch_gcu_runtime_is_applied_only_once(fresh_patch_state):
    first = ModuleType("first")
    second = ModuleType("second")

    platform_enflame._patch_gcu_runtime(first)
    platform_enflame._patch_gcu_runtime(second)

    assert hasattr(first, "ipc_collect")
    assert not hasattr(second, "ipc_collect")
