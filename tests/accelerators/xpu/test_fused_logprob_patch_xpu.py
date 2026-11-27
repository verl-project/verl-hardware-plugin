# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The XPU fused-logprob patch must switch verl's Triton kernels to the tensor-descriptor path
only on a measured device with Triton >= 3.9, and must be a no-op everywhere else."""

import sys
import types
from unittest import mock

import pytest

from verl_hardware_plugin.accelerators.xpu.patches import fused_logprob_patch as patch_mod


class _FakeKernel:
    def __init__(self):
        cfg = types.SimpleNamespace(kwargs={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP": 16})
        self.configs = [cfg]
        self.cache = {"stale": 1}


def _fake_kernels_module():
    mod = types.ModuleType(patch_mod._KERNELS_MODULE)
    mod.SUPPORT_CUDA_TMA = False
    for name in patch_mod._TILED_KERNELS:
        setattr(mod, name, _FakeKernel())
    return mod


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    patch_mod._applied = False
    monkeypatch.delenv("VERL_XPU_FUSED_LOGPROB_PATCH", raising=False)
    monkeypatch.delenv("VERL_XPU_FUSED_LOGPROB_TILE", raising=False)
    monkeypatch.delitem(sys.modules, patch_mod._KERNELS_MODULE, raising=False)
    before = list(sys.meta_path)
    try:  # the patch may replace Triton's allocator ContextVar; restore it on teardown
        import triton.runtime._allocation as _alloc

        monkeypatch.setattr(_alloc, "_allocator", _alloc._allocator)
    except (ImportError, AttributeError):
        pass
    yield
    sys.meta_path[:] = before
    patch_mod._applied = False


def _patch_env(
    monkeypatch, *, xpu=True, triton=(3, 9), name="Intel(R) Arc(TM) Pro B60 Graphics", has_td=True, names=None
):
    monkeypatch.setattr(patch_mod, "_xpu_available", lambda: xpu)
    monkeypatch.setattr(patch_mod, "_triton_version", lambda: triton)
    import torch

    devices = names or [name]
    monkeypatch.setattr(
        torch,
        "xpu",
        types.SimpleNamespace(get_device_name=lambda i: devices[i], device_count=lambda: len(devices)),
        raising=False,
    )
    import triton.language as tl

    if has_td:
        monkeypatch.setattr(tl, "make_tensor_descriptor", object(), raising=False)
    elif hasattr(tl, "make_tensor_descriptor"):
        monkeypatch.delattr(tl, "make_tensor_descriptor")


def _apply_to_loaded(monkeypatch, mod):
    monkeypatch.setitem(sys.modules, patch_mod._KERNELS_MODULE, mod)
    with mock.patch("triton.set_allocator") as set_alloc:
        patch_mod.apply()
    return set_alloc


def test_patches_loaded_module_on_b60(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    mod = _fake_kernels_module()
    set_alloc = _apply_to_loaded(monkeypatch, mod)
    assert mod.SUPPORT_CUDA_TMA is True
    set_alloc.assert_called_once()
    for name in patch_mod._TILED_KERNELS:
        k = getattr(mod, name)
        assert (k.configs[0].kwargs["BLOCK_SIZE_M"], k.configs[0].kwargs["BLOCK_SIZE_N"]) == (64, 128)
        assert k.configs[0].kwargs["GROUP"] == 16  # untouched keys survive
        assert k.configs[0].num_warps == 16 and k.configs[0].num_stages == 2
        assert k.cache == {}


def test_b70_gets_the_b60_tile(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch, name="Intel(R) Arc(TM) Pro B70 Graphics")
    mod = _fake_kernels_module()
    _apply_to_loaded(monkeypatch, mod)
    k = getattr(mod, patch_mod._TILED_KERNELS[0])
    kw = k.configs[0].kwargs
    assert (kw["BLOCK_SIZE_M"], kw["BLOCK_SIZE_N"], k.configs[0].num_stages) == (64, 128, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xpu": False},
        {"triton": (3, 8)},
        {"name": "Intel(R) Arc(TM) A770 Graphics"},
        {"has_td": False},
    ],
)
def test_noop_when_a_gate_fails(monkeypatch, kwargs):
    pytest.importorskip("triton")
    _patch_env(monkeypatch, **kwargs)
    mod = _fake_kernels_module()
    set_alloc = _apply_to_loaded(monkeypatch, mod)
    assert mod.SUPPORT_CUDA_TMA is False
    set_alloc.assert_not_called()
    assert patch_mod._applied is False


def test_disable_switch(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    monkeypatch.setenv("VERL_XPU_FUSED_LOGPROB_PATCH", "0")
    mod = _fake_kernels_module()
    _apply_to_loaded(monkeypatch, mod)
    assert mod.SUPPORT_CUDA_TMA is False


def test_tile_override_lets_unlisted_device_opt_in(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch, name="Intel(R) Arc(TM) A770 Graphics")
    monkeypatch.setenv("VERL_XPU_FUSED_LOGPROB_TILE", "128,128,32,8,3")
    mod = _fake_kernels_module()
    _apply_to_loaded(monkeypatch, mod)
    k = getattr(mod, patch_mod._TILED_KERNELS[0])
    assert k.configs[0].kwargs["BLOCK_SIZE_N"] == 128 and k.configs[0].num_warps == 8 and mod.SUPPORT_CUDA_TMA


def test_bad_override_is_rejected(monkeypatch):
    with pytest.raises(ValueError):
        monkeypatch.setenv("VERL_XPU_FUSED_LOGPROB_TILE", "1,2,3")
        patch_mod._select_tile()


def test_idempotent(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    mod = _fake_kernels_module()
    _apply_to_loaded(monkeypatch, mod)
    with mock.patch("triton.set_allocator") as again:
        patch_mod.apply()
    again.assert_not_called()


def test_import_hook_installed_when_module_not_loaded(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    n = len(sys.meta_path)
    patch_mod.apply()
    assert len(sys.meta_path) == n + 1
    assert isinstance(sys.meta_path[0], patch_mod._PatchAfterImport)


def test_mixed_host_keeps_defaults(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch, names=["Intel(R) Arc(TM) Pro B60 Graphics", "Intel(R) Arc(TM) A770 Graphics"])
    mod = _fake_kernels_module()
    set_alloc = _apply_to_loaded(monkeypatch, mod)
    assert mod.SUPPORT_CUDA_TMA is False
    set_alloc.assert_not_called()


def test_host_with_same_tile_devices_is_patched(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch, names=["Intel(R) Arc(TM) Pro B60 Graphics", "Intel(R) Arc(TM) Pro B70 Graphics"])
    mod = _fake_kernels_module()
    _apply_to_loaded(monkeypatch, mod)
    assert mod.SUPPORT_CUDA_TMA is True


def test_spec_lookup_does_not_consume_the_hook(monkeypatch, tmp_path):
    pytest.importorskip("triton")
    import importlib.machinery
    import importlib.util

    src = tmp_path / "fake_kernels.py"
    src.write_text("X = 1\n")
    hook = patch_mod._PatchAfterImport((64, 128, 32, 16, 2))
    sys.meta_path.insert(0, hook)
    spec = importlib.util.spec_from_file_location(patch_mod._KERNELS_MODULE, src)
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec", staticmethod(lambda *a, **k: spec))
    found = hook.find_spec(patch_mod._KERNELS_MODULE, None)  # what importlib.util.find_spec triggers
    assert found is spec and hook in sys.meta_path
    with mock.patch.object(patch_mod, "_patch_kernels_module") as patched:
        module = importlib.util.module_from_spec(found)
        found.loader.exec_module(module)  # the real import
    patched.assert_called_once()
    assert hook not in sys.meta_path


def test_half_imported_module_is_not_claimed_and_can_be_retried(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    partial = types.ModuleType(patch_mod._KERNELS_MODULE)  # kernels not defined yet: another thread is importing
    monkeypatch.setitem(sys.modules, patch_mod._KERNELS_MODULE, partial)
    with mock.patch("triton.set_allocator"):
        patch_mod.apply()  # must not raise
    assert patch_mod._applied is False
    for name in patch_mod._TILED_KERNELS:
        setattr(partial, name, _FakeKernel())
    partial.SUPPORT_CUDA_TMA = False
    with mock.patch("triton.set_allocator"):
        patch_mod.apply()
    assert patch_mod._applied is True and partial.SUPPORT_CUDA_TMA is True


def test_failed_patch_leaves_a_retry_possible(monkeypatch):
    pytest.importorskip("triton")
    _patch_env(monkeypatch)
    mod = _fake_kernels_module()
    monkeypatch.setitem(sys.modules, patch_mod._KERNELS_MODULE, mod)
    with mock.patch("triton.set_allocator", side_effect=RuntimeError("boom")):
        patch_mod.apply()  # must not raise
    assert patch_mod._applied is False
