# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for TPU rollout."""

import os
import types
from unittest import mock

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def test_patch_vllm_for_tpu_applies_precision_patches_and_leaves_tpu_worker_untouched():
    """patch_vllm_for_tpu installs precision patches once vllm_torchtpu is present, without patching TPUWorker."""
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    def upstream_reset_encoder_cache(self) -> None:
        pass

    class TPUWorker:
        reset_encoder_cache = upstream_reset_encoder_cache

    worker_mod.TPUWorker = TPUWorker
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }

    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "_apply_tpu_precision_patches") as apply_precision,
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()
        apply_precision.assert_called_once_with()
        assert tpu_vllm_patches._PATCHES_APPLIED is True
        assert TPUWorker.reset_encoder_cache is upstream_reset_encoder_cache


def _fake_tpu_worker_module(configure):
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")
    worker_mod.TPUWorker = type("TPUWorker", (), {})
    worker_mod._configure_tpu_process_env = configure
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }
    return worker_mod, modules


def _patch_in_fake_worker_process(modules):
    """Run patch_vllm_for_tpu as a vLLM worker would, without its process-global multiprocessing patch."""
    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "patch_multiprocessing_for_tpu"),
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()


def test_subhost_worker_hook_is_not_installed_for_other_engines():
    """A TP=8 or multi-host engine (no SUBHOST_MEGASCALE_PORT_ENV) keeps vllm-torchtpu's function as is."""

    def configure(rank, local_rank, world_size, local_world_size):
        pass

    worker_mod, modules = _fake_tpu_worker_module(configure)
    with mock.patch.dict(os.environ):
        os.environ.pop(tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV, None)
        _patch_in_fake_worker_process(modules)
    assert worker_mod._configure_tpu_process_env is configure


def test_subhost_worker_hook_sets_per_worker_ports_and_drops_pod_env():
    calls = []

    def configure(rank, local_rank, world_size, local_world_size):
        calls.append((rank, local_rank, world_size, local_world_size))

    worker_mod, modules = _fake_tpu_worker_module(configure)
    pod_env = {name: "pod" for name in tpu_vllm_patches._POD_TPU_ENV_VARS}
    with (
        mock.patch.dict(os.environ, {**pod_env, "MEGASCALE_PORT": "8081"}),
        mock.patch.object(os, "makedirs") as makedirs,
    ):
        # The second TP=4 engine of a host: device offset 4, MegaScale block from 19080.
        os.environ[tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV] = "19080"
        _patch_in_fake_worker_process(modules)
        with mock.patch.dict("sys.modules", modules):
            tpu_vllm_patches.patch_tpu_worker_subhost_env()  # a second call does not wrap twice
        worker_mod._configure_tpu_process_env(1, 5, 4, 8)

        assert calls == [(1, 5, 4, 8)]  # vllm-torchtpu's own variables are still set first
        assert os.environ["MEGASCALE_PORT"] == "19082"
        assert os.environ["MEGASCALE_DEBUG_PORT"] == "19083"
        assert os.environ["TPU_VISIBLE_DEVICES"] == os.environ["TPU_VISIBLE_CHIPS"] == "5"
        assert os.environ["TPU_LOG_DIR"] == "/tmp/tpu_logs/rank_5"
        assert not set(tpu_vllm_patches._POD_TPU_ENV_VARS) & set(os.environ)
    makedirs.assert_called_once_with("/tmp/tpu_logs/rank_5", exist_ok=True)


def test_subhost_worker_hook_drops_torch_tpu_devices_per_host():
    """A worker of an engine on part of a host must not keep a host-wide TORCH_TPU_DEVICES_PER_HOST (here the 8 of a
    whole TPU 7x host, e.g. exported for the trainer): torch_tpu sizes its node-local store by it."""
    worker_mod, modules = _fake_tpu_worker_module(lambda rank, local_rank, world_size, local_world_size: None)
    env = {"TORCH_TPU_DEVICES_PER_HOST": "8", tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21080"}
    with mock.patch.dict(os.environ, env), mock.patch.object(os, "makedirs"):
        _patch_in_fake_worker_process(modules)
        worker_mod._configure_tpu_process_env(1, 5, 4, 8)  # worker 1 of the second TP=4 engine of the host
        assert "TORCH_TPU_DEVICES_PER_HOST" not in os.environ


def _fake_fp8_quant_modules(
    *, with_native_dummy: bool = False, native_unquantized: bool = False, quantize_prefix: bool = True
):
    calls = []
    fused_moe_mod = types.ModuleType("vllm.model_executor.layers.fused_moe")

    class RoutedExperts:
        pass

    fused_moe_mod.RoutedExperts = RoutedExperts

    configs_mod = types.ModuleType("vllm_torchtpu.layers.adapter.quantization.configs")
    configs_mod.should_quantize_on_load = lambda prefix: quantize_prefix and "experts" in prefix

    fp8_mod = types.ModuleType("vllm_torchtpu.layers.adapter.quantization.fp8")
    fp8_mod.envs = types.SimpleNamespace(MOE_FUSED_EP_KERNEL_IMPL="fused")
    fp8_mod._get_moe_quant_config = mock.Mock(return_value=(None, torch.float8_e4m3fn, None))
    fp8_mod.synchronize_tensors = mock.Mock()
    fp8_mod.moe_routing = types.SimpleNamespace(
        validate_linear_ep_placement=mock.Mock(),
        register_experts_start_buffer=mock.Mock(),
    )
    fp8_mod.adaptive_moe = types.SimpleNamespace(prebuild_adaptive_fused_moe=mock.Mock(return_value="adaptive_op"))
    fp8_mod.prebuild_fused_moe_kernel = mock.Mock()

    fused_moe_ep_mod = types.ModuleType("vllm_torchtpu.layers.adapter.fused_moe_ep")
    fused_moe_ep_mod.FUSED_MOE_EP_OP_ATTR = "_fused_moe_ep_op"
    fused_moe_ep_mod.prebuild_fused_moe_ep = mock.Mock(return_value="ep_op")
    fused_moe_ep_mod.register_score_bias_buffer = mock.Mock()

    class VllmFp8Config:
        def __init__(self, *, is_checkpoint_fp8_serialized=False, activation_scheme="dynamic", vllm_config=None):
            self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
            self.activation_scheme = activation_scheme
            self.vllm_config = vllm_config

    class VllmFp8MoEMethodTPU:
        def __init__(self, quant_config, moe_config):
            self.quant_config = quant_config
            self.moe_config = moe_config

        def _set_tpu_activation(self, layer):
            return "silu"

        def create_weights(
            self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra
        ):
            calls.append(("orig_create", num_experts, hidden_size, intermediate_size_per_partition))

        def process_weights_after_loading(self, layer):
            calls.append(("orig_process", layer))

    if with_native_dummy:
        VllmFp8MoEMethodTPU._create_dummy_online_fp8_weights = lambda self, *args, **kwargs: None

    fp8_mod.VllmFp8Config = VllmFp8Config
    fp8_mod.VllmFp8MoEMethodTPU = VllmFp8MoEMethodTPU

    unquantized_mod = types.ModuleType("vllm_torchtpu.layers.adapter.quantization.unquantized")

    class VllmUnquantizedConfig:
        def __init__(self, has_bias=False):
            self._has_bias = has_bias

        def get_moe_config(self, layer):
            return types.SimpleNamespace(has_bias=self._has_bias)

        if native_unquantized:

            def get_quant_method(self, layer, prefix):
                if isinstance(layer, RoutedExperts):
                    return "native_pr2967"
                return "orig_quant"
        else:

            def get_quant_method(self, layer, prefix):
                return "orig_quant"

    unquantized_mod.VllmUnquantizedConfig = VllmUnquantizedConfig

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.fused_moe": fused_moe_mod,
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.layers": types.ModuleType("vllm_torchtpu.layers"),
        "vllm_torchtpu.layers.adapter": types.ModuleType("vllm_torchtpu.layers.adapter"),
        "vllm_torchtpu.layers.adapter.fused_moe_ep": fused_moe_ep_mod,
        "vllm_torchtpu.layers.adapter.quantization": types.ModuleType("vllm_torchtpu.layers.adapter.quantization"),
        "vllm_torchtpu.layers.adapter.quantization.configs": configs_mod,
        "vllm_torchtpu.layers.adapter.quantization.fp8": fp8_mod,
        "vllm_torchtpu.layers.adapter.quantization.unquantized": unquantized_mod,
    }
    return modules, calls, fp8_mod, fused_moe_ep_mod, unquantized_mod, RoutedExperts


def _fake_moe_layer(*, use_ep=True):
    params = {}

    def register_parameter(name, param):
        params[name] = param
        setattr(layer, name, param)

    layer = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(
            has_bias=False,
            experts_per_token=8,
            moe_parallel_config=types.SimpleNamespace(use_ep=use_ep),
        ),
        renormalize=True,
        weight_loader=object(),
        register_parameter=register_parameter,
        params=params,
    )
    layer.dummy_dtype = torch.bfloat16
    return layer


def test_patch_tpu_fp8_moe_rollout_dummy_load_allocates_fp8_and_skips_requant():
    modules, calls, fp8_mod, fused_moe_ep_mod, _, _ = _fake_fp8_quant_modules()
    with mock.patch.dict("sys.modules", modules):
        tpu_vllm_patches.patch_tpu_fp8_moe_rollout()
        vllm_cfg = types.SimpleNamespace(load_config=types.SimpleNamespace(load_format="dummy"))
        method = fp8_mod.VllmFp8MoEMethodTPU(fp8_mod.VllmFp8Config(vllm_config=vllm_cfg), moe_config=None)
        layer = _fake_moe_layer(use_ep=True)

        # intermediate=130 aligns up to 256 along the 128-element kernel tile dimension.
        method.create_weights(
            layer, num_experts=4, hidden_size=64, intermediate_size_per_partition=130, params_dtype=torch.bfloat16
        )
        assert calls == []
        assert layer._tpu_dummy_fp8_initialized is True
        assert layer.w13_weight.shape == (4, 64, 512)
        assert layer.w13_weight.dtype == torch.float8_e4m3fn
        assert layer.w13_weight.dummy_weight_value == 0
        assert layer.w2_weight.shape == (4, 256, 64)
        assert layer.w2_weight.dtype == torch.float8_e4m3fn
        assert layer.w13_weight_scale_inv.shape == (4, 1, 1, 512)
        assert layer.w13_weight_scale_inv.dtype == torch.float32
        assert layer.w13_weight_scale_inv.dummy_weight_value == 1.0
        assert layer.w2_weight_scale_inv.shape == (4, 1, 1, 64)
        assert layer.w13_weight.weight_loader is layer.weight_loader

        method.process_weights_after_loading(layer)
        assert calls == []
        fp8_mod.moe_routing.validate_linear_ep_placement.assert_called_once_with(layer)
        fp8_mod.moe_routing.register_experts_start_buffer.assert_called_once_with(layer, device=layer.w13_weight.device)
        fp8_mod.prebuild_fused_moe_kernel.assert_called_once_with(topk=8, activation="silu", use_ep=True)
        fused_moe_ep_mod.prebuild_fused_moe_ep.assert_called_once()
        fused_moe_ep_mod.register_score_bias_buffer.assert_called_once_with(layer)
        assert getattr(method, fused_moe_ep_mod.FUSED_MOE_EP_OP_ATTR) == "ep_op"

        # Non-dummy load format falls through to upstream create_weights / process_weights_after_loading.
        vllm_cfg.load_config.load_format = "auto"
        other_layer = _fake_moe_layer()
        method.create_weights(
            other_layer, num_experts=4, hidden_size=64, intermediate_size_per_partition=130, params_dtype=torch.bfloat16
        )
        method.process_weights_after_loading(other_layer)
        assert calls == [("orig_create", 4, 64, 130), ("orig_process", other_layer)]


def test_patch_tpu_fp8_moe_rollout_skips_when_native_fixes_present():
    modules, calls, fp8_mod, _, unquantized_mod, RoutedExperts = _fake_fp8_quant_modules(
        with_native_dummy=True, native_unquantized=True
    )
    orig_create = fp8_mod.VllmFp8MoEMethodTPU.create_weights
    orig_get_quant = unquantized_mod.VllmUnquantizedConfig.get_quant_method
    with mock.patch.dict("sys.modules", modules):
        tpu_vllm_patches.patch_tpu_fp8_moe_rollout()
    assert fp8_mod.VllmFp8MoEMethodTPU.create_weights is orig_create
    assert unquantized_mod.VllmUnquantizedConfig.get_quant_method is orig_get_quant
    assert unquantized_mod.VllmUnquantizedConfig().get_quant_method(RoutedExperts(), "experts") == "native_pr2967"
    assert calls == []


def test_patch_tpu_fp8_moe_rollout_installs_pr2967_routed_experts_shim():
    modules, _, fp8_mod, _, unquantized_mod, RoutedExperts = _fake_fp8_quant_modules()
    with mock.patch.dict("sys.modules", modules):
        tpu_vllm_patches.patch_tpu_fp8_moe_rollout()
        cfg = unquantized_mod.VllmUnquantizedConfig(has_bias=False)
        method = cfg.get_quant_method(RoutedExperts(), "model.layers.0.mlp.experts")
        assert isinstance(method, fp8_mod.VllmFp8MoEMethodTPU)
        assert method.quant_config.is_checkpoint_fp8_serialized is False
        assert method.quant_config.activation_scheme == "dynamic"
        assert cfg.get_quant_method(RoutedExperts(), "model.layers.0.self_attn") == "orig_quant"
        with pytest.raises(NotImplementedError, match="expert bias"):
            unquantized_mod.VllmUnquantizedConfig(has_bias=True).get_quant_method(
                RoutedExperts(), "model.layers.0.mlp.experts"
            )
