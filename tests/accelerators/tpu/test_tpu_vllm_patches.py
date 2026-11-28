# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for TPU rollout."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
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


def _fake_gpu_model_runner_modules():
    runner_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def __init__(self, *, use_async_scheduling: bool = True):
            self.use_async_scheduling = use_async_scheduling
            self.input_batch = types.SimpleNamespace(req_id_to_index={})
            self.requests = {}

        def _update_states(self, scheduler_output):
            # Mirrors vLLM's gpu_model_runner._update_states resume path.
            req_data = scheduler_output.scheduled_cached_reqs
            for i, req_id in enumerate(req_data.req_ids):
                req_index = self.input_batch.req_id_to_index.get(req_id)
                num_output_tokens = req_data.num_output_tokens[i]
                if req_index is None and self.use_async_scheduling and num_output_tokens > 0:
                    resumed = req_data.all_token_ids[req_id]
                    self.requests[req_id].output_token_ids = resumed[-num_output_tokens:]
            return "updated"

    runner_mod.GPUModelRunner = GPUModelRunner
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.gpu_model_runner": runner_mod,
    }
    return modules, runner_mod


def _req_state(prompt, output):
    return types.SimpleNamespace(prompt_token_ids=prompt, output_token_ids=output)


def test_patch_tpu_async_scheduling_update_states_backfills_missing_all_token_ids():
    modules, runner_mod = _fake_gpu_model_runner_modules()
    with mock.patch.dict("sys.modules", modules):
        tpu_vllm_patches.patch_tpu_async_scheduling_update_states()
        tpu_vllm_patches.patch_tpu_async_scheduling_update_states()  # idempotent

        runner = runner_mod.GPUModelRunner(use_async_scheduling=True)
        runner.requests["req-missing"] = _req_state([1, 2], [11, 22, 33])
        runner.requests["req-present"] = _req_state([3], [44, 55])
        runner.requests["req-in-batch"] = _req_state([4], [66])
        runner.requests["req-no-output"] = _req_state([5, 6], [])
        runner.input_batch.req_id_to_index["req-in-batch"] = 0

        req_data = types.SimpleNamespace(
            req_ids=["req-missing", "req-present", "req-in-batch", "req-no-output"],
            all_token_ids={"req-present": [3, 44, 55]},
            num_output_tokens=[3, 2, 1, 0],
        )
        sched_out = types.SimpleNamespace(scheduled_cached_reqs=req_data)
        # Without the guard the fake (like vLLM) raises KeyError on req-missing.
        assert runner._update_states(sched_out) == "updated"
        assert req_data.all_token_ids["req-missing"] == [1, 2, 11, 22, 33]
        assert runner.requests["req-missing"].output_token_ids == [11, 22, 33]
        assert runner.requests["req-present"].output_token_ids == [44, 55]
        # Requests upstream does not index are left alone.
        assert "req-in-batch" not in req_data.all_token_ids
        assert "req-no-output" not in req_data.all_token_ids


def test_patch_tpu_async_scheduling_update_states_noop_when_sync_scheduling():
    modules, runner_mod = _fake_gpu_model_runner_modules()
    with mock.patch.dict("sys.modules", modules):
        tpu_vllm_patches.patch_tpu_async_scheduling_update_states()

        runner = runner_mod.GPUModelRunner(use_async_scheduling=False)
        runner.requests["req-sync"] = _req_state([1], [11, 22])
        req_data = types.SimpleNamespace(req_ids=["req-sync"], all_token_ids={}, num_output_tokens=[2])
        assert runner._update_states(types.SimpleNamespace(scheduled_cached_reqs=req_data)) == "updated"
        assert req_data.all_token_ids == {}


# ---------------------------------------------------------------------------
# Sub-host StartSession serialization (TPUWorker.init_device)
# ---------------------------------------------------------------------------


def _fake_tpu_worker_module_with_init_device(init_device_fn):
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    class TPUWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.parallel_config = types.SimpleNamespace(world_size=world_size)

        init_device = init_device_fn

    worker_mod.TPUWorker = TPUWorker
    worker_mod._configure_tpu_process_env = lambda rank, local_rank, world_size, local_world_size: None
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }
    return worker_mod, modules


def test_subhost_start_session_hook_is_no_op_for_single_engine_hosts():
    """Engines without SUBHOST_MEGASCALE_PORT_ENV (non-sub-host engines) do not wrap init_device."""

    def orig_init(self):
        return "ok"

    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    with mock.patch.dict(os.environ):
        os.environ.pop(tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV, None)
        _patch_in_fake_worker_process(modules)
        assert not tpu_vllm_patches.subhost_serialize_start_session_enabled()
    assert worker_mod.TPUWorker.init_device is orig_init


@pytest.mark.parametrize("disabled_value", ["0", "false", "OFF", "no"])
def test_subhost_start_session_hook_can_be_disabled_via_env(disabled_value):
    def orig_init(self):
        return "ok"

    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    env = {
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21000",
        tpu_vllm_patches.SUBHOST_SERIALIZE_START_SESSION_ENV: disabled_value,
    }
    with mock.patch.dict(os.environ, env):
        _patch_in_fake_worker_process(modules)
        assert not tpu_vllm_patches.subhost_serialize_start_session_enabled()
    assert worker_mod.TPUWorker.init_device is orig_init


def test_subhost_start_session_hook_is_idempotent_and_patches_init_device(tmp_path, monkeypatch):
    calls = []

    def orig_init(self):
        calls.append((self.rank, self.parallel_config.world_size))
        return "initialized"

    lock_path = str(tmp_path / "start_session.lock")
    monkeypatch.setattr(tpu_vllm_patches, "SUBHOST_START_SESSION_LOCK_FILE", lock_path)
    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    env = {
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21000",
        tpu_vllm_patches.SUBHOST_ENGINE_ID_ENV: "engine-single",
    }
    with mock.patch.dict(os.environ, env):
        os.environ.pop(tpu_vllm_patches.SUBHOST_SERIALIZE_START_SESSION_ENV, None)
        _patch_in_fake_worker_process(modules)
        wrapped = worker_mod.TPUWorker.init_device
        assert wrapped is not orig_init
        with mock.patch.dict("sys.modules", modules):
            tpu_vllm_patches.patch_tpu_worker_subhost_start_session()
        assert worker_mod.TPUWorker.init_device is wrapped  # idempotent
        assert worker_mod.TPUWorker(rank=0, world_size=1).init_device() == "initialized"
    assert calls == [(0, 1)]
    assert os.path.exists(lock_path)


@pytest.mark.parametrize(
    "num_engines, world_size",
    [
        (2, 4),  # 2 x TP=4 on a TPU 7x host
        (4, 2),  # 4 x TP=2 on a TPU 7x host
        (2, 1),  # single-worker path
    ],
)
def test_subhost_start_session_serializes_engines_while_running_siblings_concurrently(
    tmp_path, num_engines, world_size
):
    lock_path = str(tmp_path / "start_session.lock")
    start_barrier = threading.Barrier(num_engines * world_size)
    intra_barriers = {f"eng-{e}": threading.Barrier(world_size) for e in range(num_engines)}
    state_lock = threading.Lock()
    active_engines: dict[str, int] = {}
    max_concurrent_engines = 0
    intervals: dict[str, list[tuple[float, float]]] = {f"eng-{e}": [] for e in range(num_engines)}
    errors: list[BaseException] = []

    def worker_fn(engine_id: str, rank: int):
        try:
            start_barrier.wait(timeout=10.0)
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=world_size,
                engine_id=engine_id,
                lock_path=lock_path,
                peer_appear_timeout_s=15.0,
                safety_timeout_s=15.0,
                poll_interval_s=0.005,
            ):
                t_enter = time.monotonic()
                nonlocal max_concurrent_engines
                with state_lock:
                    active_engines[engine_id] = active_engines.get(engine_id, 0) + 1
                    max_concurrent_engines = max(max_concurrent_engines, len(active_engines))
                # Every rank of this engine must reach init_device concurrently (simulating tpu_dist rendezvous).
                intra_barriers[engine_id].wait(timeout=10.0)
                time.sleep(0.03)
                t_exit = time.monotonic()
                with state_lock:
                    intervals[engine_id].append((t_enter, t_exit))
                    active_engines[engine_id] -= 1
                    if active_engines[engine_id] == 0:
                        del active_engines[engine_id]
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker_fn, args=(f"eng-{e}", r)) for e in range(num_engines) for r in range(world_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)
        assert not t.is_alive()

    assert errors == []
    assert max_concurrent_engines == 1
    # Engine intervals [min_enter, max_exit] must be strictly non-overlapping across co-located engines.
    spans = sorted((min(s for s, _ in ivs), max(e for _, e in ivs), eid) for eid, ivs in intervals.items())
    for eid, ivs in intervals.items():
        assert len(ivs) == world_size, f"engine {eid} ran {len(ivs)} workers, expected {world_size}"
    for (_, prev_end, prev_id), (next_start, _, next_id) in zip(spans[:-1], spans[1:], strict=True):
        assert prev_end <= next_start, f"{prev_id} ({prev_end}) overlapped {next_id} ({next_start})"
    # Ephemeral per-engine coordination files are cleaned up; the persistent host lockfile stays.
    remaining = sorted(os.listdir(tmp_path))
    assert remaining == ["start_session.lock"]


def test_subhost_start_session_releases_lock_before_model_load(tmp_path):
    """After init_device finishes, co-located engines run subsequent stages (load_model / compile) in parallel."""
    lock_path = str(tmp_path / "start_session.lock")
    load_barrier = threading.Barrier(4)  # 2 engines x TP=2 must enter load_model concurrently
    errors: list[BaseException] = []

    def worker_fn(engine_id: str, rank: int):
        try:
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=2,
                engine_id=engine_id,
                lock_path=lock_path,
                peer_appear_timeout_s=10.0,
                safety_timeout_s=10.0,
                poll_interval_s=0.005,
            ):
                time.sleep(0.02)
            # Only succeeds if both engines have released the StartSession lock before load_model.
            load_barrier.wait(timeout=5.0)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker_fn, args=(f"eng-{e}", r)) for e in range(2) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive()
    assert errors == []


def test_subhost_start_session_serializes_across_subprocesses(tmp_path):
    """Two real multi-worker sub-host engines (4 OS processes) serialize strictly via POSIX flock."""
    lock_path = str(tmp_path / "start_session.lock")
    log_path = str(tmp_path / "events.jsonl")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    script = (
        "import json, sys, time\n"
        "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (\n"
        "    serialize_subhost_start_session,\n"
        ")\n"
        "lock_path, log_path, engine_id = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "rank, world_size = int(sys.argv[4]), int(sys.argv[5])\n"
        "with serialize_subhost_start_session(\n"
        "    rank=rank, world_size=world_size, engine_id=engine_id,\n"
        "    lock_path=lock_path, peer_appear_timeout_s=15.0,\n"
        "    safety_timeout_s=15.0, poll_interval_s=0.005,\n"
        "):\n"
        "    t0 = time.time()\n"
        "    time.sleep(0.08)\n"
        "    t1 = time.time()\n"
        "    with open(log_path, 'a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps({'engine': engine_id, 'rank': rank, 't0': t0, 't1': t1}) + '\\n')\n"
    )
    env = {**os.environ, "PYTHONPATH": repo_root}
    procs = [
        subprocess.Popen([sys.executable, "-c", script, lock_path, log_path, f"proc-eng-{e}", str(r), "2"], env=env)
        for e in range(2)
        for r in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=20.0) == 0

    with open(log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 4
    by_engine: dict[str, list[dict]] = {}
    for rec in records:
        by_engine.setdefault(rec["engine"], []).append(rec)
    spans = sorted((min(r["t0"] for r in rs), max(r["t1"] for r in rs), eid) for eid, rs in by_engine.items())
    assert spans[0][1] <= spans[1][0], f"Subprocess engines overlapped: {spans}"


@pytest.mark.parametrize("safety_timeout_s", [10.0, None])
def test_subhost_start_session_siblings_wait_while_other_engine_holds_lock_past_appear_timeout(
    tmp_path, safety_timeout_s
):
    """If another engine holds the host lock longer than peer_appear_timeout_s (e.g. >120s in prod), siblings
    keep waiting while rank 0 holds its leader lock and never enter the guarded section before rank 0."""
    lock_path = str(tmp_path / "start_session.lock")
    peer_hold_s = 0.25
    short_appear_timeout_s = 0.05  # 5x shorter than peer_hold_s (models the old 120s window)

    peer_locked = threading.Event()
    peer_released_at: list[float] = []
    entered_before_peer_release: list[int] = []
    entered_at: dict[int, float] = {}
    intra = threading.Barrier(4)
    errors: list[BaseException] = []

    def peer_engine():
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=0,
            world_size=1,
            engine_id="eng-peer-slow",
            lock_path=lock_path,
            safety_timeout_s=10.0,
            poll_interval_s=0.005,
        ):
            peer_locked.set()
            time.sleep(peer_hold_s)
            peer_released_at.append(time.monotonic())

    def waiting_worker(rank: int):
        try:
            assert peer_locked.wait(timeout=5.0)
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=4,
                engine_id="eng-waiter",
                lock_path=lock_path,
                peer_appear_timeout_s=short_appear_timeout_s,
                safety_timeout_s=safety_timeout_s,
                poll_interval_s=0.005,
            ):
                t_enter = time.monotonic()
                if not peer_released_at:
                    entered_before_peer_release.append(rank)
                entered_at[rank] = t_enter
                intra.wait(timeout=5.0)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    peer = threading.Thread(target=peer_engine)
    workers = [threading.Thread(target=waiting_worker, args=(r,)) for r in range(4)]
    peer.start()
    for w in workers:
        w.start()
    peer.join(timeout=10.0)
    for w in workers:
        w.join(timeout=10.0)
        assert not w.is_alive()

    assert errors == []
    assert entered_before_peer_release == [], (
        f"Workers entered before peer released lock: {entered_before_peer_release}"
    )
    assert set(entered_at) == {0, 1, 2, 3}
    assert min(entered_at.values()) >= peer_released_at[0]


@pytest.mark.parametrize("failing_rank", [0, 2])
def test_subhost_start_session_releases_lock_on_exception(tmp_path, failing_rank):
    """If an engine raises during init_device (either rank 0 or a sibling), the lock is released and peers proceed."""
    lock_path = str(tmp_path / "start_session.lock")
    intra = threading.Barrier(4)
    seen_errors: list[str] = []

    def failing_worker(rank: int):
        try:
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=4,
                engine_id="eng-fail",
                lock_path=lock_path,
                peer_appear_timeout_s=5.0,
                safety_timeout_s=5.0,
                poll_interval_s=0.005,
            ):
                intra.wait(timeout=5.0)
                if rank == failing_rank:
                    raise RuntimeError(f"rank-{rank} init boom")
        except RuntimeError as exc:
            seen_errors.append(str(exc))

    threads = [threading.Thread(target=failing_worker, args=(r,)) for r in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    assert seen_errors == [f"rank-{failing_rank} init boom"]

    # Peer engine acquires the lock immediately without hanging.
    peer_ran = []
    with tpu_vllm_patches.serialize_subhost_start_session(
        rank=0,
        world_size=1,
        engine_id="eng-peer",
        lock_path=lock_path,
        safety_timeout_s=5.0,
        poll_interval_s=0.005,
    ):
        peer_ran.append(True)
    assert peer_ran == [True]


def test_subhost_start_session_aborts_siblings_when_rank0_aborts_before_go(tmp_path):
    """If rank 0 raises before GO, siblings see ABORT and raise instead of entering init_device unserialized."""
    lock_path = str(tmp_path / "start_session.lock")
    sibling_entered = threading.Event()
    sibling_errors: list[BaseException] = []
    real_open = os.open

    def fail_on_host_lock(path, flags, *args, **kwargs):
        if path == lock_path:
            raise OSError("simulated lock open failure")
        return real_open(path, flags, *args, **kwargs)

    def sibling_fn():
        try:
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=1,
                world_size=2,
                engine_id="eng-abort",
                lock_path=lock_path,
                peer_appear_timeout_s=30.0,
                safety_timeout_s=30.0,
                poll_interval_s=0.005,
            ):
                sibling_entered.set()
        except BaseException as exc:
            sibling_errors.append(exc)

    sibling = threading.Thread(target=sibling_fn)
    sibling.start()
    try:
        with mock.patch.object(tpu_vllm_patches.os, "open", side_effect=fail_on_host_lock):
            with pytest.raises(OSError, match="simulated lock open failure"):
                with tpu_vllm_patches.serialize_subhost_start_session(
                    rank=0,
                    world_size=2,
                    engine_id="eng-abort",
                    lock_path=lock_path,
                    peer_appear_timeout_s=30.0,
                    safety_timeout_s=30.0,
                    poll_interval_s=0.005,
                ):
                    pass
        sibling.join(timeout=3.0)
        assert not sibling.is_alive(), "Sibling did not abort promptly on rank-0 ABORT"
        assert not sibling_entered.is_set(), "Sibling must not enter guarded section after rank-0 ABORT"
        assert len(sibling_errors) == 1 and isinstance(sibling_errors[0], RuntimeError)
    finally:
        sibling.join(timeout=5.0)
        assert not sibling.is_alive()


def test_subhost_start_session_aborts_all_ranks_when_safety_timeout_expires(tmp_path):
    """If another engine holds the lock past safety_timeout_s, every rank aborts instead of running unserialized."""
    lock_path = str(tmp_path / "start_session.lock")
    blocker_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(blocker_fd, fcntl.LOCK_EX)
    entered: list[int] = []
    errors: dict[int, BaseException] = {}
    err_lock = threading.Lock()
    try:

        def worker_fn(rank: int):
            try:
                with tpu_vllm_patches.serialize_subhost_start_session(
                    rank=rank,
                    world_size=4,
                    engine_id="eng-safety-timeout",
                    lock_path=lock_path,
                    peer_appear_timeout_s=5.0,
                    safety_timeout_s=0.15,
                    poll_interval_s=0.005,
                ):
                    entered.append(rank)
            except BaseException as exc:
                with err_lock:
                    errors[rank] = exc

        threads = [threading.Thread(target=worker_fn, args=(r,)) for r in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        assert entered == [], f"No rank may enter guarded section on safety timeout; entered={entered}"
        assert set(errors) == {0, 1, 2, 3}
        assert isinstance(errors[0], TimeoutError)
        for r in (1, 2, 3):
            assert isinstance(errors[r], TimeoutError | RuntimeError)
    finally:
        fcntl.flock(blocker_fd, fcntl.LOCK_UN)
        os.close(blocker_fd)


def test_subhost_start_session_aborts_when_peer_never_arrives(tmp_path):
    """Missing sibling or missing leader aborts with TimeoutError instead of proceeding unserialized."""
    lock_path = str(tmp_path / "start_session.lock")
    entered = False
    with pytest.raises(TimeoutError, match="sibling TPU worker"):
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=0,
            world_size=2,
            engine_id="eng-missing-sibling",
            lock_path=lock_path,
            peer_appear_timeout_s=0.08,
            safety_timeout_s=5.0,
            poll_interval_s=0.005,
        ):
            entered = True
    assert not entered

    with pytest.raises(TimeoutError, match="waiting for rank 0"):
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=1,
            world_size=2,
            engine_id="eng-missing-leader",
            lock_path=lock_path,
            peer_appear_timeout_s=0.08,
            safety_timeout_s=5.0,
            poll_interval_s=0.005,
        ):
            entered = True
    assert not entered


def test_subhost_start_session_recovers_after_hard_process_crash(tmp_path):
    """If a process holding the host lock dies abruptly (os._exit), the OS drops flock and peer engines proceed."""
    lock_path = str(tmp_path / "start_session.lock")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    crash_script = (
        "import os, sys\n"
        "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import serialize_subhost_start_session\n"
        "with serialize_subhost_start_session(rank=0, world_size=1, engine_id='eng-crash', lock_path=sys.argv[1]):\n"
        "    os._exit(42)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", crash_script, lock_path],
        env={**os.environ, "PYTHONPATH": repo_root},
        timeout=15.0,
    )
    assert proc.returncode == 42

    ran = False
    with tpu_vllm_patches.serialize_subhost_start_session(
        rank=0,
        world_size=1,
        engine_id="eng-after-crash",
        lock_path=lock_path,
        safety_timeout_s=5.0,
        poll_interval_s=0.005,
    ):
        ran = True
    assert ran is True


def test_subhost_start_session_sibling_aborts_on_rank0_hard_crash(tmp_path):
    """If rank 0 dies abruptly (os._exit) before GO, sibling ranks abort promptly instead of proceeding unserialized."""
    lock_path = str(tmp_path / "start_session.lock")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    # Hold the host lock from the test process so rank 0 enters the leader section and blocks on host_fd.
    blocker_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(blocker_fd, fcntl.LOCK_EX)
    sibling_entered = threading.Event()
    sibling_errors: list[BaseException] = []
    try:
        rank0_script = (
            "import os, sys, threading\n"
            "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (\n"
            "    serialize_subhost_start_session,\n"
            ")\n"
            "threading.Timer(0.25, lambda: os._exit(99)).start()\n"
            "with serialize_subhost_start_session(\n"
            "    rank=0, world_size=2, engine_id='eng-r0-crash',\n"
            "    lock_path=sys.argv[1], peer_appear_timeout_s=30.0,\n"
            "    safety_timeout_s=30.0, poll_interval_s=0.005,\n"
            "):\n"
            "    pass\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", rank0_script, lock_path],
            env={**os.environ, "PYTHONPATH": repo_root},
        )

        def sibling_fn():
            try:
                with tpu_vllm_patches.serialize_subhost_start_session(
                    rank=1,
                    world_size=2,
                    engine_id="eng-r0-crash",
                    lock_path=lock_path,
                    peer_appear_timeout_s=30.0,
                    safety_timeout_s=30.0,
                    poll_interval_s=0.005,
                ):
                    sibling_entered.set()
            except BaseException as exc:
                sibling_errors.append(exc)

        sibling = threading.Thread(target=sibling_fn)
        sibling.start()
        assert proc.wait(timeout=10.0) == 99
        # Sibling detects leader_path flock drop well before safety_timeout_s=30.0 and aborts with RuntimeError.
        sibling.join(timeout=3.0)
        assert not sibling.is_alive(), "Sibling did not detect rank-0 crash promptly"
        assert not sibling_entered.is_set(), "Sibling must not enter guarded section after rank-0 death"
        assert len(sibling_errors) == 1 and isinstance(sibling_errors[0], RuntimeError)
    finally:
        fcntl.flock(blocker_fd, fcntl.LOCK_UN)
        os.close(blocker_fd)


@pytest.mark.parametrize("rank, world_size", [(-1, 2), (2, 2), (0, 0)])
def test_subhost_start_session_rejects_invalid_rank_or_world_size(tmp_path, rank, world_size):
    with pytest.raises(ValueError, match="Expected 0 <= rank < world_size"):
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=rank,
            world_size=world_size,
            engine_id="bad",
            lock_path=str(tmp_path / "start_session.lock"),
        ):
            pass


def test_subhost_worker_hook_sets_4d_single_core_bounds_for_tp1():
    """For TP=1 on TPU 7x, after dropping pod-wide env vars, the worker hook restores explicit 4-D 1,1,1,1 bounds
    and per-slot localhost process addresses so both cores of every chip can run independently."""
    worker_mod, modules = _fake_tpu_worker_module(lambda rank, local_rank, world_size, local_world_size: None)
    env = {
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21100",
        "TPU_WORKER_HOSTNAMES": "pod-worker-0",
        "TPU_PROCESS_ADDRESSES": "pod-worker-0:8476",
        "TPU_PROCESS_PORT": "8476",
        "TORCH_TPU_DEVICES_PER_HOST": "8",
    }
    with mock.patch.dict(os.environ, env), mock.patch.object(os, "makedirs") as makedirs:
        _patch_in_fake_worker_process(modules)
        worker_mod._configure_tpu_process_env(0, 5, 1, 1)  # slot 5 (odd core of chip 2)
        assert os.environ["TPU_VISIBLE_DEVICES"] == "5"
        assert os.environ["TPU_VISIBLE_CHIPS"] == "5"
        assert os.environ["TPU_HOST_BOUNDS"] == "1,1,1,1"
        assert os.environ["TPU_PROCESS_BOUNDS"] == "1,1,1,1"
        assert os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] == "1,1,1,1"
        assert os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] == "1,1,1,1"
        assert os.environ["CLOUD_TPU_TASK_ID"] == "0"
        assert os.environ["TPU_WORKER_ID"] == "0"
        assert os.environ["TPU_WORKER_HOSTNAMES"] == "localhost"
        assert os.environ["TPU_PROCESS_PORT"] == "21102"
        assert os.environ["TPU_PROCESS_ADDRESSES"] == "localhost:21102"
        assert os.environ["MEGASCALE_PORT"] == "21100"
        assert os.environ["MEGASCALE_DEBUG_PORT"] == "21101"
        assert os.environ["TPU_LOG_DIR"] == "/tmp/tpu_logs/rank_5"
        assert "TORCH_TPU_DEVICES_PER_HOST" not in os.environ
    makedirs.assert_called_once_with("/tmp/tpu_logs/rank_5", exist_ok=True)


def _tp1_worker_subprocess_script() -> str:
    """Child-process harness exercising the unified init_device wrapper across real OS processes."""
    return (
        "import json, os, sys, time, types\n"
        "from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches as p\n"
        "lock_path, base_dir, log_path, hold_after_s, sleep_init_s = (\n"
        "    sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])\n"
        ")\n"
        "core = int(os.environ['TPU_LOCAL_RANK_OFFSET'])\n"
        "p.SUBHOST_START_SESSION_LOCK_FILE = lock_path\n"
        "orig_paths = p._tp1_chip_handshake_paths\n"
        "p._tp1_chip_handshake_paths = lambda owner, chip: orig_paths(owner, chip, base_dir=base_dir)\n"
        "p._SUBHOST_INIT_POLL_INTERVAL_S = 0.01\n"
        "worker_mod = types.ModuleType('vllm_torchtpu.worker.tpu_worker')\n"
        "class TPUWorker:\n"
        "    def __init__(self):\n"
        "        self.rank = 0\n"
        "        self.parallel_config = types.SimpleNamespace(world_size=1)\n"
        "    def init_device(self):\n"
        "        t_enter = time.time()\n"
        "        time.sleep(sleep_init_s)\n"
        "        t_exit = time.time()\n"
        "        with open(log_path, 'a', encoding='utf-8') as f:\n"
        "            f.write(json.dumps({'core': core, 'enter': t_enter, 'exit': t_exit}) + '\\n')\n"
        "        return 'ok'\n"
        "worker_mod.TPUWorker = TPUWorker\n"
        "worker_mod._configure_tpu_process_env = lambda *a, **kw: None\n"
        "sys.modules['vllm_torchtpu'] = types.ModuleType('vllm_torchtpu')\n"
        "sys.modules['vllm_torchtpu.worker'] = types.ModuleType('vllm_torchtpu.worker')\n"
        "sys.modules['vllm_torchtpu.worker.tpu_worker'] = worker_mod\n"
        "p._PATCHES_APPLIED = False\n"
        "p.patch_vllm_for_tpu()\n"
        "assert TPUWorker().init_device() == 'ok'\n"
        "if hold_after_s > 0:\n"
        "    time.sleep(hold_after_s)\n"
    )


@pytest.mark.parametrize("serialize_flag", ["1", "0"])
def test_tp1_unified_wrapper_orders_odd_after_even_when_odd_arrives_first(tmp_path, serialize_flag):
    """Odd core arriving BEFORE even core must neither deadlock on the host StartSession lock nor enter init_device
    ahead of its paired even core, whether host-wide serialization is enabled ("1") or opted out ("0")."""
    lock_path = str(tmp_path / "start_session.lock")
    log_path = str(tmp_path / "events.jsonl")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    script = _tp1_worker_subprocess_script()
    base_env = {
        **os.environ,
        "PYTHONPATH": repo_root,
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21040",
        tpu_vllm_patches._SUBHOST_OWNER_ENV: "host-a:123:abcd",
        tpu_vllm_patches.SUBHOST_SERIALIZE_START_SESSION_ENV: serialize_flag,
        "TORCH_TPU_TOPOLOGY": "1,1,1,1",
    }
    # Launch odd core (core 3, chip 1) 0.2s BEFORE even core (core 2, chip 1).
    p_odd = subprocess.Popen(
        [sys.executable, "-c", script, lock_path, str(tmp_path), log_path, "0.0", "0.1"],
        env={
            **base_env,
            "TPU_LOCAL_RANK_OFFSET": "3",
            tpu_vllm_patches.SUBHOST_ENGINE_ID_ENV: "eng-odd-3",
        },
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    # Even core holds its alive lock for 0.6s after init_device so odd core observes a live even core.
    p_even = subprocess.Popen(
        [sys.executable, "-c", script, lock_path, str(tmp_path), log_path, "0.6", "0.2"],
        env={
            **base_env,
            "TPU_LOCAL_RANK_OFFSET": "2",
            tpu_vllm_patches.SUBHOST_ENGINE_ID_ENV: "eng-even-2",
        },
        stderr=subprocess.PIPE,
        text=True,
    )
    _, err_odd = p_odd.communicate(timeout=15.0)
    _, err_even = p_even.communicate(timeout=15.0)
    assert p_odd.returncode == 0, f"odd failed: {err_odd}"
    assert p_even.returncode == 0, f"even failed: {err_even}"

    with open(log_path, encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh]
    assert [e["core"] for e in events] == [2, 3]
    assert events[0]["exit"] <= events[1]["enter"]


def test_tp1_chip_order_rejects_stale_ready_marker_when_even_core_exited(tmp_path):
    """A leftover ready marker without an active even-core alive flock must NOT unblock the odd core."""
    alive_path, ready_path = tpu_vllm_patches._tp1_chip_handshake_paths("owner:1", 2, base_dir=str(tmp_path))
    with open(alive_path, "w", encoding="utf-8") as fh:
        fh.write("999999\n")
    with open(ready_path, "w", encoding="utf-8") as fh:
        fh.write("999999\n")
    with pytest.raises(TimeoutError, match="core 5 .* to appear and hold"):
        tpu_vllm_patches._wait_for_tp1_even_core_ready(
            alive_path,
            ready_path,
            core=5,
            peer_appear_timeout_s=0.05,
            safety_timeout_s=5.0,
            poll_interval_s=0.005,
        )


def test_tp1_chip_order_waits_past_peer_appear_while_even_core_holds_alive_lock(tmp_path):
    """While even core holds alive_path (e.g. queued behind other engines on the host lock), odd core waits beyond
    peer_appear_timeout_s up to safety_timeout_s, and fails at safety_timeout_s if ready never arrives."""
    alive_path, ready_path = tpu_vllm_patches._tp1_chip_handshake_paths("owner:2", 1, base_dir=str(tmp_path))
    alive_fd = os.open(alive_path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(alive_fd, fcntl.LOCK_EX)
    try:
        # Publish ready_path AFTER peer_appear_timeout_s (0.04s) has already elapsed: odd core must still succeed.
        timer = threading.Timer(0.10, lambda: tpu_vllm_patches._write_atomic(ready_path, b"123\n"))
        timer.start()
        try:
            tpu_vllm_patches._wait_for_tp1_even_core_ready(
                alive_path,
                ready_path,
                core=3,
                peer_appear_timeout_s=0.04,
                safety_timeout_s=2.0,
                poll_interval_s=0.005,
            )
        finally:
            timer.join(timeout=2.0)

        # If ready_path is removed and never published, safety_timeout_s bounds the wait.
        os.unlink(ready_path)
        with pytest.raises(TimeoutError, match="core 3 .* finish StartSession"):
            tpu_vllm_patches._wait_for_tp1_even_core_ready(
                alive_path,
                ready_path,
                core=3,
                peer_appear_timeout_s=0.02,
                safety_timeout_s=0.08,
                poll_interval_s=0.005,
            )
    finally:
        fcntl.flock(alive_fd, fcntl.LOCK_UN)
        os.close(alive_fd)
