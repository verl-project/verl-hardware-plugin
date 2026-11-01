# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The FP32 sampler / log-prob patches adapt to the installed vllm-torchtpu."""

import os
import sys
import types
from unittest import mock

import torch

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def _install_fake_vllm(runner_cls):
    runner_mod = types.ModuleType("vllm_torchtpu.runner.tpu_runner")
    runner_mod.TPUModelRunner = runner_cls
    outputs_mod = types.ModuleType("vllm.v1.outputs")
    outputs_mod.LogprobsTensors = object
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.runner": types.ModuleType("vllm_torchtpu.runner"),
        "vllm_torchtpu.runner.tpu_runner": runner_mod,
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.outputs": outputs_mod,
    }
    return (
        mock.patch.dict(sys.modules, modules),
        mock.patch.object(tpu_vllm_patches.importlib.util, "find_spec", return_value=object()),
    )


def test_sampler_patch_wraps_bf16_sampler_with_extra_parameters():
    """Newer signatures append parameters (topk_candidates); the BF16 noise is still redrawn in FP32."""
    seen = {}

    class Runner:
        def sample_from_logits(
            self, logits, temperatures, u, top_k, top_p, all_greedy=False, topk_candidates=0
        ):  # draws noise in the dtype it is given
            seen.update(u_dtype=u.dtype, topk_candidates=topk_candidates)
            return logits

    p1, p2 = _install_fake_vllm(Runner)
    with p1, p2:
        tpu_vllm_patches.patch_tpu_sampler()
        assert Runner._verl_fp32_sampler_patched is True
        Runner().sample_from_logits(
            torch.zeros(2, 4), torch.ones(2), torch.zeros(2, 4, dtype=torch.bfloat16), None, None, False, 7
        )
    assert seen == {"u_dtype": torch.float32, "topk_candidates": 7}


def test_sampler_patch_passes_fp32_noise_through():
    """vllm-torchtpu #1747 draws ``u`` in FP32 at the call sites: the wrapper must not redraw it."""
    seen = {}

    class Runner:
        def sample_from_logits(self, logits, temperatures, u, top_k, top_p, all_greedy=False, topk_candidates=0):
            seen.update(u=u, topk_candidates=topk_candidates)
            return logits

    u = torch.rand(2, 4, dtype=torch.float32)
    p1, p2 = _install_fake_vllm(Runner)
    with p1, p2:
        tpu_vllm_patches.patch_tpu_sampler()
        Runner().sample_from_logits(torch.zeros(2, 4), torch.ones(2), u, None, None, False, topk_candidates=3)
    assert seen["u"] is u
    assert seen["topk_candidates"] == 3


def test_logprob_patch_is_skipped_when_vllm_torchtpu_already_uses_fp32():
    """vllm-torchtpu #2024 normalizes log probabilities in FP32 itself: keep its implementation."""

    class Runner:
        def gather_logprobs(self, logits, sampled_tokens):
            return logits.to(torch.float32)

    original_logprobs = Runner.gather_logprobs
    p1, p2 = _install_fake_vllm(Runner)
    with p1, p2:
        tpu_vllm_patches.patch_tpu_logprobs()
    assert Runner.gather_logprobs is original_logprobs
    assert Runner._verl_fp32_logprobs_patched


def test_update_worker_block_size_invokes_platform_hook():
    """vLLM >= 0.31 finalizes hybrid Mamba/attention block sizes after ``load_model`` via ``current_platform``."""
    called = []
    platforms_mod = types.ModuleType("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace(
        update_block_size_for_backend=lambda cfg: called.append(cfg),
    )
    worker = types.SimpleNamespace(vllm_config="sentinel_vllm_config")
    with mock.patch.dict(sys.modules, {"vllm": types.ModuleType("vllm"), "vllm.platforms": platforms_mod}):
        tpu_vllm_patches._update_worker_block_size(worker)
    assert called == ["sentinel_vllm_config"]


def test_init_workers_ray_updates_worker_block_size_after_load_model():
    """``_init_workers_ray`` must run ``_update_worker_block_size`` immediately after ``load_model``."""
    platforms_mod = types.ModuleType("vllm.platforms")
    platforms_mod.current_platform = types.SimpleNamespace(ray_device_key="TPU", additional_env_vars=set())
    ray_env_mod = types.ModuleType("vllm.ray.ray_env")
    ray_env_mod.get_env_vars_to_copy = lambda **_: set()
    net_mod = types.ModuleType("vllm.utils.network_utils")
    net_mod.get_ip = lambda: "10.0.0.1"
    ports = iter([29500, 29501])
    net_mod.get_open_port = lambda: next(ports)
    net_mod.get_distributed_init_method = lambda ip, port: f"tcp://{ip}:{port}"
    ray_utils_mod = types.ModuleType("vllm.v1.executor.ray_utils")
    ray_utils_mod.WORKER_SPECIFIC_ENV_VARS = set()
    tpu_envs_mod = types.ModuleType("vllm_torchtpu.envs")
    tpu_envs_mod.TORCH_TPU_BASE_PORT = 8070
    rde_mod = types.ModuleType("vllm_torchtpu.executors.ray_distributed_executor")
    rde_mod.RayWorkerWrapper = object

    class _FakeActor:
        get_node_ip = types.SimpleNamespace(remote=lambda: "ip_ref")

    fake_pg = types.SimpleNamespace(bundle_specs=[{"TPU": 1}] * 4)
    rpc_calls = []
    executor = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(world_size=4, tensor_parallel_size=4),
        vllm_config="cfg",
        workers=[],
        _env_vars_for_all_workers=[],
        _get_env_vars_to_be_updated=lambda: [{}] * 4,
        collective_rpc=lambda method, args=(): rpc_calls.append(method),
    )

    fake_modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.platforms": platforms_mod,
        "vllm.ray": types.ModuleType("vllm.ray"),
        "vllm.ray.ray_env": ray_env_mod,
        "vllm.utils": types.ModuleType("vllm.utils"),
        "vllm.utils.network_utils": net_mod,
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.executor": types.ModuleType("vllm.v1.executor"),
        "vllm.v1.executor.ray_utils": ray_utils_mod,
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.envs": tpu_envs_mod,
        "vllm_torchtpu.executors": types.ModuleType("vllm_torchtpu.executors"),
        "vllm_torchtpu.executors.ray_distributed_executor": rde_mod,
    }
    env = {k: v for k, v in os.environ.items() if k not in ("TORCH_TPU_TOPOLOGY", "TPU_ACCELERATOR_TYPE")}
    env["VERL_TPU_PG_IDS"] = "00" * 18
    with (
        mock.patch.dict(sys.modules, fake_modules),
        mock.patch.dict("os.environ", env, clear=True),
        mock.patch.object(tpu_vllm_patches, "PlacementGroup", return_value=fake_pg),
        mock.patch.object(tpu_vllm_patches, "PlacementGroupSchedulingStrategy", return_value="sched"),
        mock.patch.object(tpu_vllm_patches, "ray_tpu_accelerator_name", return_value="v6e-4"),
        mock.patch.object(
            tpu_vllm_patches.ray,
            "remote",
            return_value=lambda _cls: types.SimpleNamespace(remote=lambda **_: _FakeActor()),
        ),
        mock.patch.object(tpu_vllm_patches.ray, "get", return_value=["10.0.0.1"] * 4),
    ):
        tpu_vllm_patches._init_workers_ray(executor, None)

    assert rpc_calls == [
        "adjust_rank",
        "update_environment_variables",
        "init_worker",
        "init_device",
        "load_model",
        tpu_vllm_patches._update_worker_block_size,
    ]
