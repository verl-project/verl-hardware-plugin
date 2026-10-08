# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


def test_bucket_length_and_env_override():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import bucket_length, get_tpu_seq_bucket_size

    assert get_tpu_seq_bucket_size() == 256
    assert bucket_length(1) == 256
    assert bucket_length(256) == 256
    assert bucket_length(257) == 512
    assert bucket_length(70, bucket_size=64) == 128

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "64"}):
        assert get_tpu_seq_bucket_size() == 64
        assert bucket_length(65) == 128


def test_unwrap_metadata():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import unwrap_metadata

    assert unwrap_metadata([torch.tensor(3.5)]) == 3.5
    assert unwrap_metadata((True, False)) is True
    assert unwrap_metadata("flex") == "flex"


def test_pad_packed_inputs_for_tpu_builds_4d_document_causal_mask():
    from tensordict import TensorDict

    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import pad_packed_inputs_for_tpu

    # Two packed documents of lengths 3 and 2 -> orig_seq_len = 5
    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        _, _, _, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
        )

    assert orig_seq_len == 5
    assert attention_masks is not None
    assert attention_masks.shape == (1, 1, 8, 8)
    assert attention_masks.dtype == torch.bool
    # Document 0 (tokens 0..2) attends causally within [0..2] and not to document 1 (tokens 3..4)
    assert attention_masks[0, 0, 2, 0].item() is True
    assert attention_masks[0, 0, 0, 2].item() is False
    assert attention_masks[0, 0, 3, 2].item() is False
    assert attention_masks[0, 0, 4, 3].item() is True
    # Padded tail (tokens 5..7) has self-attention only (no cross-token attention)
    assert attention_masks[0, 0, 6, 6].item() is True
    assert attention_masks[0, 0, 6, 5].item() is False


def test_pad_packed_inputs_for_tpu_skips_mask_and_honors_aligned_length():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import pad_packed_inputs_for_tpu

    input_ids = torch.nested.nested_tensor(
        [torch.tensor([10, 11, 12]), torch.tensor([20, 21])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_seq_len", 16)

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        padded_ids, padded_pos, padded_labels, attention_masks, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=micro_batch,
            device=torch.device("cpu"),
            build_attention_mask=False,
        )

    assert orig_seq_len == 5
    assert attention_masks is None
    assert padded_ids.shape == (1, 16)
    assert padded_pos.shape == (1, 16)
    assert padded_labels.shape == (1, 16)


def test_splash_block_size_for():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import splash_block_size_for

    assert splash_block_size_for(512) == 512
    assert splash_block_size_for(1024) == 512
    assert splash_block_size_for(768) == 256
    assert splash_block_size_for(256) == 256
    assert splash_block_size_for(384) == 128
    assert splash_block_size_for(64) == 0


def test_apply_splash_attention_tpu_and_cpu_fallback():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        TPUSplashAttention,
        apply_splash_attention_tpu,
    )

    class _DummyInnerAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = (256, 0)

        def forward(self, q, k, v, **kwargs):
            return q + k + v

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = _DummyInnerAttn()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert apply_splash_attention_tpu([model]) == 2
    assert isinstance(model[0].inner_attention, TPUSplashAttention)
    assert model[0].inner_attention.local_window_size == 256
    # Idempotent on repeated calls (e.g. __init__ followed by initialize)
    assert apply_splash_attention_tpu([model]) == 0

    q = torch.ones(1, 256, 2, 4)
    out = model[0].inner_attention(q, q, q)
    assert torch.allclose(out, q * 3)


def test_configure_torch_compile_for_tpu():
    import torch._dynamo

    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import configure_torch_compile_for_tpu

    configure_torch_compile_for_tpu(recompile_limit=64)
    assert torch._dynamo.config.automatic_dynamic_shapes is False
    assert torch._dynamo.config.assume_static_by_default is True
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert torch._dynamo.config.recompile_limit >= 64


def test_resolve_tpu_torchtitan_options_and_config_extension():
    from verl.workers.config import TorchtitanEngineConfig
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        extend_torchtitan_engine_config,
        resolve_tpu_torchtitan_options,
    )

    extend_torchtitan_engine_config()
    cfg_compiled = TorchtitanEngineConfig(use_torch_compile=True, attn_type="varlen")
    assert resolve_tpu_torchtitan_options(cfg_compiled) == (True, True, "DEFER_AND_FUSE")

    cfg_eager = TorchtitanEngineConfig(use_torch_compile=False, attn_type="varlen")
    assert resolve_tpu_torchtitan_options(cfg_eager) == (False, False, None)

    cfg_invalid = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=False,
        use_simple_fsdp=None,
        tpu_eager_mode=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_splash_attention=True"):
        resolve_tpu_torchtitan_options(cfg_invalid)

    cfg_tp = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=True,
        use_simple_fsdp=True,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_simple_fsdp supports pure FSDP/HSDP only"):
        resolve_tpu_torchtitan_options(cfg_tp)


def test_replace_varlen_attention_with_tpu_attention():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        TPUVarlenAttention,
        replace_varlen_attention_with_tpu_attention,
    )

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = torch.nn.Identity()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert replace_varlen_attention_with_tpu_attention([model]) == 2
    assert isinstance(model[0].inner_attention, TPUVarlenAttention)
    assert isinstance(model[1].inner_attention, TPUVarlenAttention)
    # Idempotent when called a second time
    assert replace_varlen_attention_with_tpu_attention([model]) == 0


def test_resolve_tpu_topology_bounds_multi_host():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import resolve_tpu_topology_bounds

    # v6e-8 across two hosts: the mesh spans hosts, so the bounds are the host bounds.
    topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
        total_chips=8, num_nodes=2
    )
    assert (topology, host_bounds, chips_per_host_bounds, chips_per_host) == ("2,4,1", "2,4,1", "1,1,1", "4")

    # v6e-16 used to fall through to "1,1,1", which silently trains on one chip.
    assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=4)[:2] == ("4,4,1", "4,4,1")


def test_resolve_tpu_topology_bounds_single_host():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import resolve_tpu_topology_bounds

    # A 4-chip slice on one host is addressed inside the host, not across hosts.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=1) == ("2,2,1", "1,1,1", "2,2,1", "4")
    # Same chip count spread over two hosts cannot use in-host bounds.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=2) == ("2,2,1", "1,1,1", "1,1,1", "2")


def test_resolve_tpu_topology_bounds_pod_type_and_env_override():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import resolve_tpu_topology_bounds

    # Pod type wins over the chip count: this job holds 8 of a 16-chip slice.
    assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2, pod_type="v6e-16")[0] == "4,4,1"

    # The env var wins over everything, including an unknown chip count.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,3,1"}):
        assert resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)[0] == "2,3,1"

    with mock.patch.dict(os.environ, {"VERL_TPU_CHIPS_PER_HOST": "2"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2)[3] == "2"


def test_resolve_tpu_topology_bounds_raises_on_unknown_slice():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import resolve_tpu_topology_bounds

    # Guessing "1,1,1" here would train on a subset of the slice without any error.
    with pytest.raises(ValueError, match="TORCH_TPU_TOPOLOGY"):
        resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)


def _two_slice_nodes():
    return [
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-1": 1.0}},
        {"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}},
        {"Alive": False, "Resources": {"TPU": 4.0, "tpu-group-9": 1.0}},
    ]


def test_auto_assign_accelerator_type_splits_trainer_and_rollout_slices():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU

    platform = PlatformTPU()
    with (
        mock.patch("ray.is_initialized", return_value=True),
        mock.patch("ray.nodes", return_value=_two_slice_nodes()),
    ):
        assert platform.auto_assign_accelerator_type("global_pool", None) == "tpu-group-0"
        assert platform.auto_assign_accelerator_type("rollout_pool_0", None) == "tpu-group-1"
        assert platform.auto_assign_accelerator_type("rollout_pool_reward_0", None) == "tpu-group-1"
        assert platform.auto_assign_accelerator_type("rollout_pool_0", "tpu-group-7") == "tpu-group-7"


def test_auto_assign_accelerator_type_single_slice_shares_slice():
    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU

    platform = PlatformTPU()
    nodes = [{"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}}]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        assert platform.auto_assign_accelerator_type("rollout_pool_0", None) == "tpu-group-0"


def test_get_ray_init_kwargs_names_the_setup_hook_by_module_path():
    import importlib

    from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU, patch_ray_worker

    hook = PlatformTPU().get_ray_init_kwargs()["runtime_env"]["worker_process_setup_hook"]
    assert hook == "verl_hardware_plugin.accelerators.tpu.platform_tpu.patch_ray_worker"
    module_name, _, func_name = hook.rpartition(".")
    assert getattr(importlib.import_module(module_name), func_name) is patch_ray_worker
