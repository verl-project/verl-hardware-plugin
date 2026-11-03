# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import ast
import dataclasses
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import pytest
import torch
import torch._dynamo
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import TorchtitanEngineConfig
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPUVarlenAttention,
    bucket_length,
    compute_global_batch_num_tokens,
    configure_torch_compile_for_tpu,
    extend_torchtitan_engine_config,
    get_tpu_seq_bucket_size,
    get_tpu_simple_fsdp_parallelize_fn,
    init_states_without_simple_fsdp_parametrization,
    install_tpu_attention,
    pad_packed_inputs_for_tpu,
    resolve_tpu_torchtitan_options,
    tpu_eager_mode_context,
    unwrap_metadata,
)
from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    PlatformTPU,
    configure_torchrun_tpu_env,
    gke_tpu_slice_topology,
    is_tpu_v7x,
    ray_tpu_accelerator_name,
    resolve_tpu_topology_bounds,
    resolve_tpu_v7x_topology,
    tpu_generation,
    tpu_mesh_env,
    tpu_v7x_process_env,
)


def _fake_torchtitan_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for the torchtitan modules these CPU tests touch, for ``mock.patch.dict(sys.modules, ...)``.

    The TPU engine imports google-pytorch/torchtitan lazily, and this repo's CPU test workflow does not install it.
    """
    attention = types.ModuleType("torchtitan.models.common.attention")
    attention.VarlenInnerAttention = type("VarlenInnerAttention", (torch.nn.Module,), {})  # type: ignore[attr-defined]
    # Same as torchtitan's: documents restart at position 0.
    attention.segment_ids_from_positions = lambda positions: (torch.cumsum((positions == 0).int(), dim=-1) - 1).int()  # type: ignore[attr-defined]

    simple_fsdp = types.ModuleType("torchtitan.experiments.graph_trainer.simple_fsdp")
    simple_fsdp._active_parametrization = True  # type: ignore[attr-defined]

    @contextmanager
    def disable_active_parametrization():
        # Like torchtitan's: not reentrant, it always re-enables the parametrization on exit.
        try:
            simple_fsdp._active_parametrization = False  # type: ignore[attr-defined]
            yield
        finally:
            simple_fsdp._active_parametrization = True  # type: ignore[attr-defined]

    simple_fsdp.disable_active_parametrization = disable_active_parametrization  # type: ignore[attr-defined]
    graph_trainer = types.ModuleType("torchtitan.experiments.graph_trainer")
    graph_trainer.simple_fsdp = simple_fsdp  # type: ignore[attr-defined]

    class SplashAttention(torch.nn.Module):
        """Stand-in for torchtitan's TPU SplashAttention: on CPU it always uses the fallback."""

        def __init__(self, is_causal=True, local_window_size=None, original_module=None, auto_block_size=False):
            super().__init__()
            self.is_causal = is_causal
            self.local_window_size = local_window_size
            self.original_module = original_module
            self.auto_block_size = auto_block_size

        def forward(self, q, k, v, **kwargs):
            return self.original_module(q, k, v, **kwargs)

    splash_attention = types.ModuleType("torchtitan.experiments.tpu.kernels.splash_attention")
    splash_attention.SplashAttention = SplashAttention  # type: ignore[attr-defined]

    class KVAllGatherCPInnerAttention(torch.nn.Module):
        def set_local_attention(self, fn):
            self._local_attn_override = fn

        def local_forward(self, q, k, v, **kwargs):
            return q

    cp_attention = types.ModuleType("torchtitan.experiments.tpu.cp_attention")
    cp_attention.KVAllGatherCPInnerAttention = KVAllGatherCPInnerAttention  # type: ignore[attr-defined]
    return {
        attention.__name__: attention,
        graph_trainer.__name__: graph_trainer,
        simple_fsdp.__name__: simple_fsdp,
        splash_attention.__name__: splash_attention,
        cp_attention.__name__: cp_attention,
    }


def test_bucket_length_and_env_override():
    assert get_tpu_seq_bucket_size() == 256
    assert bucket_length(1) == 256
    assert bucket_length(256) == 256
    assert bucket_length(257) == 512
    assert bucket_length(70, bucket_size=64) == 128

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "64"}):
        assert get_tpu_seq_bucket_size() == 64
        assert bucket_length(65) == 128


def test_unwrap_metadata():
    assert unwrap_metadata([torch.tensor(3.5)]) == 3.5
    assert unwrap_metadata((True, False)) is True
    assert unwrap_metadata("flex") == "flex"


def test_pad_packed_inputs_for_tpu_pads_to_bucket_and_aligned_length():
    # Two packed documents of lengths 3 and 2 -> orig_seq_len = 5
    input_ids = torch.nested.nested_tensor([torch.tensor([10, 11, 12]), torch.tensor([20, 21])], layout=torch.jagged)
    position_ids = torch.nested.nested_tensor([torch.tensor([0, 1, 2]), torch.tensor([0, 1])], layout=torch.jagged)

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        ids, pos, labels, orig_seq_len = pad_packed_inputs_for_tpu(
            input_ids=input_ids,
            position_ids=position_ids,
            micro_batch=TensorDict({}, batch_size=[]),
            device=torch.device("cpu"),
        )
    assert orig_seq_len == 5
    # Pad tokens cycle real input_ids so MoE routers spread pad traffic across experts.
    assert ids.tolist() == [10, 11, 12, 20, 21, 10, 11, 12]
    # Pad tokens get position 0: each one is its own document for the attention.
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 0, 0]
    assert labels.tolist() == [11, 12, 20, 21, 10, 0, 0, 0]

    # The common cross-rank length from align_micro_batch_shapes_across_ranks wins over the local bucket.
    micro_batch = TensorDict({}, batch_size=[])
    tu.assign_non_tensor_data(micro_batch, "tpu_padded_seq_len", 16)
    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "8"}):
        ids, pos, labels, _ = pad_packed_inputs_for_tpu(input_ids, position_ids, micro_batch, torch.device("cpu"))
    assert ids.shape == pos.shape == labels.shape == (16,)


def test_tpu_varlen_attention_masks_documents_and_padding():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 8, 4, 16).unbind(0)  # [T=8, H=4, D=16], packed docs of lengths 3 and 2 + 3 pads
    k, v = k[:, :2], v[:, :2]  # GQA: 2 kv heads
    positions = torch.tensor([0, 1, 2, 0, 1, 0, 0, 0])
    with mock.patch.dict(sys.modules, _fake_torchtitan_modules()):
        out = TPUVarlenAttention()(q, k, v, positions=positions)
    assert out.shape == (8, 4, 16)

    def reference(lo, hi):  # causal attention restricted to tokens [lo, hi)
        qd, kd, vd = (t[lo:hi].transpose(0, 1) for t in (q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)))
        return torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, is_causal=True).transpose(0, 1)

    for lo, hi in [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8)]:
        torch.testing.assert_close(out[lo:hi], reference(lo, hi))
    with pytest.raises(NotImplementedError, match="out_transform"):
        TPUVarlenAttention()(q, k, v, positions=positions, out_transform=lambda o, lse: o)
    with pytest.raises(ValueError, match="local_forward"):  # Q shorter than K/V belongs to KV all-gather CP
        TPUVarlenAttention()(q[:4], k, v)


def test_install_tpu_attention_splash_with_dense_cpu_fallback():
    class _DummyInnerAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = (256, 0)

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = _DummyInnerAttn()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    fake_modules = _fake_torchtitan_modules()
    SplashAttention = fake_modules["torchtitan.experiments.tpu.kernels.splash_attention"].SplashAttention
    with mock.patch.dict(sys.modules, fake_modules):
        assert install_tpu_attention([model], use_splash=True) == 2
        # torchtitan's SplashAttention, with per-call block sizes for verl's bucketed lengths and the dense fallback.
        splash = model[0].inner_attention
        assert isinstance(splash, SplashAttention)
        assert splash.auto_block_size and splash.local_window_size == 256
        assert isinstance(splash.original_module, TPUVarlenAttention)
        assert splash.original_module.local_window_size == 256
        assert install_tpu_attention([model], use_splash=True) == 0  # already installed

        # On CPU the splash module runs its dense fallback.
        q = torch.randn(256, 2, 4)
        positions = torch.cat([torch.arange(100), torch.arange(156)])
        torch.testing.assert_close(
            splash(q, q, q, positions=positions), splash.original_module(q, q, q, positions=positions)
        )

        with pytest.raises(ValueError, match="inner_attention"):
            install_tpu_attention([torch.nn.Linear(2, 2)], use_splash=True)


def test_configure_torch_compile_for_tpu():
    configure_torch_compile_for_tpu(recompile_limit=64)
    assert torch._dynamo.config.automatic_dynamic_shapes is False
    assert torch._dynamo.config.assume_static_by_default is True
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert torch._dynamo.config.recompile_limit >= 64


def test_tpu_eager_mode_context_uses_the_public_torch_tpu_api():
    """torch.tpu.set_eager_mode / torch.tpu.EagerMode (torch_tpu >= 0.2.0), not torch_tpu._internal."""
    entered = []

    @contextmanager
    def set_eager_mode(mode):
        entered.append(mode)
        yield

    tpu = SimpleNamespace(
        set_eager_mode=set_eager_mode, EagerMode=SimpleNamespace(DEFER_AND_FUSE="fuse", DEFER_NEVER="never")
    )
    with mock.patch.object(torch, "tpu", tpu, create=True):
        with tpu_eager_mode_context("DEFER_AND_FUSE"):
            assert entered == ["fuse"]
        assert isinstance(tpu_eager_mode_context(None), nullcontext)
    # Without torch_tpu (e.g. these CPU tests) the context does nothing.
    with mock.patch.object(torch, "tpu", SimpleNamespace(), create=True):
        assert isinstance(tpu_eager_mode_context("DEFER_AND_FUSE"), nullcontext)
    assert entered == ["fuse"]


def test_resolve_tpu_torchtitan_options_and_config_extension():
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
        use_simple_fsdp=None,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    assert resolve_tpu_torchtitan_options(cfg_tp) == (True, True, "DEFER_AND_FUSE")

    cfg_pp = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=True,
        use_simple_fsdp=True,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_simple_fsdp supports FSDP/HSDP"):
        resolve_tpu_torchtitan_options(cfg_pp)

    # CP: splash defaults on (CP inner attentions run it as their local attention); SimpleFSDP shards over (dp, cp).
    cfg_cp = SimpleNamespace(
        use_torch_compile=False,
        use_splash_attention=None,
        use_simple_fsdp=True,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=2,
        expert_parallel_size=1,
    )
    assert resolve_tpu_torchtitan_options(cfg_cp) == (True, True, "DEFER_AND_FUSE")

    # EP: SimpleFSDP composes with expert_parallel_size > 1 when pipeline_parallel_size == 1.
    cfg_ep = SimpleNamespace(
        use_torch_compile=True,
        use_splash_attention=None,
        use_simple_fsdp=None,
        tpu_eager_mode=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=8,
    )
    assert resolve_tpu_torchtitan_options(cfg_ep) == (True, True, "DEFER_AND_FUSE")


def test_install_tpu_attention_dense():
    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = torch.nn.Identity()

    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    assert install_tpu_attention([model], use_splash=False) == 2
    assert all(isinstance(block.inner_attention, TPUVarlenAttention) for block in model)
    assert install_tpu_attention([model], use_splash=False) == 0


def test_resolve_tpu_topology_bounds_multi_host():
    # v6e-8 across two hosts: the mesh spans hosts, so the bounds are the host bounds.
    topology, host_bounds, chips_per_host_bounds, chips_per_host = resolve_tpu_topology_bounds(
        total_chips=8, num_nodes=2
    )
    assert (topology, host_bounds, chips_per_host_bounds, chips_per_host) == ("2,4,1", "2,4,1", "1,1,1", "4")

    # v6e-16 used to fall through to "1,1,1", which silently trains on one chip.
    assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=4)[:2] == ("4,4,1", "4,4,1")


def test_resolve_tpu_topology_bounds_single_host():
    # A 4-chip slice on one host is addressed inside the host, not across hosts.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=1) == ("2,2,1", "1,1,1", "2,2,1", "4")
    # Same chip count spread over two hosts cannot use in-host bounds.
    assert resolve_tpu_topology_bounds(total_chips=4, num_nodes=2) == ("2,2,1", "1,1,1", "1,1,1", "2")


def test_resolve_tpu_topology_bounds_pod_type_and_env_override():
    # Pod type wins over the chip count: this job holds 8 of a 16-chip slice.
    assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2, pod_type="v6e-16")[0] == "4,4,1"

    # The env var wins over everything, including an unknown chip count, when its chip count matches total_chips.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,3,1"}):
        assert resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)[0] == "2,3,1"

    # When TaskRunner lands on a different TPU slice (e.g. a 4-chip v6e-4 pod with TORCH_TPU_TOPOLOGY=2,2,1),
    # that pod's webhook variable must not bleed into an 8-chip v6e-8 pool's topology.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,2,1"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2, pod_type="v6e-8") == (
            "2,4,1",
            "2,4,1",
            "1,1,1",
            "4",
        )

    with mock.patch.dict(os.environ, {"VERL_TPU_CHIPS_PER_HOST": "2"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2)[3] == "2"


_WEBHOOK_VARS = ("TORCH_TPU_TOPOLOGY", "TPU_ACCELERATOR_TYPE", "TPU_HOST_BOUNDS", "TPU_CHIPS_PER_HOST_BOUNDS")


def _env_without_webhook():
    return {k: v for k, v in os.environ.items() if k not in _WEBHOOK_VARS}


def test_tpu_generation_reads_every_spelling():
    # GKE's TPU_ACCELERATOR_TYPE, Ray's pod-type and accelerator-type labels, gcloud's pod type.
    assert tpu_generation("tpu7x") == tpu_generation("v7x-16") == tpu_generation("TPU-V7X") == "v7x"
    assert tpu_generation("tpu7x-16") == "v7x"
    assert tpu_generation("v6e-8") == tpu_generation("tpu-v6e-slice") == tpu_generation("TPU-V6E") == "v6e"
    assert tpu_generation("v5p-16") == "v5p"
    assert tpu_generation("") == ""
    with mock.patch.dict(os.environ, {**_env_without_webhook(), "TPU_ACCELERATOR_TYPE": "tpu7x"}, clear=True):
        assert is_tpu_v7x()  # the pod's own variable, as torchtitan's launcher tests it
        assert not is_tpu_v7x("v6e-8")  # the pool's pod type wins over the pod's variable
    with mock.patch.dict(os.environ, _env_without_webhook(), clear=True):
        assert not is_tpu_v7x()
        assert is_tpu_v7x("v7x-16") and is_tpu_v7x("tpu7x-8")


def test_resolve_tpu_topology_bounds_tpu7x_one_process_per_core():
    env = _env_without_webhook()
    with mock.patch.dict(os.environ, {**env, "TPU_ACCELERATOR_TYPE": "tpu7x"}, clear=True):
        # One host: 4 chips = 8 TensorCore devices, each its own process, so the mesh is laid out by process.
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=1) == ("2,2,1,2", "2,2,1,2", "1,1,1,1", "8")
        # An 8-host 2x4x4 slice: 64 devices.
        assert resolve_tpu_topology_bounds(total_chips=64, num_nodes=8) == ("2,4,4,2", "2,4,4,2", "1,1,1,1", "8")
        with pytest.raises(ValueError, match="TPU 7x mesh"):
            resolve_tpu_topology_bounds(total_chips=24, num_nodes=3)
        with pytest.raises(ValueError, match="TPU 7x mesh"):  # sub-host sizes have no validated layout
            resolve_tpu_topology_bounds(total_chips=4, num_nodes=1)
    # The pod type alone selects the TPU 7x path too.
    with mock.patch.dict(os.environ, env, clear=True):
        assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=2, pod_type="v7x-8")[0] == "2,2,2,2"
        assert resolve_tpu_topology_bounds(total_chips=16, num_nodes=4)[0] == "4,4,1"  # v6e as before


def test_resolve_tpu_v7x_topology_prefers_the_webhook_slice_shape():
    """Inside a GKE TPU pod the mesh is the slice's own shape (host bounds x chip bounds, plus the core axis),
    computed as torchtitan's launcher does; the table is only for a pool that uses part of the slice."""
    webhook = {"TPU_HOST_BOUNDS": "1,2,4", "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1"}  # an 8-host 2x4x4 slice
    assert gke_tpu_slice_topology(webhook) == ("2,4,4", 8)
    assert gke_tpu_slice_topology({}) == ("", 0)
    assert resolve_tpu_v7x_topology(64, 8, webhook) == "2,4,4,2"
    # A 4-host pool inside that slice: the table's whole-host shape for 32 devices.
    assert resolve_tpu_v7x_topology(32, 4, webhook) == "2,2,4,2"
    # A slice shape the table does not list is still resolved from the webhook.
    bigger = {"TPU_HOST_BOUNDS": "4,4,2", "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,2"}
    assert resolve_tpu_v7x_topology(512, 32, bigger) == "8,8,4,2"
    assert resolve_tpu_v7x_topology(64, 8, {**webhook, "TORCH_TPU_TOPOLOGY": "4,4,2,2"}) == "4,4,2,2"
    with pytest.raises(ValueError, match="TPU 7x mesh"):
        resolve_tpu_v7x_topology(24, 3, webhook)
    assert tpu_v7x_process_env("2,2,1,2", rank=13, local_rank=5) == {
        "TPU_PROCESS_BOUNDS": "2,2,1,2",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1,1",
        "CLOUD_TPU_TASK_ID": "13",
        "TPU_VISIBLE_DEVICES": "5",
    }


def test_tpu_mesh_env_applies_the_process_layout_on_tpu7x_only():
    env = _env_without_webhook()
    with mock.patch.dict(os.environ, env, clear=True):
        assert tpu_mesh_env(8, 1, rank=5, local_rank=5, pod_type="tpu7x") == {
            "TORCH_TPU_TOPOLOGY": "2,2,1,2",
            "TPU_HOST_BOUNDS": "2,2,1,2",
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1,1",
            "CHIPS_PER_HOST": "8",
            "TPU_PROCESS_BOUNDS": "2,2,1,2",
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1,1",
            "CLOUD_TPU_TASK_ID": "5",
            "TPU_VISIBLE_DEVICES": "5",
        }
        assert tpu_mesh_env(8, 2, rank=5, local_rank=1) == {  # v6e: no per-process variables
            "TORCH_TPU_TOPOLOGY": "2,4,1",
            "TPU_HOST_BOUNDS": "2,4,1",
            "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
            "CHIPS_PER_HOST": "4",
        }


def test_configure_torchrun_tpu_env_tpu7x_takes_the_mesh_from_the_webhook_slice():
    """torchrun on TPU 7x lays the mesh out by process: the slice the GKE webhook describes, plus the core axis.
    The webhook's own 3-D variables are left as they are; torch_tpu replaces them, the task id, the visible
    device and the process port with the per-process layout when it starts."""
    keep = {k: v for k, v in os.environ.items() if not k.startswith(("TPU_", "TORCH_TPU_", "CLOUD_TPU_", "CHIPS_"))}
    keep.pop("PJRT_DEVICE", None)
    torchrun = {
        "LOCAL_RANK": "5",
        "RANK": "13",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
        "TPU_WORKER_HOSTNAMES": "10.0.0.1,10.0.0.2",
        "TPU_ACCELERATOR_TYPE": "tpu7x",
        "TPU_HOST_BOUNDS": "1,1,2",  # the webhook's 3-D values
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
    }
    with mock.patch.dict(os.environ, {**keep, **torchrun}, clear=True):
        configure_torchrun_tpu_env()
        assert os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,2,2"
        assert os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")[13] == "10.0.0.2:8476"
        assert os.environ["TPU_HOST_BOUNDS"] == "1,1,2" and os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] == "2,2,1"
        derived = {"CLOUD_TPU_TASK_ID", "TPU_PROCESS_BOUNDS", "TPU_VISIBLE_DEVICES", "TPU_PROCESS_PORT"}
        assert not derived & set(os.environ)
    # A preset 3-D topology is replaced on TPU 7x.
    with mock.patch.dict(os.environ, {**keep, **torchrun, "TORCH_TPU_TOPOLOGY": "2,2,2"}, clear=True):
        configure_torchrun_tpu_env()
        assert os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,2,2"
    with mock.patch.dict(os.environ, {**keep, **torchrun, "TPU_ACCELERATOR_TYPE": "v6e-16"}, clear=True):
        configure_torchrun_tpu_env()
        assert os.environ["TORCH_TPU_TOPOLOGY"] == "4,4,1"


def test_ray_tpu_accelerator_name_prefers_the_pools_own_nodes(caplog):
    nodes = [
        {
            "Alive": True,
            "Resources": {"TPU": 4.0},
            "NodeManagerAddress": "10.0.0.1",
            "Labels": {"ray.io/tpu-pod-type": "v6e-4"},
        },
        {
            "Alive": True,
            "Resources": {"TPU": 8.0},
            "NodeManagerAddress": "10.0.0.2",
            "Labels": {"ray.io/accelerator-type": "TPU7X"},
        },
        {
            "Alive": False,
            "Resources": {"TPU": 8.0},
            "NodeManagerAddress": "10.0.0.3",
            "Labels": {"ray.io/tpu-pod-type": "dead"},
        },
    ]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        assert ray_tpu_accelerator_name() == "v6e-4"
        assert ray_tpu_accelerator_name(node_ips=["10.0.0.2"]) == "TPU7X"
        assert ray_tpu_accelerator_name(node_ips=["10.0.0.9"]) == ""  # unknown hosts: no guess from another node
    assert "None of the pool's hosts ['10.0.0.9']" in caplog.text
    with mock.patch("ray.is_initialized", return_value=False):
        assert ray_tpu_accelerator_name() == ""


def test_tpu7x_trainer_env_is_the_slice_addresses_and_mesh():
    """A trainer worker on a 2-host TPU 7x pool gets the processes' addresses and the 4-D mesh. torch_tpu derives
    the per-process layout (task id = rank, visible device = local rank, process bounds = the mesh) itself."""
    nodes = [
        {
            "NodeID": f"n{h}",
            "NodeManagerAddress": f"10.0.0.{h}",
            "Alive": True,
            "Resources": {"TPU": 8.0},
            "Labels": {"ray.io/tpu-pod-type": "v7x-16"},
        }
        for h in (1, 2)
    ]
    tables = {f"pg{h}": {"state": "CREATED", "bundles_to_node_id": {i: f"n{h}" for i in range(8)}} for h in (1, 2)}
    platform = PlatformTPU()
    with (
        mock.patch.dict(os.environ, _env_without_webhook(), clear=True),
        mock.patch("ray.is_initialized", return_value=True),
        mock.patch("ray.nodes", return_value=nodes),
        mock.patch("ray._private.state.state.placement_group_table", side_effect=tables.__getitem__),
    ):
        env = platform.get_tpu_env_vars(
            rank=13,
            world_size=16,
            local_rank=5,
            local_world_size=8,
            name_prefix="actor_rollout",
            pgs=[mock.Mock(id="pg1"), mock.Mock(id="pg2")],
        )
    assert env == {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"10.0.0.{h}:{8471 + c}" for h in (1, 2) for c in range(8)),
        "TORCH_TPU_TOPOLOGY": "2,2,2,2",
        "TPU_WORKER_HOSTNAMES": "10.0.0.1,10.0.0.2",
        "LOCAL_RANK": "5",
        "LOCAL_WORLD_SIZE": "8",
        "CHIPS_PER_HOST": "8",
    }


def test_resolve_tpu_topology_bounds_raises_on_unknown_slice():
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
    platform = PlatformTPU()
    nodes = [{"Alive": True, "Resources": {"TPU": 4.0, "tpu-group-0": 1.0}}]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        assert platform.auto_assign_accelerator_type("rollout_pool_0", None) == "tpu-group-0"


def _clear_forwarded_tpu_env(monkeypatch):
    """Unset the variables the platform forwards, so the test shell's own exports do not leak in."""
    for name in list(os.environ):
        if name in ("LIBTPU_INIT_ARGS", "XLA_FLAGS") or name.startswith("VERL_TPU_"):
            monkeypatch.delenv(name)


def test_get_ray_init_kwargs_pins_tpu_platform(monkeypatch):
    _clear_forwarded_tpu_env(monkeypatch)
    assert PlatformTPU().get_ray_init_kwargs() == {"runtime_env": {"env_vars": {"VERL_PLATFORM": "tpu"}}}


def test_driver_tpu_env_is_forwarded_to_ray_workers(monkeypatch):
    """The recipe's exports are set in the driver only; the platform names them in the job's runtime_env."""
    _clear_forwarded_tpu_env(monkeypatch)
    flags = "--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"
    monkeypatch.setenv("LIBTPU_INIT_ARGS", flags)
    monkeypatch.setenv("XLA_FLAGS", "")  # empty: not forwarded, it would replace the workers' own value
    monkeypatch.setenv("VERL_TPU_EAGER_MODE", "DEFER_AND_FUSE")
    monkeypatch.setenv("VERL_TPU_SYNC_RELEASE_BUFFERS", "0")
    monkeypatch.setenv("VERL_LOGGING_LEVEL", "DEBUG")  # not a TPU knob: not forwarded
    env_vars = PlatformTPU().get_ray_init_kwargs()["runtime_env"]["env_vars"]
    assert env_vars == {
        "VERL_PLATFORM": "tpu",
        "LIBTPU_INIT_ARGS": flags,
        "VERL_TPU_EAGER_MODE": "DEFER_AND_FUSE",
        "VERL_TPU_SYNC_RELEASE_BUFFERS": "0",
    }
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/xla")
    assert PlatformTPU().get_ray_init_kwargs()["runtime_env"]["env_vars"]["XLA_FLAGS"] == "--xla_dump_to=/tmp/xla"


def test_get_tpu_seq_bucket_size_respects_pad_to_length_config():
    cfg_enabled = SimpleNamespace(pad_to_length=True, pad_to_length_bucket=512)
    cfg_disabled = SimpleNamespace(pad_to_length=False, pad_to_length_bucket=512)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VERL_TPU_SEQ_BUCKET_SIZE", None)
        assert get_tpu_seq_bucket_size(cfg_enabled) == 512
        assert get_tpu_seq_bucket_size(cfg_disabled) == 256

    with mock.patch.dict(os.environ, {"VERL_TPU_SEQ_BUCKET_SIZE": "128"}):
        assert get_tpu_seq_bucket_size(cfg_enabled) == 128


@mock.patch.dict(sys.modules, _fake_torchtitan_modules())
def test_get_tpu_simple_fsdp_parallelize_fn_uses_torchtitan_tpu_recipe():
    from torchtitan.experiments.graph_trainer import simple_fsdp

    calls = []
    init_flags = []

    class _Model(torch.nn.Module):
        def init_states(self, *, buffer_device=None):
            init_flags.append(simple_fsdp._active_parametrization)

    def _recipe(model, **kwargs):
        calls.append(kwargs)
        return model

    fake_recipe_module = SimpleNamespace(parallelize_qwen3=_recipe)
    with mock.patch("importlib.import_module", return_value=fake_recipe_module) as import_module:
        parallelize_fn = get_tpu_simple_fsdp_parallelize_fn("qwen3")
    import_module.assert_called_once_with("torchtitan.experiments.tpu.qwen3.infra.parallelize")

    model = parallelize_fn(_Model(), parallel_dims="pd", parallelism="cfg", skip_dp=False)
    assert calls == [{"parallel_dims": "pd", "parallelism": "cfg", "skip_dp": False}]
    # torchtitan's Trainer initializes after parallelizing: init must see SimpleFSDP's parametrization disabled.
    model.init_states(buffer_device=None)
    assert init_flags == [False]
    assert simple_fsdp._active_parametrization is True

    with pytest.raises(ValueError, match="use_simple_fsdp=True needs a torchtitan TPU recipe"):
        get_tpu_simple_fsdp_parallelize_fn("deepseek_v3")


@mock.patch.dict(sys.modules, _fake_torchtitan_modules())
def test_init_states_without_simple_fsdp_parametrization_is_reentrant():
    from torchtitan.experiments.graph_trainer import simple_fsdp

    flags = []

    class _Model(torch.nn.Module):
        def init_states(self, *, buffer_device=None):
            flags.append(simple_fsdp._active_parametrization)

    model = _Model()
    init_states_without_simple_fsdp_parametrization(model)
    model.init_states(buffer_device=None)
    # A caller that already disabled the parametrization keeps it disabled after init returns.
    with simple_fsdp.disable_active_parametrization():
        model.init_states(buffer_device=None)
        flags.append(simple_fsdp._active_parametrization)
    assert flags == [False, False, False]
    assert simple_fsdp._active_parametrization is True


def _imported_modules(nodes) -> set[str]:
    names = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_standalone_torchtitan_tpu_engine_decoupled_from_core_torchtitan():
    """TorchTitanTPUEngine subclasses BaseEngine directly without importing verl.workers.engine.torchtitan."""
    from torch.distributed.fsdp import FSDPModule

    from verl.workers.engine.base import BaseEngine
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu, tpu_utils
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        derive_torchtitan_name_and_flavor,
        enable_fsdp_gradient_division,
    )

    assert issubclass(torchtitan_tpu.TorchTitanTPUEngine, BaseEngine)
    assert issubclass(torchtitan_tpu.TorchTitanTPUEngineWithLMHead, torchtitan_tpu.TorchTitanTPUEngine)

    # Nothing in the engine imports core verl's GPU TorchTitan engine, and torchtitan is only imported inside
    # functions, so the plugin registers on machines without torchtitan.
    for module in (torchtitan_tpu, tpu_utils):
        tree = ast.parse(open(module.__file__).read())
        assert not any(n.startswith("verl.workers.engine.torchtitan") for n in _imported_modules(ast.walk(tree)))
        assert not any(n.startswith("torchtitan") for n in _imported_modules(tree.body))

    hf_cfg = SimpleNamespace(
        model_type="qwen3",
        hidden_size=1024,
        num_hidden_layers=28,
        vocab_size=151936,
    )
    fake_cfg = SimpleNamespace(dim=1024, n_layers=28, vocab_size=151936)
    fake_qwen3 = SimpleNamespace(
        model_registry=lambda flavor_name: SimpleNamespace(model=fake_cfg),
        qwen3_configs={"0.6B": fake_cfg},
    )
    with mock.patch("importlib.import_module", return_value=fake_qwen3):
        assert derive_torchtitan_name_and_flavor(hf_cfg) == ("qwen3", "0.6B")

    fsdp_child = mock.MagicMock(spec=FSDPModule)
    root = SimpleNamespace(modules=lambda: [torch.nn.Linear(2, 2), fsdp_child])
    enable_fsdp_gradient_division(root, 4)
    fsdp_child.set_gradient_divide_factor.assert_called_once_with(4.0)


def test_fused_linear_cross_entropy_tpu_and_logprobs():
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
        fused_linear_logprobs_tpu,
        resolve_tpu_use_fused_linear_cross_entropy,
    )

    class _FakeFusedLossLinearWrapper(torch.nn.Linear):
        def forward(self, x: torch.Tensor):
            return x, self.weight.t()

    seen_calls = []

    def _fake_kernel(hidden_2d, labels_1d, weight_hv, *, reduction="sum", **kwargs):
        seen_calls.append((hidden_2d.shape, labels_1d.shape, weight_hv.shape, reduction))
        return torch.full((hidden_2d.shape[0],), 0.5, dtype=torch.float32)

    fake_loss_mod = types.ModuleType("torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss")
    fake_loss_mod.linear_softmax_cross_entropy_loss = _fake_kernel
    fake_loss_mod._verl_reduction_none_patched = True
    fake_sys_modules = {
        "torchtitan": types.ModuleType("torchtitan"),
        "torchtitan.experiments": types.ModuleType("torchtitan.experiments"),
        "torchtitan.experiments.tpu": types.ModuleType("torchtitan.experiments.tpu"),
        "torchtitan.experiments.tpu.kernels": types.ModuleType("torchtitan.experiments.tpu.kernels"),
        "torchtitan.experiments.tpu.kernels.linear_softmax_cross_entropy_loss": fake_loss_mod,
    }
    fake_sys_modules["torchtitan"].experiments = fake_sys_modules["torchtitan.experiments"]  # type: ignore[attr-defined]
    fake_sys_modules["torchtitan.experiments"].tpu = fake_sys_modules["torchtitan.experiments.tpu"]  # type: ignore[attr-defined]
    fake_sys_modules["torchtitan.experiments.tpu"].kernels = fake_sys_modules["torchtitan.experiments.tpu.kernels"]  # type: ignore[attr-defined]
    fake_sys_modules["torchtitan.experiments.tpu"].kernels.linear_softmax_cross_entropy_loss = fake_loss_mod  # type: ignore[attr-defined]

    with mock.patch.dict(sys.modules, fake_sys_modules):
        lm_head = _FakeFusedLossLinearWrapper(16, 32, bias=False)
        x = torch.randn(2, 5, 16)
        out_fused = lm_head(x)
        assert isinstance(out_fused, tuple) and len(out_fused) == 2
        assert out_fused[0] is x
        assert out_fused[1].shape == (16, 32)
        assert (
            resolve_tpu_use_fused_linear_cross_entropy(
                SimpleNamespace(use_fused_linear_cross_entropy=None, use_torch_compile=True)
            )
            is True
        )
        assert (
            resolve_tpu_use_fused_linear_cross_entropy(
                SimpleNamespace(use_fused_linear_cross_entropy=False, use_torch_compile=True)
            )
            is False
        )
        labels = torch.randint(0, 32, (2, 5))
        logprobs = fused_linear_logprobs_tpu(out_fused[0], labels, out_fused[1])
        assert logprobs.shape == (10,)
        assert torch.allclose(logprobs, torch.full((10,), -0.5))
        # Kernel receives B padded to a multiple of 1024 with reduction="none"
        assert seen_calls == [(torch.Size([1024, 16]), torch.Size([1024]), torch.Size([16, 32]), "none")]


@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_prepare_model_outputs_takes_log_probs_from_the_fused_kernel_on_every_pass(calculate_entropy):
    """With the fused LM head, log-probs come from the fused kernel also when entropy is requested.

    The reference, old-log-prob (entropy) and update passes then compute log-probs the same way, so an on-policy
    PPO ratio is exactly 1. Entropy is taken from the logits in fp32, as on GPU.
    """
    import verl.utils.torch_functional as verl_F
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu

    seq_lens, hidden_dim, vocab = [3, 5], 4, 6
    total = sum(seq_lens)
    hidden = torch.randn(1, total, hidden_dim)
    weight_t = torch.randn(hidden_dim, vocab)
    labels = torch.randint(0, vocab, (1, total))
    fused_values = torch.full((total,), -0.25)

    engine = object.__new__(torchtitan_tpu.TorchTitanTPUEngineWithLMHead)
    engine.parallel_dims = SimpleNamespace(tp_enabled=False, cp_enabled=False)
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits

    micro_batch = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [torch.zeros(n, dtype=torch.int64) for n in seq_lens], layout=torch.jagged
            )
        },
        batch_size=[len(seq_lens)],
    )
    tu.assign_non_tensor(micro_batch, use_remove_padding=True, calculate_entropy=calculate_entropy, temperature=1.0)

    with mock.patch.object(torchtitan_tpu, "fused_linear_logprobs_tpu", return_value=fused_values) as fused:
        output = engine.prepare_model_outputs((hidden, weight_t), {"labels": labels}, micro_batch)

    fused.assert_called_once()
    torch.testing.assert_close(output["log_probs"].values(), fused_values)
    if calculate_entropy:
        expected_entropy = verl_F.entropy_from_logits((hidden.squeeze(0) @ weight_t).float())
        torch.testing.assert_close(output["entropy"].values(), expected_entropy)
    else:
        assert "entropy" not in output


def test_offload_and_reload_tpu_model_reuses_cached_cpu_copy_for_frozen_models(tmp_path):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard, init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import move_tpu_params_

    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = torch.nn.Linear(4, 3, bias=False)
        local = model.weight.detach().clone()
        model.weight = torch.nn.Parameter(DTensor.from_local(local, mesh, [Shard(0)]), requires_grad=False)
        weight = model.weight

        # First offload of a frozen model caches the host shard on the parameter.
        move_tpu_params_(model, "cpu", cache_cpu_copy=True)
        cached = weight._tpu_cpu_offload_tensor
        assert weight._local_tensor is cached

        # "meta" stands in for the TPU: the reload swaps the local shard in place, keeping the DTensor and its mesh.
        move_tpu_params_(model, "meta")
        assert model.weight is weight
        assert weight._local_tensor.device.type == "meta"
        assert weight.device_mesh is mesh

        # Offloading again reuses the cached host shard (a device-to-host copy from "meta" would raise).
        move_tpu_params_(model, "cpu", cache_cpu_copy=True)
        assert weight._local_tensor is cached
        assert torch.equal(weight._local_tensor, local)
    finally:
        if owns_group:
            dist.destroy_process_group()


def test_compute_tpu_logprobs_chunked_matches_naive_and_propagates_grad():
    from verl.utils.torch_functional import logprobs_from_logits_naive
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import compute_tpu_logprobs_chunked

    torch.manual_seed(0)
    logits_ref = torch.randn(1, 20, 32, dtype=torch.bfloat16, requires_grad=True)
    logits_chk = logits_ref.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 32, (20,), dtype=torch.int64)

    lp_ref = logprobs_from_logits_naive((logits_ref.squeeze(0) / 1.25).float(), labels)
    lp_chk = compute_tpu_logprobs_chunked(logits_chk, labels, temperature=1.25, chunk_size=7)

    assert torch.allclose(lp_ref, lp_chk, atol=1e-3, rtol=1e-3)
    grad_out = torch.randn_like(lp_ref)
    lp_ref.backward(grad_out)
    lp_chk.backward(grad_out)
    assert logits_ref.grad is not None and logits_chk.grad is not None
    assert torch.allclose(logits_ref.grad, logits_chk.grad, atol=1e-3, rtol=1e-3)


def test_install_sync_dtensor_redistribute_for_tpu_forces_sync_only_in_eager():
    import torch.distributed.tensor._api as dt_api
    import torch.distributed.tensor._dispatch as dt_disp
    import torch.distributed.tensor._redistribute as redist_mod

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_utils

    seen = []

    def _fake_rlt(local_tensor, current_spec, target_spec, *, async_op=False, **kwargs):
        seen.append(async_op)
        return local_tensor

    with (
        mock.patch.object(redist_mod, "redistribute_local_tensor", _fake_rlt),
        mock.patch.object(dt_api, "redistribute_local_tensor", _fake_rlt),
        mock.patch.object(dt_disp, "redistribute_local_tensor", _fake_rlt),
        mock.patch.object(tpu_utils, "_SYNC_DTENSOR_REDISTRIBUTE_PATCHED", False),
    ):
        tpu_utils.install_sync_dtensor_redistribute_for_tpu()
        x = torch.ones(2, 2)
        with mock.patch("torch.compiler.is_compiling", return_value=False):
            redist_mod.redistribute_local_tensor(x, None, None, async_op=True)
        with mock.patch("torch.compiler.is_compiling", return_value=True):
            redist_mod.redistribute_local_tensor(x, None, None, async_op=True)
        assert seen == [False, True]


def test_build_dp_gloo_group_creates_host_gloo_subgroups_when_tp_enabled():
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    engine = object.__new__(TorchTitanTPUEngine)
    engine.engine_config = SimpleNamespace(
        data_parallel_shard_size=2,
        data_parallel_replicate_size=1,
    )
    engine.parallel_dims = SimpleNamespace(
        pp=1,
        cp=1,
        tp=2,
        _global_meshes={"dataloading": SimpleNamespace(mesh=torch.tensor([[0, 1], [2, 3]]))},
    )
    created_groups = []

    def _fake_new_group(ranks=None, backend=None):
        pg = f"gloo_pg_{tuple(ranks)}"
        created_groups.append((list(ranks), backend, pg))
        return pg

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=4),
        mock.patch("torch.distributed.get_backend", return_value="cpu:gloo,tpu:tpu_dist"),
        mock.patch("torch.distributed.get_rank", return_value=2),
        mock.patch("torch.distributed.new_group", side_effect=_fake_new_group),
    ):
        my_pg = engine._build_dp_gloo_group()
        assert my_pg == "gloo_pg_(0, 2)"
        assert [(r, b) for r, b, _ in created_groups] == [([0, 2], "gloo"), ([1, 3], "gloo")]
        assert engine._build_dp_gloo_group() == "gloo_pg_(0, 2)"
        assert len(created_groups) == 2


def _trainer_weights_worker(rank: int, world: int, init_file: str) -> None:
    """One trainer rank of ``test_trainer_weights_exports_local_fsdp_shards_or_gathers``."""
    import torch.distributed as dist
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights, fsdp_dim0_shard

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        generator = torch.Generator().manual_seed(0)  # the same full tensors on every rank
        full = {
            "a.even": torch.randn(8, 4, generator=generator).to(torch.bfloat16),  # bf16 already: .to() is a no-op
            "b.uneven": torch.randn(5, 4, generator=generator),
            "c.cols": torch.randn(4, 6, generator=generator),
            "d.replicated": torch.randn(4, generator=generator),
            "e.plain": torch.randn(3, generator=generator),
            "f.norm": torch.randn(8, generator=generator),  # 1-D: always exported whole
            "model.embed_tokens.weight": torch.randn(6, 4, generator=generator),
        }
        full["lm_head.weight"] = full["model.embed_tokens.weight"]
        stack = torch.randn(2, 3, 4, generator=generator)
        placements = {
            "a.even": Shard(0),
            "b.uneven": Shard(0),
            "c.cols": Shard(1),
            "d.replicated": Replicate(),
            "f.norm": Shard(0),
        }
        placements["model.embed_tokens.weight"] = Shard(0)
        params = {name: distribute_tensor(full[name], mesh, [p]) for name, p in placements.items()}
        params["e.plain"] = full["e.plain"]
        params["lm_head.weight"] = params["model.embed_tokens.weight"]  # tied: one DTensor under two names
        stack_dt = distribute_tensor(stack, mesh, [Shard(0)])

        assert fsdp_dim0_shard(params["a.even"]) == (world, rank)
        for name in ("b.uneven", "c.cols", "d.replicated", "e.plain", "f.norm"):
            assert fsdp_dim0_shard(params[name]) is None, name

        def weights() -> TrainerWeights:
            return TrainerWeights(dict(params), [(stack_dt, [("x.0", (3, 4)), ("x.1", (3, 4))])], "cpu")

        local_names = {"a.even", "model.embed_tokens.weight", "lm_head.weight"}
        exported = list(weights().local_shards())
        assert [name for name, _, _ in exported] == [*params, "x.0", "x.1"]
        for name, tensor, shard in exported:
            assert tensor.dtype == torch.bfloat16, name
            expected = (stack[int(name[-1])] if name.startswith("x.") else full[name]).to(torch.bfloat16)
            if name in local_names:
                assert shard == (world, rank), name
                assert torch.equal(tensor, expected.chunk(world, dim=0)[rank]), name
            else:
                assert shard is None, name
                assert torch.equal(tensor, expected), name
        # The exported block is a copy: an optimizer step after the export must not change what is sent.
        assert exported[0][1].data_ptr() != params["a.even"].to_local().data_ptr()

        # Single use: the parameters are released as they are exported.
        once = weights()
        list(once.local_shards())
        try:
            list(once)
            raise AssertionError("a second export must raise")
        except RuntimeError as e:
            assert "only be exported once" in str(e)

        # The dense parameters may come as several dicts (one per layer), consumed lazily; same export.
        names = list(params)
        chunks = iter([{n: params[n] for n in names[:2]}, {n: params[n] for n in names[2:]}])
        chunked = list(TrainerWeights(chunks, [(stack_dt, [("x.0", (3, 4)), ("x.1", (3, 4))])], "cpu").local_shards())
        assert [(n, s) for n, _, s in chunked] == [(n, s) for n, _, s in exported]
        for (name, tensor, _), (_, ref, _) in zip(chunked, exported, strict=True):
            assert torch.equal(tensor, ref), name

        gathered = list(weights())  # plain iteration: every tensor full, as verl expects
        assert [name for name, _ in gathered] == [*params, "x.0", "x.1"]
        for name, tensor in gathered:
            expected = (stack[int(name[-1])] if name.startswith("x.") else full[name]).to(torch.bfloat16)
            assert torch.equal(tensor, expected), name
    finally:
        dist.destroy_process_group()


def test_trainer_weights_exports_local_fsdp_shards_or_gathers(tmp_path):
    """``local_shards()`` keeps even dim-0 FSDP shards local and gathers the rest; iteration gathers everything."""
    torch.multiprocessing.start_processes(
        _trainer_weights_worker, args=(2, str(tmp_path / "pg")), nprocs=2, join=True, start_method="fork"
    )


def _expert_stacks_worker(rank: int, world: int, init_file: str) -> None:
    """One trainer rank of ``test_trainer_weights_sends_expert_stacks_locally_or_gathers``."""
    import torch.distributed as dist
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights
    from verl_hardware_plugin.accelerators.tpu.rollout.qwen3_moe_layout import expert_stack_wire_name

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        generator = torch.Generator().manual_seed(0)  # the same full stacks on every rank
        # layer -> (experts, placement): whole experts per rank; torchtitan's dim-1 expert sharding (FSDP degree above
        # the expert count, torchtitan 73ebc5f2); an uneven expert split; replicated.
        layouts = {0: (4, Shard(0)), 1: (4, Shard(1)), 2: (3, Shard(0)), 3: (4, Replicate())}
        stacks, full = [], {}
        for layer, (num_experts, placement) in layouts.items():
            stack = torch.randn(num_experts, 6, 4, generator=generator)
            slots = [(f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight", (6, 4)) for e in range(num_experts)]
            stacks.append((distribute_tensor(stack, mesh, [placement]), slots))
            full[layer] = stack

        def weights() -> TrainerWeights:
            return TrainerWeights({}, list(stacks), "cpu", expert_stack_wire=expert_stack_wire_name)

        exported = list(weights().local_shards())
        assert [name for name, _, _ in exported] == [
            f"model.layers.{layer}.mlp.experts.gate_proj.weight" for layer in layouts
        ]
        for layer, (name, tensor, shard) in enumerate(exported):
            stack = full[layer].to(torch.bfloat16)
            assert tensor.dtype == torch.bfloat16 and tensor.ndim == 3, name
            if layer == 0:  # a dim-0 block of whole experts of the 3-D stack: no gather
                assert shard == (world, rank), name
                assert tensor._base is None, name  # a copy in its own buffer, as tpu-sync needs
                assert torch.equal(tensor, stack.chunk(world, dim=0)[rank]), name
            else:  # gathered (or already whole) and sent full
                assert shard is None, name
                assert torch.equal(tensor, stack), name

        gathered = dict(weights())  # plain iteration keeps one HF tensor per expert
        assert len(gathered) == sum(n for n, _ in layouts.values())
        assert torch.equal(gathered["model.layers.1.mlp.experts.3.gate_proj.weight"], full[1][3].to(torch.bfloat16))
    finally:
        dist.destroy_process_group()


def test_trainer_weights_sends_expert_stacks_locally_or_gathers(tmp_path):
    """Expert stacks go out as 3-D tensors: whole-expert dim-0 FSDP shards locally, every other layout gathered."""
    torch.multiprocessing.start_processes(
        _expert_stacks_worker, args=(2, str(tmp_path / "pg")), nprocs=2, join=True, start_method="fork"
    )


def test_install_tpu_attention_keeps_cp_wrappers():
    """CP inner attentions keep their communication and take the TPU attention as their local attention."""

    class _DummyCPInnerAttn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = (256, 0)

        def set_local_attention(self, fn):
            self._local_attn_override = fn

    class _DummyAttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = _DummyCPInnerAttn()

    fake_modules = _fake_torchtitan_modules()
    SplashAttention = fake_modules["torchtitan.experiments.tpu.kernels.splash_attention"].SplashAttention
    model = torch.nn.Sequential(_DummyAttnBlock(), _DummyAttnBlock())
    with mock.patch.dict(sys.modules, fake_modules):
        assert install_tpu_attention([model], use_splash=True) == 2
        assert isinstance(model[0].inner_attention, _DummyCPInnerAttn)
        local = model[0].inner_attention._local_attn_override
        assert isinstance(local, SplashAttention) and local.local_window_size == 256
        assert isinstance(local.original_module, TPUVarlenAttention)
        assert install_tpu_attention([model], use_splash=True) == 0

    model_dense = torch.nn.Sequential(_DummyAttnBlock())
    assert install_tpu_attention([model_dense], use_splash=False) == 1
    assert isinstance(model_dense[0].inner_attention._local_attn_override, TPUVarlenAttention)
    assert install_tpu_attention([model_dense], use_splash=False) == 0


def test_kv_all_gather_cp_keeps_torchtitan_reference_attention():
    """KV all-gather CP uses torchtitan's local_forward (Q shorter than K/V), not TPUVarlenAttention."""
    fake_modules = _fake_torchtitan_modules()
    KVAllGatherCP = fake_modules["torchtitan.experiments.tpu.cp_attention"].KVAllGatherCPInnerAttention

    class _AttnBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_attention = KVAllGatherCP()

    model = torch.nn.Sequential(_AttnBlock())
    with mock.patch.dict(sys.modules, fake_modules):
        # Dense path: nothing to replace, the CP inner attention keeps its own local_forward.
        assert install_tpu_attention([model], use_splash=False) == 0
        assert not hasattr(model[0].inner_attention, "_local_attn_override")
        # Splash path: the kernel falls back to torchtitan's local_forward.
        assert install_tpu_attention([model], use_splash=True) == 1
    assert model[0].inner_attention._local_attn_override.original_module == model[0].inner_attention.local_forward


def test_cp_bucket_size_and_token_count():
    # Sequence bucket size aligns to 2 * cp_size when needed
    cfg_cp3 = SimpleNamespace(pad_to_length=True, pad_to_length_bucket=256, context_parallel_size=3)
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VERL_TPU_SEQ_BUCKET_SIZE", None)
        assert get_tpu_seq_bucket_size(cfg_cp3) == 768

    # Under all_gather CP, bucket aligns to 128 * cp_size so local Q shard (S // cp_size) is a multiple of 128
    cfg_ag_cp4 = SimpleNamespace(
        pad_to_length=False,
        pad_to_length_bucket=256,
        context_parallel_size=4,
        engine_kwargs={"tpu": {"context_parallel_strategy": "all_gather"}},
    )
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VERL_TPU_SEQ_BUCKET_SIZE", None)
        assert get_tpu_seq_bucket_size(cfg_ag_cp4) == 512

    # compute_global_batch_num_tokens divides by tp_size * cp_size when distributed is initialized
    data = TensorDict({"loss_mask": torch.ones(4, 10, dtype=torch.int64)}, batch_size=[4])
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.all_reduce", side_effect=lambda t, op=None: t.mul_(4)),
    ):
        assert compute_global_batch_num_tokens(data, None, tp_size=1, cp_size=2) == 80


@pytest.mark.parametrize(
    ("strategy", "expected_lb"),
    [("all_gather", "headtail"), ("ulysses", None)],
)
def test_enable_context_parallel_follows_torchtitan_cp_recipe(strategy, expected_lb):
    """Sets SPMD backend + load balancer, applies ContextParallelTransform, validates, patches."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    calls = []

    class _SDPA:
        pass

    class _CPSDPA:
        pass

    class _ContextParallelTransform:
        def __init__(self, inner_attention):
            self.inner_attention = inner_attention

    def _transform_model_config_(model, transforms):
        calls.append(("transform", [t.inner_attention for t in transforms]))
        return model

    fake_transform = SimpleNamespace(
        ContextParallelTransform=_ContextParallelTransform, transform_model_config_=_transform_model_config_
    )
    fake_cp_attention = SimpleNamespace(
        install_tpu_cp_patches=lambda: calls.append("patches"),
        tpu_cp_inner_attention=lambda base, strat: calls.append(("inner", base, strat)) or _CPSDPA,
        tpu_cp_load_balancer=lambda strat: "headtail" if strat == "all_gather" else None,
        validate_tpu_context_parallel=lambda model, parallelism: calls.append(
            ("validate", parallelism.spmd_backend, parallelism.context_parallel_load_balancer)
        ),
    )
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(engine_kwargs={"tpu": {"context_parallel_strategy": strategy}}),
        _use_splash_attention=True,
    )
    parallelism = SimpleNamespace(
        context_parallel_degree=2, spmd_backend="partial_dtensor", context_parallel_load_balancer="headtail"
    )
    model = SimpleNamespace(first_full_attention_backend=SimpleNamespace(_owner=_SDPA))
    with mock.patch.dict(
        sys.modules,
        {
            "torchtitan": SimpleNamespace(),
            "torchtitan.config": SimpleNamespace(transform=fake_transform),
            "torchtitan.config.transform": fake_transform,
            "torchtitan.experiments": SimpleNamespace(),
            "torchtitan.experiments.tpu": SimpleNamespace(cp_attention=fake_cp_attention),
            "torchtitan.experiments.tpu.cp_attention": fake_cp_attention,
        },
    ):
        TorchTitanTPUEngine._enable_context_parallel(engine, parallelism, model)

    assert parallelism.spmd_backend == "spmd_types"
    assert parallelism.context_parallel_load_balancer == expected_lb
    assert calls == [
        ("inner", _SDPA, strategy),
        ("transform", [_CPSDPA]),
        ("validate", "spmd_types", expected_lb),
        "patches",
    ]


def test_enable_context_parallel_all_gather_requires_splash():
    """KV all-gather CP fails before the model is built when splash is off; torchtitan would refuse it on TPU."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    fake_transform = SimpleNamespace(
        ContextParallelTransform=lambda inner_attention: inner_attention,
        transform_model_config_=lambda model, transforms: model,
    )
    fake_cp_attention = SimpleNamespace(
        install_tpu_cp_patches=lambda: None,
        tpu_cp_inner_attention=lambda base, strat: base,
        tpu_cp_load_balancer=lambda strat: "headtail" if strat == "all_gather" else None,
        validate_tpu_context_parallel=lambda model, parallelism: None,
    )
    model = SimpleNamespace(first_full_attention_backend=SimpleNamespace(_owner=object))
    with mock.patch.dict(
        sys.modules,
        {
            "torchtitan": SimpleNamespace(),
            "torchtitan.config": SimpleNamespace(transform=fake_transform),
            "torchtitan.config.transform": fake_transform,
            "torchtitan.experiments": SimpleNamespace(),
            "torchtitan.experiments.tpu": SimpleNamespace(cp_attention=fake_cp_attention),
            "torchtitan.experiments.tpu.cp_attention": fake_cp_attention,
        },
    ):
        for strategy, raises in (("all_gather", True), ("ulysses", False)):
            engine = SimpleNamespace(
                engine_config=SimpleNamespace(engine_kwargs={"tpu": {"context_parallel_strategy": strategy}}),
                _use_splash_attention=False,
            )
            parallelism = SimpleNamespace(
                context_parallel_degree=2, spmd_backend="partial_dtensor", context_parallel_load_balancer=None
            )
            if raises:
                with pytest.raises(ValueError, match="needs use_splash_attention=True"):
                    TorchTitanTPUEngine._enable_context_parallel(engine, parallelism, model)
                assert parallelism.spmd_backend == "partial_dtensor"
            else:
                TorchTitanTPUEngine._enable_context_parallel(engine, parallelism, model)
                assert parallelism.spmd_backend == "spmd_types"


def test_cp_attention_owner_dispatches_to_inner_attention_cp_shard():
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    class _CPAttention:
        @classmethod
        def cp_shard(cls, input_dict, input_shardings, cp_mesh, load_balancer_type, ptrr_mask_key):
            return input_dict

    def _engine(inner_attention):
        model = SimpleNamespace(first_full_attention_backend=inner_attention)
        return SimpleNamespace(config=SimpleNamespace(model_spec=SimpleNamespace(model=model)))

    cp_inner = SimpleNamespace(_owner=_CPAttention)
    assert TorchTitanTPUEngine._cp_attention_owner(_engine(cp_inner)) is _CPAttention

    # A non-CP inner attention means _enable_context_parallel did not run: fail loudly
    # instead of falling back to sharding that ignores the CP strategy.
    plain_inner = SimpleNamespace(_owner=object)
    with pytest.raises(RuntimeError, match="has no cp_shard"):
        TorchTitanTPUEngine._cp_attention_owner(_engine(plain_inner))


def test_chunked_optimizer_step_matches_unchunked_adamw():
    """Chunked ``optimizer_step`` produces the exact same grad_norm and parameter updates as un-chunked AdamW."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import (
        TorchTitanTPUEngine,
        _partition_params_into_chunks,
    )

    torch.manual_seed(0)
    names = ["tok_embeddings.weight"] + [f"layers.{i}.wq.weight" for i in range(8)] + ["norm.weight"]
    params_a = [torch.nn.Parameter(torch.randn(4, 4)) for _ in names]
    params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
    for pa, pb in zip(params_a, params_b, strict=True):
        g = torch.randn_like(pa) * 2.0
        pa.grad = g.clone()
        pb.grad = g.clone()

    chunks = _partition_params_into_chunks(params_a, names, layers_per_chunk=4)
    # 2 four-layer chunks (layers 0..3, layers 4..7) + 1 non-layer chunk (tok_embeddings, norm)
    assert [len(c) for c in chunks] == [4, 4, 2]

    opt_a = torch.optim.AdamW([{"params": params_a, "param_names": names, "lr": 1e-2, "foreach": True}])
    opt_b = torch.optim.AdamW([{"params": params_b, "param_names": names, "lr": 1e-2, "foreach": True}])

    ref_norm = torch.nn.utils.clip_grad_norm_(params_b, max_norm=1.0, foreach=False).item()
    opt_b.step()

    fake_engine = SimpleNamespace(
        _tpu_eager_mode=None,
        module=[SimpleNamespace(named_parameters=lambda: list(zip(names, params_a, strict=True)))],
        optimizer=SimpleNamespace(optimizers=[opt_a], zero_grad=opt_a.zero_grad),
        parallel_dims=SimpleNamespace(ep_enabled=False, get_optional_mesh=lambda _: None),
        config=SimpleNamespace(training=SimpleNamespace(max_norm=1.0)),
    )
    for helper in ("_optimizer_param_chunks", "_chunked_grad_norm", "_chunked_clip_and_step"):
        setattr(fake_engine, helper, getattr(TorchTitanTPUEngine, helper).__get__(fake_engine))
    chunked_norm = TorchTitanTPUEngine.optimizer_step(fake_engine)
    assert chunked_norm == pytest.approx(ref_norm, rel=1e-6, abs=1e-6)
    for pa, pb in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-6)
        assert pa.grad is None
    assert opt_a.param_groups[0]["params"] == params_a


@pytest.mark.parametrize("host_offload", [False, True])
def test_chunked_optimizer_step_fp32_master(host_offload):
    """``fp32_master``: bf16 params follow an FP32 AdamW run (cast to bf16) and keep tiny updates bf16 alone loses."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine
    from verl_hardware_plugin.accelerators.tpu.options import TPUEngineOptions

    torch.manual_seed(0)
    names = [f"layers.{i}.wq.weight" for i in range(6)] + ["norm.weight"]
    ref = [torch.nn.Parameter(torch.randn(8, 8)) for _ in names]
    params = [torch.nn.Parameter(p.detach().to(torch.bfloat16)) for p in ref]
    for p in ref:  # the reference starts from the same (bf16-representable) weights
        p.data.copy_(p.data.to(torch.bfloat16).float())
    lr = 1e-6  # ~3 orders of magnitude below a bf16 ulp of these weights
    opt_ref = torch.optim.AdamW([{"params": ref, "lr": lr, "foreach": True}])
    opt = torch.optim.AdamW([{"params": params, "param_names": names, "lr": lr, "foreach": True}])

    fake_engine = SimpleNamespace(
        _tpu_eager_mode=None,
        _tpu_options=TPUEngineOptions(param_dtype="bfloat16", fp32_master=True, optimizer_host_offload=host_offload),
        module=[SimpleNamespace(named_parameters=lambda: list(zip(names, params, strict=True)))],
        optimizer=SimpleNamespace(optimizers=[opt], zero_grad=opt.zero_grad),
        parallel_dims=SimpleNamespace(ep_enabled=False, get_optional_mesh=lambda _: None),
        config=SimpleNamespace(training=SimpleNamespace(max_norm=1e9)),
    )
    for helper in ("_optimizer_param_chunks", "_chunked_grad_norm", "_chunked_clip_and_step"):
        setattr(fake_engine, helper, getattr(TorchTitanTPUEngine, helper).__get__(fake_engine))

    initial = [p.detach().clone() for p in params]
    for _ in range(300):
        for p, r in zip(params, ref, strict=True):
            g = torch.randn_like(r)
            r.grad = g.clone()
            p.grad = g.to(torch.bfloat16)
        opt_ref.step()
        TorchTitanTPUEngine.optimizer_step(fake_engine)
    store = fake_engine._fp32_master_store
    for p, r in zip(params, ref, strict=True):
        assert p.dtype == torch.bfloat16 and p.grad is None
        master, state = store._entries[id(p)]
        assert master.dtype == torch.float32 and state["exp_avg"].dtype == torch.float32
        torch.testing.assert_close(master, r.detach(), rtol=1e-3, atol=1e-6)  # bf16 grads, fp32 everything else
        torch.testing.assert_close(p.detach(), master.to(torch.bfloat16), rtol=0, atol=0)
    # 300 steps x 1e-6 move the masters by ~3e-4, enough to flip some bf16 weights.
    assert any(not torch.equal(p.detach(), i) for p, i in zip(params, initial, strict=True))
    assert opt.state == {} and opt.param_groups[0]["params"] == params


def test_fp32_master_host_offload_cpu_fused_adamw_bitwise_parity_and_buffer_reuse():
    """CPU ``fused=True`` host-offload matches fused FP32 AdamW to 0 ULP, matches device masters, & reuses RAM."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import (
        TorchTitanTPUEngine,
        _cpu_fused_adamw_supported,
        _move_optimizer_state,
    )
    from verl_hardware_plugin.accelerators.tpu.options import TPUEngineOptions

    # 12 layers -> 3 chunks (> 2 double-buffer slots) + 2 param groups (decayed & non-decayed).
    names_decayed = [f"layers.{i}.wq.weight" for i in range(12)]
    names_nodecay = [f"layers.{i}.norm.weight" for i in range(12)]
    all_names = names_decayed + names_nodecay

    def _build(host_offload: bool):
        gen = torch.Generator().manual_seed(42)
        p_dec = [torch.nn.Parameter(torch.randn(16, 8, generator=gen, dtype=torch.bfloat16)) for _ in names_decayed]
        p_nod = [torch.nn.Parameter(torch.randn(8, generator=gen, dtype=torch.bfloat16)) for _ in names_nodecay]
        all_p = p_dec + p_nod
        opt = torch.optim.AdamW(
            [
                {"params": p_dec, "param_names": names_decayed, "lr": 3e-4, "weight_decay": 0.05, "foreach": True},
                {"params": p_nod, "param_names": names_nodecay, "lr": 1e-4, "weight_decay": 0.0, "foreach": True},
            ]
        )
        eng = SimpleNamespace(
            _tpu_eager_mode=None,
            _tpu_options=TPUEngineOptions(
                param_dtype="bfloat16", fp32_master=True, optimizer_host_offload=host_offload
            ),
            module=[SimpleNamespace(named_parameters=lambda: list(zip(all_names, all_p, strict=True)))],
            optimizer=SimpleNamespace(optimizers=[opt], zero_grad=opt.zero_grad),
            parallel_dims=SimpleNamespace(ep_enabled=False, get_optional_mesh=lambda _: None),
            config=SimpleNamespace(training=SimpleNamespace(max_norm=0.25)),
        )
        for helper in ("_optimizer_param_chunks", "_chunked_grad_norm", "_chunked_clip_and_step"):
            setattr(eng, helper, getattr(TorchTitanTPUEngine, helper).__get__(eng))
        return eng, opt, all_p

    eng_dev, opt_dev, p_dev = _build(host_offload=False)
    eng_cpu, opt_cpu, p_cpu = _build(host_offload=True)

    # Pure FP32 reference using the exact same CPU ``fused`` kernel selection as ``_Fp32Masters`` (0-ULP oracle).
    ref_p = [torch.nn.Parameter(p.detach().to(torch.float32)) for p in p_cpu]
    use_fused = _cpu_fused_adamw_supported()
    opt_ref = torch.optim.AdamW(
        [
            {"params": ref_p[:12], "lr": 3e-4, "weight_decay": 0.05, "fused": use_fused, "foreach": not use_fused},
            {"params": ref_p[12:], "lr": 1e-4, "weight_decay": 0.0, "fused": use_fused, "foreach": not use_fused},
        ]
    )

    master_ptrs_step1: dict[int, tuple[int, int, int]] = {}
    slot_ptrs_step1: tuple[dict, dict] = ({}, {})

    for step in range(1, 9):
        lr_scale = 0.5 + 0.1 * step
        for opt in (opt_dev, opt_cpu, opt_ref):
            opt.param_groups[0]["lr"] = 3e-4 * lr_scale
            opt.param_groups[1]["lr"] = 1e-4 * lr_scale
        g_gen = torch.Generator().manual_seed(1000 + step)
        raw_grads = [torch.randn(pd.shape, generator=g_gen, dtype=torch.bfloat16) for pd in p_dev]
        for pd, pc, g in zip(p_dev, p_cpu, raw_grads, strict=True):
            pd.grad = g.clone()
            pc.grad = g.clone()

        norm_dev = TorchTitanTPUEngine.optimizer_step(eng_dev)
        norm_cpu = TorchTitanTPUEngine.optimizer_step(eng_cpu)
        assert torch.equal(torch.as_tensor(norm_dev), torch.as_tensor(norm_cpu))
        # Apply the exact same bf16 gradient clipping before FP32 cast as ``_Fp32Masters``:
        ref_bf16 = [torch.nn.Parameter(torch.empty_like(g), requires_grad=False) for g in raw_grads]
        for rb, g in zip(ref_bf16, raw_grads, strict=True):
            rb.grad = g.clone()
        torch.nn.utils.clip_grads_with_norm_(
            ref_bf16, 0.25, torch.as_tensor(norm_cpu, dtype=torch.bfloat16), foreach=False
        )
        for pr, rb in zip(ref_p, ref_bf16, strict=True):
            pr.grad = rb.grad.to(torch.float32)
        opt_ref.step()
        opt_ref.zero_grad(set_to_none=True)

        store_cpu = eng_cpu._fp32_master_store
        if step == 1:
            for pc in p_cpu:
                m, st = store_cpu._entries[id(pc)]
                master_ptrs_step1[id(pc)] = (m.data_ptr(), st["exp_avg"].data_ptr(), st["exp_avg_sq"].data_ptr())
            slot_ptrs_step1 = (
                {k: b.data_ptr() for k, b in store_cpu._staging_slots[0].items()},
                {k: b.data_ptr() for k, b in store_cpu._staging_slots[1].items()},
            )

    store_dev = eng_dev._fp32_master_store
    store_cpu = eng_cpu._fp32_master_store
    for pd, pc, pr in zip(p_dev, p_cpu, ref_p, strict=True):
        md, sd = store_dev._entries[id(pd)]
        mc, sc = store_cpu._entries[id(pc)]
        sr = opt_ref.state[pr]
        # Exact 0-ULP bitwise parity against reference fused FP32 AdamW across all CPU ISAs:
        assert torch.equal(mc, pr.detach())
        assert torch.equal(pc.detach(), pr.detach().to(torch.bfloat16))
        assert torch.equal(sc["exp_avg"], sr["exp_avg"])
        assert torch.equal(sc["exp_avg_sq"], sr["exp_avg_sq"])
        # Device ``foreach=True`` vs CPU ``fused=True`` differs only by single-rounding FMA (~1 FP32 ULP):
        torch.testing.assert_close(pc.detach(), pd.detach(), rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(mc, md, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(sc["exp_avg"], sd["exp_avg"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(sc["exp_avg_sq"], sd["exp_avg_sq"], rtol=1e-6, atol=1e-7)
        # Zero per-step reallocation of FP32 masters or moments across steps 1..8:
        assert (mc.data_ptr(), sc["exp_avg"].data_ptr(), sc["exp_avg_sq"].data_ptr()) == master_ptrs_step1[id(pc)]

    assert {k: b.data_ptr() for k, b in store_cpu._staging_slots[0].items()} == slot_ptrs_step1[0]
    assert {k: b.data_ptr() for k, b in store_cpu._staging_slots[1].items()} == slot_ptrs_step1[1]

    # Verify ``_move_optimizer_state`` initializes and reuses ``_tpu_pinned_host_state`` cache.
    p_plain = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
    p_plain.grad = torch.ones_like(p_plain)
    opt_plain = torch.optim.AdamW([p_plain], lr=1e-3)
    opt_plain.step()
    _move_optimizer_state(opt_plain, [p_plain], "cpu")
    assert isinstance(opt_plain._tpu_pinned_host_state, dict)


def _fp32_master_host_offload_dtensor_worker(rank: int, world: int, init_file: str) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor, init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine
    from verl_hardware_plugin.accelerators.tpu.options import TPUEngineOptions

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))

        def _make(host_offload: bool):
            gen = torch.Generator().manual_seed(17)
            params = [
                torch.nn.Parameter(
                    distribute_tensor(torch.randn(8, 4, generator=gen, dtype=torch.bfloat16), mesh, [Shard(0)])
                ),
                torch.nn.Parameter(
                    distribute_tensor(torch.randn(5, 4, generator=gen, dtype=torch.bfloat16), mesh, [Shard(0)])
                ),
                torch.nn.Parameter(
                    distribute_tensor(torch.randn(4, generator=gen, dtype=torch.bfloat16), mesh, [Replicate()])
                ),
            ]
            names = ["layers.0.wq.weight", "layers.4.wo.weight", "norm.weight"]
            opt = torch.optim.AdamW([{"params": params, "param_names": names, "lr": 2e-4, "foreach": True}])
            eng = SimpleNamespace(
                _tpu_eager_mode=None,
                _tpu_options=TPUEngineOptions(
                    param_dtype="bfloat16", fp32_master=True, optimizer_host_offload=host_offload
                ),
                module=[SimpleNamespace(named_parameters=lambda: list(zip(names, params, strict=True)))],
                optimizer=SimpleNamespace(optimizers=[opt], zero_grad=opt.zero_grad),
                parallel_dims=SimpleNamespace(ep_enabled=False, get_optional_mesh=lambda _: None),
                config=SimpleNamespace(training=SimpleNamespace(max_norm=0.5)),
            )
            for helper in ("_optimizer_param_chunks", "_chunked_grad_norm", "_chunked_clip_and_step"):
                setattr(eng, helper, getattr(TorchTitanTPUEngine, helper).__get__(eng))
            return eng, params

        eng_dev, p_dev = _make(host_offload=False)
        eng_cpu, p_cpu = _make(host_offload=True)
        for step in range(1, 5):
            g_gen = torch.Generator().manual_seed(200 + step)
            grads = [
                distribute_tensor(torch.randn(8, 4, generator=g_gen, dtype=torch.bfloat16), mesh, [Shard(0)]),
                distribute_tensor(torch.randn(5, 4, generator=g_gen, dtype=torch.bfloat16), mesh, [Shard(0)]),
                distribute_tensor(torch.randn(4, generator=g_gen, dtype=torch.bfloat16), mesh, [Replicate()]),
            ]
            for pd, pc, g in zip(p_dev, p_cpu, grads, strict=True):
                pd.grad = g
                pc.grad = g.clone()
            TorchTitanTPUEngine.optimizer_step(eng_dev)
            TorchTitanTPUEngine.optimizer_step(eng_cpu)

        for pd, pc in zip(p_dev, p_cpu, strict=True):
            md, sd = eng_dev._fp32_master_store._entries[id(pd)]
            mc, sc = eng_cpu._fp32_master_store._entries[id(pc)]
            assert isinstance(mc, DTensor) and isinstance(sc["exp_avg"], DTensor)
            torch.testing.assert_close(pc._local_tensor, pd._local_tensor, rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(mc._local_tensor, md._local_tensor, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(sc["exp_avg"]._local_tensor, sd["exp_avg"]._local_tensor, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(
                sc["exp_avg_sq"]._local_tensor, sd["exp_avg_sq"]._local_tensor, rtol=1e-6, atol=1e-7
            )
    finally:
        dist.destroy_process_group()


def test_fp32_master_host_offload_dtensor_shards(tmp_path):
    """2-rank ``DTensor`` shards match 0-ULP between CPU ``fused=True`` host-offload and device ``fp32_master``."""
    torch.multiprocessing.start_processes(
        _fp32_master_host_offload_dtensor_worker,
        args=(2, str(tmp_path / "pg_dt")),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_to_hf_named_param_chunks_converts_one_layer_at_a_time():
    """``get_per_tensor_param`` converts the state dict to HF names one transformer layer at a time, lazily."""
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine

    calls = []

    class _Adapter:
        def to_hf(self, sd):
            calls.append(sorted(sd))
            return {f"hf.{k}": v for k, v in sd.items()}

    engine = SimpleNamespace(checkpointer=SimpleNamespace(sd_adapter=_Adapter()))
    engine._to_hf_named_params = lambda params: TorchTitanTPUEngine._to_hf_named_params(engine, params)
    names = [
        "tok_embeddings.weight",
        "layers.0.attn.wq.weight",
        "layers.0.moe.w1",
        "layers.1.attn.wq.weight",
        "layers.10.attn.wq.weight",
        "vision_encoder.layers.0.attn.wq.weight",
        "norm.weight",
    ]
    params = {n: torch.zeros(1) for n in names}
    chunks = TorchTitanTPUEngine._to_hf_named_param_chunks(engine, params)
    assert params == {} and calls == []  # nothing is converted before the export asks for it
    first = next(chunks)
    assert calls == [["norm.weight", "tok_embeddings.weight"]] and set(first) == {
        "hf.norm.weight",
        "hf.tok_embeddings.weight",
    }
    rest = list(chunks)
    assert calls[1:] == [
        ["layers.0.attn.wq.weight", "layers.0.moe.w1"],
        ["layers.1.attn.wq.weight"],
        ["layers.10.attn.wq.weight"],
        ["vision_encoder.layers.0.attn.wq.weight"],
    ]
    assert sum(len(c) for c in rest) == 5


def test_resolve_tpu_model_kernels(monkeypatch):
    """``engine_kwargs.tpu.model_kernels`` selects (or auto-detects from torchtitan) the optional TPU kernels."""
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    resolve = kernels_mod.resolve_tpu_model_kernels
    # torchtitan main: no optional kernels, so "auto" (the default) enables none and explicit requests fail.
    monkeypatch.setattr(kernels_mod, "_available_kernels", lambda name: frozenset())
    assert resolve("qwen3_5") == frozenset()
    assert resolve("qwen3_5", "auto") == frozenset()
    assert resolve("qwen3_5", "none") == frozenset()
    with pytest.raises(ValueError, match="not provided by the installed torchtitan"):
        resolve("qwen3_5", "all")

    # An optimized torchtitan build (google-pytorch/torchtitan#73-#80) with Tokamax.
    available = frozenset({"moe", "kda", "lce"})
    monkeypatch.setattr(kernels_mod, "_available_kernels", lambda name: available if name == "qwen3_5" else frozenset())
    assert resolve("qwen3_5") == available
    assert resolve("qwen3_5", "all") == available
    assert resolve("qwen3_5", "none") == frozenset()
    assert resolve("qwen3") == frozenset()
    kernels = resolve("qwen3_5", "moe, lce")
    assert kernels == {"moe", "lce"}
    assert kernels_mod.override_imports(kernels) == []
    lce = {"b_block_size": 1024, "h_block_size": 2048, "v_block_size": 640}
    assert kernels_mod.lce_block_sizes(kernels, hidden_size=2048, vocab_size=248320) == lce
    assert kernels_mod.lce_block_sizes(kernels, hidden_size=2560, vocab_size=248320) == {}
    assert kernels_mod.lce_block_sizes(frozenset(), hidden_size=2048, vocab_size=248320) == {}
    with pytest.raises(ValueError, match="not supported"):
        resolve("qwen3_5", "bogus")


def _fake_torchtitan_moe_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for torchtitan's MoE config classes and TPU MoE converters (fields as in torchtitan ``ea7fec0f``)."""

    @dataclasses.dataclass
    class _ConverterConfig:
        pass

    class MoE:
        @dataclasses.dataclass
        class Config:
            router: Any
            routed_experts: Any
            load_balance_coeff: Optional[float] = 1e-3

    class GroupedExperts:
        @dataclasses.dataclass
        class Config:
            num_experts: int = 128

    class TpuGroupedExperts(GroupedExperts):
        @dataclasses.dataclass
        class Config(GroupedExperts.Config):
            use_pallas_grouped_mm: bool = False
            use_pallas_paired_grouped_mm: bool = False
            pallas_grouped_mm_tile_m: int = 1024
            pallas_grouped_mm_tgmm_tiling: tuple[int, int, int] = (128, 128, 128)
            use_kernel_native_weight_layout: bool = False

    moe_mod = types.ModuleType("torchtitan.models.common.moe")
    moe_mod.MoE = MoE  # type: ignore[attr-defined]
    moe_mod.GroupedExperts = GroupedExperts  # type: ignore[attr-defined]
    grouped_experts = types.ModuleType("torchtitan.experiments.tpu.grouped_experts")
    grouped_experts.TpuGroupedExperts = TpuGroupedExperts  # type: ignore[attr-defined]
    grouped_experts.TpuGroupedExpertsConverter = type(  # type: ignore[attr-defined]
        "TpuGroupedExpertsConverter", (), {"Config": type("Config", (_ConverterConfig,), {})}
    )
    token_dispatcher = types.ModuleType("torchtitan.experiments.tpu.token_dispatcher")
    token_dispatcher.TpuTokenDispatcherConverter = type(  # type: ignore[attr-defined]
        "TpuTokenDispatcherConverter", (), {"Config": type("Config", (_ConverterConfig,), {})}
    )
    return {m.__name__: m for m in (moe_mod, grouped_experts, token_dispatcher)}


def _fake_qwen3_moe_model_spec(modules, *, n_layers=2, top_k=8, tpu_experts=True):
    """A ``ModelSpec`` stand-in whose model config ``traverse``s like torchtitan (``(fqn, config, parent, field)``)."""
    MoE = modules["torchtitan.models.common.moe"].MoE
    experts_cls = (
        modules["torchtitan.experiments.tpu.grouped_experts"].TpuGroupedExperts
        if tpu_experts
        else modules["torchtitan.models.common.moe"].GroupedExperts
    )
    nodes = []
    for _ in range(n_layers):
        experts = experts_cls.Config()
        router = SimpleNamespace(top_k=top_k, aux_loss="flavor aux loss setting")
        nodes += [MoE.Config(router=router, routed_experts=experts), experts]

    def traverse(cls):
        return [(f"layers.{i}", node, None, None) for i, node in enumerate(nodes) if isinstance(node, cls)]

    return SimpleNamespace(model=SimpleNamespace(traverse=traverse), nodes=nodes)


def test_only_qwen3_moe_gets_the_qwen3_moe_setup():
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    def hf(model_type):
        return SimpleNamespace(model_type=model_type)

    assert kernels_mod.is_qwen3_moe("qwen3", hf("qwen3_moe"))
    assert not kernels_mod.is_qwen3_moe("qwen3", hf("qwen3"))  # dense Qwen3: same torchtitan model, unchanged
    assert not kernels_mod.is_qwen3_moe("qwen3", hf("qwen2"))
    assert not kernels_mod.is_qwen3_moe("qwen3_5", hf("qwen3_5_moe"))
    assert not kernels_mod.is_qwen3_moe("qwen3_5", hf("qwen3_5_moe_text"))
    assert not kernels_mod.is_qwen3_moe("llama3", hf("qwen3_moe"))
    # The optional ``model_kernels`` do not cover it: torchtitan's qwen3 has none.
    assert kernels_mod.resolve_tpu_model_kernels("qwen3", "auto") == frozenset()


def test_qwen3_moe_model_registry_converters_dispatcher_first():
    """Same converters, in the same order, as torchtitan's Qwen3-30B-A3B TPU recipe."""
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    modules = _fake_torchtitan_moe_modules()
    with mock.patch.dict(sys.modules, modules):
        kwargs = kernels_mod.qwen3_moe_model_registry_kwargs()
    dispatcher = modules["torchtitan.experiments.tpu.token_dispatcher"].TpuTokenDispatcherConverter
    grouped_experts = modules["torchtitan.experiments.tpu.grouped_experts"].TpuGroupedExpertsConverter
    assert list(kwargs) == ["converters"]
    assert [type(c) for c in kwargs["converters"]] == [dispatcher.Config, grouped_experts.Config]


def test_model_setup_table_selects_qwen3_moe_only():
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    def hf(model_type):
        return SimpleNamespace(model_type=model_type)

    setup = kernels_mod.model_setup("qwen3", hf("qwen3_moe"))
    assert setup is not None and setup.name == "qwen3_moe"
    for name, model_type in [
        ("qwen3", "qwen3"),
        ("qwen3", "qwen2_moe"),
        ("qwen3_5", "qwen3_5_moe"),
        ("llama3", "llama"),
    ]:
        assert kernels_mod.model_setup(name, hf(model_type)) is None
    # configure forwards the engine settings to configure_qwen3_moe.
    calls = []
    with mock.patch.object(kernels_mod, "configure_qwen3_moe", lambda spec, **kw: calls.append((spec, kw))):
        setup.configure("spec", kernels_mod.ModelSetupContext(expert_parallel_size=1, seq_bucket_size=1024))
    assert calls == [("spec", {"expert_parallel_size": 1, "seq_bucket_size": 1024})]


def test_configure_qwen3_moe_routes_like_hf_with_tpu_grouped_experts():
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    modules = _fake_torchtitan_moe_modules()
    spec = _fake_qwen3_moe_model_spec(modules)
    with mock.patch.dict(sys.modules, modules):
        kernels_mod.configure_qwen3_moe(spec, expert_parallel_size=1, seq_bucket_size=1024)

    moes, experts = spec.nodes[0::2], spec.nodes[1::2]
    # Expert bias off; aux loss untouched.
    assert all(m.load_balance_coeff is None for m in moes)
    assert all(m.router.aux_loss == "flavor aux loss setting" for m in moes)
    for e in experts:
        assert e.use_pallas_grouped_mm and not e.use_pallas_paired_grouped_mm
        assert not e.use_kernel_native_weight_layout  # canonical w1/w2/w3 layout, which the weight export assumes
        assert e.pallas_grouped_mm_tile_m == 256
        assert e.pallas_grouped_mm_tgmm_tiling == (256, 0, 0)  # full K/N dWeight tiles


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expert_parallel_size": 2, "seq_bucket_size": 256}, "expert_parallel_size=2 is not supported"),
        ({"expert_parallel_size": 1, "seq_bucket_size": 16}, "multiple of 256"),  # 16 * 8 = 128 routed rows
        ({"expert_parallel_size": 1, "seq_bucket_size": 0}, "multiple of 256"),  # no bucketing: any token count
    ],
)
def test_configure_qwen3_moe_rejects_ep_and_untiled_buckets(kwargs, match):
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    modules = _fake_torchtitan_moe_modules()
    spec = _fake_qwen3_moe_model_spec(modules)
    with mock.patch.dict(sys.modules, modules), pytest.raises(ValueError, match=match):
        kernels_mod.configure_qwen3_moe(spec, **kwargs)
    # 32 * 8 = 256 routed rows is tile-aligned.
    with mock.patch.dict(sys.modules, modules):
        kernels_mod.configure_qwen3_moe(spec, expert_parallel_size=1, seq_bucket_size=32)


def test_configure_qwen3_moe_needs_the_tpu_converters():
    """A model built without ``qwen3_moe_model_registry_kwargs`` would silently keep the native grouped mm."""
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels as kernels_mod

    modules = _fake_torchtitan_moe_modules()
    with mock.patch.dict(sys.modules, modules), pytest.raises(ValueError, match="0 of 2 MoE layers"):
        kernels_mod.configure_qwen3_moe(
            _fake_qwen3_moe_model_spec(modules, tpu_experts=False), expert_parallel_size=1, seq_bucket_size=256
        )


def test_packed_document_budget_keeps_every_rl_sequence_apart():
    """Two sequences packed into a 768-token row restart the KDA recurrence each (plus one padding segment)."""
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_model_kernels import set_packed_document_budget
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import _document_ids

    class TokamaxGatedDeltaKernel(torch.nn.Module):  # stands in for torchtitan's kernel module (same name)
        def __init__(self):
            super().__init__()
            self.max_num_segments = None

    class ParallelizedTokamaxGatedDeltaKernel(TokamaxGatedDeltaKernel):  # parallelization may subclass modules
        pass

    model = torch.nn.Sequential(TokamaxGatedDeltaKernel(), torch.nn.Linear(2, 2), ParallelizedTokamaxGatedDeltaKernel())
    assert set_packed_document_budget([model], 3) == 2
    assert all(m.max_num_segments == 3 for m in model if isinstance(m, TokamaxGatedDeltaKernel))
    assert set_packed_document_budget([model], 3) == 0  # unchanged: no recompile

    # A packed RL row: 2 sequences (300 + 308 tokens) bucket-padded to 768; every pad token has position 0.
    positions = torch.cat([torch.arange(300), torch.arange(308), torch.zeros(160, dtype=torch.int64)]).unsqueeze(0)
    with mock.patch.dict(sys.modules, _fake_torchtitan_modules()):
        segment_ids = _document_ids(positions, None)

    def tokamax_ids(budget):  # torchtitan's tokamax_segment_ids (a9093690)
        return (segment_ids - segment_ids[:, :1] + 1).clamp(max=budget)

    # torchtitan's default budget for T=768 is max(1, 768 // 512) = 1: both sequences share one recurrence.
    assert tokamax_ids(1)[0, 300].item() == tokamax_ids(1)[0, 299].item()
    ids = tokamax_ids(3)[0]
    assert ids[:300].eq(1).all() and ids[300:608].eq(2).all() and ids[608:].eq(3).all()


def test_ep_capacity_factor_and_strided_pad_tokens():
    """Non-Module ``RoutedExperts.token_dispatcher`` gets ``capacity_factor``, and padding strides across vocab."""
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_model_kernels
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_model_kernels import set_ep_capacity_factor
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import pad_packed_inputs_for_tpu

    class TpuQwen35TokenDispatcher:  # plain Configurable object in torchtitan (NOT an nn.Module)
        def __init__(self):
            self.capacity_factor = 1.25

    class RoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_dispatcher = TpuQwen35TokenDispatcher()

    model = torch.nn.Sequential(RoutedExperts(), torch.nn.Linear(2, 2), RoutedExperts())
    assert set_ep_capacity_factor([model], 4.0) == 2
    assert all(m.token_dispatcher.capacity_factor == 4.0 for m in model if isinstance(m, RoutedExperts))
    assert set_ep_capacity_factor([model], 4.0) == 0

    ids = torch.nested.nested_tensor_from_jagged(torch.tensor([11, 22, 33, 44, 55]), torch.tensor([0, 3, 5]))
    pos = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 2, 0, 1]), torch.tensor([0, 3, 5]))
    mb = TensorDict({}, batch_size=[2])
    tu.assign_non_tensor_data(mb, "tpu_padded_seq_len", 12)
    padded_ids, padded_pos, padded_labels, orig_len = pad_packed_inputs_for_tpu(
        ids, pos, mb, device="cpu", bucket_size=8, vocab_size=1000
    )
    assert orig_len == 5 and padded_ids.shape == (12,)
    assert padded_ids[:5].tolist() == [11, 22, 33, 44, 55]
    assert padded_ids[5:].tolist() == [(i * 9973) % 1000 for i in range(7)]
    assert len(set(padded_ids[5:].tolist())) == 7
    assert padded_pos.tolist() == [0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert padded_labels.tolist() == [22, 33, 44, 55, 11, 0, 0, 0, 0, 0, 0, 0]

    # Fallback when vocab_size is omitted: cycle real token IDs so tiny synthetic vocabs never go out of bounds.
    fallback_ids, _, _, _ = pad_packed_inputs_for_tpu(ids, pos, mb, device="cpu", bucket_size=8)
    assert fallback_ids.tolist() == [11, 22, 33, 44, 55, 11, 22, 33, 44, 55, 11, 22]

    # Wide-MoE (dim > 2048) dWeight TGMM tiling vs narrow-MoE (dim <= 2048), and EP > 1 paired-gmm disable.
    class _FakeExpertsCfg:
        def __init__(self, dim):
            self.dim = dim
            self.use_pallas_paired_grouped_mm = True
            self.pallas_grouped_mm_tile_m = None
            self.pallas_grouped_mm_tgmm_tiling = None
            self.pallas_grouped_mm_w2_tgmm_tiling = None

    class _FakeTpuGroupedExperts:
        Config = _FakeExpertsCfg

    narrow_exp = _FakeExpertsCfg(dim=2048)
    wide_exp = _FakeExpertsCfg(dim=4096)
    fake_spec = SimpleNamespace(
        model=SimpleNamespace(traverse=lambda cls: [(None, narrow_exp, None, None), (None, wide_exp, None, None)])
    )
    fake_ge = types.ModuleType("torchtitan.experiments.tpu.grouped_experts")
    fake_ge.TpuGroupedExperts = _FakeTpuGroupedExperts  # type: ignore[attr-defined]
    fake_reg = SimpleNamespace(_use_tpu_moe_kernels=lambda _: None)
    with (
        mock.patch.dict(sys.modules, {"torchtitan.experiments.tpu.grouped_experts": fake_ge}),
        mock.patch.object(tpu_model_kernels, "_qwen3_5_config_registry", return_value=fake_reg),
    ):
        tpu_model_kernels.configure_model_spec(fake_spec, frozenset({"moe"}))
        assert narrow_exp.use_pallas_paired_grouped_mm is True
        assert wide_exp.use_pallas_paired_grouped_mm is True
        tpu_model_kernels.configure_model_spec(fake_spec, frozenset({"moe"}), expert_parallel_degree=4)
    assert narrow_exp.use_pallas_paired_grouped_mm is False
    assert wide_exp.use_pallas_paired_grouped_mm is False
    assert narrow_exp.pallas_grouped_mm_tile_m == 512
    assert narrow_exp.pallas_grouped_mm_tgmm_tiling == (512, 0, 0)
    assert narrow_exp.pallas_grouped_mm_w2_tgmm_tiling is None
    assert wide_exp.pallas_grouped_mm_tile_m is None
    assert wide_exp.pallas_grouped_mm_tgmm_tiling == (256, 1024, 0)
    assert wide_exp.pallas_grouped_mm_w2_tgmm_tiling == (256, 1024, 1024)


def test_torchtitan_head_compatibility_for_kda_multi_mesh_and_ep_simple_fsdp(monkeypatch):
    """Compat with torchtitan HEAD before #73-#80/#6 land: tokamax kda probe, multi-mesh chunks, and EP SimpleFSDP."""
    from verl_hardware_plugin.accelerators.tpu.engines import torchtitan_tpu, tpu_model_kernels, tpu_utils

    # 1. PyPI tokamax==0.0.14 has ``tokamax`` without ``tokamax._src.ops.kda``: "auto" must not enable ``kda``.
    fake_reg = SimpleNamespace(_use_tpu_moe_kernels=lambda _: None, TOKAMAX_GATED_DELTA_OVERRIDE="kda_override")
    monkeypatch.setattr(tpu_model_kernels, "_qwen3_5_config_registry", lambda: fake_reg)
    monkeypatch.setattr(
        tpu_model_kernels.importlib.util,
        "find_spec",
        lambda name: object() if name == "tokamax" else None,
    )
    assert tpu_model_kernels.resolve_tpu_model_kernels("qwen3_5", "auto") == frozenset({"moe", "lce"})
    monkeypatch.setattr(
        tpu_model_kernels.importlib.util,
        "find_spec",
        lambda name: object() if name in ("tokamax", "tokamax._src.ops.kda") else None,
    )
    assert tpu_model_kernels.resolve_tpu_model_kernels("qwen3_5", "auto") == frozenset({"moe", "kda", "lce"})

    # 2. Multi-mesh optimizer param group (torchtitan HEAD without _split_optimizer_groups_by_mesh):
    #    _partition_params_into_chunks separates (fsdp,) and (efsdp, ep) params deterministically by first-seen mesh.
    mesh_fsdp, mesh_ep = object(), object()
    p_dense0 = torch.nn.Parameter(torch.ones(2))
    p_exp0 = torch.nn.Parameter(torch.ones(2))
    p_dense1 = torch.nn.Parameter(torch.ones(2))
    p_exp1 = torch.nn.Parameter(torch.ones(2))
    p_dense0._spec = SimpleNamespace(mesh=mesh_fsdp)  # type: ignore[attr-defined]
    p_dense1._spec = SimpleNamespace(mesh=mesh_fsdp)  # type: ignore[attr-defined]
    p_exp0._spec = SimpleNamespace(mesh=mesh_ep)  # type: ignore[attr-defined]
    p_exp1._spec = SimpleNamespace(mesh=mesh_ep)  # type: ignore[attr-defined]
    params = [p_dense0, p_exp0, p_dense1, p_exp1]
    names = [
        "layers.0.attention.wq.weight",
        "layers.0.moe.experts.w1",
        "layers.1.attention.wq.weight",
        "layers.1.moe.experts.w1",
    ]
    chunks = torchtitan_tpu._partition_params_into_chunks(params, names, layers_per_chunk=4)
    assert chunks == [[p_dense0, p_dense1], [p_exp0, p_exp1]]
    assert torchtitan_tpu._partition_params_into_chunks(params, None) == [[p_dense0, p_dense1], [p_exp0, p_exp1]]

    # 3. Non-static EP dispatcher (AllToAllTokenDispatcher on torchtitan HEAD) relaxes MoE block compile to
    #    fullgraph=False while static TpuQwen35TokenDispatcher keeps fullgraph=True.
    compile_calls: list[tuple[str, str, bool]] = []

    class AllToAllTokenDispatcher:
        pass

    class TpuQwen35TokenDispatcher:
        pass

    class _Block(torch.nn.Module):
        def __init__(self, name: str, dispatcher: object):
            super().__init__()
            self._name = name
            self._compiled_call_impl = object()
            self.routed = torch.nn.Module()
            if dispatcher is not None:
                self.routed.token_dispatcher = dispatcher

        def compile(self, *, backend: str = "tpu", fullgraph: bool = True, **kwargs: object):
            compile_calls.append((self._name, backend, fullgraph))

    fake_model = torch.nn.Module()
    fake_model.layers = torch.nn.ModuleDict(
        {
            "0": _Block("head_moe", AllToAllTokenDispatcher()),
            "1": _Block("static_ep_moe", TpuQwen35TokenDispatcher()),
            "2": _Block("dense", None),
        }
    )
    tpu_utils._relax_non_static_ep_moe_compile(
        fake_model,
        parallel_dims=SimpleNamespace(ep_enabled=True),
        compile_config=SimpleNamespace(enable=True, backend="tpu"),
    )
    assert compile_calls == [("head_moe", "tpu", False)]


def _fp32_master_ckpt_worker(rank: int, world: int, init_file: str, ckpt_root: str, host_offload: bool) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import TorchTitanTPUEngine
    from verl_hardware_plugin.accelerators.tpu.options import TPUEngineOptions

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))

        def _make_engine(seed: int, offload: bool = host_offload):
            gen = torch.Generator().manual_seed(seed)
            mod = torch.nn.Module()
            w_even = distribute_tensor(torch.randn(8, 4, generator=gen).to(torch.bfloat16), mesh, [Shard(0)])
            w_uneven = distribute_tensor(torch.randn(5, 4, generator=gen).to(torch.bfloat16), mesh, [Shard(0)])
            w_rep = distribute_tensor(torch.randn(4, generator=gen).to(torch.bfloat16), mesh, [Replicate()])
            w_plain = torch.randn(3, 3, generator=gen).to(torch.bfloat16)
            w_frozen = distribute_tensor(torch.randn(4, 2, generator=gen).to(torch.bfloat16), mesh, [Shard(0)])
            mod.register_parameter("layers_0_wq", torch.nn.Parameter(w_even, requires_grad=True))
            mod.register_parameter("layers_0_wo_uneven", torch.nn.Parameter(w_uneven, requires_grad=True))
            mod.register_parameter("norm", torch.nn.Parameter(w_rep, requires_grad=True))
            mod.register_parameter("plain", torch.nn.Parameter(w_plain, requires_grad=True))
            mod.register_parameter("frozen", torch.nn.Parameter(w_frozen, requires_grad=False))
            mod.register_buffer(
                "expert_bias",
                distribute_tensor(torch.zeros(4, dtype=torch.float32), mesh, [Shard(0)]),
            )
            dt_names = ["layers.0.wq.weight", "layers.0.wo.weight", "norm.weight"]
            dt_params = [mod.layers_0_wq, mod.layers_0_wo_uneven, mod.norm]
            stepped_params = [*dt_params, mod.plain]
            opt = torch.optim.AdamW(
                [
                    {"params": dt_params, "param_names": dt_names, "lr": 1e-6, "foreach": True},
                    {"params": [mod.plain], "param_names": ["plain.weight"], "lr": 1e-6, "foreach": True},
                ]
            )
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: max(0.1, (s + 1) / 5.0))
            eng = SimpleNamespace(
                _is_offload_param=False,
                _is_offload_optimizer=False,
                _tpu_eager_mode=None,
                _tpu_options=TPUEngineOptions(param_dtype="bfloat16", fp32_master=True, optimizer_host_offload=offload),
                module=[mod],
                optimizer=SimpleNamespace(optimizers=[opt], zero_grad=opt.zero_grad),
                lr_scheduler=sched,
                parallel_dims=SimpleNamespace(ep_enabled=False, get_optional_mesh=lambda _: None),
                config=SimpleNamespace(training=SimpleNamespace(max_norm=0.0625)),
            )
            for helper in (
                "_optimizer_param_chunks",
                "_chunked_grad_norm",
                "_chunked_clip_and_step",
                "_save_fp32_master_checkpoint",
                "_load_fp32_master_checkpoint",
                "save_checkpoint",
                "load_checkpoint",
            ):
                setattr(eng, helper, getattr(TorchTitanTPUEngine, helper).__get__(eng))
            return eng, mod, stepped_params

        def _apply_step_grads(mod, stepped_params, step_idx: int) -> None:
            g_gen = torch.Generator().manual_seed(100 + step_idx)
            full_grads = [
                distribute_tensor(torch.randn(8, 4, generator=g_gen).to(torch.bfloat16), mesh, [Shard(0)]),
                distribute_tensor(torch.randn(5, 4, generator=g_gen).to(torch.bfloat16), mesh, [Shard(0)]),
                distribute_tensor(torch.randn(4, generator=g_gen).to(torch.bfloat16), mesh, [Replicate()]),
                torch.randn(3, 3, generator=g_gen).to(torch.bfloat16),
            ]
            for p, g in zip(stepped_params, full_grads, strict=True):
                p.grad = g
            mod.expert_bias._local_tensor.add_(float(step_idx + 1 + rank))

        eng_cont, mod_cont, params_cont = _make_engine(seed=7)
        for s in range(1, 4):
            _apply_step_grads(mod_cont, params_cont, s)
            TorchTitanTPUEngine.optimizer_step(eng_cont)
            eng_cont.lr_scheduler.step()
            eng_cont.save_checkpoint(
                local_path=os.path.join(ckpt_root, f"global_step_{s}", "actor"),
                global_step=s,
                max_ckpt_to_keep=2,
            )

        assert not os.path.exists(os.path.join(ckpt_root, "global_step_1"))
        assert os.path.isdir(os.path.join(ckpt_root, "global_step_2", "actor"))
        assert os.path.isdir(os.path.join(ckpt_root, "global_step_3", "actor"))

        # Fresh engines (both same offload mode and opposite offload mode) resume from global_step_3/actor:
        eng_res, mod_res, params_res = _make_engine(seed=999, offload=host_offload)
        eng_cross, mod_cross, params_cross = _make_engine(seed=555, offload=not host_offload)
        eng_res.load_checkpoint(local_path=os.path.join(ckpt_root, "global_step_3", "actor"))
        eng_cross.load_checkpoint(local_path=os.path.join(ckpt_root, "global_step_3", "actor"))

        for s in range(4, 6):
            for eng, mod, ps in (
                (eng_cont, mod_cont, params_cont),
                (eng_res, mod_res, params_res),
                (eng_cross, mod_cross, params_cross),
            ):
                _apply_step_grads(mod, ps, s)
                TorchTitanTPUEngine.optimizer_step(eng)
                eng.lr_scheduler.step()

        # Same-mode resume is 0-ULP bitwise identical; cross-mode resume (CPU fused AVX-512 FMA vs foreach) matches
        # bf16 weights & step counters bit-for-bit and FP32 masters/moments within 1 FP32 FMA ULP.
        for is_same_mode, (other_eng, other_mod) in (
            (True, (eng_res, mod_res)),
            (False, (eng_cross, mod_cross)),
        ):
            for (_, p_c), (_, p_r) in zip(mod_cont.named_parameters(), other_mod.named_parameters(), strict=True):
                loc_c = p_c._local_tensor if hasattr(p_c, "_local_tensor") else p_c.data
                loc_r = p_r._local_tensor if hasattr(p_r, "_local_tensor") else p_r.data
                assert torch.equal(loc_c, loc_r)
                in_c = id(p_c) in eng_cont._fp32_master_store._entries
                in_r = id(p_r) in other_eng._fp32_master_store._entries
                assert in_c == in_r
                if in_c:
                    m_c, st_c = eng_cont._fp32_master_store._entries[id(p_c)]
                    m_r, st_r = other_eng._fp32_master_store._entries[id(p_r)]
                    ml_c = m_c._local_tensor if hasattr(m_c, "_local_tensor") else m_c
                    ml_r = m_r._local_tensor if hasattr(m_r, "_local_tensor") else m_r
                    if is_same_mode:
                        assert torch.equal(ml_c, ml_r)
                    else:
                        torch.testing.assert_close(ml_c, ml_r, rtol=1e-6, atol=1e-7)
                    assert set(st_c) == set(st_r)
                    for k in st_c:
                        vc = st_c[k]._local_tensor if hasattr(st_c[k], "_local_tensor") else st_c[k]
                        vr = st_r[k]._local_tensor if hasattr(st_r[k], "_local_tensor") else st_r[k]
                        if is_same_mode or k == "step":
                            assert torch.equal(vc, vr), k
                        else:
                            torch.testing.assert_close(vc, vr, rtol=1e-6, atol=1e-7)

            assert torch.equal(mod_cont.expert_bias._local_tensor, other_mod.expert_bias._local_tensor)
            assert eng_cont.lr_scheduler.state_dict() == other_eng.lr_scheduler.state_dict()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("host_offload", [False, True])
def test_fp32_master_checkpoint_save_load_lossless_dtensor_and_pruning(tmp_path, host_offload):
    """Saving & resuming ``fp32_master=True`` across distributed DTensor shards is 0-ULP bitwise lossless."""
    ckpt_root = tmp_path / f"ckpts_{host_offload}"
    ckpt_root.mkdir()
    torch.multiprocessing.start_processes(
        _fp32_master_ckpt_worker,
        args=(2, str(tmp_path / f"pg_{host_offload}"), str(ckpt_root), host_offload),
        nprocs=2,
        join=True,
        start_method="fork",
    )


def test_gcs_ckpt_blob_target_resolution(monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.engines.torchtitan_tpu import _gcs_ckpt_blob_target

    monkeypatch.delenv("VERL_GCS_CKPT_BUCKET", raising=False)
    monkeypatch.delenv("VERL_GCS_CKPT_MOUNT", raising=False)
    monkeypatch.delenv("VERL_GCS_CKPT_PREFIX", raising=False)

    assert _gcs_ckpt_blob_target("/gcs_ckpt/exp/global_step_5/actor") is None
    assert _gcs_ckpt_blob_target("gs://my-bucket/ckpts/global_step_5/actor/rank0.pt") == (
        "my-bucket",
        "ckpts/global_step_5/actor/rank0.pt",
    )

    monkeypatch.setenv("VERL_GCS_CKPT_BUCKET", "my-ckpt-bucket")
    monkeypatch.setenv("VERL_GCS_CKPT_PREFIX", "verl/checkpoints")
    assert _gcs_ckpt_blob_target("/gcs_ckpt/exp1/global_step_5/actor/shard.pt") == (
        "my-ckpt-bucket",
        "verl/checkpoints/exp1/global_step_5/actor/shard.pt",
    )
    assert _gcs_ckpt_blob_target("/tmp/local_ckpt/global_step_5/actor/shard.pt") is None


def test_should_enable_tpu_topo_mesh_and_install_ep_dispatcher(monkeypatch):
    from dataclasses import dataclass

    import torch.distributed.device_mesh as dm_mod

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_ragged_ep, tpu_utils

    @dataclass
    class _NoTopoParallelism:
        expert_parallel_degree: int = 1

    @dataclass
    class _TopoParallelism:
        enable_tpu_topo_mesh: bool = False

    assert tpu_ragged_ep.should_enable_tpu_topo_mesh(None, _NoTopoParallelism) is None
    assert tpu_ragged_ep.should_enable_tpu_topo_mesh(True, _TopoParallelism) is True
    assert tpu_ragged_ep.should_enable_tpu_topo_mesh(False, _TopoParallelism) is False

    class _OldDeviceMesh:
        def __init__(self, device_type: str, mesh: object):
            pass

    class _NewDeviceMesh:
        def __init__(self, device_type: str, mesh: object, *, preserve_rank_order: bool = False):
            pass

    monkeypatch.setattr(dm_mod, "DeviceMesh", _OldDeviceMesh)
    assert tpu_ragged_ep.should_enable_tpu_topo_mesh(None, _TopoParallelism) is False
    monkeypatch.setattr(dm_mod, "DeviceMesh", _NewDeviceMesh)
    assert tpu_ragged_ep.should_enable_tpu_topo_mesh(None, _TopoParallelism) is True

    # Verify auto/static/ragged installation on non-nn.Module Configurable-style TorchTitan dispatchers,
    # Pallas tile alignment, score_before_experts=False default, and restoring fullgraph=True on relaxed blocks.
    compile_calls: list[tuple[str, str, bool]] = []

    class AllToAllTokenDispatcher:
        def __init__(self):
            self.num_experts = 8
            self.top_k = 2
            self.fuse_route_score_backward = True

        def dispatch(self, *args: object):
            return args

        def combine(self, *args: object):
            return args[0]

    class TpuQwen35TokenDispatcher(AllToAllTokenDispatcher):
        def __init__(self):
            super().__init__()
            self.capacity_factor = None

    class _FakePallasExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_pallas_grouped_mm = True
            self.pallas_grouped_mm_tile_m = 512
            self.pallas_grouped_mm_tgmm_tiling = (256, 1024, 0)
            self.pallas_grouped_mm_w2_tgmm_tiling = (384, 1024, 1024)

    class _MoeBlock(torch.nn.Module):
        def __init__(self, dispatcher: object):
            super().__init__()
            self._compiled_call_impl = object()
            self.routed = torch.nn.Module()
            self.routed.inner_experts = _FakePallasExperts()
            self.routed.token_dispatcher = dispatcher

        def compile(self, *, backend: str = "tpu", fullgraph: bool = True, **kwargs: object):
            compile_calls.append(("moe_block", backend, fullgraph))

    static_root = torch.nn.Module()
    static_root.layers = torch.nn.ModuleDict({"0": _MoeBlock(TpuQwen35TokenDispatcher())})
    mode_s, n_s = tpu_ragged_ep.install_tpu_ep_token_dispatcher(
        [static_root],
        ep_dispatcher="static",
        capacity_factor=4.0,
        spmd_backend="default",
    )
    assert (mode_s, n_s) == ("static", 1)
    assert static_root.layers["0"].routed.token_dispatcher.capacity_factor == 4.0

    root = torch.nn.Module()
    root.layers = torch.nn.ModuleDict({"0": _MoeBlock(AllToAllTokenDispatcher())})
    compile_cfg = SimpleNamespace(enable=True, backend="tpu")
    tpu_utils._relax_non_static_ep_moe_compile(
        root,
        parallel_dims=SimpleNamespace(ep_enabled=True),
        compile_config=compile_cfg,
    )
    assert compile_calls == [("moe_block", "tpu", False)]

    mode, n_installed = tpu_ragged_ep.install_tpu_ep_token_dispatcher(
        [root],
        ep_dispatcher="ragged",
        capacity_factor=3.5,
        spmd_backend="default",
        compile_config=compile_cfg,
    )
    assert (mode, n_installed) == ("ragged", 1)
    td = root.layers["0"].routed.token_dispatcher
    assert isinstance(td, tpu_ragged_ep.TpuRaggedEPTokenDispatcher)
    assert not isinstance(td, torch.nn.Module)
    assert td.score_before_experts is False
    assert td.fuse_route_score_backward is True
    assert td.capacity_factor == 3.5
    assert td.unpadded_expert_counts is True
    assert td.row_align == 1536  # lcm(512, 256, 384)
    assert compile_calls == [("moe_block", "tpu", False), ("moe_block", "tpu", True)]


def _dense_static_ep_reference(
    x_TD: torch.Tensor,
    scores_TK: torch.Tensor,
    expert_ids_TK: torch.Tensor,
    counts_E: torch.Tensor,
    weights_EDE: torch.Tensor,
    *,
    ep_mesh: object,
    capacity_factor: float | None,
    score_before_experts: bool,
) -> torch.Tensor:
    """Pure-PyTorch reference matching ``TpuQwen35TokenDispatcher`` + per-local-expert matmul."""
    import math as _math

    import torch.distributed._functional_collectives as funcol

    from verl_hardware_plugin.accelerators.tpu.engines.tpu_ragged_ep import (
        _default_combine_reduce,
        _default_local_reorder,
        _wait_collective,
    )

    ep_size = ep_mesh.size()  # type: ignore[attr-defined]
    pg = ep_mesh.get_group()  # type: ignore[attr-defined]
    num_experts = int(counts_E.shape[0])
    e = num_experts // ep_size
    routed_ND, tok_idx_N, scores_N = _default_local_reorder(x_TD, scores_TK, expert_ids_TK, score_before_experts)
    n_routed = int(routed_ND.shape[0])
    cap = (
        max(n_routed, 1)
        if capacity_factor is None
        else max(1, min(max(n_routed, 1), _math.ceil(capacity_factor * n_routed / ep_size)))
    )
    local_EP_e = counts_E.to(torch.int32).reshape(ep_size, e)
    if capacity_factor is not None:
        cum = local_EP_e.cumsum(dim=1).clamp(max=cap)
        capped_EP_e = torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], dim=1)
    else:
        capped_EP_e = local_EP_e
    send_EP = local_EP_e.sum(dim=1)
    send_cap_EP = capped_EP_e.sum(dim=1)
    starts_EP = send_EP.cumsum(dim=0) - send_EP

    # Pack dense [ep_size, cap, D] send buffer:
    send_buf = routed_ND.new_zeros((ep_size, cap, routed_ND.shape[1]))
    for dst in range(ep_size):
        cnt = int(send_cap_EP[dst].item())
        if cnt > 0:
            st = int(starts_EP[dst].item())
            send_buf[dst, :cnt] = routed_ND[st : st + cnt]

    recv_counts = _wait_collective(funcol.all_to_all_single(capped_EP_e.reshape(-1), None, None, group=pg)).reshape(
        ep_size, e
    )
    recv_buf = _wait_collective(
        funcol.all_to_all_single(send_buf.reshape(ep_size * cap, -1), None, None, group=pg)
    ).reshape(ep_size, cap, -1)

    # Apply local expert linear layers weights_EDE [e, D, D] per sender & local expert:
    out_recv_buf = torch.zeros_like(recv_buf)
    for sender in range(ep_size):
        cur = 0
        for le in range(e):
            cnt = int(recv_counts[sender, le].item())
            if cnt > 0:
                tokens = recv_buf[sender, cur : cur + cnt]
                out_recv_buf[sender, cur : cur + cnt] = tokens @ weights_EDE[le]
                cur += cnt

    send_back = _wait_collective(
        funcol.all_to_all_single(out_recv_buf.reshape(ep_size * cap, -1), None, None, group=pg)
    ).reshape(ep_size, cap, -1)

    out_ND = routed_ND.new_zeros(routed_ND.shape)
    for dst in range(ep_size):
        cnt = int(send_cap_EP[dst].item())
        if cnt > 0:
            st = int(starts_EP[dst].item())
            out_ND[st : st + cnt] = send_back[dst, :cnt]

    return _default_combine_reduce(
        out_ND,
        tok_idx_N,
        scores_N,
        x_TD,
        top_k=int(expert_ids_TK.shape[1]),
        score_before_experts=score_before_experts,
    )


def _ragged_ep_dist_worker(rank: int, world: int, init_file: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_ragged_ep
    from verl_hardware_plugin.accelerators.tpu.engines.tpu_ragged_ep import (
        TpuRaggedEPTokenDispatcher,
        _default_local_reorder,
    )

    def _exact_2arg_scale_route_weights(routed_output_ND: torch.Tensor, scores_N: torch.Tensor) -> torch.Tensor:
        return (routed_output_ND.to(torch.float32) * scores_N.reshape(-1, 1).to(torch.float32)).to(
            routed_output_ND.dtype
        )

    def _exact_3arg_stable_grouped_reduce(src_ND: torch.Tensor, index_N: torch.Tensor, top_k: int) -> torch.Tensor:
        num_tokens = src_ND.shape[0] // top_k
        perm = torch.argsort(index_N, stable=True)
        return src_ND.index_select(0, perm).to(torch.float32).reshape(num_tokens, top_k, -1).sum(dim=1).to(src_ND.dtype)

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        ep_mesh = init_device_mesh("cpu", (world,))
        num_experts = 8
        e = num_experts // world
        top_k = 2
        dim = 16

        # Case 1: Skewed routing with capacity dropping (capacity_factor=0.75) + unpadded MegaBlox NaN tail rows!
        # Case 2: Rank 1 sends 0 tokens to Rank 0 + real TorchTitan base_dispatcher (fuse_route_score_backward=True,
        #         no score_before_experts attr -> False, exact 2-arg scale_route_weights & 3-arg stable_grouped_reduce)!
        scenarios = [
            (0.75, True, True, False),
            (4.0, False, False, True),
        ]
        for sc_idx, (cap_factor, score_before, unpadded, use_tt_base) in enumerate(scenarios):
            gen = torch.Generator().manual_seed(500 + sc_idx * 10 + rank)
            t_tokens = 12
            x_ref = torch.randn(t_tokens, dim, generator=gen, dtype=torch.float32, requires_grad=True)
            scores_ref = torch.rand(t_tokens, top_k, generator=gen, dtype=torch.float32, requires_grad=True)
            w_ref = torch.randn(e, dim, dim, generator=gen, dtype=torch.float32, requires_grad=True)
            grad_out = torch.randn(t_tokens, dim, generator=gen, dtype=torch.float32)

            if sc_idx == 0:
                # Heavy skew onto rank 1's local experts (experts 4..7) to exercise capacity clamping:
                exp_ids = torch.tensor(
                    [[4, 5], [4, 6], [5, 7], [4, 5], [6, 7], [0, 4], [4, 5], [5, 6], [4, 7], [1, 4], [4, 6], [5, 7]],
                    dtype=torch.int64,
                )
            else:
                # Rank 0 routes only to experts 0..1; Rank 1 routes only to experts 4..5 (0 cross-rank traffic!).
                base_e = 0 if rank == 0 else 4
                exp_ids = torch.tensor([[base_e, base_e + 1]] * t_tokens, dtype=torch.int64)
            counts_E = torch.bincount(exp_ids.reshape(-1), minlength=num_experts).to(torch.int32)

            out_ref = _dense_static_ep_reference(
                x_ref,
                scores_ref,
                exp_ids,
                counts_E,
                w_ref,
                ep_mesh=ep_mesh,
                capacity_factor=cap_factor,
                score_before_experts=score_before,
            )
            (out_ref * grad_out).sum().backward()

            x_rag = x_ref.detach().clone().requires_grad_(True)
            scores_rag = scores_ref.detach().clone().requires_grad_(True)
            w_rag = w_ref.detach().clone().requires_grad_(True)

            if use_tt_base:
                tpu_ragged_ep._tt_scale_route_weights = _exact_2arg_scale_route_weights
                tpu_ragged_ep._tt_stable_grouped_reduce = _exact_3arg_stable_grouped_reduce

                class _FakeTTBase:
                    def __init__(self):
                        self.num_experts = num_experts
                        self.top_k = top_k
                        self.fuse_route_score_backward = True
                        self.ep_mesh = ep_mesh

                    def _local_reorder(self, x_t, s_tk, id_tk):
                        return _default_local_reorder(x_t, s_tk, id_tk, False)

                disp = TpuRaggedEPTokenDispatcher(
                    _FakeTTBase(),
                    capacity_factor=cap_factor,
                    unpadded_expert_counts=unpadded,
                    row_align=16,
                )
                assert disp.score_before_experts is False
            else:
                disp = TpuRaggedEPTokenDispatcher(
                    num_experts=num_experts,
                    top_k=top_k,
                    score_before_experts=score_before,
                    capacity_factor=cap_factor,
                    unpadded_expert_counts=unpadded,
                    row_align=16,
                    ep_mesh=ep_mesh,
                )
            routed_in_RD, counts_local_e, meta = disp.dispatch(x_rag, scores_rag, exp_ids, counts_E)
            assert routed_in_RD.shape[0] % 16 == 0
            assert counts_local_e.dtype == torch.int32
            for field_name in (
                "disp_in_off_EP",
                "disp_send_sz_EP",
                "disp_out_off_EP",
                "disp_recv_sz_EP",
                "comb_in_off_EP",
                "comb_out_off_EP",
                "perm_idx_R",
                "inv_perm_idx_R",
            ):
                assert getattr(meta, field_name).dtype == torch.int32, field_name
            total_valid = int(meta.valid_mask_R1.sum().item())
            if unpadded:
                assert int(counts_local_e.sum().item()) == total_valid

                # Inject NaNs into backward grad of invalid tail rows >= total_valid to verify NaN-safe torch.where!
                def _poison_tail_grad(g: torch.Tensor, tv: int = total_valid) -> torch.Tensor:
                    poisoned = g.clone()
                    poisoned[tv:] = float("nan")
                    return poisoned

                routed_in_RD.register_hook(_poison_tail_grad)
            else:
                assert int(counts_local_e.sum().item()) == routed_in_RD.shape[0]

            # Apply local experts on valid segments (and fill tail >= total_valid with NaN when unpadded, mimicking
            # MegaBlox gmm leaving out[total_valid:] uninitialized in recycled HBM):
            pieces: list[torch.Tensor] = []
            cursor = 0
            for le in range(e):
                cnt = (
                    int(counts_local_e[le].item())
                    if unpadded
                    else int(min(max(total_valid - cursor, 0), int(counts_local_e[le].item())))
                )
                if cnt > 0:
                    pieces.append(routed_in_RD[cursor : cursor + cnt] @ w_rag[le])
                    cursor += cnt
            valid_out = torch.cat(pieces, dim=0) if pieces else routed_in_RD[:0]
            tail_rows = routed_in_RD.shape[0] - valid_out.shape[0]
            tail_fill = float("nan") if unpadded else 0.0
            tail = torch.full((tail_rows, dim), tail_fill, dtype=routed_in_RD.dtype)
            routed_out_RD = torch.cat([valid_out, tail], dim=0)

            out_rag = disp.combine(routed_out_RD, meta, x_rag)
            (out_rag * grad_out).sum().backward()

            assert not torch.isnan(out_rag).any()
            assert x_rag.grad is not None and not torch.isnan(x_rag.grad).any()
            assert scores_rag.grad is not None and not torch.isnan(scores_rag.grad).any()
            assert w_rag.grad is not None and not torch.isnan(w_rag.grad).any()
            torch.testing.assert_close(out_rag, out_ref, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(x_rag.grad, x_ref.grad, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(scores_rag.grad, scores_ref.grad, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(w_rag.grad, w_ref.grad, rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_ragged_ep_token_dispatcher_distributed_parity_and_nan_tail_immunity(tmp_path):
    """2-rank ``gloo`` parity for ``TpuRaggedEPTokenDispatcher`` (skew, dropping, 0-token peers, & NaN tail safety)."""
    torch.multiprocessing.start_processes(
        _ragged_ep_dist_worker,
        args=(2, str(tmp_path / "pg_ragged_ep")),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def _ragged_ep_fullgraph_compile_worker(rank: int, world: int, init_file: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_ragged_ep

    # Register a test ``tpu::_ragged_all_to_all`` custom op with CPU + Meta kernels so ``torch.compile(...,
    # fullgraph=True)`` traces the exact custom-op path used on TPU!
    if not tpu_ragged_ep.has_tpu_ragged_all_to_all():
        lib = torch.library.Library("tpu", "DEF")
        lib.define(
            "_ragged_all_to_all(Tensor operand, Tensor output, Tensor input_offsets, Tensor send_sizes, "
            "Tensor output_offsets, Tensor recv_sizes, str group_name) -> Tensor"
        )
        impl_cpu = torch.library.Library("tpu", "IMPL", "CPU")
        impl_meta = torch.library.Library("tpu", "IMPL", "Meta")

        def _cpu_kernel(operand, output, in_off, s_sz, out_off, r_sz, group_name):
            for t in (in_off, s_sz, out_off, r_sz):
                assert t.dtype == torch.int32, f"ragged_all_to_all metadata must be int32, got {t.dtype}"
            pg = dist.distributed_c10d._resolve_process_group(group_name)
            return tpu_ragged_ep._cpu_ragged_all_to_all(operand, output, in_off, s_sz, out_off, r_sz, pg)

        def _meta_kernel(operand, output, in_off, s_sz, out_off, r_sz, group_name):
            for t in (in_off, s_sz, out_off, r_sz):
                assert t.dtype == torch.int32, f"ragged_all_to_all metadata must be int32, got {t.dtype}"
            return torch.empty_like(output)

        impl_cpu.impl("_ragged_all_to_all", _cpu_kernel)
        impl_meta.impl("_ragged_all_to_all", _meta_kernel)
        # Keep library handles alive for the worker lifetime:
        tpu_ragged_ep._TEST_TPU_LIBS = (lib, impl_cpu, impl_meta)  # type: ignore[attr-defined]

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        ep_mesh = init_device_mesh("cpu", (world,))
        tpu_ragged_ep._FORCE_TPU_CUSTOM_OP_ON_CPU = True
        disp = tpu_ragged_ep.TpuRaggedEPTokenDispatcher(
            num_experts=4,
            top_k=2,
            score_before_experts=True,
            capacity_factor=2.0,
            unpadded_expert_counts=True,
            row_align=8,
            ep_mesh=ep_mesh,
        )

        def _step_fn(x, scores, exp_ids, counts, w):
            rin, _, meta = disp.dispatch(x, scores, exp_ids, counts)
            rout = rin @ w
            return disp.combine(rout, meta, x)

        compiled_fn = torch.compile(_step_fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(6, 8, requires_grad=True)
        scores = torch.rand(6, 2, requires_grad=True)
        w = torch.randn(8, 8, requires_grad=True)
        exp_ids = torch.tensor([[0, 2], [1, 3], [2, 3], [0, 1], [2, 3], [1, 2]], dtype=torch.int64)
        counts = torch.bincount(exp_ids.reshape(-1), minlength=4).to(torch.int32)

        out = compiled_fn(x, scores, exp_ids, counts, w)
        out.sum().backward()
        assert out.shape == (6, 8) and not torch.isnan(out).any()
        assert x.grad is not None and not torch.isnan(x.grad).any()
    finally:
        tpu_ragged_ep._FORCE_TPU_CUSTOM_OP_ON_CPU = False
        dist.destroy_process_group()


def test_ragged_ep_token_dispatcher_fullgraph_compile(tmp_path):
    """``TpuRaggedEPTokenDispatcher`` traces through ``torch.compile(..., fullgraph=True)`` with 0 graph breaks."""
    torch.multiprocessing.start_processes(
        _ragged_ep_fullgraph_compile_worker,
        args=(2, str(tmp_path / "pg_ragged_compile")),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
