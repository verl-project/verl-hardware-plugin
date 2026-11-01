# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import ast
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
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
    assert ids.tolist() == [10, 11, 12, 20, 21, 0, 0, 0]
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


def test_install_parallel_dims_cache_for_tpu_reuses_parallel_dims_and_mesh():
    from verl_hardware_plugin.accelerators.tpu.engines import tpu_utils

    build_calls = []

    class _FakeParallelDims:
        def __init__(self, cfg, ws):
            self.cfg = cfg
            self.ws = ws
            self._world_mesh = None
            self._single_axis_meshes = None

        @classmethod
        def from_config(cls, cfg, ws):
            return cls(cfg, ws)

        def build_mesh(self):
            build_calls.append(1)
            self._world_mesh = object()
            self._single_axis_meshes = {"tp": object()}
            return self._world_mesh

    fake_pd_mod = types.ModuleType("torchtitan.distributed.parallel_dims")
    fake_pd_mod.ParallelDims = _FakeParallelDims  # type: ignore[attr-defined]
    fake_dist_mod = types.ModuleType("torchtitan.distributed")
    fake_dist_mod.parallel_dims = fake_pd_mod  # type: ignore[attr-defined]

    with (
        mock.patch.dict(
            sys.modules,
            {
                "torchtitan.distributed": fake_dist_mod,
                "torchtitan.distributed.parallel_dims": fake_pd_mod,
            },
        ),
        mock.patch.object(tpu_utils, "_PARALLEL_DIMS_PATCHED", False),
        mock.patch.object(tpu_utils, "_PARALLEL_DIMS_CACHE", {}),
    ):
        tpu_utils.install_parallel_dims_cache_for_tpu()
        cfg1 = SimpleNamespace(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=2,
            context_parallel_degree=1,
            tensor_parallel_degree=2,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            spmd_backend="spmd_types",
            enable_tpu_topo_mesh=False,
        )
        cfg2 = SimpleNamespace(**vars(cfg1))
        pd1 = _FakeParallelDims.from_config(cfg1, 4)
        pd2 = _FakeParallelDims.from_config(cfg2, 4)
        assert pd1 is pd2
        m1 = pd1.build_mesh()
        m2 = pd2.build_mesh()
        assert m1 is m2
        assert len(build_calls) == 1


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
