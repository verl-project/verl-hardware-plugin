# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the TPU TorchTitan engine utilities on CPU."""

import ast
import os
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch._dynamo
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config import ActorConfig, TorchtitanEngineConfig
from verl.workers.utils.losses import ppo_loss, sft_loss
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    TPUSplashAttention,
    TPUVarlenAttention,
    bucket_length,
    configure_torch_compile_for_tpu,
    extend_torchtitan_engine_config,
    get_tpu_seq_bucket_size,
    get_tpu_simple_fsdp_parallelize_fn,
    init_states_without_simple_fsdp_parametrization,
    install_tpu_attention,
    pad_packed_inputs_for_tpu,
    prepare_tpu_loss_inputs,
    resolve_tpu_torchtitan_options,
    splash_block_size_for,
    unwrap_metadata,
)
from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    PlatformTPU,
    resolve_tpu_topology_bounds,
)


def _fake_torchtitan_modules() -> dict[str, types.ModuleType]:
    """Stand-ins for the two torchtitan modules these CPU tests touch, for ``mock.patch.dict(sys.modules, ...)``.

    The TPU engine imports google-pytorch/torchtitan lazily, and this repo's CPU test workflow does not install it.
    """
    attention = types.ModuleType("torchtitan.models.common.attention")
    attention.VarlenInnerAttention = type("VarlenInnerAttention", (torch.nn.Module,), {})  # type: ignore[attr-defined]

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
    return {
        attention.__name__: attention,
        graph_trainer.__name__: graph_trainer,
        simple_fsdp.__name__: simple_fsdp,
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
    out = TPUVarlenAttention()(q, k, v, positions=positions)
    assert out.shape == (8, 4, 16)

    def reference(lo, hi):  # causal attention restricted to tokens [lo, hi)
        qd, kd, vd = (t[lo:hi].transpose(0, 1) for t in (q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)))
        return torch.nn.functional.scaled_dot_product_attention(qd, kd, vd, is_causal=True).transpose(0, 1)

    for lo, hi in [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8)]:
        torch.testing.assert_close(out[lo:hi], reference(lo, hi))
    with pytest.raises(NotImplementedError, match="out_transform"):
        TPUVarlenAttention()(q, k, v, positions=positions, out_transform=lambda o, lse: o)


def test_splash_block_size_for():
    assert splash_block_size_for(512) == 512
    assert splash_block_size_for(1024) == 512
    assert splash_block_size_for(768) == 256
    assert splash_block_size_for(256) == 256
    assert splash_block_size_for(384) == 128
    assert splash_block_size_for(64) == 0


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
    assert install_tpu_attention([model], use_splash=True) == 2
    splash = model[0].inner_attention
    assert isinstance(splash, TPUSplashAttention)
    assert splash.local_window_size == 256
    assert isinstance(splash.fallback, TPUVarlenAttention) and splash.fallback.local_window_size == 256
    assert install_tpu_attention([model], use_splash=True) == 0  # already installed

    # On CPU the splash module runs its dense fallback.
    q = torch.randn(256, 2, 4)
    positions = torch.cat([torch.arange(100), torch.arange(156)])
    torch.testing.assert_close(splash(q, q, q, positions=positions), splash.fallback(q, q, q, positions=positions))

    with pytest.raises(ValueError, match="inner_attention"):
        install_tpu_attention([torch.nn.Linear(2, 2)], use_splash=True)


def test_configure_torch_compile_for_tpu():
    configure_torch_compile_for_tpu(recompile_limit=64)
    assert torch._dynamo.config.automatic_dynamic_shapes is False
    assert torch._dynamo.config.assume_static_by_default is True
    assert torch._dynamo.config.capture_scalar_outputs is True
    assert torch._dynamo.config.recompile_limit >= 64


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
        use_simple_fsdp=True,
        tpu_eager_mode="DEFER_AND_FUSE",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        expert_parallel_size=1,
    )
    with pytest.raises(ValueError, match="use_simple_fsdp supports pure FSDP/HSDP only"):
        resolve_tpu_torchtitan_options(cfg_tp)


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

    # The env var wins over everything, including an unknown chip count.
    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "2,3,1"}):
        assert resolve_tpu_topology_bounds(total_chips=6, num_nodes=2)[0] == "2,3,1"

    with mock.patch.dict(os.environ, {"VERL_TPU_CHIPS_PER_HOST": "2"}):
        assert resolve_tpu_topology_bounds(total_chips=8, num_nodes=2)[3] == "2"


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


def test_get_ray_init_kwargs_pins_tpu_platform():
    assert PlatformTPU().get_ray_init_kwargs() == {"runtime_env": {"env_vars": {"VERL_PLATFORM": "tpu"}}}


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


def test_prepare_tpu_loss_inputs_matches_unmodified_sft_loss():
    # 2 packed samples of lengths 3 and 2 (orig_seq_len = 5), padded to bucket 8
    padded_log_probs = torch.tensor(
        [-1.0, -2.0, -3.0, -4.0, -5.0, -100.0, -200.0, -300.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    unpadded_log_probs = padded_log_probs.detach().clone()[:5]
    nested_log_probs = torch.nested.nested_tensor_from_jagged(unpadded_log_probs, cu_seqlens)
    setattr(nested_log_probs, TPU_PADDED_VALUES_ATTR, padded_log_probs)

    loss_mask = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 1]), torch.tensor([0, 1])],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({"loss_mask": loss_mask}, batch_size=[2])
    tu.assign_non_tensor(
        micro_batch,
        pad_mode=DatasetPadMode.NO_PADDING,
        dp_size=2,
        batch_num_tokens=6,
    )

    model_output = {"log_probs": nested_log_probs}
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = sft_loss(config=None, model_output=loss_model_output, data=loss_data)
    assert metrics == {}
    loss.backward()

    # Rolled unpadded mask is [1, 1, 0, 1, 0], padded with 3 zeros -> [1, 1, 0, 1, 0, 0, 0, 0].
    # Selected log_probs are -1.0, -2.0, -4.0 -> sum = -7.0 -> loss = -(-7.0) / 6 * 2 = 7/3.
    assert loss.item() == pytest.approx(7.0 / 3.0)
    assert padded_log_probs.grad is not None
    expected_grad = torch.tensor([-1.0 / 3.0, -1.0 / 3.0, 0.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(padded_log_probs.grad, expected_grad)


def _ppo_micro_batch(lengths, max_response_len=None, seed=0):
    """Packed PPO micro-batch as `TorchTitanTPUEngineWithLMHead` builds it: nested prompts/responses on CPU."""
    gen = torch.Generator().manual_seed(seed)

    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    prompt_lens = [p for p, _ in lengths]
    response_lens = [r for _, r in lengths]
    micro_batch = TensorDict(
        {
            "prompts": nested([torch.randint(0, 100, (p,), generator=gen) for p in prompt_lens]),
            "responses": nested([torch.randint(0, 100, (r,), generator=gen) for r in response_lens]),
            "input_ids": nested([torch.randint(0, 100, (p + r,), generator=gen) for p, r in lengths]),
            "response_mask": nested([torch.ones(r, dtype=torch.int64) for r in response_lens]),
            "old_log_probs": nested([-torch.rand(r, generator=gen) for r in response_lens]),
            "advantages": nested([torch.randn(r, generator=gen) for r in response_lens]),
            "ref_log_prob": nested([-torch.rand(r, generator=gen) for r in response_lens]),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(micro_batch, dp_size=2, batch_num_tokens=sum(response_lens) * 2, global_batch_size=8)
    if max_response_len is not None:
        tu.assign_non_tensor_data(micro_batch, "max_response_len", max_response_len)
    return micro_batch


def _ppo_model_output(lengths, padded_len, with_entropy, seed=0):
    """Bucket-padded device outputs + their trimmed CPU views, as `prepare_model_outputs` returns them."""
    gen = torch.Generator().manual_seed(seed + 1)
    total = sum(p + r for p, r in lengths)
    cu_seqlens = torch.tensor([0] + [p + r for p, r in lengths]).cumsum(0)
    leaves = {"log_probs": -torch.rand(padded_len, generator=gen)}
    if with_entropy:
        leaves["entropy"] = torch.rand(padded_len, generator=gen)
    model_output = {}
    for key, leaf in leaves.items():
        leaf.requires_grad_(True)
        nested = torch.nested.nested_tensor_from_jagged(leaf.detach()[:total], cu_seqlens)
        setattr(nested, TPU_PADDED_VALUES_ATTR, leaf)
        model_output[key] = nested
    return model_output, leaves, cu_seqlens


def _ppo_config():
    return ActorConfig(
        strategy="fsdp",
        rollout_n=8,
        ppo_micro_batch_size_per_gpu=4,
        loss_agg_mode="token-mean",
        use_kl_loss=True,
        kl_loss_coef=0.001,
        entropy_coeff=0.01,
    )


@pytest.mark.parametrize("lengths", [[(5, 7)], [(3, 2), (6, 9), (4, 1)]])
@pytest.mark.parametrize("with_entropy", [False, True])
def test_prepare_tpu_loss_inputs_ppo_matches_unmodified_ppo_loss(lengths, with_entropy):
    padded_len = bucket_length(sum(p + r for p, r in lengths), 16)
    response_bucket = bucket_length(max(r for _, r in lengths), 16)

    # Reference: core ppo_loss on exact (unbucketed) packed outputs, as on GPU.
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len, with_entropy)
    total = int(cu_seqlens[-1])
    reference_output = {
        key: torch.nested.nested_tensor_from_jagged(leaf[:total], cu_seqlens) for key, leaf in leaves.items()
    }
    ref_loss, ref_metrics = ppo_loss(_ppo_config(), reference_output, _ppo_micro_batch(lengths))
    ref_loss.backward()
    ref_grads = {key: leaf.grad.clone() for key, leaf in leaves.items()}
    for leaf in leaves.values():
        leaf.grad = None

    # TPU adapter: bucket-padded outputs, static layout, same unmodified core loss.
    micro_batch = _ppo_micro_batch(lengths, max_response_len=response_bucket)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
    loss, metrics = ppo_loss(_ppo_config(), loss_model_output, loss_data)
    loss.backward()

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=1e-6)
    for key, leaf in leaves.items():
        torch.testing.assert_close(leaf.grad, ref_grads[key], rtol=0, atol=1e-7)
    assert set(metrics) == set(ref_metrics)
    # The source micro-batch keeps its real sequence metadata.
    assert micro_batch["responses"].offsets().diff().tolist() == [r for _, r in lengths]


def test_prepare_tpu_loss_inputs_ppo_shapes_depend_only_on_buckets():
    """Different exact lengths in the same buckets must give the loss identical shapes (no XLA recompiles)."""

    def loss_inputs(lengths):
        model_output, _, _ = _ppo_model_output(lengths, padded_len=48, with_entropy=True)
        micro_batch = _ppo_micro_batch(lengths, max_response_len=16)
        return prepare_tpu_loss_inputs(model_output, micro_batch)

    out_a, data_a = loss_inputs([(5, 7), (9, 11)])
    out_b, data_b = loss_inputs([(12, 16), (3, 2)])
    for key in ("log_probs", "entropy"):
        assert out_a[key].values().shape == out_b[key].values().shape == torch.Size([2 * (16 + 1)])
    for key in ("prompts", "responses"):
        assert data_a[key].offsets().tolist() == data_b[key].offsets().tolist()
    for key in ("response_mask", "old_log_probs", "advantages", "ref_log_prob"):
        assert data_a[key].shape == data_b[key].shape == torch.Size([2, 16])
    assert tu.get_non_tensor_data(data_a, "max_response_len", None) == 16


def test_prepare_tpu_loss_inputs_ppo_falls_back_to_trimmed_outputs():
    """A packed output without its bucket-padded device tensor keeps the exact-length (trimmed) path."""
    lengths = [(3, 2), (4, 5)]
    model_output, leaves, cu_seqlens = _ppo_model_output(lengths, padded_len=32, with_entropy=False)
    model_output["values"] = torch.nested.nested_tensor_from_jagged(torch.zeros(int(cu_seqlens[-1])), cu_seqlens)
    loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, _ppo_micro_batch(lengths, 16))
    assert loss_model_output["log_probs"].values().shape == torch.Size([int(cu_seqlens[-1])])
    assert loss_data["responses"].offsets().diff().tolist() == [2, 5]


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
    engine.parallel_dims = SimpleNamespace(tp_enabled=False)
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
