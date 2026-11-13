# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct TorchTitan adapter tests with CPU tensors and no accelerator runtime."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("spmd_types")
pytest.importorskip("torchtitan.components.lr_scheduler")

from verl_hardware_plugin.accelerators.neuron.engines.torchtitan_neuron import (
    TorchTitanNeuronEngineWithLMHead,
    native_config_overrides,
)


def test_forward_preserves_loss_autograd_and_row_boundaries(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    inputs = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    batch = TensorDict({"input_ids": inputs, "temperature": torch.ones(2)}, batch_size=[2])
    logits = torch.randn(2, 8, 7, requires_grad=True)
    captured = {}

    def forward(*, inputs, extra_inputs, extra_kwargs):
        captured.update(ids=inputs, positions=extra_inputs["positions"], kwargs=extra_kwargs)
        return logits

    engine = SimpleNamespace(
        engine_config=SimpleNamespace(max_seq_len=8),
        trainer=SimpleNamespace(device="cpu"),
        model_forward_step=forward,
        get_data_parallel_group=lambda: None,
    )

    def loss_function(model_output, data, dp_group):
        values = model_output["log_probs"].values()
        return -values[torch.tensor([0, 1, 3])].sum(), {}

    loss, output = TorchTitanNeuronEngineWithLMHead.forward_step(engine, batch, loss_function, False)
    expected = -(logits[0, 0].log_softmax(-1)[2] + logits[0, 1].log_softmax(-1)[3] + logits[1, 0].log_softmax(-1)[5])
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert captured["kwargs"] == {"attention_masks": None}
    assert captured["ids"].shape == (2, 8)
    assert captured["positions"][1].tolist() == list(range(8))
    assert not output["model_output"]["log_probs"].requires_grad


def test_native_overrides_restore_after_failure():
    from torchtitan.models import qwen3

    from verl_hardware_plugin.accelerators.neuron.engines.torchtitan_neuron import titan_impl

    original_compile = titan_impl.CompileConfig
    original_training = titan_impl.TrainingConfig
    original_registry = qwen3.model_registry
    original_init = titan_impl.Trainer.init_distributed
    original_checkpointer = titan_impl.CheckpointManager
    dimensions = object()
    with pytest.raises(RuntimeError, match="sentinel"):
        with native_config_overrides(SimpleNamespace(tie_word_embeddings=False), parallel_dims=dimensions):
            assert titan_impl.CompileConfig(enable=True).backend == "neuron"
            assert not titan_impl.ParallelismConfig().enable_sequence_parallel
            assert titan_impl.Trainer.init_distributed(None) is dimensions
            assert not titan_impl.TrainingConfig(enable_cpu_offload=True).enable_cpu_offload
            assert not qwen3.model_registry("0.6B", attn_backend="sdpa").model.enable_weight_tying
            from verl_hardware_plugin.accelerators.neuron.engines.checkpoint import NeuronCheckpointManager

            assert titan_impl.CheckpointManager is NeuronCheckpointManager
            raise RuntimeError("sentinel")
    assert titan_impl.CompileConfig is original_compile
    assert titan_impl.TrainingConfig is original_training
    assert qwen3.model_registry is original_registry
    assert titan_impl.Trainer.init_distributed is original_init
    assert titan_impl.CheckpointManager is original_checkpointer


def test_forward_does_not_gather_vocabulary_logits():
    from torch.distributed.tensor import DTensor

    predictions = Mock(spec=DTensor)
    model = Mock(return_value=predictions)
    engine = SimpleNamespace(module=[model])
    inputs, positions = object(), object()
    output = TorchTitanNeuronEngineWithLMHead.model_forward_step(
        engine, inputs=inputs, extra_inputs={"positions": positions}, extra_kwargs={"attention_masks": None}
    )
    assert output is predictions
    predictions.full_tensor.assert_not_called()
    model.assert_called_once_with(inputs, positions=positions, attention_masks=None)


def test_mesh_is_reused_and_loss_group_is_not_fake_batch_group():
    dimensions = SimpleNamespace(world_mesh=object(), get_optional_mesh=lambda name: name, tp=1)
    engine = SimpleNamespace(trainer=SimpleNamespace(parallel_dims=dimensions))
    TorchTitanNeuronEngineWithLMHead._init_device_mesh(engine)
    assert engine.parallel_dims is dimensions
    assert engine.device_mesh is dimensions.world_mesh
    assert TorchTitanNeuronEngineWithLMHead._get_data_parallel_mesh(engine) == "loss"


@pytest.mark.parametrize("world_size,tp_size,rank,expected", [(4, 2, 3, [1, 3]), (4, 4, 2, [2])])
def test_data_parallel_groups_exclude_tp_peers(monkeypatch, world_size, tp_size, rank, expected):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    created = []

    def new_group(ranks, backend):
        assert backend == "cpu:gloo,neuron:neuron"
        created.append(ranks)
        return ranks

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    dimensions = SimpleNamespace(world_mesh=object(), tp=tp_size)
    engine = SimpleNamespace(trainer=SimpleNamespace(parallel_dims=dimensions))
    TorchTitanNeuronEngineWithLMHead._init_device_mesh(engine)
    assert TorchTitanNeuronEngineWithLMHead.get_data_parallel_group(engine) == expected
    assert len(created) == tp_size


def test_checkpoint_retention_across_verl_step_directories(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    checkpointer = SimpleNamespace(save=Mock(return_value=True))
    engine = SimpleNamespace(checkpointer=checkpointer, trainer=SimpleNamespace(step=0))
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    paths = [tmp_path / f"global_step_{step}" / "actor" for step in range(1, 5)]
    for step, target in enumerate(paths[:3], 1):
        TorchTitanNeuronEngineWithLMHead.save_checkpoint(engine, str(target), global_step=step, max_ckpt_to_keep=2)
    assert not paths[0].exists()
    assert paths[1].is_dir() and paths[2].is_dir() and unrelated.is_dir()
    checkpointer.save.return_value = False
    with pytest.raises(RuntimeError, match="did not write"):
        TorchTitanNeuronEngineWithLMHead.save_checkpoint(engine, str(paths[3]), global_step=4, max_ckpt_to_keep=2)
    assert paths[1].is_dir() and paths[2].is_dir()


def test_checkpoint_uses_requested_directory_and_checks_completion(tmp_path, monkeypatch):
    barrier = Mock()
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    checkpointer = SimpleNamespace(save=Mock(return_value=True), load=Mock(return_value=True))
    engine = SimpleNamespace(checkpointer=checkpointer, trainer=SimpleNamespace(step=0))
    target = tmp_path / "global_step_1" / "actor"
    TorchTitanNeuronEngineWithLMHead.save_checkpoint(engine, str(target), global_step=1)
    assert checkpointer.folder == str(target)
    assert checkpointer.interval == 1
    assert engine.trainer.step == 1
    checkpointer.save.assert_called_once_with(curr_step=1)
    TorchTitanNeuronEngineWithLMHead.load_checkpoint(engine, str(target))
    checkpointer.load.assert_called_once_with(step=-1)
    assert barrier.call_count == 2
    checkpointer.load.return_value = False
    with pytest.raises(RuntimeError, match="No complete"):
        TorchTitanNeuronEngineWithLMHead.load_checkpoint(engine, str(target))
    checkpointer.save.return_value = False
    with pytest.raises(RuntimeError, match="did not write"):
        TorchTitanNeuronEngineWithLMHead.save_checkpoint(engine, str(target), global_step=2)


def test_nixl_bridge_refuses_cpu_fallback(monkeypatch):
    from verl_hardware_plugin.accelerators.neuron.engines import torchtitan_neuron as engine_module
    from verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine import NeuronNixlCheckpointEngine

    monkeypatch.setattr(
        engine_module, "actor_engine", lambda: SimpleNamespace(trainer=SimpleNamespace(device=torch.device("cpu")))
    )
    with pytest.raises(ValueError, match="real Trainium"):
        NeuronNixlCheckpointEngine(bucket_size=1024)


def test_weight_schedule_uses_bfloat16():
    engine = SimpleNamespace(hf_parameters=lambda: {"model.norm.weight": torch.ones(4)})
    schedule = TorchTitanNeuronEngineWithLMHead.weight_sync_metadata(engine)
    assert schedule == [("model.norm.weight", (4,), "bfloat16")]


@pytest.mark.parametrize("cpu_threshold", [0, 8, 9])
def test_weight_export_gathers_bfloat16_without_mutating_master_weights(cpu_threshold):
    from torch.distributed.tensor import DTensor

    master = torch.tensor([[1.12345, -2.34567], [3.45678, 4.56789]], requires_grad=True)
    original = master.detach().clone()
    sharded = Mock(spec=DTensor)
    sharded.detach.return_value = sharded
    converted = Mock(spec=DTensor)
    sharded.to.return_value = converted
    converted.numel.return_value = master.numel()
    converted.element_size.return_value = 2
    converted.device = master.device
    host_sharded = Mock(spec=DTensor)
    converted.to.return_value = host_sharded
    host_sharded.full_tensor.return_value = master.detach().bfloat16()
    converted.full_tensor.return_value = master.detach().bfloat16()
    # A large FP32 gather is the failure this export must avoid.
    sharded.full_tensor.side_effect = AssertionError("FP32 weight gather exceeds runtime limits")
    engine = SimpleNamespace(
        hf_parameters=lambda: {"lm_head.weight": sharded, "model.norm.weight": master},
        engine_config=SimpleNamespace(weight_sync_cpu_gather_threshold_bytes=cpu_threshold),
    )

    weights, metadata = TorchTitanNeuronEngineWithLMHead.get_per_tensor_param(engine)
    exported = dict(weights)

    assert metadata is None
    sharded.to.assert_called_once_with(dtype=torch.bfloat16)
    if cpu_threshold == 8:
        converted.to.assert_called_once_with("cpu")
        host_sharded.full_tensor.assert_called_once_with()
        converted.full_tensor.assert_not_called()
    else:
        converted.full_tensor.assert_called_once_with()
        converted.to.assert_not_called()
    for value in exported.values():
        torch.testing.assert_close(value, original.bfloat16(), rtol=0, atol=0)
        assert value.is_contiguous() and not value.requires_grad
    torch.testing.assert_close(master, original, rtol=0, atol=0)
    assert master.dtype == torch.float32 and master.requires_grad
