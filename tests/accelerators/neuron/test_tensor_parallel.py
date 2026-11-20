# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Distributed vocabulary loss/entropy and gradient parity against an unsharded oracle."""

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from verl_hardware_plugin.accelerators.neuron.engines.tensor_parallel import token_statistics


def _check_token_statistics(mesh, device):
    from torchtitan.components.loss import _LossParallelCrossEntropy

    rank, world_size = mesh.get_local_rank(), mesh.size()
    torch.manual_seed(42)
    for temperature in (0.8, torch.tensor([0.7, 1.3])[:, None, None]):
        for statistic in ("log_probs", "both", "entropy"):
            full = (torch.randn(2, 5, 32) * 20).requires_grad_()
            labels = torch.tensor([[0, 7, 8, 15, 16], [23, 24, 31, 0, 31]])
            local = full.detach().chunk(world_size, dim=-1)[rank].to(device).clone().requires_grad_()
            sharded = DTensor.from_local(local, mesh, (Shard(2),))
            device_temperature = temperature.to(device) if isinstance(temperature, torch.Tensor) else temperature
            calculate_entropy = statistic != "log_probs"
            with (
                patch.object(_LossParallelCrossEntropy, "apply", wraps=_LossParallelCrossEntropy.apply) as titan_loss,
                patch.object(
                    DTensor, "full_tensor", side_effect=AssertionError("Token statistics must not gather logits")
                ),
            ):
                log_probs, entropy = token_statistics(sharded, labels.to(device), device_temperature, calculate_entropy)
                assert titan_loss.call_count == 1
                # Match the engine's differentiable host loss boundary.
                log_probs = log_probs.cpu()
                entropy = entropy.cpu() if entropy is not None else None
                expected_log_probs, expected_entropy = token_statistics(full, labels, temperature, calculate_entropy)
                torch.testing.assert_close(log_probs, expected_log_probs, atol=1e-5, rtol=1e-5)
                coefficient = torch.randn_like(log_probs)
                loss = (coefficient * log_probs).sum()
                expected_loss = (coefficient * expected_log_probs).sum()
                if calculate_entropy:
                    torch.testing.assert_close(entropy, expected_entropy, atol=1e-5, rtol=1e-5)
                    entropy_loss = entropy.square().sum() * 0.03
                    expected_entropy_loss = expected_entropy.square().sum() * 0.03
                    loss = entropy_loss if statistic == "entropy" else loss + entropy_loss
                    expected_loss = (
                        expected_entropy_loss if statistic == "entropy" else expected_loss + expected_entropy_loss
                    )
                loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(
                    local.grad.cpu(), full.grad.chunk(world_size, dim=-1)[rank], atol=1e-5, rtol=1e-5
                )


def _check_statistics(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        from verl_hardware_plugin.accelerators.neuron.engines.snapshots import restore_model_parts, save_model_parts

        module = torch.nn.Module()
        module.register_parameter("column", torch.nn.Parameter(DTensor.from_local(torch.ones(8, 8), mesh, (Shard(0),))))
        module.register_parameter("row", torch.nn.Parameter(DTensor.from_local(torch.ones(8, 8), mesh, (Shard(1),))))
        snapshot = save_model_parts([module])
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.to_local().zero_()
        restore_model_parts([module], snapshot)
        for parameter in module.parameters():
            torch.testing.assert_close(parameter.to_local(), torch.ones(8, 8))
        _check_token_statistics(mesh, "cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_vocabulary_parallel_statistics_and_gradients(world_size, tmp_path):
    pytest.importorskip("torchtitan.components.loss")
    mp.spawn(_check_statistics, args=(world_size, f"file://{tmp_path / 'rendezvous'}"), nprocs=world_size, join=True)


@pytest.mark.skipif(os.environ.get("RUN_NEURON_HARDWARE_TESTS") != "1", reason="requires allocated Trainium cores")
def test_native_vocabulary_parallel_statistics_and_gradients():
    from verl_hardware_plugin.accelerators.neuron.runtime import ensure_neuron

    ensure_neuron()
    device = torch.device("neuron", int(os.environ["LOCAL_RANK"]))
    torch.neuron.set_device(device)
    dist.init_process_group("cpu:gloo,neuron:neuron")
    try:
        mesh = init_device_mesh("neuron", (dist.get_world_size(),))
        _check_token_statistics(mesh, device)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4, 8])
def test_config_accepts_tensor_parallelism(tensor_parallel_size):
    from verl_hardware_plugin.accelerators.neuron.engines.config import NeuronTorchtitanEngineConfig

    config = NeuronTorchtitanEngineConfig(tensor_parallel_size=tensor_parallel_size)
    assert config.tensor_parallel_size == tensor_parallel_size
