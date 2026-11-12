# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Policy swaps must update master shards after forward-only FSDP passes."""

import pytest
import torch
import torch.distributed as dist
from packaging.version import Version
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from verl_hardware_plugin.accelerators.neuron.engines.snapshots import restore_model_parts, save_model_parts


def test_snapshot_is_independent_and_preserves_optimizer_parameters():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameters = list(model.parameters())
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    snapshot = save_model_parts([model])
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    restore_model_parts([model], snapshot)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert all(actual is original for actual, original in zip(model.parameters(), parameters, strict=True))


def test_snapshot_rejects_changed_model():
    model = torch.nn.Linear(4, 2)
    snapshot = save_model_parts([model])
    model.double()
    with pytest.raises(ValueError, match="shape or dtype changed"):
        restore_model_parts([model], snapshot)
    model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="parameter names changed"):
        restore_model_parts([model], snapshot)


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


def _snapshot_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = _TiedModel()
        mesh = init_device_mesh("cpu", (2,))
        fully_shard(
            [model.embedding, model.head],
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            reshard_after_forward=False,
        )
        fully_shard(model, mesh=mesh)
        tokens = torch.tensor([[1, 2, 3]])
        old_policy = save_model_parts([model])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(tokens).float().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        expected = model.embedding.weight.to_local().detach().clone()

        # Match verl's eval context: it reshards only the root, leaving the
        # embedding/head FSDP group gathered in BF16.
        with torch.no_grad():
            expected_logits = model(tokens).clone()
        model.reshard()
        current_policy = save_model_parts([model])
        assert current_policy[0]["embedding.weight"][0].dtype == torch.float32
        torch.testing.assert_close(current_policy[0]["embedding.weight"][0], expected, rtol=0, atol=0)

        restore_model_parts([model], old_policy)
        with torch.no_grad():
            model(tokens)
        model.reshard()
        restore_model_parts([model], current_policy)
        torch.testing.assert_close(model.embedding.weight.to_local(), expected, rtol=0, atol=0)
        assert model.embedding.weight is model.head.weight
        with torch.no_grad():
            torch.testing.assert_close(model(tokens), expected_logits, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(Version(torch.__version__).release < (2, 14), reason="CPU FSDP requires the Neuron PyTorch version")
def test_snapshot_swap_after_forward_only_fsdp(tmp_path):
    torch.multiprocessing.spawn(_snapshot_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)
