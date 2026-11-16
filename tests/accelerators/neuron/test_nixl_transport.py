# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Control-plane and fail-closed transport tests without loading NIXL or Neuron."""

import asyncio
import importlib
import sys
from collections import deque
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

from verl_hardware_plugin.accelerators.neuron.rollout.worker_extension import NeuronWorkerExtension
from verl_hardware_plugin.accelerators.neuron.weight_sync import nixl_transport
from verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport import NixlEndpoint, NixlTopology, slot_address


def test_transport_refuses_cpu_and_cpu_slot():
    with pytest.raises(ValueError, match="real Trainium"):
        NixlEndpoint(None, "pair", "cpu", 0, sender=True)
    with pytest.raises(ValueError, match="Neuron allocation"):
        slot_address(torch.zeros(4))


def test_native_sync_is_scoped_to_the_assigned_device(monkeypatch):
    native = SimpleNamespace(current_device=lambda: 3, synchronize=Mock())
    monkeypatch.setattr(torch, "neuron", native, raising=False)
    nixl_transport.synchronize()
    native.synchronize.assert_called_once_with(3)


def test_topology_replica_bounds():
    topology = NixlTopology("localhost", 12345, 2, 2, 30)
    assert topology.receiver(1)["replica"] == 1
    with pytest.raises(ValueError):
        topology.receiver(2)


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setattr(nixl_transport, "synchronize", Mock())
    return SimpleNamespace(
        sender=True,
        lite=False,
        transfers={},
        remote="receiver",
        address=1024,
        remote_address=2048,
        registration=object(),
        buffer=torch.ones(4, dtype=torch.bfloat16),
        sequence=3,
        timeout_s=1,
        store=Mock(),
        agent=Mock(),
    )


@pytest.mark.parametrize("state,error", [("ERR", RuntimeError), ("PROC", TimeoutError)])
def test_failed_write_never_signals_receiver(endpoint, state, error):
    endpoint.timeout_s = 0
    endpoint.agent.transfer.return_value = state
    with pytest.raises(error):
        NixlEndpoint.push(endpoint, 3)
    endpoint.store.set.assert_not_called()
    assert endpoint.sequence == 3


def test_sender_requires_receiver_apply_acknowledgement(endpoint):
    endpoint.agent.transfer.return_value = "DONE"
    endpoint.store.get.return_value = b"apply failed"
    with pytest.raises(RuntimeError, match="receiver rejected"):
        NixlEndpoint.push(endpoint, 3)
    assert endpoint.sequence == 3
    endpoint.store.get.return_value = b"ok"
    NixlEndpoint.push(endpoint, 3)
    assert endpoint.sequence == 4
    endpoint.store.set.assert_called_with("written/3", b"ok")


def test_writes_only_payload_bytes_and_reuses_handles(endpoint):
    endpoint.agent.get_xfer_descs.side_effect = lambda descriptors, kind: descriptors
    handles = [object(), object(), object()]
    endpoint.agent.initialize_xfer.side_effect = handles
    endpoint.agent.transfer.return_value = "DONE"
    endpoint.store.get.return_value = b"ok"
    for numel in (4, 1, 3, 1, 4):
        NixlEndpoint.push(endpoint, numel)
    assert endpoint.agent.initialize_xfer.call_args_list == [
        call("WRITE", [(1024, size, 0)], [(2048, size, 0)], "receiver", b"") for size in (8, 2, 6)
    ]
    assert endpoint.agent.transfer.call_args_list == [call(handles[i]) for i in (0, 1, 2, 1, 0)]
    assert endpoint.sequence == 8
    endpoint.agent.register_memory.assert_not_called()
    endpoint.agent.deregister_memory.assert_not_called()


@pytest.mark.parametrize("numel", [-1, 0, 5])
def test_invalid_payload_never_starts_a_transfer(endpoint, numel):
    with pytest.raises(ValueError, match="payload"):
        NixlEndpoint.push(endpoint, numel)
    endpoint.agent.get_xfer_descs.assert_not_called()
    endpoint.agent.transfer.assert_not_called()
    endpoint.store.set.assert_not_called()
    nixl_transport.synchronize.assert_not_called()
    assert endpoint.sequence == 3


def test_slot_registration_covers_full_allocation_without_starting_a_write(endpoint, monkeypatch):
    monkeypatch.setattr(nixl_transport, "slot_address", lambda *args, **kwargs: (1024, 8))
    endpoint.buffer = None
    endpoint.device = "cpu"
    endpoint.store.get.side_effect = [b"[2048, 8]", b"remote metadata"]
    endpoint.agent.add_remote_agent.return_value = b"receiver"
    NixlEndpoint.register(endpoint, 4)
    buffer = endpoint.buffer
    NixlEndpoint.register(endpoint, 4)
    assert endpoint.buffer is buffer
    endpoint.agent.get_reg_descs.assert_called_once_with([(1024, 8, 0, "")], "VRAM")
    endpoint.agent.register_memory.assert_called_once()
    endpoint.agent.initialize_xfer.assert_not_called()
    assert (endpoint.address, endpoint.remote_address, endpoint.remote) == (1024, 2048, "receiver")
    with pytest.raises(ValueError, match="slot size cannot change"):
        NixlEndpoint.register(endpoint, 8)


def test_close_releases_every_cached_handle_before_memory(endpoint):
    handles = [object(), object()]
    endpoint.transfers = dict(zip((2, 6), handles, strict=True))
    registration = endpoint.registration
    NixlEndpoint.close(endpoint)
    NixlEndpoint.close(endpoint)
    assert endpoint.agent.mock_calls == [
        call.release_xfer_handle(handles[0]),
        call.release_xfer_handle(handles[1]),
        call.remove_remote_agent("receiver"),
        call.deregister_memory(registration),
    ]
    assert endpoint.transfers == {}
    assert endpoint.buffer is endpoint.address is endpoint.remote_address is None


def test_receiver_acknowledges_only_after_apply_and_device_sync(endpoint, monkeypatch):
    events = []
    monkeypatch.setattr(nixl_transport, "synchronize", lambda **kwargs: events.append("sync"))
    endpoint.sender = False
    endpoint.store.set.side_effect = lambda *args: events.append("ack")
    assert NixlEndpoint.receive(endpoint, lambda slot: events.append("apply") or 4) == 4
    assert events == ["apply", "sync", "ack"]
    assert endpoint.sequence == 4


def test_receiver_failure_is_reported_without_advancing_sequence(endpoint):
    endpoint.sender = False
    with pytest.raises(ValueError, match="sentinel"):
        NixlEndpoint.receive(endpoint, Mock(side_effect=ValueError("sentinel")))
    endpoint.store.set.assert_called_once_with("applied/3", b"error: sentinel")
    nixl_transport.synchronize.assert_not_called()
    assert endpoint.sequence == 3


@pytest.fixture(scope="module")
def checkpoint_engine():
    """Like TPU, stub unused HTTP dependencies while importing the real bridge."""
    stubs = {}
    for name in ("uvicorn", "fastapi"):
        try:
            importlib.import_module(name)
        except ImportError:
            stubs[name] = ModuleType(name)
            vars(stubs[name]).update(Server=object, Config=object, FastAPI=object)
    with patch.dict(sys.modules, stubs):
        yield importlib.import_module("verl_hardware_plugin.accelerators.neuron.weight_sync.checkpoint_engine")


def test_nixl_topology_is_local_and_assigns_every_replica(checkpoint_engine):
    actor = {
        "schedule": [("model.norm.weight", (4,), "bfloat16")],
        "bucket_size": 1024,
        "host": "localhost",
        "port": 12345,
        "timeout_s": 30,
        "checksums": {},
    }
    metadata = [actor, actor.copy(), {"rollout_tp": 2}, {"rollout_tp": 2}]
    trainer, rollout = checkpoint_engine.NeuronNixlCheckpointEngine.build_topology(2, 2, metadata)
    assert len(trainer["spec"]) == 2
    assert rollout["replica_index"] == [0, 1]
    assert trainer["spec"][0].receiver(1) == {
        "store_host": "localhost",
        "store_port": 12345,
        "timeout_s": 30,
        "rollout_tp": 2,
        "replicas": 2,
        "replica": 1,
    }
    metadata[-1]["rollout_tp"] = 4
    with pytest.raises(ValueError, match="same TP"):
        checkpoint_engine.NeuronNixlCheckpointEngine.build_topology(2, 2, metadata)


def test_export_ranks_cannot_advance_past_sender(checkpoint_engine, monkeypatch):
    events = []
    group = object()
    monkeypatch.setattr(checkpoint_engine, "plan_buckets", lambda *args, **kwargs: [object(), object()])

    def buffers(*args):
        for index in range(2):
            events.append(f"export-{index}")
            yield torch.ones(4)

    monkeypatch.setattr(checkpoint_engine, "iter_weight_buckets", buffers)
    monkeypatch.setattr(checkpoint_engine, "synchronize", lambda: events.append("sync"))

    def barrier(*, group):
        assert group is instance.export_group
        events.append("barrier")

    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    instance = SimpleNamespace(
        connected=True, schedule=[], bucket_size=8, endpoints={}, export_group=group, device="cpu", rank=1
    )
    asyncio.run(checkpoint_engine.NeuronNixlCheckpointEngine.send_weights(instance, []))
    assert events == ["barrier", "export-0", "sync", "barrier", "barrier", "export-1", "sync", "barrier", "barrier"]


@pytest.mark.parametrize("tp_rank", range(4))
def test_sync_transfers_only_resident_weights_and_skips_empty_buckets(checkpoint_engine, monkeypatch, tp_rank):
    monkeypatch.setattr(checkpoint_engine, "synchronize", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    weights = {
        "linear.weight": torch.arange(48, dtype=torch.bfloat16).reshape(12, 4),
        "other.weight": torch.arange(48, dtype=torch.bfloat16).reshape(4, 12),
        "norm.weight": torch.arange(4, dtype=torch.bfloat16),
    }
    schedule = [(name, tuple(tensor.shape), "bfloat16") for name, tensor in weights.items()]
    tensors = {
        "linear_weight": torch.full((4, 3), torch.nan, dtype=torch.bfloat16),
        "other_weight": torch.full((3, 4), torch.nan, dtype=torch.bfloat16),
        "norm.weight": torch.full((4,), torch.nan, dtype=torch.bfloat16),
    }
    inventory = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    payloads = deque()
    sizes = []
    sender = SimpleNamespace(buffer=None)
    receiver = SimpleNamespace(buffer=None)

    def register(endpoint, numel):
        if endpoint.buffer is None:
            endpoint.buffer = torch.full((numel,), torch.nan, dtype=torch.bfloat16)
        assert endpoint.buffer.numel() == numel

    def push(numel):
        assert numel > 0
        sizes.append(numel)
        payloads.append(sender.buffer[:numel].clone())

    def receive(apply):
        payload = payloads.popleft()
        receiver.buffer[: payload.numel()].copy_(payload)
        return apply(receiver.buffer)

    sender.register = lambda numel: register(sender, numel)
    sender.push = push
    receiver.register = lambda numel: register(receiver, numel)
    receiver.receive = receive
    engine = SimpleNamespace(
        connected=True,
        schedule=schedule,
        bucket_size=32,
        endpoints={(0, tp_rank): sender},
        inventories={(0, tp_rank): inventory},
        spec=SimpleNamespace(rollout_tp=4),
        export_group=object(),
        device="cpu",
        rank=1,
    )
    worker = SimpleNamespace(
        _weight_sync_state=SimpleNamespace(
            endpoint=receiver, inventory=inventory, tensors=tensors, rank=tp_rank, size=4
        )
    )
    bucket_count = len(checkpoint_engine.plan_buckets(schedule, bucket_bytes=32))
    for step in range(2):
        updated = {name: tensor + 100 * step for name, tensor in weights.items()}
        sizes.clear()
        asyncio.run(checkpoint_engine.NeuronNixlCheckpointEngine.send_weights(engine, updated.items()))
        NeuronWorkerExtension.weight_sync_recv(worker, schedule, 32)
        assert not payloads
        assert 0 < len(sizes) < bucket_count
        assert len(set(sizes)) > 1
        assert sum(sizes) == sum(tensor.numel() for tensor in tensors.values()) == 28
        torch.testing.assert_close(tensors["linear_weight"], updated["linear.weight"].chunk(4, dim=0)[tp_rank].T)
        torch.testing.assert_close(tensors["other_weight"], updated["other.weight"].chunk(4, dim=1)[tp_rank].T)
        torch.testing.assert_close(tensors["norm.weight"], updated["norm.weight"])


@pytest.mark.parametrize("world_size,replicas,rollout_tp", [(32, 4, 8), (2, 1, 2), (3, 2, 4), (8, 1, 2)])
def test_nixl_fanout_assigns_every_receiver_once(checkpoint_engine, monkeypatch, world_size, replicas, rollout_tp):
    stores = []

    def control_store(spec, *, master):
        stores.append(master)
        return object()

    monkeypatch.setattr(checkpoint_engine, "control_store", control_store)
    monkeypatch.setattr(
        checkpoint_engine, "NixlEndpoint", lambda *args, **kwargs: SimpleNamespace(store=Mock(get=lambda key: b"{}"))
    )
    spec = SimpleNamespace(replicas=replicas, rollout_tp=rollout_tp, timeout_s=30)
    ownership = []
    loads = []
    for rank in range(world_size):
        instance = SimpleNamespace(
            rank=rank, world_size=world_size, connected=False, endpoints={}, inventories={}, device="neuron", core=rank
        )
        checkpoint_engine.NeuronNixlCheckpointEngine.init_process_group(instance, spec)
        assert instance.connected
        ownership.extend(instance.endpoints)
        loads.append(len(instance.endpoints))
    assert sorted(ownership) == [(replica, rank) for replica in range(replicas) for rank in range(rollout_tp)]
    assert max(loads) - min(loads) <= 1
    assert stores == [True, *([False] * (world_size - 1))]
