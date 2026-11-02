# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the tpu-sync checkpoint engine and its vLLM worker extension.

tpu-sync, torch_tpu and vLLM are not installed on CI. A fake ``tpu_sync`` stands in for the transfer
library: its controller "transfers" by copying the trainer's bound tensors into the rollout workers'
receive buffers, resharded as the variable metadata registered by both sides describes, so the
end-to-end tests check that this metadata is consistent. Ray actor handles are in-process stand-ins
around the real ``TpuSyncRendezvousState``.
"""

import asyncio
import importlib
import itertools
import json
import logging
import sys
import threading
from collections import namedtuple
from types import ModuleType, SimpleNamespace

import pytest
import ray
import torch
import torch.nn as nn

for _mod_name in ("uvicorn", "fastapi"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub = ModuleType(_mod_name)
            _stub.FastAPI = object  # type: ignore[attr-defined]
            _stub.Server = object  # type: ignore[attr-defined]
            _stub.Config = object  # type: ignore[attr-defined]
            sys.modules[_mod_name] = _stub

# Imported up front, so that no test imports (and registers) them a second time. Importing
# tpu_sync_checkpoint_engine also installs the CheckpointEngineManager hook that routes backend=tpu-sync.
import verl.checkpoint_engine.base as ckpt_base  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.weight_sync import (  # noqa: E402
    tpu_sync_checkpoint_engine as tpu_sync_engine,
)
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_parity as parity  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_rendezvous  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_rendezvous import TpuSyncRendezvousState  # noqa: E402

TPU_SYNC_WORKER_EXTENSION = "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension"
TpuSyncId = namedtuple("TpuSyncId", ["job_name", "job_replica_id", "data_name"])
TIMING_KEYS = {
    f"timing_s/tpu-sync/{phase}"
    for phase in (
        "quiesce",
        "trainer_init",
        "sampler_init",
        "barrier",
        "p2p_transfer",
        "sampler_h2d",
        "sampler_resume",
        "trainer_release",
        "total_sync",
        "sampler_h2d_pure",
    )
}

# A one-layer Qwen3-like model in HF naming: hidden size, vocab, MLP intermediate size, q rows, k/v rows, and
# KV heads (so head_dim = KV_ROWS // KV_HEADS; TP > KV_HEADS replicates KV heads like GQA in vLLM).
HIDDEN, VOCAB, INTERMEDIATE, Q_ROWS, KV_ROWS, KV_HEADS = 8, 16, 16, 8, 4, 2
LAYER = "model.layers.0."
TRAINER_SHAPES = {
    "lm_head.weight": (VOCAB, HIDDEN),
    "model.embed_tokens.weight": (VOCAB, HIDDEN),
    LAYER + "input_layernorm.weight": (HIDDEN,),
    LAYER + "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
    LAYER + "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
    LAYER + "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
    LAYER + "post_attention_layernorm.weight": (HIDDEN,),
    LAYER + "self_attn.k_proj.weight": (KV_ROWS, HIDDEN),
    LAYER + "self_attn.o_proj.weight": (HIDDEN, Q_ROWS),
    LAYER + "self_attn.q_proj.weight": (Q_ROWS, HIDDEN),
    LAYER + "self_attn.v_proj.weight": (KV_ROWS, HIDDEN),
    "model.norm.weight": (HIDDEN,),
}
QKV = [LAYER + f"self_attn.{p}_proj.weight" for p in "qkv"]


def _trainer_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    """Random bf16 weights (every copy is exact, so the tests compare with ``torch.equal``)."""
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=generator).to(torch.bfloat16) for name, shape in TRAINER_SHAPES.items()}


# ---------------------------------------------------------------------------
# Fakes: Ray handles, tpu_sync, the trainer worker group, the rollout server and vLLM model
# ---------------------------------------------------------------------------


class _Ref:
    """Stands in for a Ray ObjectRef: awaitable, and resolved by the patched ``ray.get``.

    ``fn`` defers the work to resolution time, like a remote task that ``ray.get`` waits for.
    """

    def __init__(self, value=None, fn=None):
        self._value, self._fn = value, fn

    def result(self):
        if self._fn is not None:
            self._value, self._fn = self._fn(), None
        return self._value

    def __await__(self):
        yield from ()
        return self.result()


def _fake_ray_get(refs, timeout=None):
    if isinstance(refs, list):
        return [_fake_ray_get(ref) for ref in refs]
    return refs.result() if isinstance(refs, _Ref) else refs


class _FakeActorHandle:
    """In-process stand-in for a Ray actor handle: ``handle.method.remote(...)`` runs the method right away."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        method = getattr(self._obj, name)
        return SimpleNamespace(remote=lambda *args, **kwargs: _Ref(method(*args, **kwargs)))


class _FakeTpuSync:
    """Fake ``tpu_sync`` and ``torch_tpu``: records synchronizers, registrations and transfers."""

    def __init__(self):
        self.synchronizers = []
        self.registrations = {}
        self.controller = None
        self.servers_started = 0
        self.transfers = []
        self.tpu_syncs = 0
        self.fail_tpu_sync = False
        self.ports = itertools.count(20000)
        self.registry_state = TpuSyncRendezvousState()
        self.registry = _FakeActorHandle(self.registry_state)
        self.rendezvous_creations = 0
        self.rendezvous_lookups = []

    def create_rendezvous(self):
        """Stands in for ``create_tpu_sync_rendezvous``: this job's rendezvous actor and its name."""
        self.rendezvous_creations += 1
        return self.registry, "TpuSyncRendezvous-test"

    def get_rendezvous(self, name=None):
        """Stands in for ``get_tpu_sync_rendezvous``; records the name the caller looked up."""
        self.rendezvous_lookups.append(name)
        return self.registry

    def synchronizer_at(self, address):
        port = int(address.rsplit(":", 1)[1])
        return next(ws for ws in reversed(self.synchronizers) if ws.local_port == port)

    def transfer(self, src_units, dst_units):
        """Copies the trainer units' tensors into the rollout units' buffers, resharded per their metadata."""
        full = {}
        blocks: dict[str, tuple[int, dict[int, torch.Tensor]]] = {}
        for unit in src_units:
            registration = self.registrations[unit]
            ws = self.synchronizer_at(registration.data_addresses[0])
            for var, tensor in zip(registration.variables, ws.tensors, strict=True):
                rest = len(var.shape) - 1
                indices = getattr(var, "global_shard_indices", None)
                if indices:
                    # A local FSDP shard: block `index` of `mesh_shape[0]` equal dim-0 blocks of the full tensor.
                    num_shards = var.mesh_shape[0]
                    assert var.mesh_shape[1:] == [1] * rest and var.sharding_spec == ["fsdp"] + [""] * rest, var.name
                    assert registration.mesh_shape[0] >= num_shards, var.name
                    assert list(tensor.shape) == [var.shape[0] // num_shards] + var.shape[1:], var.name
                    (index,) = indices
                    shards = blocks.setdefault(var.name, (num_shards, {}))[1]
                    assert index not in shards, f"two units registered block {index} of {var.name}"
                    shards[index] = tensor
                else:
                    # A full, unsharded tensor (the same on every trainer rank).
                    assert var.mesh_shape == [1] * len(var.shape), var.name
                    assert var.sharding_spec == [""] * len(var.shape), var.name
                    assert list(tensor.shape) == list(var.shape), var.name
                    full[var.name] = tensor
        for name, (num_shards, shards) in blocks.items():
            assert sorted(shards) == list(range(num_shards)), f"missing blocks of {name}"
            full[name] = torch.cat([shards[i] for i in range(num_shards)])
        for unit in dst_units:
            registration = self.registrations[unit]
            ws = self.synchronizer_at(registration.data_addresses[0])
            tp_rank = int(unit.job_replica_id)
            for var, buffer in zip(registration.variables, ws.tensors, strict=True):
                src = full[var.name]
                assert list(src.shape) == list(var.shape), var.name
                for dim, axis in enumerate(var.sharding_spec):
                    if axis == "tp":
                        size = src.shape[dim] // var.mesh_shape[dim]
                        # An explicit global_shard_indices names the slice; else it is the unit's rank in the mesh.
                        indices = getattr(var, "global_shard_indices", None)
                        index = indices[0] if indices else tp_rank
                        src = src.narrow(dim, index * size, size)
                buffer.copy_(src)

    def modules(self) -> dict[str, ModuleType]:
        world = self

        class WeightSynchronizer:
            def __init__(self, device_tensors, local_port=0, parallelism=8, listener_port=0, bind_ip="", **kwargs):
                self.tensors = [tensors[0] for tensors in device_tensors]
                self.local_port = local_port or next(world.ports)
                self.listener_port = listener_port or next(world.ports)
                self.parallelism = parallelism
                self.bind_ip = bind_ip
                self.kwargs = kwargs
                self.d2h_calls = 0
                self.h2d_calls = 0
                self.bind_calls = 0
                self.unbind_calls = 0
                self.bound = True
                self.skip_tiling = None
                world.synchronizers.append(self)

            def test_only_set_skip_tiling(self, plan):
                self.skip_tiling = list(plan)

            def bind_weights(self, device_tensors):
                assert len(device_tensors) == len(self.tensors), "bind_weights must keep the registered tensor count"
                self.tensors = [tensors[0] for tensors in device_tensors]
                self.bind_calls += 1
                self.bound = True

            def unbind_weights(self):
                self.unbind_calls += 1
                self.bound = False

            def d2h(self):
                assert self.bound, "D2H on an unbound WeightSynchronizer"
                self.d2h_calls += 1

            def h2d(self):
                self.h2d_calls += 1

        class FakeController:
            def __init__(self, port=0):
                self._lock = threading.Lock()
                self._registered_shards = {}

            def start_transfer(self, **kwargs):
                world.transfers.append(kwargs)

                async def wait():
                    world.transfer(kwargs["src_units"], kwargs["dst_units"])

                return SimpleNamespace(wait=wait)

        class FakeControllerServer:
            def __init__(self, controller):
                self.controller = controller

            def start(self):
                world.controller = self.controller
                world.servers_started += 1
                return 7777

        class FakeControllerClient:
            def __init__(self, address):
                self.address = address

            def register_work_unit(self, unit_id, data_addresses, listener_address, mesh_shape, variables, mesh_axes):
                registration = SimpleNamespace(
                    address=self.address,
                    data_addresses=data_addresses,
                    listener_address=listener_address,
                    mesh_shape=mesh_shape,
                    variables=list(variables),
                    mesh_axes=mesh_axes,
                )
                world.registrations[unit_id] = registration
                if world.controller is not None:
                    with world.controller._lock:
                        world.controller._registered_shards[unit_id] = registration

        def synchronize(wait=False):
            if world.fail_tpu_sync:
                raise RuntimeError("TPU runtime unavailable")
            world.tpu_syncs += 1

        names = (
            "tpu_sync",
            "tpu_sync.api",
            "tpu_sync.api.common",
            "tpu_sync.api.torch",
            "tpu_sync.api.torch.weight_synchronizer",
            "tpu_sync.rpc",
            "tpu_sync.rpc.raiden_controller",  # the tpu-sync library's own module names
            "tpu_sync.rpc.raiden_service_pb2",
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.sync",
        )
        mods = {name: ModuleType(name) for name in names}
        attrs: dict[str, dict[str, object]] = {
            "tpu_sync.api.common": {"RaidenId": TpuSyncId},
            "tpu_sync.api.torch.weight_synchronizer": {"WeightSynchronizer": WeightSynchronizer},
            "tpu_sync.rpc.raiden_controller": {
                "RaidenController": FakeController,
                "RaidenControllerServer": FakeControllerServer,
                "RaidenControllerClientFacade": FakeControllerClient,
                "RaidenId": TpuSyncId,
                "RaidenMemoryType": SimpleNamespace(DRAM="DRAM"),
            },
            "tpu_sync.rpc.raiden_service_pb2": {"VariableMetadataProto": lambda **fields: SimpleNamespace(**fields)},
            "torch_tpu._internal.sync": {"synchronize": synchronize},
        }
        for name, module in mods.items():
            vars(module).update(attrs.get(name, {}))
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(mods[parent], child, module)
        return mods


class _FakeActorWorkerGroup:
    """The trainer worker group: rank ``r`` runs ``engines[r].send_weights(weights_for_rank(r))``."""

    def __init__(self, engines, weights_for_rank, events, backend="tpu-sync"):
        self.engines, self.weights_for_rank, self.events = engines, weights_for_rank, events
        self.world_size = len(engines)
        self.backend = backend

    def update_weights(self, global_steps=None, mode=None):
        assert mode == self.backend
        self.events.append("update_weights")

        def send(engine, rank):
            return asyncio.run(engine.send_weights(self.weights_for_rank(rank), global_steps=global_steps))

        return [_Ref(fn=lambda e=engine, r=rank: send(e, r)) for rank, engine in enumerate(self.engines)]

    def execute_checkpoint_engine(self, methods):
        (method,) = set(methods)
        self.events.append(method)
        return [_Ref(fn=getattr(engine, method)) for engine in self.engines]


class _FakeServer:
    """The rollout server actor: ``collective_rpc`` runs ``method`` on every vLLM worker and returns the results."""

    def __init__(self, workers, events):
        self.workers, self.events = workers, events
        self.global_steps = None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.events.append(method)
        return [getattr(worker, method)(*args, **(kwargs or {})) for worker in self.workers]

    def set_global_steps(self, global_steps):
        self.events.append("set_global_steps")
        self.global_steps = global_steps


def _make_manager(
    engines, weights_for_rank, samplers, events, backend="tpu-sync", replica_samplers=None, **engine_kwargs_
):
    """A ``CheckpointEngineManager`` stand-in: one rollout server over ``samplers``, or, with ``replica_samplers``
    (a list of sampler lists), one rollout server per replica over its own samplers (``replica_rank`` = position).

    ``engine_kwargs_`` go to ``engine_kwargs[backend]``, where verl reads the engine kwargs of the backend. Returns
    the manager and the first server.
    """
    servers = [_FakeServer(group, events) for group in (replica_samplers or [samplers])]
    server = servers[0]

    async def abort_replicas():
        events.append("abort")

    async def resume_generation_replicas():
        events.append("resume")

    manager = SimpleNamespace(
        backend=backend,
        config=SimpleNamespace(backend=backend, engine_kwargs={backend: engine_kwargs_}),
        actor_wg=_FakeActorWorkerGroup(engines, weights_for_rank, events, backend=backend),
        replicas=[
            SimpleNamespace(server_handle=_FakeActorHandle(server), world_size=len(server.workers), replica_rank=rank)
            for rank, server in enumerate(servers)
        ],
        abort_replicas=abort_replicas,
        resume_generation_replicas=resume_generation_replicas,
    )
    return manager, server


def _trainer_engine(rank, **engine_kwargs):
    engine = tpu_sync_engine.TPUSyncCheckpointEngine(is_master=rank == 0, **engine_kwargs)
    engine.rank = rank
    return engine


class _LocalShardWeights(TrainerWeights):
    """Stands in for ``TrainerWeights`` on trainer rank ``rank`` of ``world`` FSDP ranks.

    ``local_shards()`` yields this rank's dim-0 block of every tensor not in ``full`` and the whole tensor of those in
    ``full`` (replicated, or a layout the exporter gathered).
    """

    def __init__(self, weights, rank, world, full=("model.norm.weight",)):
        self.weights, self.rank, self.world, self.full = weights, rank, world, set(full)

    def local_shards(self):
        for name, tensor in self.weights.items():
            if name in self.full:
                yield name, tensor, None
            else:
                yield name, tensor.chunk(self.world, dim=0)[self.rank].clone(), (self.world, self.rank)

    def __iter__(self):
        raise AssertionError("the tpu-sync engine must export through local_shards(), not gather")


class _Linear(nn.Module):
    """Holds one bf16 ``weight``; ``flipped`` marks vllm-torchtpu's transposed ``(in, out)`` layout."""

    def __init__(self, *shape, flipped=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
        self._tpu_weight_flipped = flipped


class _FakeVllmModel(nn.Module):
    """One TP rank of the vLLM model: fused qkv/gate_up, column and vocab parallel on dim 0, row parallel on dim 1.

    With ``tp > KV_HEADS`` every rank holds one whole KV head (replicated on ``tp // KV_HEADS`` ranks).
    """

    def __init__(self, tp, flip_o_proj=False):
        super().__init__()
        attn = nn.Module()
        attn.qkv_proj = _Linear(Q_ROWS // tp + 2 * KV_ROWS // min(tp, KV_HEADS), HIDDEN)
        attn.o_proj = _Linear(Q_ROWS // tp, HIDDEN, flipped=True) if flip_o_proj else _Linear(HIDDEN, Q_ROWS // tp)
        mlp = nn.Module()
        mlp.gate_up_proj = _Linear(2 * INTERMEDIATE // tp, HIDDEN)
        mlp.down_proj = _Linear(HIDDEN, INTERMEDIATE // tp)
        layer = nn.Module()
        layer.self_attn = attn
        layer.mlp = mlp
        layer.input_layernorm = _Linear(HIDDEN)
        layer.post_attention_layernorm = _Linear(HIDDEN)
        self.model = nn.Module()
        self.model.embed_tokens = _Linear(VOCAB // tp, HIDDEN)
        self.model.layers = nn.ModuleList([layer])
        self.model.norm = _Linear(HIDDEN)
        self.lm_head = _Linear(VOCAB // tp, HIDDEN)


def _expected_vllm_state(weights, rank, tp, lm_head_from_embed=False, flip_o_proj=False):
    """The state dict that rank ``rank`` of a TP=``tp`` vLLM model must hold after a sync of ``weights``."""

    def rows(name):
        return weights[name].chunk(tp, dim=0)[rank]

    def kv_rows(name):
        # One whole KV head per rank once KV heads are replicated (tp > KV_HEADS); else a plain dim-0 split.
        if tp <= KV_HEADS:
            return rows(name)
        return weights[name].chunk(KV_HEADS, dim=0)[rank // (tp // KV_HEADS)]

    def cols(name):
        return weights[name].chunk(tp, dim=1)[rank]

    q_name, k_name, v_name = QKV
    o_proj = cols(LAYER + "self_attn.o_proj.weight")
    return {
        "model.embed_tokens.weight": rows("model.embed_tokens.weight"),
        "model.norm.weight": weights["model.norm.weight"],
        LAYER + "input_layernorm.weight": weights[LAYER + "input_layernorm.weight"],
        LAYER + "post_attention_layernorm.weight": weights[LAYER + "post_attention_layernorm.weight"],
        LAYER + "self_attn.qkv_proj.weight": torch.cat([rows(q_name), kv_rows(k_name), kv_rows(v_name)]),
        LAYER + "self_attn.o_proj.weight": o_proj.T if flip_o_proj else o_proj,
        LAYER + "mlp.gate_up_proj.weight": torch.cat(
            [rows(LAYER + "mlp.gate_proj.weight"), rows(LAYER + "mlp.up_proj.weight")]
        ),
        LAYER + "mlp.down_proj.weight": cols(LAYER + "mlp.down_proj.weight"),
        "lm_head.weight": rows("model.embed_tokens.weight" if lm_head_from_embed else "lm_head.weight"),
    }


def _assert_state(model, expected):
    state = model.state_dict()
    assert sorted(state) == sorted(expected)
    for name, tensor in expected.items():
        assert torch.equal(state[name], tensor), name


class _WorkerExtensionBase:
    """Stands in for upstream ``vLLMColocateWorkerExtension``, whose module needs vLLM."""


def _make_sampler(module, rank, tp, model):
    """A vLLM worker with the tpu-sync extension: rank ``rank`` of a TP=``tp`` engine serving ``model``."""
    sampler = module.vLLMTPUSyncWorkerExtension()
    sampler.rank = rank
    sampler.model_runner = SimpleNamespace(model=model)
    sampler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: KV_HEADS),
    )
    return sampler


@pytest.fixture
def fake_tpu_sync(monkeypatch):
    """Installs the fake tpu_sync / torch_tpu, runs tpu-sync on CPU and fakes the RayWeightRegistry actor."""
    world = _FakeTpuSync()
    for name, module in world.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(tpu_sync_engine, "TPU_SYNC_DEVICE", "cpu")
    monkeypatch.setattr(tpu_sync_engine, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr(tpu_sync_engine, "create_tpu_sync_rendezvous", world.create_rendezvous)
    monkeypatch.setattr(parity, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "10.0.0.1")
    monkeypatch.setattr(ray, "get", _fake_ray_get)
    monkeypatch.delenv("RANK", raising=False)
    # The parity results are logged by tpu_sync_parity; let INFO through to caplog like the engine's own logger.
    parity_level = parity.logger.level
    parity.logger.setLevel(logging.INFO)
    yield world
    parity.logger.setLevel(parity_level)


@pytest.fixture
def worker_ext(fake_tpu_sync, monkeypatch):
    """Imports ``tpu_sync_worker_extension`` against a stubbed upstream worker extension."""
    utils = ModuleType("verl.workers.rollout.vllm_rollout.utils")
    utils.vLLMColocateWorkerExtension = _WorkerExtensionBase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.utils", utils)
    # vLLM's mp workers are not Ray processes: the extension takes its bind address from vLLM.
    network_utils = ModuleType("vllm.utils.network_utils")
    network_utils.get_ip = lambda: "10.0.0.1"  # type: ignore[attr-defined]
    for name, mod in {
        "vllm": ModuleType("vllm"),
        "vllm.utils": ModuleType("vllm.utils"),
        "vllm.utils.network_utils": network_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules.pop(TPU_SYNC_WORKER_EXTENSION, None)
    module = importlib.import_module(TPU_SYNC_WORKER_EXTENSION)
    yield module
    sys.modules.pop(TPU_SYNC_WORKER_EXTENSION, None)


# ---------------------------------------------------------------------------
# Registration, engine options and RayWeightRegistry
# ---------------------------------------------------------------------------


def test_tpu_sync_engine_registered():
    assert ckpt_base.CheckpointEngineRegistry.get("tpu-sync") is tpu_sync_engine.TPUSyncCheckpointEngine


@pytest.mark.parametrize(
    ("backend", "engine_kwargs", "expected"),
    [
        ("tpu-sync", {"tpu-sync": {"parallelism": 4}}, {"parallelism": 4}),
        # Like verl's trainer-side engine, only the entry of the configured backend counts.
        ("tpu-sync", {"ray-plasma-store": {"parallelism": 2}}, {}),
        ("tpu-sync", {}, {}),
    ],
)
def test_tpu_sync_engine_kwargs_follow_configured_backend(backend, engine_kwargs, expected):
    config = SimpleNamespace(backend=backend, engine_kwargs=engine_kwargs)
    assert tpu_sync_engine._tpu_sync_engine_kwargs(config, backend) == expected
    assert tpu_sync_engine._tpu_sync_engine_kwargs(config) == expected


def test_engine_kwargs(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    engine = tpu_sync_engine.TPUSyncCheckpointEngine(bucket_size=16, is_master=True)
    assert (engine.bucket_size, engine.is_master, engine.rank) == (16, True, 3)
    assert (engine.parallelism, engine.verify_parity, engine.tie_word_embeddings) == (8, False, None)

    # Command-line overrides may arrive as strings.
    engine = tpu_sync_engine.TPUSyncCheckpointEngine(parallelism="4", verify_parity="true", tie_word_embeddings="1")
    assert (engine.parallelism, engine.verify_parity, engine.tie_word_embeddings) == (4, True, True)
    # Trainer ranks only post norm stats; the exact check runs on the rollout workers.
    assert tpu_sync_engine.TPUSyncCheckpointEngine(verify_parity="exact").verify_parity is False
    assert tpu_sync_engine.TPUSyncCheckpointEngine(verify_parity="all").verify_parity is True

    with pytest.raises(NotImplementedError):
        engine.receive_weights()


def test_rendezvous_state():
    state = TpuSyncRendezvousState()
    assert (state.get_controller_address(), state.get_global_shapes(), state.get_stats(1)) == (None, {}, None)
    state.set_controller_address("10.0.0.1:7777")
    state.set_global_shapes({"w": [2, 3]})
    state.set_stats(1, {"l1_norm": 1.0})
    state.set_rank_stats(2, 0, {"l1_norm": 1.0})
    state.set_rank_stats(2, 1, {"l1_norm": 2.0})
    assert state.get_controller_address() == "10.0.0.1:7777" and state.get_global_shapes() == {"w": [2, 3]}
    assert state.get_stats(1) == {"master": {"l1_norm": 1.0}}
    assert state.get_stats(2) == {"ranks": {0: {"l1_norm": 1.0}, 1: {"l1_norm": 2.0}}}
    for step in range(3, 3 + state.MAX_STATS_STEPS):
        state.set_rank_stats(step, 0, {})
    assert state.get_stats(1) is None and state.get_stats(2) is None  # only the latest steps are kept


def test_rendezvous_is_named_after_the_job_and_not_detached(monkeypatch):
    """Each job gets its own actor, owned by the driver: jobs sharing a cluster never touch each other's."""
    actors: dict = {}
    options_calls = []

    def get_actor(name, namespace=None):
        assert namespace == "verl"
        if name not in actors:
            raise ValueError(f"Failed to look up actor {name}")
        return actors[name]

    def options(**kwargs):
        options_calls.append(kwargs)

        def remote():
            actors[kwargs["name"]] = handle = object()
            return handle

        return SimpleNamespace(remote=remote)

    job = {"id": "01000000"}
    monkeypatch.setattr(ray, "get_actor", get_actor)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(get_job_id=lambda: job["id"]))
    monkeypatch.setattr(tpu_sync_rendezvous, "TpuSyncRendezvous", SimpleNamespace(options=options))

    handle, name = tpu_sync_rendezvous.create_tpu_sync_rendezvous()
    assert name == "TpuSyncRendezvous-01000000"
    assert options_calls == [{"name": name, "namespace": "verl", "scheduling_strategy": "DEFAULT"}]  # no lifetime
    assert tpu_sync_rendezvous.create_tpu_sync_rendezvous() == (handle, name)  # reused within the job
    assert tpu_sync_rendezvous.get_tpu_sync_rendezvous() is handle
    assert tpu_sync_rendezvous.get_tpu_sync_rendezvous(name) is handle  # e.g. from vLLM's workers

    job["id"] = "02000000"  # another job on the same cluster gets its own actor
    other, other_name = tpu_sync_rendezvous.create_tpu_sync_rendezvous()
    assert other_name == "TpuSyncRendezvous-02000000" and other is not handle
    assert len(options_calls) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_filter_tied_embeddings():
    items = [("model.embed_tokens.weight", 0), ("lm_head.weight", 1), ("model.norm.weight", 2)]
    assert tpu_sync_engine.filter_tied_embeddings(items, tie_word_embeddings=True) == [items[0], items[2]]
    assert tpu_sync_engine.filter_tied_embeddings(items, tie_word_embeddings=False) == items
    torchtitan_names = [("model.lm_head.weight", 1), ("model.tok_embeddings.weight", 0)]
    assert tpu_sync_engine.filter_tied_embeddings(torchtitan_names) == [("model.tok_embeddings.weight", 0)]
    # Without the input embedding there is nothing to copy lm_head from: keep it.
    assert tpu_sync_engine.filter_tied_embeddings([("lm_head.weight", 1)], tie_word_embeddings=True) == [
        ("lm_head.weight", 1)
    ]


def test_validate_and_sanitize_tensors():
    weight = torch.arange(6.0).reshape(2, 3)
    named = [
        ("none", None),
        ("not_a_tensor", 3.0),
        ("empty", torch.empty(0)),
        ("meta", torch.empty(2, 2, device="meta")),
        ("param", nn.Parameter(weight)),
        ("transposed", weight.t()),
    ]
    sanitized = tpu_sync_engine.validate_and_sanitize_tensors(named, device=torch.device("cpu"))
    assert [name for name, _ in sanitized] == ["param", "transposed"]
    assert all(type(t) is torch.Tensor and t.is_contiguous() and t.device.type == "cpu" for _, t in sanitized)
    assert torch.equal(sanitized[0][1], weight)
    assert torch.equal(sanitized[1][1], weight.t())


def test_compute_tensor_stats():
    weights = _trainer_weights()
    names = ("model.embed_tokens.weight", "model.norm.weight")
    stats = parity.compute_tensor_stats([(name, weights[name]) for name in names])
    assert (stats["total_numel"], stats["num_tensors"]) == (VOCAB * HIDDEN + HIDDEN, 2)
    l1 = sum(float(weights[name].float().abs().sum()) for name in names)
    l2_sq = sum(float(weights[name].float().pow(2).sum()) for name in names)
    assert stats["l1_norm"] == pytest.approx(l1, rel=1e-6)
    assert stats["l2_norm"] == pytest.approx(l2_sq**0.5, rel=1e-6)
    embed = stats["per_tensor"]["model.embed_tokens.weight"]
    assert (embed["numel"], embed["shape"], embed["dtype"]) == (VOCAB * HIDDEN, [VOCAB, HIDDEN], "torch.bfloat16")
    assert embed["l2"] == pytest.approx(embed["l2_sq"] ** 0.5)

    unnamed = parity.compute_tensor_stats([torch.ones(2, 2)])
    assert (unnamed["total_numel"], unnamed["l1_norm"], unnamed["per_tensor"]) == (4, 4.0, {})


def test_tpu_synchronize_strict_and_lenient(fake_tpu_sync, caplog):
    tpu_sync_engine.tpu_synchronize(strict=True)
    assert fake_tpu_sync.tpu_syncs == 1

    fake_tpu_sync.fail_tpu_sync = True
    with pytest.raises(RuntimeError, match="TPU synchronization failed"):
        tpu_sync_engine.tpu_synchronize(strict=True)
    with caplog.at_level(logging.WARNING, logger=tpu_sync_engine.__name__):
        tpu_sync_engine.tpu_synchronize()
    assert "Could not synchronize via torch_tpu" in caplog.text


def test_setup_tpu_sync_controller_publishes_address(fake_tpu_sync):
    state = fake_tpu_sync.registry_state
    state.set_controller_address("10.9.9.9:1")  # left behind in the detached actor by a previous job

    controller, server, address, rendezvous, rendezvous_name = tpu_sync_engine.setup_tpu_sync_controller()
    assert address == "10.0.0.1:7777"
    assert server.controller is controller is fake_tpu_sync.controller
    assert state.get_controller_address() == address
    assert rendezvous_name == "TpuSyncRendezvous-test" and fake_tpu_sync.rendezvous_creations == 1
    assert rendezvous is fake_tpu_sync.registry


def test_setup_tpu_sync_controller_wraps_registry_errors(fake_tpu_sync, monkeypatch):
    def fail(address):
        raise ValueError("registry unavailable")

    broken = _FakeActorHandle(SimpleNamespace(set_controller_address=fail))
    monkeypatch.setattr(tpu_sync_engine, "create_tpu_sync_rendezvous", lambda: (broken, "TpuSyncRendezvous-test"))
    with pytest.raises(
        RuntimeError, match=r"Failed to store the tpu-sync controller address \(10\.0\.0\.1:7777\)"
    ) as err:
        tpu_sync_engine.setup_tpu_sync_controller()
    assert isinstance(err.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Trainer side: TPUSyncCheckpointEngine.send_weights
# ---------------------------------------------------------------------------


def test_send_weights_registers_full_tensors_and_recreates_synchronizer(fake_tpu_sync):
    tpu_sync_engine.setup_tpu_sync_controller()
    weights = _trainer_weights()
    names = sorted(weights)  # lm_head is sent unless the model is known to tie it
    engine = _trainer_engine(rank=0)
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))

    (ws,) = fake_tpu_sync.synchronizers
    assert [t.data_ptr() for t in ws.tensors] == [weights[n].data_ptr() for n in names]  # bound in place
    assert (ws.parallelism, ws.bind_ip) == (8, "10.0.0.1")
    assert ws.kwargs == {"unsafe_skip_buffer_lock": True, "auto_h2d": False}
    assert ws.d2h_calls == 0  # the controller-driven push does the D2H; send_weights only binds
    registration = fake_tpu_sync.registrations[TpuSyncId("trainer", "0", "weights")]
    assert registration.address == "10.0.0.1:7777"
    assert (registration.mesh_shape, registration.mesh_axes) == ([1, 1], ["fsdp", "tp"])
    assert registration.data_addresses == [f"10.0.0.1:{ws.local_port}"]
    assert registration.listener_address == f"10.0.0.1:{ws.listener_port}"
    for idx, (var, name) in enumerate(zip(registration.variables, names, strict=True)):
        shape = list(TRAINER_SHAPES[name])
        assert (var.name, var.shape, var.mesh_shape, var.sharding_spec, var.layer_idx, var.item_size) == (
            name,
            shape,
            [1] * len(shape),
            [""] * len(shape),
            idx,
            2,
        )
    assert fake_tpu_sync.registry_state.get_global_shapes() == {n: list(s) for n, s in TRAINER_SHAPES.items()}

    # After the transfer, release_sync_buffers unbinds the send tensors but keeps the synchronizer and its
    # controller registration; the next sync rebinds the new tensors without re-registering.
    engine.release_sync_buffers()
    assert engine._trainer_tpu_sync_ws is ws
    assert (ws.unbind_calls, ws.bound, engine._bound_tensors) == (1, False, None)
    new_weights = _trainer_weights(seed=1)
    asyncio.run(engine.send_weights(iter(new_weights.items()), global_steps=2))
    assert fake_tpu_sync.synchronizers == [ws]
    assert (ws.bind_calls, ws.bound, ws.d2h_calls) == (1, True, 0)
    assert [t.data_ptr() for t in ws.tensors] == [new_weights[n].data_ptr() for n in names]
    assert fake_tpu_sync.registrations[TpuSyncId("trainer", "0", "weights")] is registration

    # Without a release in between (a sync that failed after send_weights), the synchronizer is rebound too.
    asyncio.run(engine.send_weights(iter(new_weights.items()), global_steps=3))
    assert fake_tpu_sync.synchronizers == [ws] and ws.bind_calls == 2

    # A changed tensor signature (here: a dropped tensor) re-creates and re-registers the synchronizer.
    engine.release_sync_buffers()
    fewer = {n: t for n, t in new_weights.items() if n != "lm_head.weight"}
    asyncio.run(engine.send_weights(iter(fewer.items()), global_steps=4))
    assert len(fake_tpu_sync.synchronizers) == 2
    new_ws = fake_tpu_sync.synchronizers[1]
    assert engine._trainer_tpu_sync_ws is new_ws and new_ws.d2h_calls == 0
    registration = fake_tpu_sync.registrations[TpuSyncId("trainer", "0", "weights")]
    assert registration.data_addresses == [f"10.0.0.1:{new_ws.local_port}"]
    assert [var.name for var in registration.variables] == sorted(fewer)

    engine.finalize()
    assert (engine._trainer_tpu_sync_ws, engine._registered_signature, engine._bound_tensors) == (None, None, None)


def test_release_sync_buffers_is_noop_when_disabled_or_before_first_sync(fake_tpu_sync):
    tpu_sync_engine.setup_tpu_sync_controller()
    engine = _trainer_engine(rank=0, release_buffers_after_sync=False)
    assert engine.release_sync_buffers() == {}  # nothing bound yet

    weights = _trainer_weights()
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))
    (ws,) = fake_tpu_sync.synchronizers
    assert engine.release_sync_buffers() == {}
    assert (ws.unbind_calls, ws.bound) == (0, True)
    assert engine._bound_tensors is not None  # the send tensors stay resident between syncs


def test_release_sync_buffers_falls_back_to_destroying_synchronizer_without_unbind(fake_tpu_sync, caplog):
    tpu_sync_engine.setup_tpu_sync_controller()
    engine = _trainer_engine(rank=0)
    weights = _trainer_weights()
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))
    (ws,) = fake_tpu_sync.synchronizers
    del type(ws).unbind_weights  # an older tpu_sync build

    with caplog.at_level(logging.WARNING, logger=tpu_sync_engine.__name__):
        engine.release_sync_buffers()
        engine.release_sync_buffers()  # idempotent
    assert engine._trainer_tpu_sync_ws is None and engine._registered_signature is None
    assert caplog.text.count("has no unbind_weights()") == 1  # warned once

    # The next sync creates and registers a new synchronizer.
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=2))
    assert len(fake_tpu_sync.synchronizers) == 2 and engine._trainer_tpu_sync_ws is fake_tpu_sync.synchronizers[1]


@pytest.mark.parametrize(
    ("kwargs", "env", "expected"),
    [
        ({}, None, True),
        ({"release_buffers_after_sync": False}, None, False),
        ({"release_buffers_after_sync": "0"}, None, False),
        ({"release_buffers_after_sync": False}, "1", True),  # the deprecated environment variable still wins
        ({}, "false", False),
        ({}, "", True),  # unset / empty is ignored
    ],
)
def test_release_buffers_flag_from_kwargs_and_env(monkeypatch, kwargs, env, expected):
    if env is None:
        monkeypatch.delenv("VERL_TPU_SYNC_RELEASE_BUFFERS", raising=False)
    else:
        monkeypatch.setenv("VERL_TPU_SYNC_RELEASE_BUFFERS", env)
    assert tpu_sync_engine.TPUSyncCheckpointEngine(**kwargs).release_buffers_after_sync is expected


def test_send_weights_drops_tied_lm_head(fake_tpu_sync):
    tpu_sync_engine.setup_tpu_sync_controller()
    engine = _trainer_engine(rank=0, tie_word_embeddings=True)
    asyncio.run(engine.send_weights(_trainer_weights(), global_steps=1))  # a dict works too

    registration = fake_tpu_sync.registrations[TpuSyncId("trainer", "0", "weights")]
    assert [var.name for var in registration.variables] == sorted(set(TRAINER_SHAPES) - {"lm_head.weight"})
    assert "lm_head.weight" not in fake_tpu_sync.registry_state.get_global_shapes()


def test_send_weights_registers_local_shards(fake_tpu_sync):
    tpu_sync_engine.setup_tpu_sync_controller()
    rank, world = 1, 2
    weights = _trainer_weights()
    names = sorted(weights)
    engine = _trainer_engine(rank=rank, verify_parity=True)
    asyncio.run(engine.send_weights(_LocalShardWeights(weights, rank, world), global_steps=1))

    (ws,) = fake_tpu_sync.synchronizers
    registration = fake_tpu_sync.registrations[TpuSyncId("trainer", str(rank), "weights")]
    assert registration.mesh_shape == [world, 1]
    for var, tensor, name in zip(registration.variables, ws.tensors, names, strict=True):
        shape = list(TRAINER_SHAPES[name])
        rest = [1] * (len(shape) - 1)
        assert (var.name, var.shape) == (name, shape)  # every variable carries its full shape
        if name == "model.norm.weight":
            assert (var.mesh_shape, var.sharding_spec) == ([1], [""])
            assert not hasattr(var, "global_shard_indices")
            assert torch.equal(tensor, weights[name])
        else:
            assert (var.mesh_shape, var.global_shard_indices) == ([world] + rest, [rank])
            assert var.sharding_spec == ["fsdp"] + [""] * len(rest)
            assert torch.equal(tensor, weights[name].chunk(world, dim=0)[rank])  # only this rank's block is bound
    # The rollout sizes its receive buffers from the full shapes.
    assert fake_tpu_sync.registry_state.get_global_shapes() == {n: list(s) for n, s in TRAINER_SHAPES.items()}
    # Parity: this rank posts the stats of its own blocks; the full tensors are counted by rank 0 only.
    (posted,) = fake_tpu_sync.registry_state.get_stats(1)["ranks"].values()
    assert sorted(posted["per_tensor"]) == sorted(set(names) - {"model.norm.weight"})
    assert posted["total_numel"] == sum(weights[n].numel() // world for n in names if n != "model.norm.weight")

    # Same layout next step: rebind, no new registration.
    engine.release_sync_buffers()
    asyncio.run(engine.send_weights(_LocalShardWeights(_trainer_weights(seed=1), rank, world), global_steps=2))
    assert fake_tpu_sync.synchronizers == [ws] and ws.bind_calls == 1
    assert fake_tpu_sync.registrations[TpuSyncId("trainer", str(rank), "weights")] is registration

    # A different shard layout with identical local shapes (block 0 instead of 1) re-registers.
    engine.release_sync_buffers()
    asyncio.run(engine.send_weights(_LocalShardWeights(weights, 0, world), global_steps=3))
    assert len(fake_tpu_sync.synchronizers) == 2
    registration = fake_tpu_sync.registrations[TpuSyncId("trainer", str(rank), "weights")]
    assert {tuple(getattr(var, "global_shard_indices", ())) for var in registration.variables} == {(), (0,)}


def test_merge_rank_stats_matches_full_tensor_stats():
    weights = _trainer_weights()
    world = 4
    rank_stats = {}
    for rank in range(world):
        shards = _LocalShardWeights(weights, rank, world).local_shards()
        items = [(n, t) for n, t, shard in shards if shard is not None or rank == 0]
        rank_stats[rank] = parity.compute_tensor_stats(items)
    merged = parity.merge_rank_stats(rank_stats)
    reference = parity.compute_tensor_stats(list(weights.items()))
    assert (merged["total_numel"], merged["num_tensors"]) == (reference["total_numel"], reference["num_tensors"])
    for key in ("l1_norm", "l2_norm", "l2_sq"):
        assert merged[key] == pytest.approx(reference[key], rel=1e-6)
    for name, ref in reference["per_tensor"].items():
        got = merged["per_tensor"][name]
        assert got["numel"] == ref["numel"]
        assert (got["l1"], got["l2"]) == (pytest.approx(ref["l1"], rel=1e-6), pytest.approx(ref["l2"], rel=1e-6))


# ---------------------------------------------------------------------------
# Driver side: update_tpu_sync_weights, end to end through the fake transfer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_update_tpu_sync_weights_end_to_end(fake_tpu_sync, worker_ext, caplog, tie_word_embeddings):
    tp = trainer_ranks = 2
    weights = {"current": _trainer_weights(seed=0)}
    engines = [
        _trainer_engine(rank, verify_parity=True, tie_word_embeddings=tie_word_embeddings)
        for rank in range(trainer_ranks)
    ]
    models = [_FakeVllmModel(tp) for _ in range(tp)]
    samplers = [_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(models)]
    events: list = []
    manager, server = _make_manager(
        engines, lambda rank: iter(weights["current"].items()), samplers, events, verify_parity=True, parallelism=4
    )

    # Initial sync (step 0): nothing to abort and no parity check.
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        metrics = asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    assert events == [
        "update_weights",
        "init_tpu_sync_on_worker",
        "release_sync_buffers",
        "install_tpu_sync_weights",
        "set_global_steps",
        "resume",
    ]
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights["current"], rank, tp, lm_head_from_embed=tie_word_embeddings))
    assert server.global_steps == 0
    # The manager holds the rendezvous handle: the actor is not detached, so Ray would destroy it otherwise.
    assert manager._tpu_sync_rendezvous is fake_tpu_sync.registry
    # The rollout workers get the controller address and the shapes from the driver, not from the rendezvous.
    assert set(fake_tpu_sync.rendezvous_lookups) <= {None}
    assert set(metrics) == TIMING_KEYS and min(metrics.values()) >= 0
    (transfer,) = fake_tpu_sync.transfers
    assert transfer["src_units"] == [TpuSyncId("trainer", str(r), "weights") for r in range(trainer_ranks)]
    assert transfer["dst_units"] == [TpuSyncId("sampler0", str(r), "weights") for r in range(tp)]
    assert (transfer["parallelism"], transfer["req_id"], transfer["dst_mem_type"]) == (4, "verl_step_0", "DRAM")
    # The listener port is OS-assigned (0 at creation) and the bound port is what the controller learns.
    listener_ports = [sampler._tpu_sync_ws.listener_port for sampler in samplers]
    assert len(set(listener_ports)) == tp and all(port > 0 for port in listener_ports)
    for rank, sampler in enumerate(samplers):
        registration = fake_tpu_sync.registrations[TpuSyncId("sampler0", str(rank), "weights")]
        assert registration.listener_address == f"10.0.0.1:{sampler._tpu_sync_ws.listener_port}"
    assert [sampler._tpu_sync_ws.parallelism for sampler in samplers] == [4, 4]
    assert ("lm_head.weight" in fake_tpu_sync.registry_state.get_global_shapes()) is not tie_word_embeddings
    # Released after the transfer: the send tensors are unbound, the synchronizers are kept for the next sync.
    assert all(not engine._trainer_tpu_sync_ws.bound and engine._bound_tensors is None for engine in engines)

    # Next step: new weights. The controller, the rollout registrations and the trainer synchronizers are
    # reused; every trainer rank rebinds and stages its new tensors.
    weights["current"] = _trainer_weights(seed=1)
    events.clear()
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        metrics = asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    # The pause runs concurrently with the trainer send, which is requested first.
    assert events == [
        "update_weights",
        "abort",
        "init_tpu_sync_on_worker",
        "release_sync_buffers",
        "install_tpu_sync_weights",
        "set_global_steps",
        "get_model_weights_stats",
        "resume",
    ]
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights["current"], rank, tp, lm_head_from_embed=tie_word_embeddings))
    assert server.global_steps == 1
    assert set(metrics) == TIMING_KEYS
    assert fake_tpu_sync.servers_started == 1
    assert [sampler._tpu_sync_ws.h2d_calls for sampler in samplers] == [2, 2]
    sampler_synchronizers = [sampler._tpu_sync_ws for sampler in samplers]
    trainer_synchronizers = [ws for ws in fake_tpu_sync.synchronizers if ws not in sampler_synchronizers]
    assert len(trainer_synchronizers) == trainer_ranks  # one synchronizer per rank
    assert [ws.d2h_calls for ws in trainer_synchronizers] == [0] * trainer_ranks  # D2H belongs to the push
    bind_state = [(ws.bind_calls, ws.unbind_calls, ws.bound) for ws in trainer_synchronizers]
    assert bind_state == [(1, 2, False)] * trainer_ranks  # rebound once, unbound after each transfer
    assert "TPU-SYNC PARITY VERIFIED | Step 1" in caplog.text
    assert "MISMATCH" not in caplog.text


@pytest.mark.parametrize("backend", ["tpu-sync"])
def test_update_tpu_sync_weights_installs_tp_shards_into_transposed_weights(fake_tpu_sync, worker_ext, backend):
    """Every trainer rank registers the full tensors; vLLM ranks get TP shards, also of transposed parameters."""
    tp, trainer_ranks = 2, 4
    weights = _trainer_weights()
    models = [_FakeVllmModel(tp, flip_o_proj=True) for _ in range(tp)]
    samplers = [_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(models)]
    manager, _ = _make_manager(
        [_trainer_engine(rank) for rank in range(trainer_ranks)],
        lambda rank: iter(weights.items()),
        samplers,
        [],
        backend=backend,
        parallelism=3,
    )
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))

    assert [sampler._tpu_sync_ws.parallelism for sampler in samplers] == [3] * tp
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights, rank, tp, flip_o_proj=True))
    for rank in range(trainer_ranks):
        registration = fake_tpu_sync.registrations[TpuSyncId("trainer", str(rank), "weights")]
        assert registration.mesh_shape == [1, 1]
        assert [(var.name, var.shape) for var in registration.variables] == [
            (name, list(TRAINER_SHAPES[name])) for name in sorted(TRAINER_SHAPES)
        ]


@pytest.mark.parametrize(("trainer_ranks", "tp"), [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_update_tpu_sync_weights_from_local_shards(
    fake_tpu_sync, worker_ext, caplog, trainer_ranks, tp, tie_word_embeddings
):
    """Every trainer rank registers only its FSDP blocks; the controller reassembles and reshards them for vLLM TP.

    Includes TP > KV heads (tp=4), where vLLM replicates each KV head, and FSDP degree != TP.
    """
    weights = {"current": _trainer_weights(seed=0)}
    engines = [
        _trainer_engine(rank, verify_parity=True, tie_word_embeddings=tie_word_embeddings)
        for rank in range(trainer_ranks)
    ]
    models = [_FakeVllmModel(tp) for _ in range(tp)]
    samplers = [_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(models)]
    manager, _ = _make_manager(
        engines,
        lambda rank: _LocalShardWeights(weights["current"], rank, trainer_ranks),
        samplers,
        [],
        verify_parity=True,
    )

    for step in (1, 2):
        weights["current"] = _trainer_weights(seed=step)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
            asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=step))
        for rank, model in enumerate(models):
            expected = _expected_vllm_state(weights["current"], rank, tp, lm_head_from_embed=tie_word_embeddings)
            _assert_state(model, expected)
        assert f"TPU-SYNC PARITY VERIFIED | Step {step}" in caplog.text
        assert "MISMATCH" not in caplog.text

    for rank in range(trainer_ranks):
        registration = fake_tpu_sync.registrations[TpuSyncId("trainer", str(rank), "weights")]
        assert registration.mesh_shape == [trainer_ranks, 1]
    sampler_synchronizers = [sampler._tpu_sync_ws for sampler in samplers]
    trainer_synchronizers = [ws for ws in fake_tpu_sync.synchronizers if ws not in sampler_synchronizers]
    assert [ws.bind_calls for ws in trainer_synchronizers] == [1] * trainer_ranks  # registered once, then rebound


def test_parity_check_with_local_shards_waits_for_every_rank_and_reports_mismatch(
    fake_tpu_sync, worker_ext, caplog, monkeypatch
):
    tp, trainer_ranks = 2, 2
    weights = _trainer_weights()
    samplers = [_make_sampler(worker_ext, rank, tp, _FakeVllmModel(tp)) for rank in range(tp)]
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=True) for rank in range(trainer_ranks)],
        lambda rank: _LocalShardWeights(weights, rank, trainer_ranks),
        samplers,
        [],
        verify_parity=True,
    )
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    state = fake_tpu_sync.registry_state
    assert sorted(state.get_stats(1)["ranks"]) == list(range(trainer_ranks))

    # A rank whose stats never arrive: the check gives up waiting and fails on the incomplete stats.
    ranks = state.stats[1]["ranks"]
    missing = ranks.pop(1)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(tpu_sync_engine.asyncio, "sleep", lambda _: real_sleep(0))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        with pytest.raises(parity.TPUSyncParityError, match=r"Step 1\] Incomplete trainer stats: 1 of 2 trainer"):
            asyncio.run(tpu_sync_engine._verify_parity(manager, global_steps=1))
    assert "PARITY VERIFIED" not in caplog.text
    ranks[1] = missing

    # A corrupted rollout tensor is reported against the merged trainer stats.
    samplers[1]._tpu_sync_staging["model.embed_tokens.weight"].zero_()
    with pytest.raises(parity.TPUSyncParityError, match=r"TPU-SYNC PARITY MISMATCH \| Step 1\]") as excinfo:
        asyncio.run(tpu_sync_engine._verify_parity(manager, global_steps=1))
    assert "model.embed_tokens.weight" in str(excinfo.value)


def test_parity_check_reports_mismatch(fake_tpu_sync, worker_ext):
    tp = 2
    weights = _trainer_weights()
    samplers = [_make_sampler(worker_ext, rank, tp, _FakeVllmModel(tp)) for rank in range(tp)]
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=True) for rank in range(2)],
        lambda rank: iter(weights.items()),
        samplers,
        [],
        verify_parity=True,
    )
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))

    samplers[1]._tpu_sync_staging["model.embed_tokens.weight"].zero_()
    with pytest.raises(parity.TPUSyncParityError, match=r"TPU-SYNC PARITY MISMATCH \| Step 1\]") as excinfo:
        asyncio.run(tpu_sync_engine._verify_parity(manager, global_steps=1))
    assert "model.embed_tokens.weight" in str(excinfo.value)


def _norm_parity_setup(worker_ext, weights, trainer_verify_parity=True):
    """TP=2 rollout and 2 trainer ranks with ``verify_parity=norm`` on the driver (and, by default, the trainer)."""
    samplers = [_make_sampler(worker_ext, rank, 2, _FakeVllmModel(2)) for rank in range(2)]
    events: list = []
    manager, server = _make_manager(
        [_trainer_engine(rank, verify_parity=trainer_verify_parity) for rank in range(2)],
        lambda rank: iter(weights.items()),
        samplers,
        events,
        verify_parity="norm",
    )
    return manager, server, samplers, events


def _rpc_without_results(server, monkeypatch, methods):
    """Makes ``server`` run ``methods`` on its workers but return no results, like a server that drops them."""
    real_rpc = server.collective_rpc

    def collective_rpc(method, **kwargs):
        result = real_rpc(method, **kwargs)
        return None if method in methods else result

    monkeypatch.setattr(server, "collective_rpc", collective_rpc)


def test_norm_parity_mismatch_fails_the_sync_without_resuming(fake_tpu_sync, worker_ext):
    weights = _trainer_weights()
    manager, _, samplers, events = _norm_parity_setup(worker_ext, weights)
    # One rollout worker ends up with a wrong tensor after its install.
    real_install = samplers[1].install_tpu_sync_weights

    def install_and_corrupt(**kwargs):
        result = real_install(**kwargs)
        samplers[1]._tpu_sync_staging["model.embed_tokens.weight"].zero_()
        return result

    samplers[1].install_tpu_sync_weights = install_and_corrupt
    with pytest.raises(parity.TPUSyncParityError, match=r"TPU-SYNC PARITY MISMATCH \| Step 1\]") as excinfo:
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    assert "model.embed_tokens.weight" in str(excinfo.value)
    assert events[-2:] == ["set_global_steps", "get_model_weights_stats"]  # checked, then stopped: no "resume"


def test_norm_parity_fails_the_sync_without_trainer_stats(fake_tpu_sync, worker_ext, monkeypatch):
    # The trainer ranks post no norms (here: norm check off on the trainer side), so there is nothing to compare.
    manager, _, _, events = _norm_parity_setup(worker_ext, _trainer_weights(), trainer_verify_parity=False)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(parity.asyncio, "sleep", lambda _: real_sleep(0))
    with pytest.raises(parity.TPUSyncParityError, match=r"Step 1\] Incomplete trainer stats: 0 of 2 trainer ranks"):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    assert "install_tpu_sync_weights" in events and "resume" not in events


def test_norm_parity_fails_the_sync_without_rollout_stats(fake_tpu_sync, worker_ext, monkeypatch):
    manager, server, _, events = _norm_parity_setup(worker_ext, _trainer_weights())
    _rpc_without_results(server, monkeypatch, {"get_model_weights_stats"})
    with pytest.raises(parity.TPUSyncParityError, match=r"Step 1\] No sampler worker stats"):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    assert "resume" not in events


def test_exact_parity_fails_the_sync_without_rollout_results(fake_tpu_sync, worker_ext, monkeypatch):
    weights = _trainer_weights()
    manager, _, events = _exact_parity_setup(worker_ext, weights, 2, "exact")
    _rpc_without_results(manager.replicas[0].server_handle._obj, monkeypatch, {"install_tpu_sync_weights"})
    with pytest.raises(parity.TPUSyncParityError, match=r"Step 0\] no rollout worker returned a result"):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    assert "resume" not in events


def test_update_tpu_sync_weights_returns_phase_timings_and_slowest_worker_h2d(fake_tpu_sync, worker_ext, monkeypatch):
    """The ``timing_s/tpu-sync/*`` step metrics: ``sampler_h2d_pure`` is the slowest worker's H2D time over every
    replica, and is left out when the rollout servers do not pass the worker results back."""
    tp, num_replicas = 2, 2
    weights = _trainer_weights()
    samplers = [
        [_make_sampler(worker_ext, rank, tp, _FakeVllmModel(tp)) for rank in range(tp)] for _ in range(num_replicas)
    ]

    def with_h2d(install, h2d):
        """``install``, reporting ``h2d`` as its host-to-device time."""
        return lambda **kwargs: {**install(**kwargs), "h2d": h2d}

    h2d_times = [0.25, 0.5, 0.75, 0.125]  # The slowest worker is rank 0 of replica 1.
    for sampler, h2d in zip(itertools.chain.from_iterable(samplers), h2d_times, strict=True):
        sampler.install_tpu_sync_weights = with_h2d(sampler.install_tpu_sync_weights, h2d)
    manager, _ = _make_manager(
        [_trainer_engine(0)], lambda rank: iter(weights.items()), None, [], replica_samplers=samplers
    )
    assert len(manager.replicas) == num_replicas

    metrics = asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    assert set(metrics) == TIMING_KEYS and min(metrics.values()) >= 0
    assert metrics["timing_s/tpu-sync/sampler_h2d_pure"] == 0.75

    for replica in manager.replicas:
        _rpc_without_results(replica.server_handle._obj, monkeypatch, {"install_tpu_sync_weights"})
    metrics = asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=2))
    assert set(metrics) == TIMING_KEYS - {"timing_s/tpu-sync/sampler_h2d_pure"}


def test_update_tpu_sync_weights_pauses_concurrently_and_resumes_before_the_release(
    fake_tpu_sync, worker_ext, monkeypatch
):
    """The pause overlaps the trainer send, and generation resumes before the trainer frees its send buffers."""
    tp = 2
    weights = _trainer_weights()
    samplers = [_make_sampler(worker_ext, rank, tp, _FakeVllmModel(tp)) for rank in range(tp)]
    events: list = []
    manager, _ = _make_manager(
        [_trainer_engine(rank) for rank in range(2)], lambda rank: iter(weights.items()), samplers, events
    )
    # Record when the trainer's buffer release completes (ray.get on its refs), not only when it is requested.
    real_get = tpu_sync_engine.ray.get

    def get(refs, timeout=None):
        fns = [getattr(ref, "_fn", None) for ref in refs] if isinstance(refs, list) else []
        names = {getattr(fn, "__name__", "") for fn in fns}
        result = real_get(refs, timeout=timeout)
        if "release_sync_buffers" in names:
            events.append("release_done")
        return result

    monkeypatch.setattr(tpu_sync_engine.ray, "get", get)
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    # update_weights is requested before the pause task first runs: the two overlap.
    assert events.index("update_weights") < events.index("abort") < events.index("init_tpu_sync_on_worker")
    assert events.index("resume") < events.index("release_done")


def _exact_parity_setup(worker_ext, weights, tp, verify_parity, flip_o_proj=False, loaded=True):
    """Rollout models that already hold the checkpoint (``loaded``), as vLLM does before the step-0 sync."""
    models = [_FakeVllmModel(tp, flip_o_proj=flip_o_proj) for _ in range(tp)]
    if loaded:
        for rank, model in enumerate(models):
            model.load_state_dict(_expected_vllm_state(weights, rank, tp, flip_o_proj=flip_o_proj))
    samplers = [_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(models)]
    events: list = []
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=verify_parity) for rank in range(2)],
        lambda rank: _LocalShardWeights(weights, rank, 2),
        samplers,
        events,
        verify_parity=verify_parity,
    )
    return manager, models, events


@pytest.mark.parametrize("verify_parity", ["exact", "all"])
@pytest.mark.parametrize("tp", [2, 4])
def test_exact_parity_passes_on_step_0_when_rollout_holds_the_checkpoint(
    fake_tpu_sync, worker_ext, caplog, verify_parity, tp
):
    weights = _trainer_weights()
    manager, models, _ = _exact_parity_setup(worker_ext, weights, tp, verify_parity, flip_o_proj=True)
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    # Every tensor goes through copy_into: 2 fused (qkv, gate_up) + 7 others, lm_head received (not tied).
    assert f"TPU-SYNC PARITY EXACT | Step 0] {tp} rollout ranks, [9] tensors checked per rank, 0 mismatched" in (
        caplog.text
    )
    assert "bit for bit" in caplog.text
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights, rank, tp, flip_o_proj=True))
    # Trainer ranks post norm stats only for norm / all; exact alone needs nothing from them.
    assert (fake_tpu_sync.registry_state.get_stats(0) is not None) is (verify_parity == "all")


def test_exact_parity_mismatch_fails_the_sync_without_resuming(fake_tpu_sync, worker_ext):
    tp = 2
    weights = _trainer_weights()
    manager, models, events = _exact_parity_setup(worker_ext, weights, tp, "exact")
    # vLLM's copy of one tensor differs from the checkpoint the trainer sends (e.g. a mis-sliced shard).
    with torch.no_grad():
        models[1].model.layers[0].mlp.down_proj.weight.add_(1)
    with pytest.raises(parity.TPUSyncParityError) as excinfo:
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    message = str(excinfo.value)
    assert "TPU-SYNC PARITY EXACT | Step 0] 2 rollout ranks, [9] tensors checked per rank, 1 mismatched" in message
    assert "rank 1: model.layers.0.mlp.down_proj.weight: max|diff|=1" in message
    # The check runs after the install (the received weights are in place) and before generation resumes.
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights, rank, tp))
    assert "install_tpu_sync_weights" in events and "resume" not in events


def test_exact_parity_runs_only_on_the_step_0_sync(fake_tpu_sync, worker_ext, caplog):
    weights = _trainer_weights()
    manager, _, events = _exact_parity_setup(worker_ext, weights, 2, "all", loaded=False)
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    # Trained weights no longer match what vLLM loaded: no exact check, the norm check runs as usual.
    assert "PARITY EXACT" not in caplog.text
    assert "TPU-SYNC PARITY VERIFIED | Step 1" in caplog.text
    assert "get_model_weights_stats" in events


def test_report_exact_parity_names_replicas_and_flags_skipped_or_missing_results(caplog):
    ok = {"rank": 0, "checked": 3, "num_mismatched": 0, "mismatched": [], "unresolved": []}
    skipped = {"rank": 0, "skipped": "not supported with the vLLM-loader install"}
    install_results = [
        [{"h2d": 0.1, "exact_parity": ok}],
        [{"h2d": 0.1, "exact_parity": skipped}],
        [{"h2d": 0.1}],
        None,  # A rollout server that does not pass the worker results back.
    ]
    labels = [" | replica 0", " | replica 1", " | replica 2", " | replica 3"]
    with caplog.at_level(logging.INFO, logger=parity.__name__):
        # Every replica is checked before the check fails, and the error names each failing one.
        with pytest.raises(parity.TPUSyncParityError) as excinfo:
            parity.report_exact_parity(0, install_results, labels)
    assert "PARITY EXACT | Step 0 | replica 0] 1 rollout ranks, [3] tensors checked per rank, 0 mismatched" in (
        caplog.text
    )
    message = str(excinfo.value)
    assert "PARITY EXACT | Step 0 | replica 1] not run: not supported with the vLLM-loader install" in message
    assert "PARITY EXACT | Step 0 | replica 2] no rollout worker returned a result" in message
    assert "PARITY EXACT | Step 0 | replica 3] no rollout worker returned a result" in message
    assert "replica 0" not in message

    # A replica the check cannot run on (vLLM-loader install) fails the check too: exact was asked for and did not run.
    with pytest.raises(parity.TPUSyncParityError, match=r"replica 1\] not run: .*use verify_parity=norm"):
        parity.report_exact_parity(0, install_results[:2], labels[:2])
    parity.report_exact_parity(0, install_results[:1], labels[:1])  # all checked and matched: no error


def test_report_exact_parity_fails_on_unresolved_tensors():
    unresolved = {"rank": 1, "checked": 3, "num_mismatched": 0, "mismatched": [], "unresolved": ["extra.weight"]}
    with pytest.raises(parity.TPUSyncParityError, match="0 mismatched, 1 unresolved") as excinfo:
        parity.report_exact_parity(None, [[{"exact_parity": unresolved}]], [""])
    assert "rank 1: unresolved ['extra.weight']" in str(excinfo.value)


def test_exact_parity_check_flags_shape_and_value_differences():
    check = parity.ExactParityCheck()
    target = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    check.check("same", target, target.clone())
    check.check("shape", target, target.T)
    check.check("value", target, target + 2)
    result = check.result(rank=3, unresolved=["extra"])
    assert (result["rank"], result["checked"], result["num_mismatched"], result["unresolved"]) == (3, 2, 2, ["extra"])
    assert result["mismatched"][0].startswith("shape: shape target=(2, 3) received=(3, 2)")
    assert result["mismatched"][1].startswith("value: max|diff|=2")


def test_update_tpu_sync_weights_syncs_several_replicas_in_one_transfer(fake_tpu_sync, worker_ext, caplog):
    """Two TP=2 replicas: a job name and a port range per replica, one transfer, the parity check per replica."""
    tp, trainer_ranks, num_replicas = 2, 2, 2
    weights = _trainer_weights()
    models = [[_FakeVllmModel(tp) for _ in range(tp)] for _ in range(num_replicas)]
    samplers = [[_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(group)] for group in models]
    events: list = []
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=True) for rank in range(trainer_ranks)],
        lambda rank: iter(weights.items()),
        None,
        events,
        replica_samplers=samplers,
        verify_parity=True,
    )
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))

    for group in models:
        for rank, model in enumerate(group):
            _assert_state(model, _expected_vllm_state(weights, rank, tp))
    assert events.count("init_tpu_sync_on_worker") == num_replicas
    (transfer,) = fake_tpu_sync.transfers
    assert transfer["src_units"] == [TpuSyncId("trainer", str(r), "weights") for r in range(trainer_ranks)]
    assert transfer["dst_units"] == [
        TpuSyncId(f"sampler{replica}", str(rank), "weights") for replica in range(num_replicas) for rank in range(tp)
    ]
    assert transfer["req_id"] == "verl_step_1"
    listener_ports = [sampler._tpu_sync_ws.listener_port for group in samplers for sampler in group]
    assert len(set(listener_ports)) == num_replicas * tp  # OS-assigned: distinct even when replicas share a host
    assert "[TPU-SYNC PARITY VERIFIED | Step 1 | replica 0]" in caplog.text
    assert "[TPU-SYNC PARITY VERIFIED | Step 1 | replica 1]" in caplog.text
    assert "MISMATCH" not in caplog.text

    # A replica with wrong weights fails the check by name; the other one still verifies.
    samplers[1][0]._tpu_sync_staging["model.embed_tokens.weight"].zero_()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        with pytest.raises(parity.TPUSyncParityError) as excinfo:
            asyncio.run(tpu_sync_engine._verify_parity(manager, global_steps=1))
    assert "[TPU-SYNC PARITY MISMATCH | Step 1 | replica 1]" in str(excinfo.value)
    assert "replica 0" not in str(excinfo.value)
    assert "[TPU-SYNC PARITY VERIFIED | Step 1 | replica 0]" in caplog.text

    # Both replicas wrong: each one is checked and named in the error.
    samplers[0][1]._tpu_sync_staging["model.embed_tokens.weight"].zero_()
    with pytest.raises(parity.TPUSyncParityError) as excinfo:
        asyncio.run(tpu_sync_engine._verify_parity(manager, global_steps=1))
    assert "[TPU-SYNC PARITY MISMATCH | Step 1 | replica 0]" in str(excinfo.value)
    assert "[TPU-SYNC PARITY MISMATCH | Step 1 | replica 1]" in str(excinfo.value)


def test_sampler_job_names_follow_replica_rank():
    # Keyed on replica_rank, so a replica keeps its name when others are added or removed.
    assert tpu_sync_engine._sampler_job_name(SimpleNamespace(replica_rank=3), 0) == "sampler3"
    assert tpu_sync_engine._sampler_job_name(SimpleNamespace(), 2) == "sampler2"  # no replica_rank: list position


def test_wait_for_registration_times_out():
    controller = SimpleNamespace(_lock=threading.Lock(), _registered_shards={TpuSyncId("trainer", "0", "weights"): 0})
    src_units = [TpuSyncId("trainer", str(rank), "weights") for rank in range(2)]
    dst_units = [TpuSyncId("sampler", "0", "weights")]
    with pytest.raises(RuntimeError, match="Timeout") as excinfo:
        asyncio.run(tpu_sync_engine._wait_for_registration(controller, src_units, dst_units, timeout_s=0.01))
    message = str(excinfo.value)
    assert "Missing Trainer: [TpuSyncId(job_name='trainer', job_replica_id='1', data_name='weights')]" in message
    assert "Missing Sampler: [TpuSyncId(job_name='sampler', job_replica_id='0', data_name='weights')]" in message


@pytest.mark.parametrize("backend", ["tpu-sync"])
def test_manager_hook_routes_tpu_sync_backend(monkeypatch, backend):
    calls = []

    async def fake_update_tpu_sync_weights(manager, global_steps=None):
        calls.append((manager.backend, global_steps))
        return {"timing_s/tpu-sync/total_sync": 1.0}

    monkeypatch.setattr(tpu_sync_engine, "update_tpu_sync_weights", fake_update_tpu_sync_weights)
    assert getattr(ckpt_base, "_verl_tpu_ckpt_patched", False)
    result = ckpt_base.CheckpointEngineManager.update_weights(SimpleNamespace(backend=backend), global_steps=3)
    assert result == {"timing_s/tpu-sync/total_sync": 1.0}
    assert calls == [(backend, 3)]


# ---------------------------------------------------------------------------
# Rollout side: vLLMTPUSyncWorkerExtension
# ---------------------------------------------------------------------------


def test_sampler_sharding(worker_ext):
    def shard(name, shape, tp, **kwargs):
        s = worker_ext._sampler_sharding(name, shape, tp, **kwargs)
        return s.sharding_spec, s.local_shape, s.mesh_shape, s.global_shard_indices, s.replicas

    assert shard(LAYER + "self_attn.o_proj.weight", [8, 16], 2) == (["", "tp"], [8, 8], [1, 2], [], 1)
    assert shard(LAYER + "mlp.down_proj.weight", [8, 16], 4) == (["", "tp"], [8, 4], [1, 4], [], 1)
    assert shard(LAYER + "self_attn.q_proj.weight", [16, 8], 2) == (["tp", ""], [8, 8], [2, 1], [], 1)
    # indivisible: replicated on every rank
    assert shard("model.embed_tokens.weight", [15, 8], 2) == (["", ""], [15, 8], [1, 1], [], 2)
    assert shard("model.norm.weight", [8], 2) == ([""], [8], [1], [], 2)


def test_sampler_sharding_gqa_kv_heads(worker_ext):
    """With tp_size > num_kv_heads, k/v are sliced per KV head and every rank names the head it holds."""

    def shard(name, shape, tp, rank, kv_heads):
        s = worker_ext._sampler_sharding(name, shape, tp, tp_rank=rank, num_kv_heads=kv_heads)
        return s.sharding_spec, s.local_shape, s.mesh_shape, s.global_shard_indices, s.replicas

    k, v, q = (LAYER + f"self_attn.{p}_proj.weight" for p in "kvq")
    # Qwen3-32B at TP=32: 8 KV heads of 128 rows, each on 4 ranks.
    assert shard(k, [1024, 5120], 32, 13, 8) == (["tp", ""], [128, 5120], [8, 1], [3], 4)
    assert shard(v, [1024, 5120], 32, 31, 8) == (["tp", ""], [128, 5120], [8, 1], [7], 4)
    # q is not replicated, and neither are k/v while tp_size <= num_kv_heads.
    assert shard(q, [8192, 5120], 32, 13, 8) == (["tp", ""], [256, 5120], [32, 1], [], 1)
    assert shard(k, [1024, 5120], 8, 3, 8) == (["tp", ""], [128, 5120], [8, 1], [], 1)
    # Without the KV head count (older callers) k/v fall back to the plain dim-0 split.
    assert shard(k, [1024, 5120], 32, 13, None) == (["tp", ""], [32, 5120], [32, 1], [], 1)


def test_sampler_sharding_load_weights_mode(worker_ext):
    """use_vllm_loaders=True shards >=2D tensors on dim 0 and keeps 1-D tensors replicated."""

    def shard(name, shape, tp):
        s = worker_ext._sampler_sharding(name, shape, tp, use_vllm_loaders=True)
        return s.sharding_spec, s.local_shape, s.mesh_shape, s.replicas

    assert shard(LAYER + "linear_attn.norm.weight", [128], 8) == ([""], [128], [1], 8)
    assert shard(LAYER + "mlp.experts.down_proj", [256, 2048, 512], 8) == (
        ["tp", "", ""],
        [32, 2048, 512],
        [8, 1, 1],
        1,
    )
    # Row-parallel names are still split on dim 0: the install all-gathers them before vLLM's loaders run.
    assert shard(LAYER + "self_attn.o_proj.weight", [16, 8], 2) == (["tp", ""], [8, 8], [2, 1], 1)


def test_uses_vllm_weight_loader(worker_ext):
    """Auto-selects vLLM's loaders for hybrid (linear attention) or MoE models, direct copy for dense ones."""
    dense = {LAYER + "self_attn.q_proj.weight": [16, 8], "model.norm.weight": [8]}
    moe = {**dense, LAYER + "mlp.experts.down_proj": [4, 8, 2]}
    model = object()
    assert not worker_ext._uses_vllm_weight_loader(model, dense)
    assert worker_ext._uses_vllm_weight_loader(model, moe)


@pytest.mark.parametrize("verify_parity", [False, True])
def test_update_tpu_sync_weights_gqa_replicated_kv_heads(fake_tpu_sync, worker_ext, caplog, verify_parity):
    """TP=4 with 2 KV heads: each rank receives one whole KV head, and the parity check counts each head once."""
    tp = 4
    assert tp > KV_HEADS
    weights = _trainer_weights()
    models = [_FakeVllmModel(tp) for _ in range(tp)]
    samplers = [_make_sampler(worker_ext, rank, tp, model) for rank, model in enumerate(models)]
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=verify_parity) for rank in range(2)],
        lambda rank: iter(weights.items()),
        samplers,
        [],
        verify_parity=verify_parity,
    )
    with caplog.at_level(logging.INFO, logger=tpu_sync_engine.__name__):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))

    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights, rank, tp))
    for rank in range(tp):
        variables = {
            v.name: v for v in fake_tpu_sync.registrations[TpuSyncId("sampler0", str(rank), "weights")].variables
        }
        for name in QKV[1:]:
            assert (variables[name].mesh_shape, variables[name].global_shard_indices) == ([KV_HEADS, 1], [rank // 2])
        assert (variables[QKV[0]].mesh_shape, variables[QKV[0]].global_shard_indices) == ([tp, 1], [])
    if verify_parity:
        assert "TPU-SYNC PARITY VERIFIED | Step 1" in caplog.text
        assert "MISMATCH" not in caplog.text


def test_sampler_init_allocates_tp_shards_and_registers(fake_tpu_sync, worker_ext):
    shapes = {
        "model.norm.weight": [128],
        LAYER + "self_attn.o_proj.weight": [128, 256],
        LAYER + "self_attn.k_proj.weight": [16, 64],
    }
    sampler = _make_sampler(worker_ext, rank=1, tp=2, model=_FakeVllmModel(2))
    assert sampler.init_tpu_sync_on_worker("10.0.0.2:7777", shapes, parallelism=4) is True

    (ws,) = fake_tpu_sync.synchronizers
    assert (ws.parallelism, ws.bind_ip) == (4, "10.0.0.1")
    assert ws.listener_port > 0  # OS-assigned (requested as 0), registered with the controller below
    assert [tuple(t.shape) for t in ws.tensors] == [(8, 64), (128, 128), (128,)]
    assert all(t.dtype == torch.bfloat16 for t in ws.tensors)
    assert ws.skip_tiling is None  # skip-tiling is left to the planner, which sees both sides of the transfer
    registration = fake_tpu_sync.registrations[TpuSyncId("sampler0", "1", "weights")]
    assert registration.address == "10.0.0.2:7777"
    assert (registration.mesh_shape, registration.mesh_axes) == ([1, 2], ["fsdp", "tp"])
    assert registration.data_addresses == [f"10.0.0.1:{ws.local_port}"]
    assert registration.listener_address == f"10.0.0.1:{ws.listener_port}"
    assert [
        (v.name, v.shape, v.mesh_shape, v.sharding_spec, v.global_shard_indices) for v in registration.variables
    ] == [
        (LAYER + "self_attn.k_proj.weight", [16, 64], [2, 1], ["tp", ""], []),
        (LAYER + "self_attn.o_proj.weight", [128, 256], [1, 2], ["", "tp"], []),
        ("model.norm.weight", [128], [1], [""], []),
    ]
    assert sampler._tpu_sync_replicas == {
        LAYER + "self_attn.k_proj.weight": 1,
        LAYER + "self_attn.o_proj.weight": 1,
        "model.norm.weight": 2,
    }

    # Later syncs reuse the receive buffers and the registration.
    assert sampler.init_tpu_sync_on_worker("10.0.0.2:7777", shapes, parallelism=4) is True
    assert len(fake_tpu_sync.synchronizers) == 1


def test_sampler_init_registers_under_its_replica_job_name(fake_tpu_sync, worker_ext):
    sampler = _make_sampler(worker_ext, rank=1, tp=2, model=_FakeVllmModel(2))
    assert (
        sampler.init_tpu_sync_on_worker(
            "10.0.0.2:7777", {"model.norm.weight": [128]}, parallelism=4, job_name="sampler1"
        )
        is True
    )

    (ws,) = fake_tpu_sync.synchronizers
    assert TpuSyncId("sampler0", "1", "weights") not in fake_tpu_sync.registrations
    registration = fake_tpu_sync.registrations[TpuSyncId("sampler1", "1", "weights")]
    assert registration.listener_address == f"10.0.0.1:{ws.listener_port}"


@pytest.mark.parametrize("address, shapes", [(None, {"model.norm.weight": [128]}), ("10.0.0.2:7777", {})])
def test_sampler_init_requires_the_controller_address_and_shapes(fake_tpu_sync, worker_ext, address, shapes):
    sampler = _make_sampler(worker_ext, rank=0, tp=2, model=_FakeVllmModel(2))
    with pytest.raises(RuntimeError, match="the driver passed no tpu-sync controller address"):
        sampler.init_tpu_sync_on_worker(address, shapes)
    assert fake_tpu_sync.synchronizers == []


def test_sampler_methods_before_init(worker_ext):
    sampler = _make_sampler(worker_ext, rank=0, tp=2, model=_FakeVllmModel(2))
    assert sampler.install_tpu_sync_weights() == {}
    with pytest.raises(RuntimeError, match="only supported for tpu-sync after initialization"):
        sampler.get_model_weights_stats()


def test_unknown_engine_kwarg_is_an_error():
    # verl passes checkpoint_engine.engine_kwargs.tpu-sync to the constructor; a typo must not be ignored.
    with pytest.raises(ValueError, match="paralelism"):
        tpu_sync_engine.TPUSyncCheckpointEngine(paralelism=4)


def test_update_tpu_sync_weights_prints_timeline(fake_tpu_sync, worker_ext, monkeypatch, capsys):
    from verl_hardware_plugin.accelerators.tpu.profilers import timeline

    monkeypatch.setenv(timeline.ENV_VAR, "1")
    monkeypatch.delenv(timeline.RANKS_ENV_VAR, raising=False)
    monkeypatch.delenv("RANK", raising=False)
    weights = _trainer_weights(seed=0)
    engines = [_trainer_engine(0, verify_parity=False)]
    samplers = [_make_sampler(worker_ext, 0, 1, _FakeVllmModel(1))]
    manager, _ = _make_manager(engines, lambda _: iter(weights.items()), samplers, [], verify_parity=False)

    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))

    events = [
        json.loads(line[len(timeline.MARKER) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(timeline.MARKER)
    ]
    loop = [(e["name"], e["step"]) for e in events if e["group"] == "trainer" and not e["name"].startswith("sync.")]
    assert loop == [("initial_weight_sync", 0), ("update_weights", 1)]
    worker = {e["name"] for e in events if e["group"] == "trainer_workers" and e["lane"] == "rank0"}
    assert {"update_weights", "gather", "bind", "release_tpu_sync"} <= worker
