# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the Raiden (tpu-sync) checkpoint engine and its vLLM worker extension.

tpu-sync, torch_tpu and vLLM are not installed on CI. A fake ``tpu_sync`` stands in for the transfer
library: its controller "transfers" by copying the trainer's bound tensors into the rollout workers'
receive buffers, resharded as the variable metadata registered by both sides describes, so the
end-to-end tests check that this metadata is consistent. Ray actor handles are in-process stand-ins
around the real ``RayWeightRegistryState``.
"""

import asyncio
import importlib
import itertools
import logging
import sys
import threading
from collections import namedtuple
from types import ModuleType, SimpleNamespace
from unittest import mock

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
# tpu_checkpoint_engine also installs the CheckpointEngineManager hook that routes backend=raiden.
import verl.checkpoint_engine.base as ckpt_base  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.engines import raiden_checkpoint_engine as raiden  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.engines import ray_weight_registry, tpu_checkpoint_engine  # noqa: E402, F401
from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import RayWeightRegistryState  # noqa: E402

TPU_RAIDEN = "verl_hardware_plugin.accelerators.tpu.rollout.tpu_raiden"
RaidenId = namedtuple("RaidenId", ["job_name", "job_replica_id", "data_name"])
TIMING_KEYS = {
    f"timing_s/tpu-sync/{phase}"
    for phase in (
        "quiesce",
        "trainer_init",
        "sampler_init",
        "barrier",
        "p2p_transfer",
        "sampler_h2d",
        "total_sync",
        "sampler_h2d_pure",
    )
}

# A one-layer Qwen3-like model in HF naming: hidden size, vocab, MLP intermediate size, q rows, k/v rows.
HIDDEN, VOCAB, INTERMEDIATE, Q_ROWS, KV_ROWS = 8, 16, 16, 8, 4
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
        self.registry_state = RayWeightRegistryState()
        self.registry = _FakeActorHandle(self.registry_state)

    def synchronizer_at(self, address):
        port = int(address.rsplit(":", 1)[1])
        return next(ws for ws in reversed(self.synchronizers) if ws.local_port == port)

    def transfer(self, src_units, dst_units):
        """Copies the trainer units' tensors into the rollout units' buffers, resharded per their metadata."""
        full = {}
        for unit in src_units:
            registration = self.registrations[unit]
            ws = self.synchronizer_at(registration.data_addresses[0])
            for var, tensor in zip(registration.variables, ws.tensors, strict=True):
                # Every trainer rank registers the full, unsharded tensor.
                assert var.mesh_shape == [1] * len(var.shape) and var.sharding_spec == [""] * len(var.shape), var.name
                assert list(tensor.shape) == list(var.shape), var.name
                full[var.name] = tensor
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
                        src = src.narrow(dim, tp_rank * size, size)
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

        class RaidenController:
            def __init__(self, port=0):
                self._lock = threading.Lock()
                self._registered_shards = {}

            def start_transfer(self, **kwargs):
                world.transfers.append(kwargs)

                async def wait():
                    world.transfer(kwargs["src_units"], kwargs["dst_units"])

                return SimpleNamespace(wait=wait)

        class RaidenControllerServer:
            def __init__(self, controller):
                self.controller = controller

            def start(self):
                world.controller = self.controller
                world.servers_started += 1
                return 7777

        class RaidenControllerClientFacade:
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
            "tpu_sync.rpc.raiden_controller",
            "tpu_sync.rpc.raiden_service_pb2",
            "torch_tpu",
            "torch_tpu._internal",
            "torch_tpu._internal.sync",
        )
        mods = {name: ModuleType(name) for name in names}
        attrs: dict[str, dict[str, object]] = {
            "tpu_sync.api.common": {"RaidenId": RaidenId},
            "tpu_sync.api.torch.weight_synchronizer": {"WeightSynchronizer": WeightSynchronizer},
            "tpu_sync.rpc.raiden_controller": {
                "RaidenController": RaidenController,
                "RaidenControllerServer": RaidenControllerServer,
                "RaidenControllerClientFacade": RaidenControllerClientFacade,
                "RaidenId": RaidenId,
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

    def __init__(self, engines, weights_for_rank, events):
        self.engines, self.weights_for_rank, self.events = engines, weights_for_rank, events
        self.world_size = len(engines)

    def update_weights(self, global_steps=None, mode=None):
        assert mode == "raiden"
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


def _make_manager(engines, weights_for_rank, samplers, events, num_replicas=1, **raiden_kwargs):
    """A ``CheckpointEngineManager`` stand-in with ``num_replicas`` handles to one rollout server."""
    server = _FakeServer(samplers, events)

    async def abort_replicas():
        events.append("abort")

    async def resume_generation_replicas():
        events.append("resume")

    manager = SimpleNamespace(
        backend="raiden",
        config=SimpleNamespace(engine_kwargs={"raiden": raiden_kwargs}),
        actor_wg=_FakeActorWorkerGroup(engines, weights_for_rank, events),
        replicas=[SimpleNamespace(server_handle=_FakeActorHandle(server), world_size=len(samplers))] * num_replicas,
        abort_replicas=abort_replicas,
        resume_generation_replicas=resume_generation_replicas,
    )
    return manager, server


def _trainer_engine(rank, **engine_kwargs):
    engine = raiden.RaidenCheckpointEngine(is_master=rank == 0, **engine_kwargs)
    engine.rank = rank
    return engine


class _Linear(nn.Module):
    """Holds one bf16 ``weight``; ``flipped`` marks vllm-torchtpu's transposed ``(in, out)`` layout."""

    def __init__(self, *shape, flipped=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False)
        self._tpu_weight_flipped = flipped


class _FakeVllmModel(nn.Module):
    """One TP rank of the vLLM model: fused qkv/gate_up, column and vocab parallel on dim 0, row parallel on dim 1."""

    def __init__(self, tp, flip_o_proj=False):
        super().__init__()
        attn = nn.Module()
        attn.qkv_proj = _Linear((Q_ROWS + 2 * KV_ROWS) // tp, HIDDEN)
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

    def cols(name):
        return weights[name].chunk(tp, dim=1)[rank]

    o_proj = cols(LAYER + "self_attn.o_proj.weight")
    return {
        "model.embed_tokens.weight": rows("model.embed_tokens.weight"),
        "model.norm.weight": weights["model.norm.weight"],
        LAYER + "input_layernorm.weight": weights[LAYER + "input_layernorm.weight"],
        LAYER + "post_attention_layernorm.weight": weights[LAYER + "post_attention_layernorm.weight"],
        LAYER + "self_attn.qkv_proj.weight": torch.cat([rows(name) for name in QKV]),
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
    """A vLLM worker with the Raiden extension: rank ``rank`` of a TP=``tp`` engine serving ``model``."""
    sampler = module.vLLMRaidenWorkerExtension()
    sampler.rank = rank
    sampler.model_runner = SimpleNamespace(model=model)
    sampler.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp))
    return sampler


@pytest.fixture
def fake_tpu_sync(monkeypatch):
    """Installs the fake tpu_sync / torch_tpu, runs Raiden on CPU and fakes the RayWeightRegistry actor."""
    world = _FakeTpuSync()
    for name, module in world.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(raiden, "RAIDEN_DEVICE", "cpu")
    monkeypatch.setattr(raiden, "get_ray_weight_registry", lambda: world.registry)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "10.0.0.1")
    monkeypatch.setattr(ray, "get", _fake_ray_get)
    monkeypatch.delenv("RANK", raising=False)
    return world


@pytest.fixture
def tpu_raiden(fake_tpu_sync, monkeypatch):
    """Imports ``tpu_raiden`` against a stubbed upstream worker extension."""
    utils = ModuleType("verl.workers.rollout.vllm_rollout.utils")
    utils.vLLMColocateWorkerExtension = _WorkerExtensionBase  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "verl.workers.rollout.vllm_rollout.utils", utils)
    sys.modules.pop(TPU_RAIDEN, None)
    module = importlib.import_module(TPU_RAIDEN)
    monkeypatch.setattr(module, "get_ray_weight_registry", lambda: fake_tpu_sync.registry)
    monkeypatch.setattr(module, "_REGISTRY_POLL_INTERVAL_S", 0.0)
    yield module
    sys.modules.pop(TPU_RAIDEN, None)


# ---------------------------------------------------------------------------
# Registration, engine options and RayWeightRegistry
# ---------------------------------------------------------------------------


def test_raiden_engine_registered():
    assert ckpt_base.CheckpointEngineRegistry.get("raiden") is raiden.RaidenCheckpointEngine


def test_engine_kwargs(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    engine = raiden.RaidenCheckpointEngine(bucket_size=16, is_master=True)
    assert (engine.bucket_size, engine.is_master, engine.rank) == (16, True, 3)
    assert (engine.parallelism, engine.verify_parity, engine.tie_word_embeddings) == (8, False, None)

    # Command-line overrides may arrive as strings.
    engine = raiden.RaidenCheckpointEngine(parallelism="4", verify_parity="true", tie_word_embeddings="1")
    assert (engine.parallelism, engine.verify_parity, engine.tie_word_embeddings) == (4, True, True)

    with pytest.raises(NotImplementedError):
        engine.receive_weights()


def test_registry_state_raiden_fields():
    reg = RayWeightRegistryState()
    assert (reg.get_controller_address(), reg.get_global_shapes(), reg.get_stats(1)) == (None, {}, None)

    reg.set_controller_address("10.0.0.1:7777")
    reg.set_global_shapes({"w": [2, 3]})
    reg.set_stats(1, {"l1_norm": 1.0})
    assert reg.get_controller_address() == "10.0.0.1:7777"
    assert reg.get_global_shapes() == {"w": [2, 3]}
    assert reg.get_stats(1) == {"master": {"l1_norm": 1.0}}

    latest = list(range(3, 3 + reg.MAX_STATS_STEPS))
    for step in latest:
        reg.set_stats(step, {})
    assert sorted(reg.stats) == latest  # only the latest steps are kept

    reg.set_weights(0, ["ref"])
    reg.clear()
    assert (reg.get_weights(0), reg.get_controller_address()) == (None, None)
    assert (reg.get_global_shapes(), reg.stats) == ({}, {})


def test_get_ray_weight_registry_finds_creates_or_loses_race(monkeypatch):
    existing, created = object(), object()
    lookups: list = []
    create_error: list = []
    options_calls = []

    def get_actor(name, namespace=None):
        assert (name, namespace) == ("RayWeightRegistry", "verl")
        result = lookups.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def options(**kwargs):
        options_calls.append(kwargs)

        def remote():
            if create_error:
                raise create_error[0]
            return created

        return SimpleNamespace(remote=remote)

    monkeypatch.setattr(ray, "get_actor", get_actor)
    monkeypatch.setattr(ray_weight_registry, "RayWeightRegistry", SimpleNamespace(options=options))

    lookups[:] = [existing]
    assert ray_weight_registry.get_ray_weight_registry() is existing
    assert options_calls == []

    lookups[:] = [ValueError("Failed to look up actor")]
    assert ray_weight_registry.get_ray_weight_registry() is created
    assert options_calls == [{"name": "RayWeightRegistry", "namespace": "verl", "lifetime": "detached"}]

    # Another process creates the actor between the lookup and the creation.
    lookups[:] = [ValueError("Failed to look up actor"), existing]
    create_error[:] = [ValueError("Actor name is already taken")]
    assert ray_weight_registry.get_ray_weight_registry() is existing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_filter_tied_embeddings():
    items = [("model.embed_tokens.weight", 0), ("lm_head.weight", 1), ("model.norm.weight", 2)]
    assert raiden.filter_tied_embeddings(items, tie_word_embeddings=True) == [items[0], items[2]]
    assert raiden.filter_tied_embeddings(items, tie_word_embeddings=False) == items
    torchtitan_names = [("model.lm_head.weight", 1), ("model.tok_embeddings.weight", 0)]
    assert raiden.filter_tied_embeddings(torchtitan_names) == [("model.tok_embeddings.weight", 0)]
    # Without the input embedding there is nothing to copy lm_head from: keep it.
    assert raiden.filter_tied_embeddings([("lm_head.weight", 1)], tie_word_embeddings=True) == [("lm_head.weight", 1)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" Yes ", True), ("on", True), ("0", False), ("false", False), ("", False)]
    + [("off", False), (True, True), (False, False), (1, True), (0, False)],
)
def test_as_bool(value, expected):
    assert raiden._as_bool(value) is expected


def test_tile_alignment_and_skip_tiling_api():
    assert raiden.raiden_is_tile_aligned([8, 128])
    assert raiden.raiden_is_tile_aligned([2, 16, 256])
    assert not raiden.raiden_is_tile_aligned([4, 128])
    assert not raiden.raiden_is_tile_aligned([8, 64])
    assert not raiden.raiden_is_tile_aligned([1024])

    # The setter's name differs across tpu_sync builds; builds with neither are left alone.
    test_only_api = SimpleNamespace(test_only_set_skip_tiling=mock.Mock())
    public_api = SimpleNamespace(set_skip_tiling=mock.Mock())
    raiden.apply_raiden_skip_tiling(test_only_api, [True])
    raiden.apply_raiden_skip_tiling(public_api, [False])
    raiden.apply_raiden_skip_tiling(SimpleNamespace(), [True])
    test_only_api.test_only_set_skip_tiling.assert_called_once_with([True])
    public_api.set_skip_tiling.assert_called_once_with([False])


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
    sanitized = raiden.validate_and_sanitize_tensors(named, device=torch.device("cpu"))
    assert [name for name, _ in sanitized] == ["param", "transposed"]
    assert all(type(t) is torch.Tensor and t.is_contiguous() and t.device.type == "cpu" for _, t in sanitized)
    assert torch.equal(sanitized[0][1], weight)
    assert torch.equal(sanitized[1][1], weight.t())


def test_compute_tensor_stats():
    weights = _trainer_weights()
    names = ("model.embed_tokens.weight", "model.norm.weight")
    stats = raiden.compute_tensor_stats([(name, weights[name]) for name in names])
    assert (stats["total_numel"], stats["num_tensors"]) == (VOCAB * HIDDEN + HIDDEN, 2)
    l1 = sum(float(weights[name].float().abs().sum()) for name in names)
    l2_sq = sum(float(weights[name].float().pow(2).sum()) for name in names)
    assert stats["l1_norm"] == pytest.approx(l1, rel=1e-6)
    assert stats["l2_norm"] == pytest.approx(l2_sq**0.5, rel=1e-6)
    embed = stats["per_tensor"]["model.embed_tokens.weight"]
    assert (embed["numel"], embed["shape"], embed["dtype"]) == (VOCAB * HIDDEN, [VOCAB, HIDDEN], "torch.bfloat16")
    assert embed["l2"] == pytest.approx(embed["l2_sq"] ** 0.5)

    unnamed = raiden.compute_tensor_stats([torch.ones(2, 2)])
    assert (unnamed["total_numel"], unnamed["l1_norm"], unnamed["per_tensor"]) == (4, 4.0, {})


def test_tpu_synchronize_strict_and_lenient(fake_tpu_sync, caplog):
    raiden.tpu_synchronize(strict=True)
    assert fake_tpu_sync.tpu_syncs == 1

    fake_tpu_sync.fail_tpu_sync = True
    with pytest.raises(RuntimeError, match="TPU synchronization failed"):
        raiden.tpu_synchronize(strict=True)
    with caplog.at_level(logging.WARNING, logger=raiden.__name__):
        raiden.tpu_synchronize()
    assert "Could not synchronize via torch_tpu" in caplog.text


def test_setup_raiden_controller_publishes_address(fake_tpu_sync):
    state = fake_tpu_sync.registry_state
    state.set_controller_address("10.9.9.9:1")  # left behind in the detached actor by a previous job

    controller, server, address = raiden.setup_raiden_controller()
    assert address == "10.0.0.1:7777"
    assert server.controller is controller is fake_tpu_sync.controller
    assert state.get_controller_address() == address


def test_setup_raiden_controller_wraps_registry_errors(fake_tpu_sync, monkeypatch):
    def fail(address):
        raise ValueError("registry unavailable")

    broken = _FakeActorHandle(SimpleNamespace(set_controller_address=fail))
    monkeypatch.setattr(raiden, "get_ray_weight_registry", lambda: broken)
    with pytest.raises(RuntimeError, match=r"Failed to store RaidenController address \(10\.0\.0\.1:7777\)") as err:
        raiden.setup_raiden_controller()
    assert isinstance(err.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Trainer side: RaidenCheckpointEngine.send_weights
# ---------------------------------------------------------------------------


def test_send_weights_registers_full_tensors_and_recreates_synchronizer(fake_tpu_sync):
    raiden.setup_raiden_controller()
    weights = _trainer_weights()
    names = sorted(weights)  # lm_head is sent unless the model is known to tie it
    engine = _trainer_engine(rank=0)
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))

    (ws,) = fake_tpu_sync.synchronizers
    assert [t.data_ptr() for t in ws.tensors] == [weights[n].data_ptr() for n in names]  # bound in place
    assert (ws.parallelism, ws.bind_ip) == (8, "10.0.0.1")
    assert ws.kwargs == {"unsafe_skip_buffer_lock": True, "auto_h2d": False}
    assert ws.d2h_calls == 1  # staged in host memory before send_weights returns
    registration = fake_tpu_sync.registrations[RaidenId("trainer", "0", "weights")]
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
    # controller registration; the next sync rebinds the new tensors and stages them, without re-registering.
    engine.release_sync_buffers()
    assert engine._trainer_raiden_ws is ws
    assert (ws.unbind_calls, ws.bound, engine._bound_tensors) == (1, False, None)
    new_weights = _trainer_weights(seed=1)
    asyncio.run(engine.send_weights(iter(new_weights.items()), global_steps=2))
    assert fake_tpu_sync.synchronizers == [ws]
    assert (ws.bind_calls, ws.bound, ws.d2h_calls) == (1, True, 2)
    assert [t.data_ptr() for t in ws.tensors] == [new_weights[n].data_ptr() for n in names]
    assert fake_tpu_sync.registrations[RaidenId("trainer", "0", "weights")] is registration

    # Without a release in between (a sync that failed after send_weights), the synchronizer is rebound too.
    asyncio.run(engine.send_weights(iter(new_weights.items()), global_steps=3))
    assert fake_tpu_sync.synchronizers == [ws] and ws.bind_calls == 2

    # A changed tensor signature (here: a dropped tensor) re-creates and re-registers the synchronizer.
    engine.release_sync_buffers()
    fewer = {n: t for n, t in new_weights.items() if n != "lm_head.weight"}
    asyncio.run(engine.send_weights(iter(fewer.items()), global_steps=4))
    assert len(fake_tpu_sync.synchronizers) == 2
    new_ws = fake_tpu_sync.synchronizers[1]
    assert engine._trainer_raiden_ws is new_ws and new_ws.d2h_calls == 1
    registration = fake_tpu_sync.registrations[RaidenId("trainer", "0", "weights")]
    assert registration.data_addresses == [f"10.0.0.1:{new_ws.local_port}"]
    assert [var.name for var in registration.variables] == sorted(fewer)

    engine.finalize()
    assert (engine._trainer_raiden_ws, engine._registered_signature, engine._bound_tensors) == (None, None, None)


def test_release_sync_buffers_is_noop_when_disabled_or_before_first_sync(fake_tpu_sync):
    raiden.setup_raiden_controller()
    engine = _trainer_engine(rank=0, release_buffers_after_sync=False)
    assert engine.release_sync_buffers() == {}  # nothing bound yet

    weights = _trainer_weights()
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))
    (ws,) = fake_tpu_sync.synchronizers
    assert engine.release_sync_buffers() == {}
    assert (ws.unbind_calls, ws.bound) == (0, True)
    assert engine._bound_tensors is not None  # the send tensors stay resident between syncs


def test_release_sync_buffers_falls_back_to_destroying_synchronizer_without_unbind(fake_tpu_sync, caplog):
    raiden.setup_raiden_controller()
    engine = _trainer_engine(rank=0)
    weights = _trainer_weights()
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=1))
    (ws,) = fake_tpu_sync.synchronizers
    del type(ws).unbind_weights  # an older tpu_sync build

    with caplog.at_level(logging.WARNING, logger=raiden.__name__):
        engine.release_sync_buffers()
        engine.release_sync_buffers()  # idempotent
    assert engine._trainer_raiden_ws is None and engine._registered_signature is None
    assert caplog.text.count("has no unbind_weights()") == 1  # warned once

    # The next sync creates and registers a new synchronizer.
    asyncio.run(engine.send_weights(iter(weights.items()), global_steps=2))
    assert len(fake_tpu_sync.synchronizers) == 2 and engine._trainer_raiden_ws is fake_tpu_sync.synchronizers[1]


@pytest.mark.parametrize(
    ("kwargs", "env", "expected"),
    [
        ({}, None, True),
        ({"release_buffers_after_sync": False}, None, False),
        ({"release_buffers_after_sync": "0"}, None, False),
        ({"release_buffers_after_sync": False}, "1", True),  # the environment variable wins
        ({}, "false", False),
        ({}, "", True),  # unset / empty is ignored
    ],
)
def test_release_buffers_flag_from_kwargs_and_env(monkeypatch, kwargs, env, expected):
    if env is None:
        monkeypatch.delenv("VERL_RAIDEN_RELEASE_BUFFERS", raising=False)
    else:
        monkeypatch.setenv("VERL_RAIDEN_RELEASE_BUFFERS", env)
    assert raiden.RaidenCheckpointEngine(**kwargs).release_buffers_after_sync is expected


def test_send_weights_drops_tied_lm_head(fake_tpu_sync):
    raiden.setup_raiden_controller()
    engine = _trainer_engine(rank=0, tie_word_embeddings=True)
    asyncio.run(engine.send_weights(_trainer_weights(), global_steps=1))  # a dict works too

    registration = fake_tpu_sync.registrations[RaidenId("trainer", "0", "weights")]
    assert [var.name for var in registration.variables] == sorted(set(TRAINER_SHAPES) - {"lm_head.weight"})
    assert "lm_head.weight" not in fake_tpu_sync.registry_state.get_global_shapes()


# ---------------------------------------------------------------------------
# Driver side: update_raiden_weights, end to end through the fake transfer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_update_raiden_weights_end_to_end(fake_tpu_sync, tpu_raiden, caplog, tie_word_embeddings):
    tp = trainer_ranks = 2
    weights = {"current": _trainer_weights(seed=0)}
    engines = [
        _trainer_engine(rank, verify_parity=True, tie_word_embeddings=tie_word_embeddings)
        for rank in range(trainer_ranks)
    ]
    models = [_FakeVllmModel(tp) for _ in range(tp)]
    samplers = [_make_sampler(tpu_raiden, rank, tp, model) for rank, model in enumerate(models)]
    events: list = []
    manager, server = _make_manager(
        engines, lambda rank: iter(weights["current"].items()), samplers, events, verify_parity=True, parallelism=4
    )

    # Initial sync (step 0): nothing to abort and no parity check.
    with caplog.at_level(logging.INFO, logger=raiden.__name__):
        metrics = asyncio.run(raiden.update_raiden_weights(manager, global_steps=0))
    assert events == [
        "update_weights",
        "init_raiden_sync_on_worker",
        "release_sync_buffers",
        "install_raiden_weights",
        "set_global_steps",
        "resume",
    ]
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights["current"], rank, tp, lm_head_from_embed=tie_word_embeddings))
    assert server.global_steps == 0
    assert set(metrics) == TIMING_KEYS and min(metrics.values()) >= 0
    (transfer,) = fake_tpu_sync.transfers
    assert transfer["src_units"] == [RaidenId("trainer", str(r), "weights") for r in range(trainer_ranks)]
    assert transfer["dst_units"] == [RaidenId("sampler", str(r), "weights") for r in range(tp)]
    assert (transfer["parallelism"], transfer["req_id"], transfer["dst_mem_type"]) == (4, "verl_step_0", "DRAM")
    assert [sampler._raiden_ws.listener_port for sampler in samplers] == [12000, 12001]
    assert [sampler._raiden_ws.parallelism for sampler in samplers] == [4, 4]
    assert ("lm_head.weight" in fake_tpu_sync.registry_state.get_global_shapes()) is not tie_word_embeddings
    # Released after the transfer: the send tensors are unbound, the synchronizers are kept for the next sync.
    assert all(not engine._trainer_raiden_ws.bound and engine._bound_tensors is None for engine in engines)

    # Next step: new weights. The controller, the rollout registrations and the trainer synchronizers are
    # reused; every trainer rank rebinds and stages its new tensors.
    weights["current"] = _trainer_weights(seed=1)
    events.clear()
    with caplog.at_level(logging.INFO, logger=raiden.__name__):
        metrics = asyncio.run(raiden.update_raiden_weights(manager, global_steps=1))
    assert events == [
        "abort",
        "update_weights",
        "init_raiden_sync_on_worker",
        "release_sync_buffers",
        "install_raiden_weights",
        "set_global_steps",
        "get_model_weights_stats",
        "resume",
    ]
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights["current"], rank, tp, lm_head_from_embed=tie_word_embeddings))
    assert server.global_steps == 1
    assert set(metrics) == TIMING_KEYS
    assert fake_tpu_sync.servers_started == 1
    assert [sampler._raiden_ws.h2d_calls for sampler in samplers] == [2, 2]
    sampler_synchronizers = [sampler._raiden_ws for sampler in samplers]
    trainer_synchronizers = [ws for ws in fake_tpu_sync.synchronizers if ws not in sampler_synchronizers]
    assert [ws.d2h_calls for ws in trainer_synchronizers] == [2] * trainer_ranks  # one synchronizer per rank
    bind_state = [(ws.bind_calls, ws.unbind_calls, ws.bound) for ws in trainer_synchronizers]
    assert bind_state == [(1, 2, False)] * trainer_ranks  # rebound once, unbound after each transfer
    assert "RAIDEN PARITY VERIFIED | Step 1" in caplog.text
    assert "MISMATCH" not in caplog.text


def test_update_raiden_weights_installs_tp_shards_into_transposed_weights(fake_tpu_sync, tpu_raiden):
    """Every trainer rank registers the full tensors; vLLM ranks get TP shards, also of transposed parameters."""
    tp, trainer_ranks = 2, 4
    weights = _trainer_weights()
    models = [_FakeVllmModel(tp, flip_o_proj=True) for _ in range(tp)]
    manager, _ = _make_manager(
        [_trainer_engine(rank) for rank in range(trainer_ranks)],
        lambda rank: iter(weights.items()),
        [_make_sampler(tpu_raiden, rank, tp, model) for rank, model in enumerate(models)],
        [],
    )
    asyncio.run(raiden.update_raiden_weights(manager, global_steps=1))

    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights, rank, tp, flip_o_proj=True))
    for rank in range(trainer_ranks):
        registration = fake_tpu_sync.registrations[RaidenId("trainer", str(rank), "weights")]
        assert registration.mesh_shape == [1, 1]
        assert [(var.name, var.shape) for var in registration.variables] == [
            (name, list(TRAINER_SHAPES[name])) for name in sorted(TRAINER_SHAPES)
        ]


def test_parity_check_reports_mismatch(fake_tpu_sync, tpu_raiden, caplog):
    tp = 2
    weights = _trainer_weights()
    samplers = [_make_sampler(tpu_raiden, rank, tp, _FakeVllmModel(tp)) for rank in range(tp)]
    manager, _ = _make_manager(
        [_trainer_engine(rank, verify_parity=True) for rank in range(2)],
        lambda rank: iter(weights.items()),
        samplers,
        [],
        verify_parity=True,
    )
    asyncio.run(raiden.update_raiden_weights(manager, global_steps=1))

    samplers[1]._raiden_staging["model.embed_tokens.weight"].zero_()
    with caplog.at_level(logging.INFO, logger=raiden.__name__):
        asyncio.run(raiden._verify_parity_async(manager, global_steps=1))
    assert "RAIDEN PARITY MISMATCH | Step 1" in caplog.text
    assert "model.embed_tokens.weight" in caplog.text


def test_update_raiden_weights_requires_single_replica(fake_tpu_sync):
    manager, _ = _make_manager([], lambda rank: iter(()), [], [], num_replicas=2)
    with pytest.raises(NotImplementedError, match="single rollout replica"):
        asyncio.run(raiden.update_raiden_weights(manager, global_steps=1))
    assert fake_tpu_sync.servers_started == 0


def test_wait_for_registration_times_out():
    controller = SimpleNamespace(_lock=threading.Lock(), _registered_shards={RaidenId("trainer", "0", "weights"): 0})
    src_units = [RaidenId("trainer", str(rank), "weights") for rank in range(2)]
    dst_units = [RaidenId("sampler", "0", "weights")]
    with pytest.raises(RuntimeError, match="Timeout") as excinfo:
        asyncio.run(raiden._wait_for_registration(controller, src_units, dst_units, timeout_s=0.01))
    message = str(excinfo.value)
    assert "Missing Trainer: [RaidenId(job_name='trainer', job_replica_id='1', data_name='weights')]" in message
    assert "Missing Sampler: [RaidenId(job_name='sampler', job_replica_id='0', data_name='weights')]" in message


def test_manager_hook_routes_raiden_backend(monkeypatch):
    calls = []

    async def fake_update_raiden_weights(manager, global_steps=None):
        calls.append((manager.backend, global_steps))
        return {"timing_s/tpu-sync/total_sync": 1.0}

    monkeypatch.setattr(raiden, "update_raiden_weights", fake_update_raiden_weights)
    assert getattr(ckpt_base, "_verl_tpu_ckpt_patched", False)
    result = ckpt_base.CheckpointEngineManager.update_weights(SimpleNamespace(backend="raiden"), global_steps=3)
    assert result == {"timing_s/tpu-sync/total_sync": 1.0}
    assert calls == [("raiden", 3)]


# ---------------------------------------------------------------------------
# Rollout side: vLLMRaidenWorkerExtension
# ---------------------------------------------------------------------------


def test_sampler_sharding(tpu_raiden):
    shard = tpu_raiden._sampler_sharding
    assert shard(LAYER + "self_attn.o_proj.weight", [8, 16], 2) == (["", "tp"], [8, 8])
    assert shard(LAYER + "mlp.down_proj.weight", [8, 16], 4) == (["", "tp"], [8, 4])
    assert shard(LAYER + "self_attn.q_proj.weight", [16, 8], 2) == (["tp", ""], [8, 8])
    assert shard("model.embed_tokens.weight", [15, 8], 2) == (["", ""], [15, 8])  # indivisible: replicated
    assert shard("model.norm.weight", [8], 2) == ([""], [8])


def test_sampler_init_allocates_tp_shards_and_registers(fake_tpu_sync, tpu_raiden):
    fake_tpu_sync.registry_state.set_controller_address("10.0.0.2:7777")
    fake_tpu_sync.registry_state.set_global_shapes(
        {
            "model.norm.weight": [128],
            LAYER + "self_attn.o_proj.weight": [128, 256],
            LAYER + "self_attn.k_proj.weight": [16, 64],
        }
    )
    sampler = _make_sampler(tpu_raiden, rank=1, tp=2, model=_FakeVllmModel(2))
    assert sampler.init_raiden_sync_on_worker(parallelism=4) is True

    (ws,) = fake_tpu_sync.synchronizers
    assert (ws.listener_port, ws.parallelism, ws.bind_ip) == (12001, 4, "10.0.0.1")
    assert [tuple(t.shape) for t in ws.tensors] == [(8, 64), (128, 128), (128,)]
    assert all(t.dtype == torch.bfloat16 for t in ws.tensors)
    assert ws.skip_tiling == [False, True, False]  # only (8, 128)-tile aligned 2-D shards skip the tiling pass
    registration = fake_tpu_sync.registrations[RaidenId("sampler", "1", "weights")]
    assert registration.address == "10.0.0.2:7777"
    assert (registration.mesh_shape, registration.mesh_axes) == ([1, 2], ["fsdp", "tp"])
    assert registration.data_addresses == [f"10.0.0.1:{ws.local_port}"]
    assert registration.listener_address == "10.0.0.1:12001"
    assert [(v.name, v.shape, v.mesh_shape, v.sharding_spec) for v in registration.variables] == [
        (LAYER + "self_attn.k_proj.weight", [16, 64], [2, 1], ["tp", ""]),
        (LAYER + "self_attn.o_proj.weight", [128, 256], [1, 2], ["", "tp"]),
        ("model.norm.weight", [128], [1], [""]),
    ]

    # Later syncs reuse the receive buffers and the registration.
    assert sampler.init_raiden_sync_on_worker(parallelism=4) is True
    assert len(fake_tpu_sync.synchronizers) == 1


def test_sampler_init_times_out_without_registry(fake_tpu_sync, tpu_raiden, monkeypatch):
    monkeypatch.setattr(tpu_raiden, "_REGISTRY_POLL_ATTEMPTS", 2)
    sampler = _make_sampler(tpu_raiden, rank=0, tp=2, model=_FakeVllmModel(2))
    with pytest.raises(RuntimeError, match="No RaidenController address"):
        sampler.init_raiden_sync_on_worker()
    assert fake_tpu_sync.synchronizers == []


def test_sampler_methods_before_init(tpu_raiden):
    sampler = _make_sampler(tpu_raiden, rank=0, tp=2, model=_FakeVllmModel(2))
    assert sampler.install_raiden_weights() == {}
    with pytest.raises(RuntimeError, match="only supported for Raiden after initialization"):
        sampler.get_model_weights_stats()
