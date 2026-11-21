# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU-side tests for the Enflame ECCL checkpoint engine.

The module refuses to import without a live GCU (it owns the ``nccl`` key, so it
must not shadow the stock NCCL engine on a CUDA host). These tests therefore
fake ``torch.gcu`` and the platform probe to exercise registration and the
rank layout; moving weights over ECCL needs real hardware.
"""

import asyncio
import importlib
import logging
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("ray", reason="the ECCL engine imports ray.util.collective")
torch = pytest.importorskip("torch")

# verl.checkpoint_engine.base imports verl.workers.rollout.utils, which imports
# uvicorn and fastapi at module scope. They are not in the CPU test image and are
# unused here, so stand them in (same approach as the TPU checkpoint engine tests).
# The stub needs a __spec__: transformers probes fastapi with importlib.util.find_spec().
for _mod_name in ("uvicorn", "fastapi"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub = ModuleType(_mod_name)
            _stub.FastAPI = object  # type: ignore[attr-defined]
            _stub.Server = object  # type: ignore[attr-defined]
            _stub.Config = object  # type: ignore[attr-defined]
            _stub.__spec__ = ModuleSpec(_mod_name, None)
            sys.modules[_mod_name] = _stub

from verl.checkpoint_engine import CheckpointEngineRegistry  # noqa: E402
from verl_hardware_plugin.accelerators.enflame import platform_enflame  # noqa: E402

MODULE = "verl_hardware_plugin.accelerators.enflame.engines.eccl_checkpoint_engine"
REGISTRATION = "verl_hardware_plugin.accelerators.enflame.registration"
REGISTRY_KEYS = ("eccl", "nccl")
_MISSING = object()


class FakeGCU:
    """Minimal ``torch.gcu`` used by the module import-time guard and the
    ``init_process_group`` diagnostics."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def synchronize() -> None:
        return

    @staticmethod
    def empty_cache() -> None:
        return


@pytest.fixture
def imported_engine(monkeypatch):
    """Import the engine with a fake GCU, restoring the registry and modules."""
    saved = {key: CheckpointEngineRegistry._registry.get(key, _MISSING) for key in REGISTRY_KEYS}
    saved_errors = dict(CheckpointEngineRegistry._import_errors)
    monkeypatch.setattr(torch, "gcu", FakeGCU(), raising=False)
    monkeypatch.setattr(platform_enflame, "_ensure_torch_gcu", lambda: True)
    sys.modules.pop(MODULE, None)
    try:
        yield importlib.import_module(MODULE)
    finally:
        sys.modules.pop(MODULE, None)
        for key, engine in saved.items():
            if engine is _MISSING:
                CheckpointEngineRegistry._registry.pop(key, None)
            else:
                CheckpointEngineRegistry._registry[key] = engine
        CheckpointEngineRegistry._import_errors.clear()
        CheckpointEngineRegistry._import_errors.update(saved_errors)


def test_registered_under_eccl_and_nccl(imported_engine):
    assert CheckpointEngineRegistry.get("eccl") is imported_engine.ECCLCheckpointEngine
    # "nccl" is a compatibility alias used by explicit async scripts; it must
    # reach this engine, exactly as it reaches HCCL on Ascend.
    assert CheckpointEngineRegistry.get("nccl") is imported_engine.ECCLCheckpointEngine
    assert CheckpointEngineRegistry.get("eccl") is CheckpointEngineRegistry.get("nccl")


@pytest.mark.parametrize(("ensure_gcu", "is_available"), [(True, False), (False, False)])
def test_import_fails_without_a_usable_gcu(monkeypatch, ensure_gcu, is_available):
    sentinel = object()
    monkeypatch.setitem(CheckpointEngineRegistry._registry, "nccl", sentinel)
    sys.modules.pop(MODULE, None)
    monkeypatch.setattr(torch, "gcu", SimpleNamespace(is_available=lambda: is_available), raising=False)
    monkeypatch.setattr(platform_enflame, "_ensure_torch_gcu", lambda: ensure_gcu)

    with pytest.raises(ImportError, match="no Enflame GCU"):
        importlib.import_module(MODULE)

    assert CheckpointEngineRegistry._registry["nccl"] is sentinel


@pytest.mark.parametrize(
    ("gcu_expected", "log_level"),
    [(True, logging.WARNING), (False, logging.DEBUG)],
)
def test_registration_records_engine_import_failure(monkeypatch, caplog, gcu_expected, log_level):
    """Exercise the registration except-branch: record the error, keep nccl, log by host type."""
    saved_errors = dict(CheckpointEngineRegistry._import_errors)
    registration = importlib.import_module(REGISTRATION)
    engines_package = importlib.import_module(REGISTRATION.rsplit(".", 1)[0] + ".engines")
    module_state = dict(registration.__dict__)
    saved_logger = registration.logger
    saved_level = saved_logger.level
    sentinel = object()
    monkeypatch.setitem(CheckpointEngineRegistry._registry, "nccl", sentinel)
    CheckpointEngineRegistry._import_errors.pop("eccl_checkpoint_engine", None)
    monkeypatch.setenv("VERL_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setattr(platform_enflame, "gcu_available", lambda: gcu_expected)
    # Drop both caches: the parent package attribute and sys.modules. Otherwise
    # ``from ...engines import eccl_checkpoint_engine`` returns the cached class.
    monkeypatch.delattr(engines_package, "eccl_checkpoint_engine", raising=False)

    try:
        # A None entry makes importing the engine raise ImportError, without
        # depending on a real GCU probe.
        with mock.patch.dict(sys.modules, {MODULE: None}):
            with caplog.at_level(logging.DEBUG):
                importlib.reload(registration)

        recorded = CheckpointEngineRegistry._import_errors["eccl_checkpoint_engine"]
        assert isinstance(recorded, ImportError)
        assert CheckpointEngineRegistry._registry["nccl"] is sentinel

        messages = [
            record
            for record in caplog.records
            if record.name == registration.__name__
            and "ECCL checkpoint engine registration failed" in record.getMessage()
        ]
        assert messages
        assert messages[-1].levelno == log_level
    finally:
        registration.__dict__.clear()
        registration.__dict__.update(module_state)
        # reload() re-read VERL_LOGGING_LEVEL; a logger is a process-wide
        # singleton, so put its level back explicitly.
        saved_logger.setLevel(saved_level)
        CheckpointEngineRegistry._import_errors.clear()
        CheckpointEngineRegistry._import_errors.update(saved_errors)


@pytest.fixture
def make_engine(imported_engine, monkeypatch):
    """Build an engine whose platform resolves to GCU, whatever host runs the tests."""
    platform = SimpleNamespace(device_name="gcu", communication_backend_name=lambda: "eccl")
    monkeypatch.setattr(imported_engine, "get_platform", lambda: platform)

    def build(bucket_size=16, **kwargs):
        return imported_engine.ECCLCheckpointEngine(bucket_size=bucket_size, **kwargs)

    return build


def test_a_rank_outside_the_group_does_not_join_it(make_engine, imported_engine, monkeypatch):
    calls = []
    monkeypatch.setattr(imported_engine.collective, "init_collective_group", lambda *a, **k: calls.append("init"))
    monkeypatch.setattr(imported_engine.collective, "barrier", lambda *a, **k: calls.append("barrier"))
    engine = make_engine()

    engine.init_process_group(rank=-1, world_size=2, master_metadata=object())

    assert calls == []
    assert engine.rank == -1


def test_a_rank_outside_the_group_still_drains_the_weight_generator(make_engine):
    engine = make_engine()
    engine.init_process_group(rank=-1, world_size=2, master_metadata=object())
    seen = []

    # get_per_tensor_param() hands the engine a sync generator, so the drain
    # path must consume it synchronously, exactly like the stock NCCL engine.
    def weights():
        for name in ("a", "b"):
            seen.append(name)
            yield name, torch.zeros(1)

    asyncio.run(engine.send_weights(weights()))

    assert seen == ["a", "b"]


def test_the_zmq_subscription_is_created_once_across_weight_updates(make_engine, imported_engine, monkeypatch):
    engine = make_engine()
    engine.rank, engine.world_size = 1, 2
    metadata = SimpleNamespace(zmq_ip="127.0.0.1", zmq_port=5555)
    connects = []

    monkeypatch.setattr(imported_engine.collective, "is_group_initialized", lambda name: True)
    monkeypatch.setattr(imported_engine.collective, "barrier", lambda name: None)

    def connect(meta):
        connects.append(meta)
        engine.socket = object()

    monkeypatch.setattr(engine, "_connect_zmq_client", connect)

    engine.init_process_group(rank=1, world_size=2, master_metadata=metadata)
    engine.init_process_group(rank=1, world_size=2, master_metadata=metadata)

    assert connects == [metadata]


def test_prepare_allocates_two_buffers_on_the_gcu_device(make_engine, imported_engine, monkeypatch):
    allocations = []

    def zeros(size, dtype=None, device=None):
        allocations.append((size, dtype, device))
        return object()

    monkeypatch.setattr(imported_engine.torch, "zeros", zeros)
    engine = make_engine(bucket_size=64)

    assert engine.prepare() is None
    assert allocations == [(64, torch.uint8, "gcu:0"), (64, torch.uint8, "gcu:0")]


def test_collective_backend_follows_the_platform(imported_engine, monkeypatch):
    for backend in ("flagcx", "eccl"):
        platform = SimpleNamespace(device_name="gcu", communication_backend_name=lambda backend=backend: backend)
        monkeypatch.setattr(imported_engine, "get_platform", lambda platform=platform: platform)

        engine = imported_engine.ECCLCheckpointEngine(bucket_size=16)

        assert engine.collective_backend == backend


def test_build_topology_keeps_only_actor_rank0_in_the_group(imported_engine):
    metadata = [{"zmq_ip": "10.0.0.1", "zmq_port": 5555}]

    actor_kwargs, rollout_kwargs = imported_engine.ECCLCheckpointEngine.build_topology(
        actor_wg_world_size=2, rollout_world_size=3, metadata=metadata
    )

    assert actor_kwargs["rank"] == [0, -1]
    assert rollout_kwargs["rank"] == [1, 2, 3]
    assert actor_kwargs["world_size"] == [4, 4]
    assert rollout_kwargs["world_size"] == [4, 4, 4]
    assert actor_kwargs["master_metadata"] == metadata * 2
    assert rollout_kwargs["master_metadata"] == metadata * 3


def test_build_topology_requires_master_metadata(imported_engine):
    with pytest.raises(AssertionError):
        imported_engine.ECCLCheckpointEngine.build_topology(actor_wg_world_size=1, rollout_world_size=1, metadata=[])
