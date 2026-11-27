# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the mori_publish checkpoint engine with an in-process fake of MoRI-IO.

No GPU, NIC or Ray. The fake IOEngine implements READ as a byte copy on a background thread after a random delay,
reading the source only then (as a NIC does), so a snapshot that changes under a reader is seen changing. The tests
drive engines the way CheckpointEngineManager does in publish mode: prepare -> build_topology -> init_process_group
-> trainer send_weights, with the rollout's receive_weights running independently -> finalize.
"""

import asyncio
import importlib
import importlib.machinery
import itertools
import pickle
import random
import sys
import threading
import time
import types

import pytest
import torch

# verl.checkpoint_engine.base imports verl's rollout utilities, which import uvicorn and fastapi; the CPU CI does not
# install them and nothing here uses them
for _mod_name in ("uvicorn", "fastapi"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub = types.ModuleType(_mod_name)
            _stub.FastAPI = object  # type: ignore[attr-defined]
            _stub.Server = object  # type: ignore[attr-defined]
            _stub.Config = object  # type: ignore[attr-defined]
            sys.modules[_mod_name] = _stub

BUCKET = 64 << 10  # small, so a stripe spans many chunks
_ENGINES: list = []  # every engine a test creates; closed by the fixture


def _fake_mori():
    fab = {"mems": {}, "ids": itertools.count(1), "threads": [], "reads": [], "delay": 0.002}

    class Status:
        def __init__(self):
            self.done = threading.Event()

        def InProgress(self):
            return not self.done.is_set()

        def Succeeded(self):
            return self.done.is_set()

        def Message(self):
            return ""

    class EngineDesc:
        def __init__(self, key):
            self.key = key

        def pack(self):
            return pickle.dumps(self.key)

        @staticmethod
        def unpack(b):
            return EngineDesc(pickle.loads(b))

    class MemoryDesc:
        def __init__(self, mid, size):
            self.id, self.size = mid, size

        def pack(self):
            return pickle.dumps((self.id, self.size))

        @staticmethod
        def unpack(b):
            return MemoryDesc(*pickle.loads(b))

    class IOEngine:
        def __init__(self, key, config):
            self.key, self.remotes, self.uid = key, set(), itertools.count(1)

        def create_backend(self, backend_type, config):
            assert backend_type == "RDMA"

        def get_engine_desc(self):
            return EngineDesc(self.key)

        def register_remote_engine(self, desc):
            self.remotes.add(desc.key)

        def deregister_remote_engine(self, desc):
            self.remotes.discard(desc.key)

        def register_torch_tensor(self, t):
            mid = next(fab["ids"])
            fab["mems"][mid] = (self.key, t)
            return MemoryDesc(mid, t.numel())

        def deregister_memory(self, mem):
            fab["mems"].pop(mem.id, None)

        def allocate_transfer_uid(self):
            return next(self.uid)

        def read(self, local, local_off, remote, remote_off, size, uid):
            owner, src = fab["mems"][remote.id]
            assert owner in self.remotes, "read from an engine that was not registered"
            assert local_off + size <= local.size and remote_off + size <= remote.size, "out of bounds"
            dst = fab["mems"][local.id][1]
            status = Status()

            def land():
                time.sleep(random.uniform(0, fab["delay"]))
                dst[local_off : local_off + size].copy_(src[remote_off : remote_off + size])
                fab["reads"].append((owner, self.key, size))
                status.done.set()

            thread = threading.Thread(target=land, daemon=True)
            fab["threads"].append(thread)
            thread.start()
            return status

    mori = types.ModuleType("mori")
    mori.__spec__ = importlib.machinery.ModuleSpec("mori", None)
    io = types.ModuleType("mori.io")
    io.BackendType = types.SimpleNamespace(RDMA="RDMA", XGMI="XGMI")
    io.IOEngine, io.EngineDesc, io.MemoryDesc = IOEngine, EngineDesc, MemoryDesc
    io.IOEngineConfig = io.RdmaBackendConfig = lambda **kwargs: kwargs
    mori.io = io
    return mori, io, fab


@pytest.fixture
def publish_env(monkeypatch):
    """Install the fake MoRI (the engine imports it lazily); yield (engine module, fake fabric)."""
    mori, io, fab = _fake_mori()
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.io", io)
    mod = importlib.import_module("verl_hardware_plugin.accelerators.rocm.engines.mori_publish_checkpoint_engine")
    yield mod, fab
    for thread in fab["threads"]:  # let in-flight fake reads land before the next test
        thread.join()
    while _ENGINES:
        _ENGINES.pop().close()


def _make_weights(seed):
    g = torch.Generator().manual_seed(seed)
    ws = [("embed", torch.randn(300, 77, generator=g).to(torch.bfloat16))]
    dtypes = [torch.bfloat16, torch.float32, torch.int64, torch.float16]
    for i in range(30):
        n, dt = 1 + (seed * 7919 + i * 104729) % 3000, dtypes[i % 4]
        t = torch.randint(-1000, 1000, (n,), generator=g) if dt == torch.int64 else torch.randn(n, generator=g).to(dt)
        ws.append((f"layer{i}.w", t))
    ws.append(("big", torch.randn(100_000, generator=g)))  # 400 KB: spans several chunks, put together on receive
    ws.append(("tail.bias", torch.randn(3, generator=g).to(torch.bfloat16)))
    return ws


def _same(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    )


def _new(mod, **kwargs):
    engine = mod.MoriPublishCheckpointEngine(
        **{"bucket_size": BUCKET, "num_stripes": 2, "host": "127.0.0.1", "device": "cpu", "timeout_s": 30, **kwargs}
    )
    _ENGINES.append(engine)
    return engine


def _wire(actors, rollouts):
    meta = [e.prepare() for e in actors + rollouts]
    akw, rkw = type(actors[0]).build_topology(len(actors), len(rollouts), meta)
    for i, e in enumerate(actors):
        e.init_process_group(**{k: v[i] for k, v in akw.items()})
    for i, e in enumerate(rollouts):
        e.init_process_group(**{k: v[i] for k, v in rkw.items()})


async def _publish(actors, weights, step):
    await asyncio.gather(*[e.send_weights(iter(weights), global_steps=step) for e in actors])


async def _pull(engine, step, delay=0.0):
    """Keeps every tensor exactly as yielded, as a consumer that batches them before loading does. The dict keeps
    the order they came in."""
    got = {}
    async for name, tensor in engine.receive_weights(global_steps=step):
        assert name not in got, f"{name} received twice"
        got[name] = tensor
        if delay:
            await asyncio.sleep(delay)
    return got


def _assert_received(got, weights):
    assert set(got) == {name for name, _ in weights}
    for name, tensor in weights:
        assert _same(got[name], tensor), name


def test_registered_in_publish_mode(publish_env):
    mod, _ = publish_env
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("mori_publish") is mod.MoriPublishCheckpointEngine
    assert mod.MoriPublishCheckpointEngine.support_publish_weights


def test_the_trainer_publishes_with_no_reader_and_every_reader_then_gets_every_tensor(publish_env):
    """Publish mode: send_weights may not depend on a rollout worker. Three trainer ranks, two stripes: the third
    rank only drives the generator."""
    mod, fab = publish_env
    actors, rollouts = [_new(mod) for _ in range(3)], [_new(mod) for _ in range(3)]
    weights = _make_weights(1)

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, weights, step=5)
        assert fab["reads"] == [], "publishing read something"
        return await asyncio.gather(*[_pull(e, step=5) for e in rollouts])

    pulled = asyncio.run(run())
    for got in pulled:
        _assert_received(got, weights)
    # verl's SGLang adapter loads one bucket across the TP ranks of a replica: they must see the same order
    assert [list(got) for got in pulled] == [list(pulled[0])] * len(pulled)
    assert list(pulled[0]) != [name for name, _ in weights], "one stripe after the other would not use every NIC"
    # ...while each reads the chunks of a round one at a time, starting at its own stripe: the trainers' NICs
    # stop answering when every reader pulls from all of them at once
    for index, rollout in enumerate(rollouts):
        chunk_reads = [src for src, dst, size in fab["reads"] if dst == rollout._key and size > 4096]
        assert chunk_reads[0] == actors[index % 2]._key
    publishers = {actors[0]._key, actors[1]._key}
    for rollout in rollouts:  # every reader took both stripes from the trainers themselves
        assert {src for src, dst, _ in fab["reads"] if dst == rollout._key} == publishers
    assert not actors[2]._chunks


def test_float_dtype_publishes_wide_floats_narrowed_and_leaves_everything_else(publish_env):
    """A rollout that holds bf16 weights casts what it loads anyway; cast on the trainer, the bytes installed are
    the same and half as many travel."""
    mod, _ = publish_env
    actors = [_new(mod, float_dtype="bfloat16") for _ in range(2)]
    rollouts = [_new(mod, float_dtype="bfloat16")]
    weights = _make_weights(9)
    expected = [(name, t.to(torch.bfloat16) if t.dtype == torch.float32 else t) for name, t in weights]
    assert any(t.dtype == torch.float32 for _, t in weights) and any(t.dtype == torch.int64 for _, t in weights)

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, weights, step=1)
        return await _pull(rollouts[0], step=1)

    _assert_received(asyncio.run(run()), expected)


def test_reads_in_flight_bounds_the_concurrent_chunk_reads_and_keeps_the_order(publish_env):
    """More reads at once must not change what a reader yields or in which order; the bound is what keeps the
    trainers' NICs answering."""
    mod, fab = publish_env
    weights = _make_weights(10)

    async def run(reads_in_flight):
        actors = [_new(mod, num_stripes=4, reads_in_flight=reads_in_flight) for _ in range(4)]
        rollouts = [_new(mod, num_stripes=4, reads_in_flight=reads_in_flight)]
        _wire(actors, rollouts)
        await _publish(actors, weights, step=1)
        return await _pull(rollouts[0], step=1)

    # chunk reads in flight when each one starts, itself included; a status record or a manifest is smaller
    engine_cls = sys.modules["mori.io"].IOEngine
    read, statuses, in_flight = engine_cls.read, [], []

    def counted(self, *args):
        is_chunk = args[4] > 4096
        if is_chunk:
            in_flight.append(sum(status.InProgress() for status in statuses) + 1)
        status = read(self, *args)
        if is_chunk:
            statuses.append(status)
        return status

    engine_cls.read = counted
    one = asyncio.run(run(1))
    assert max(in_flight) == 1
    in_flight.clear()
    three = asyncio.run(run(3))
    assert 1 < max(in_flight) <= 3
    _assert_received(three, weights)
    assert list(three) == list(one)


def test_a_reader_that_starts_first_waits_for_the_publication(publish_env):
    mod, _ = publish_env
    actors, rollouts = [_new(mod) for _ in range(2)], [_new(mod) for _ in range(2)]
    weights = _make_weights(2)

    async def run():
        _wire(actors, rollouts)
        pulls = [asyncio.ensure_future(_pull(e, step=1)) for e in rollouts]
        await asyncio.sleep(0.05)
        assert not any(p.done() for p in pulls)
        await _publish(actors, weights, step=1)
        return await asyncio.gather(*pulls)

    for got in asyncio.run(run()):
        _assert_received(got, weights)


def test_the_next_sync_delivers_its_own_weights_not_the_snapshot_still_in_place(publish_env):
    """The previous snapshot stays readable after finalize; a reader of the next sync must wait for the new one."""
    mod, _ = publish_env
    actors, rollouts = [_new(mod) for _ in range(2)], [_new(mod) for _ in range(2)]
    first, second = _make_weights(3), _make_weights(4)

    async def sync(weights, step, readers_first):
        _wire(actors, rollouts)
        if readers_first:
            pulls = [asyncio.ensure_future(_pull(e, step=step)) for e in rollouts]
            await asyncio.sleep(0.05)
            await _publish(actors, weights, step=step)
            got = await asyncio.gather(*pulls)
        else:
            await _publish(actors, weights, step=step)
            got = await asyncio.gather(*[_pull(e, step=step) for e in rollouts])
        for e in actors + rollouts:
            e.finalize()
        return got

    async def run():
        return await sync(first, 1, readers_first=False), await sync(second, 2, readers_first=True)

    got_first, got_second = asyncio.run(run())
    for got in got_first:
        _assert_received(got, first)
    for got in got_second:
        _assert_received(got, second)


def test_a_stripe_republished_while_it_is_read_fails_the_reader(publish_env):
    """The manager publishes the next version only after every reader committed; if that is ever violated, a reader
    must not hand over a mix of two versions."""
    mod, _ = publish_env
    actors, rollouts = [_new(mod)], [_new(mod)]

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, _make_weights(5), step=1)
        pull = asyncio.ensure_future(_pull(rollouts[0], step=1, delay=0.01))
        await asyncio.sleep(0.05)
        assert not pull.done()
        await _publish(actors, _make_weights(6), step=1)
        await pull

    with pytest.raises(RuntimeError, match="republished while it was read|does not match"):
        asyncio.run(run())


def test_a_chunk_that_differs_from_what_was_published_fails_the_reader(publish_env):
    """What the NIC delivers is checked against the checksum the trainer put in the manifest."""
    mod, _ = publish_env
    actors, rollouts = [_new(mod)], [_new(mod)]

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, _make_weights(8), step=1)
        actors[0]._chunks[1][0][:64] ^= 0xFF  # after publication: the manifest still describes the old bytes
        await _pull(rollouts[0], step=1)

    with pytest.raises(RuntimeError, match="does not match the checksum in its manifest"):
        asyncio.run(run())


def test_the_checksum_covers_the_bytes_after_the_last_whole_word(publish_env):
    mod, _ = publish_env
    actors, rollouts = [_new(mod, num_stripes=1)], [_new(mod, num_stripes=1)]

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, [("bias", torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16))], step=1)  # 6 bytes
        actors[0]._chunks[0][0][0] ^= 1
        await _pull(rollouts[0], step=1)

    with pytest.raises(RuntimeError, match="does not match the checksum in its manifest"):
        asyncio.run(run())


def test_a_reader_cancelled_during_a_read_is_not_reused_and_keeps_its_registrations(publish_env, monkeypatch):
    """The NIC goes on writing into the reader's regions after its task is gone."""
    mod, fab = publish_env
    actors, rollouts = [_new(mod, num_stripes=1)], [_new(mod, num_stripes=1)]
    monkeypatch.setattr(random, "uniform", lambda low, high: 0.3)  # every fake read is still landing at the cancel

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, [("w", torch.ones(8))], step=1)
        task = asyncio.create_task(_pull(rollouts[0], step=1))
        while not fab["threads"]:  # the first read is on the wire
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    reader = rollouts[0]
    assert "cancelled while in flight" in reader._unusable
    with pytest.raises(RuntimeError, match="engine unusable"):
        reader.prepare()
    header_id = reader._header_mem.id
    reader.close()
    assert header_id in fab["mems"], "close() deregistered a region a read may still be landing in"
    kept_engine, kept_status = mod._KEEPALIVE[-1]
    assert kept_engine is reader and kept_status.InProgress(), "the read still landing needs its status object too"
    with pytest.raises(RuntimeError, match="engine unusable"):  # nothing more goes out through this engine
        asyncio.run(reader._read(None, 0, None, 0, 8, "anything"))


def test_a_read_that_times_out_keeps_its_status_and_stops_the_reads_queued_behind_it(publish_env, monkeypatch):
    mod, fab = publish_env
    actors, rollouts = [_new(mod)], [_new(mod, timeout_s=0.2)]
    monkeypatch.setattr(random, "uniform", lambda low, high: 1.0)  # no fake read lands within the timeout

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, _make_weights(9), step=1)
        await _pull(rollouts[0], step=1)

    with pytest.raises(TimeoutError, match="did not finish in 0.2s"):
        asyncio.run(run())
    kept_engine, kept_status = mod._KEEPALIVE[-1]
    assert kept_engine is rollouts[0] and kept_status.InProgress()
    assert len(fab["threads"]) == 1, "a read went out through an engine already marked unusable"


def test_a_manifest_larger_than_the_header_region_is_refused_with_the_size_to_set(publish_env):
    mod, _ = publish_env
    actors, rollouts = [_new(mod, header_bytes=1024)], [_new(mod, header_bytes=1024)]

    async def run():
        _wire(actors, rollouts)
        await _publish(actors, _make_weights(7), step=1)

    with pytest.raises(ValueError, match="raise engine_kwargs.mori_publish.header_bytes above"):
        asyncio.run(run())
