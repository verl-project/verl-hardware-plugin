# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the mori checkpoint engine with an in-process fake of MoRI-IO.

No GPU, NIC or Ray: every engine runs in one asyncio loop and talks over real ZMQ on localhost. The fake IOEngine
implements WRITE as a byte copy on a background thread after a random delay: the source is read only then (as a NIC
reads it), the initiator's status completes after the copy, and an RDMA write then raises a target-side notification
(an XGMI one does not), so buckets of different stripes complete out of order and reusing a slot early corrupts data.
The tests drive engines the way CheckpointEngineManager does: prepare -> build_topology -> init_process_group ->
send/receive -> finalize, several syncs in a row.
"""

import asyncio
import gc
import importlib
import importlib.machinery
import itertools
import pickle
import random
import sys
import threading
import time
import types
import weakref

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

BUCKET = 64 << 10  # small, so models span many buckets and chains
_ENGINES = []  # every engine a test creates; closed by the fixture


def _fake_mori():
    """Fake `mori` / `mori.io` modules backed by a dict of registered tensors."""
    fab = {
        "engines": weakref.WeakValueDictionary(),  # key -> fake IOEngine, as long as its owner keeps it
        "mems": {},
        "ids": itertools.count(1),
        "lock": threading.Lock(),
        "writes": [],
        "threads": [],
        "node_of": {},  # engine key -> node, filled by _wire after prepare(); same node + XGMI on both = XGMI
        "delay": 0.003,  # max seconds a write takes
        "inflight": 0,  # writes submitted and not yet landed
        "fail_writes": 0,  # the next N writes report FAILED at once while their copy still runs (orphaned WRs)
    }

    class Status:
        def __init__(self, ok=True):
            self.ok, self.done = ok, threading.Event()

        def InProgress(self):
            return not self.done.is_set()

        def Succeeded(self):
            return not self.InProgress() and self.ok

        def Failed(self):
            return not self.InProgress() and not self.ok

        def Message(self):
            return ""

        def WaitFor(self, ms):
            self.done.wait(ms / 1000)

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
            self.key, self.remotes, self.inbound, self.uid = key, set(), {}, itertools.count(1)
            self.endpoints = set()  # like MoRI 1.2.4: built on first write, kept until the engine is destroyed
            self.xgmi = False
            fab["engines"][key] = self

        def create_backend(self, backend_type, config):
            if backend_type == "XGMI":
                self.xgmi = True

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

        def allocate_transfer_uid(self):
            return next(self.uid)

        def write(self, local, local_off, remote, remote_off, size, uid):
            owner, dst = fab["mems"][remote.id]
            assert owner in self.remotes, "write to an engine that was not registered"
            assert local_off + size <= local.size and remote_off + size <= remote.size, "out of bounds"
            src = fab["mems"][local.id][1]
            target = fab["engines"][owner]
            node = fab["node_of"].get(self.key)
            # like MoRI's XGMI backend, a same-node XGMI copy raises no notification at the target
            xgmi = self.xgmi and target.xgmi and node is not None and node == fab["node_of"].get(owner)
            delay, st = random.uniform(0, fab["delay"]), Status()
            self.endpoints.add(owner)
            with fab["lock"]:
                fab["inflight"] += 1
                failed = fab["fail_writes"] > 0
                fab["fail_writes"] -= failed
            if failed:  # FAILED right away, but the copy below still happens later
                st.ok = False
                st.done.set()

            def land():
                time.sleep(delay)
                dst[remote_off : remote_off + size].copy_(src[local_off : local_off + size])  # the NIC reads now
                with fab["lock"]:
                    fab["inflight"] -= 1
                    if not xgmi:
                        done = Status()
                        done.done.set()
                        target.inbound[(self.key, uid)] = done
                if not failed:
                    st.done.set()

            th = threading.Thread(target=land, daemon=True)
            th.start()
            fab["threads"].append(th)
            fab["writes"].append((self.key, owner, remote_off))
            if xgmi:
                fab["xgmi_writes"] = fab.get("xgmi_writes", 0) + 1
            return st

        def pop_inbound_transfer_status(self, remote_key, uid):
            with fab["lock"]:
                return self.inbound.pop((remote_key, uid), None)

        def pop_peer_failure(self):
            return None

        def deregister_memory(self, mem):
            fab["mems"].pop(mem.id, None)

    io = types.ModuleType("mori.io")
    io.__spec__ = importlib.machinery.ModuleSpec("mori.io", None)
    for name, obj in dict(
        IOEngine=IOEngine,
        EngineDesc=EngineDesc,
        MemoryDesc=MemoryDesc,
        IOEngineConfig=lambda host="", port=0: None,
        RdmaBackendConfig=lambda **kw: None,
        XgmiBackendConfig=lambda **kw: None,
        BackendType=types.SimpleNamespace(RDMA="RDMA", XGMI="XGMI"),
    ).items():
        setattr(io, name, obj)
    mori = types.ModuleType("mori")
    mori.__spec__ = importlib.machinery.ModuleSpec("mori", None)
    mori.io = io
    return mori, io, fab


@pytest.fixture
def mori_env(monkeypatch):
    """Install the fake MoRI (its module is imported lazily by the engine); yield (engine module, fake fabric)."""
    mori, io, fab = _fake_mori()
    monkeypatch.setitem(sys.modules, "mori", mori)
    monkeypatch.setitem(sys.modules, "mori.io", io)
    mod = importlib.import_module("verl_hardware_plugin.accelerators.rocm.engines.mori_checkpoint_engine")
    yield mod, fab
    for th in fab["threads"]:  # let in-flight fake writes land before the next test
        th.join()
    while _ENGINES:
        _ENGINES.pop().close()


def _make_weights(seed):
    g, r = torch.Generator().manual_seed(seed), random.Random(seed)
    ws = [("embed", torch.randn(300, 77, generator=g).to(torch.bfloat16))]
    dtypes = [torch.bfloat16, torch.float32, torch.int64, torch.float16]
    for i in range(30):
        n, dt = r.randint(1, 3000), dtypes[i % 4]
        t = torch.randint(-1000, 1000, (n,), generator=g) if dt == torch.int64 else torch.randn(n, generator=g).to(dt)
        ws.append((f"layer{i}.w", t))
    # low-precision rollout: quantized on the trainer side and exported as (weight, scale) pairs; the engine moves
    # bytes, so every dtype must arrive bit-exact
    ws.append(("mlp.fp8.weight", torch.randn(120, 70, generator=g).to(torch.float8_e4m3fn)))
    ws.append(("mlp.fp8.weight_scale_inv", torch.rand(1, 3, generator=g) + 0.5))
    ws.append(("attn.fp8.weight", torch.randn(333, generator=g).to(torch.float8_e5m2)))
    ws.append(("mlp.mxfp4.weight_packed", torch.randint(0, 256, (257, 32), dtype=torch.uint8, generator=g)))
    if hasattr(torch, "float8_e8m0fnu"):
        ws.append(("mlp.mxfp4.weight_scale", (torch.rand(257, generator=g) * 8 + 0.5).to(torch.float8_e8m0fnu)))
    ws.append(("big", torch.randn(100_000, generator=g)))  # 400 KB: spans many buckets, merged on receive
    ws.append(("tail.bias", torch.randn(3, generator=g).to(torch.bfloat16)))
    return ws


def _same(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    )


def _new(mod, num_chains, node="n0", **kw):
    e = mod.MoriCheckpointEngine(
        bucket_size=BUCKET, num_chains=num_chains, host="127.0.0.1", device="cpu", timeout_s=30, node_id=node, **kw
    )
    _ENGINES.append(e)
    return e


def _wire(actors, rollouts, fab=None):
    meta = [e.prepare() for e in actors + rollouts]
    if fab is not None:
        fab["node_of"].update({e._key: e.node_id for e in actors + rollouts})
    akw, rkw = type(actors[0]).build_topology(len(actors), len(rollouts), meta)
    for i, e in enumerate(actors):
        e.init_process_group(**{k: v[i] for k, v in akw.items()})
    for i, e in enumerate(rollouts):
        e.init_process_group(**{k: v[i] for k, v in rkw.items()})
    return meta


async def _consume(e, step, delay=0.0, fail_at=None):
    got = {}
    async for name, w in e.receive_weights(global_steps=step):
        assert name not in got, f"{name} received twice"
        got[name] = w.clone()
        if fail_at is not None and len(got) == fail_at:
            raise ValueError("simulated load_weights failure")
        if delay:
            await asyncio.sleep(delay)
    return got


def _check_traffic(fab, actors, rollouts, nodes, P):
    """v2 traffic: each rollout node takes each stripe exactly once from outside, and one GPU of the node (the
    stripe's holder) copies it to the others; different stripes use different holders (so different NICs)."""
    node_of = {e._key: e.node_id for e in actors + rollouts}
    cb, slots = actors[0].chain_bucket, actors[0].num_slots
    inbound, local = {}, {}  # (node, stripe) -> writers from outside / inside the node
    for src, dst, off in fab["writes"]:
        key = (node_of[dst], off // cb // slots)
        (inbound if node_of[src] != node_of[dst] else local).setdefault(key, set()).add(src)
    for ni, size in enumerate(nodes):
        for c in range(P):
            assert len(inbound.get((f"r{ni}", c), ())) == 1, (ni, c, inbound.get((f"r{ni}", c)))
            if size > 1:
                assert len(local.get((f"r{ni}", c), ())) == 1, (ni, c, local.get((f"r{ni}", c)))
        if size > 1:
            holders = {w for c in range(P) for w in local[(f"r{ni}", c)]}
            assert len(holders) == min(P, size)
    if max(nodes) > 1:  # same-node copies took the XGMI path, which raises no inbound notification
        assert fab.get("xgmi_writes", 0) > 0
    fab["xgmi_writes"] = 0


def test_registered(mori_env):
    mod, _ = mori_env
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("mori") is mod.MoriCheckpointEngine


@pytest.mark.parametrize(
    "n_actors, nodes, num_chains, num_slots",
    [
        (2, [3], 1, 2),
        (4, [3], 8, 2),
        (4, [3], 2, 2),
        (4, [2], 8, 2),
        (2, [6], 8, 2),
        (4, [3, 3], 8, 2),
        (8, [4, 4], 8, 2),
        (2, [2, 2, 2, 2], 8, 2),
        (4, [3, 3], 1, 2),
        (8, [2, 2], 8, 2),
        (2, [3], 1, 4),
        (4, [3, 3], 8, 3),
    ],
    ids=[
        "single-stripe",
        "stripe-per-actor",
        "fewer-stripes-than-actors",
        "more-stripes-than-gpus",
        "one-node-2x6",
        "2-nodes",
        "2-nodes-8-stripes",
        "4-nodes",
        "2-nodes-1-stripe",
        "stripes-wrap-gpus",
        "single-stripe-4-slots",
        "2-nodes-3-slots",
    ],
)
def test_every_rollout_gets_every_tensor(mori_env, n_actors, nodes, num_chains, num_slots):
    """Trainer on its own node, rollout workers on len(nodes) nodes. Every tensor must arrive bit-exact, and traffic
    must follow v2: each rollout node takes each stripe exactly once from outside, and one GPU of the node (the
    stripe's holder) copies it to the others; different stripes use different holders (so different NICs)."""
    mod, fab = mori_env
    actors = [_new(mod, num_chains, node="trainer", is_master=(i == 0), num_slots=num_slots) for i in range(n_actors)]
    rollouts = [
        _new(mod, num_chains, node=f"r{ni}", num_slots=num_slots) for ni, size in enumerate(nodes) for _ in range(size)
    ]

    async def run():
        first = None
        for step in range(3):  # buckets, registrations and sockets must be reused across syncs
            meta = _wire(actors, rollouts, fab)
            first = first or meta
            assert meta == first
            ws = _make_weights(step)
            fab["writes"].clear()
            res = await asyncio.gather(
                *[a.send_weights(iter(ws), global_steps=step) for a in actors],
                *[_consume(r, step, 0.002 if i == 1 else 0) for i, r in enumerate(rollouts)],
            )
            for got in res[n_actors:]:
                assert set(got) == {n for n, _ in ws}
                for n, w in ws:
                    assert _same(got[n], w), n
            P = actors[0]._P
            assert P == min(num_chains, n_actors)
            _check_traffic(fab, actors, rollouts, nodes, P)
            for e in actors + rollouts:
                e.finalize()

    asyncio.run(run())


def test_version_mismatch_fails_everyone_fast(mori_env):
    mod, _ = mori_env
    actors = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
    rollouts = [_new(mod, 4, node=f"r{i % 2}") for i in range(3)]
    _wire(actors, rollouts)

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(_make_weights(7)), global_steps=7) for a in actors],
            *[_consume(r, 8) for r in rollouts],
            return_exceptions=True,
        )

    t = time.time()
    res = asyncio.run(run())
    assert time.time() - t < 5  # far below timeout_s: the failure is propagated, not waited out
    assert all(isinstance(x, RuntimeError) for x in res)
    assert any("step 7 while expecting step 8" in str(x) for x in res)


def test_consumer_failure_fails_everyone_fast(mori_env):
    mod, _ = mori_env
    actors = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
    rollouts = [_new(mod, 4, node=f"r{i % 2}") for i in range(3)]
    _wire(actors, rollouts)

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(_make_weights(3)), global_steps=3) for a in actors],
            _consume(rollouts[0], 3),
            _consume(rollouts[1], 3, fail_at=10),
            _consume(rollouts[2], 3),
            return_exceptions=True,
        )

    t = time.time()
    res = asyncio.run(run())
    assert time.time() - t < 5
    assert isinstance(res[5], ValueError)
    assert all(isinstance(x, RuntimeError) for i, x in enumerate(res) if i != 5)


async def _consume_like_worker(e, step, fail_at=None):
    """Consume through a caller that keeps the stream (as a traceback would) and closes it explicitly when its
    consumer fails, so the failure path runs now rather than when the stream is garbage-collected."""
    weights = e.receive_weights(global_steps=step)
    try:
        got = {}
        async for name, w in weights:
            got[name] = w.clone()
            if fail_at is not None and len(got) == fail_at:
                raise ValueError("simulated load_weights failure")
        return got
    except BaseException:
        await weights.aclose()
        raise


def _unusable(e):
    try:
        e.prepare()
    except RuntimeError as err:
        return "unusable" in str(err)
    return False


def _fail_stop_holds(engines, results):
    """Every worker whose sync raised is unusable; a worker whose sync completed (it may finish its part before
    another worker fails) has nothing in flight."""
    for e, x in zip(engines, results, strict=True):
        if isinstance(x, BaseException):
            assert _unusable(e), (e._key, x)
        else:
            assert not e._inflight and e._unusable is None, e._key


def test_failed_sync_is_fail_stop(mori_env):
    """A consumer fails while writes are in flight: every worker that failed refuses any further use (MoRI 1.2.4 does
    not guarantee that abandoned transfers stopped), keeps its MoRI objects alive, and fresh workers sync fine."""
    mod, fab = mori_env
    actors = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
    rollouts = [_new(mod, 4, node=f"r{i % 2}") for i in range(4)]

    async def run():
        _wire(actors, rollouts, fab)
        fab["delay"] = 0.05  # slow writes, so the failure catches many in flight
        res = await asyncio.gather(
            *[a.send_weights(iter(_make_weights(1)), global_steps=1) for a in actors],
            *[_consume_like_worker(r, 1, fail_at=5 if i == 1 else None) for i, r in enumerate(rollouts)],
            return_exceptions=True,
        )
        assert isinstance(res[5], ValueError)
        _fail_stop_holds(actors + rollouts, res)
        failed = [e for e, x in zip(actors + rollouts, res, strict=True) if isinstance(x, BaseException)]
        assert all(e._engine is not None and e._region is not None for e in failed)  # nothing released
        assert any(st for (*_, inflight) in mod._KEEPALIVE for st in inflight)  # unfinished writes stay referenced
        fab["delay"] = 0.003
        fresh_a = [_new(mod, 4, node="trainer", is_master=(i == 0)) for i in range(4)]
        fresh_r = [_new(mod, 4, node=f"r{i % 2}") for i in range(4)]
        _wire(fresh_a, fresh_r, fab)
        ws = _make_weights(2)
        res = await asyncio.gather(
            *[a.send_weights(iter(ws), global_steps=2) for a in fresh_a], *[_consume(r, 2) for r in fresh_r]
        )
        for got in res[4:]:
            assert all(_same(got[n], w) for n, w in ws)

    asyncio.run(run())


def test_failed_status_with_running_copy_is_kept_and_refused(mori_env):
    """MoRI may report FAILED while work requests of the transfer are still posted: the status must stay referenced
    and the engine must refuse reuse, even though the status is no longer in progress."""
    mod, fab = mori_env
    actor = _new(mod, 1, node="trainer", is_master=True)
    rollout = _new(mod, 1, node="r0")

    async def run():
        _wire([actor], [rollout], fab)
        fab["fail_writes"], fab["delay"] = 1, 0.05
        return await asyncio.gather(
            actor.send_weights(iter(_make_weights(0)), global_steps=0), _consume(rollout, 0), return_exceptions=True
        )

    sent, received = asyncio.run(run())
    assert isinstance(sent, RuntimeError) and isinstance(received, RuntimeError)
    assert any(not st.Succeeded() for st in actor._inflight)
    assert _unusable(actor) and _unusable(rollout)


def test_cancelled_sync_is_fail_stop(mori_env):
    """A sync cancelled from outside (e.g. a timeout around update_weights) leaves the engines unusable, also when
    the cancellation hits again during the cleanup."""
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node="r0") for _ in range(2)]

    async def run():
        _wire(actors, rollouts, fab)
        fab["delay"] = 0.05
        tasks = [asyncio.ensure_future(a.send_weights(iter(_make_weights(0)), global_steps=0)) for a in actors]
        tasks += [asyncio.ensure_future(_consume_like_worker(r, 0)) for r in rollouts]
        await asyncio.sleep(0.1)
        tasks[2].cancel()  # one receiver
        await asyncio.sleep(0.005)
        tasks[2].cancel()  # and again, while it cleans up
        return await asyncio.gather(*tasks, return_exceptions=True)

    res = asyncio.run(run())
    assert isinstance(res[2], asyncio.CancelledError) and _unusable(rollouts[0])
    _fail_stop_holds(actors + rollouts, res)


def test_retry_after_first_error_is_refused_while_workers_unwind(mori_env):
    """ray.get raises on the first failed worker without waiting for the others: a retry must not get hold of a
    worker whose sync is still running (it refuses prepare), and no worker that failed is reusable afterwards."""
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node="r0") for _ in range(3)]

    async def run():
        _wire(actors, rollouts, fab)
        fab["delay"] = 0.05
        tasks = [asyncio.ensure_future(a.send_weights(iter(_make_weights(0)), global_steps=0)) for a in actors]
        tasks += [
            asyncio.ensure_future(_consume_like_worker(r, 0, fail_at=2 if i == 0 else None))
            for i, r in enumerate(rollouts)
        ]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        assert pending, "the others should still be unwinding"
        for e in actors + rollouts:  # what a retry's prepare() would hit right now
            with pytest.raises(RuntimeError):
                e.prepare()
        return await asyncio.gather(*tasks, return_exceptions=True)

    _fail_stop_holds(actors + rollouts, asyncio.run(run()))


@pytest.mark.parametrize(
    "membership",
    [
        [["A"], ["A", "B"]],
        [["A", "B"], ["A"], ["A", "B"]],
        [["A", "B"], ["A", "C"], ["C", "D"]],
    ],
    ids=["join-after-a-sync", "leave-and-rejoin", "replace"],
)
def test_rollout_membership_changes(mori_env, membership):
    """Rollout workers join, leave and are replaced between syncs (new topology each time): every sync must reach
    exactly the current workers, a newcomer must accept the others' messages, and connections to workers that left
    must be released."""
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    pool = {}

    def rollout(name):
        if name not in pool:
            pool[name] = _new(mod, 2, node=f"r-{name}")
        return pool[name]

    async def run():
        for step, names in enumerate(membership):
            rollouts = [rollout(n) for n in names]
            for e in actors + rollouts:
                e.timeout_s = 5
            _wire(actors, rollouts, fab)
            ws = _make_weights(step)
            res = await asyncio.gather(
                *[a.send_weights(iter(ws), global_steps=step) for a in actors],
                *[_consume(r, step) for r in rollouts],
            )
            for got in res[len(actors) :]:
                assert set(got) == {n for n, _ in ws}
                assert all(_same(got[n], w) for n, w in ws)
            current = {r._key for r in rollouts}
            for a in actors:  # actors talk to the current workers only
                assert set(a._peers) <= current and len(a._push) == len(a._peers), (set(a._peers), current)
                assert fab["engines"][a._key].remotes <= current
            for e in actors + rollouts:
                e.finalize()

    asyncio.run(run())


def test_rollout_churn_replaces_the_mori_engine(mori_env, monkeypatch):
    """MoRI 1.2.4 keeps the endpoints of a deregistered peer until the engine is destroyed: with rollout workers
    replaced again and again, the engine must be replaced once enough peers have left, so what it keeps stays
    bounded, and every sync still delivers the weights."""
    mod, fab = mori_env
    monkeypatch.setattr(mod, "_MAX_RETIRED_PEERS", 2)
    actor = _new(mod, 1, node="trainer", is_master=True)
    keys, mori_engines = set(), []

    async def run():
        for step in range(10):
            rollout = _new(mod, 1, node="r0")  # a new worker every sync
            _wire([actor], [rollout], fab)
            keys.add(actor._key)
            mori_engines.append(weakref.ref(fab["engines"][actor._key]))
            ws = _make_weights(step)
            _, got = await asyncio.gather(actor.send_weights(iter(ws), global_steps=step), _consume(rollout, step))
            assert all(_same(got[n], w) for n, w in ws)
            assert len(fab["engines"][actor._key].endpoints) <= 2 + 2  # retired before the replacement + current
            actor.finalize()
            rollout.finalize()

    asyncio.run(run())
    assert len(keys) > 1, "the MoRI engine was never replaced"
    gc.collect()
    alive = {id(r()) for r in mori_engines if r() is not None}
    assert alive == {id(fab["engines"][actor._key])}, "a replaced MoRI engine is still referenced"


def test_worker_refuses_a_new_sync_while_its_sync_runs(mori_env):
    """prepare / init_process_group / send / receive on a worker whose sync is still running must refuse (a retry
    racing the old sync), and the worker is usable again once that sync completed."""
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node="r0") for _ in range(2)]

    async def run():
        _wire(actors, rollouts, fab)
        fab["delay"] = 0.05
        tasks = [asyncio.ensure_future(a.send_weights(iter(_make_weights(0)), global_steps=0)) for a in actors]
        tasks += [asyncio.ensure_future(_consume(r, 0)) for r in rollouts]
        await asyncio.sleep(0.01)  # every worker is inside its sync: no bucket can have completed yet
        for e in actors + rollouts:
            with pytest.raises(RuntimeError, match="has not finished"):
                e.prepare()
        await asyncio.gather(*tasks)
        for e in actors + rollouts:
            e.finalize()
        _wire(actors, rollouts, fab)  # usable again

    asyncio.run(run())


def test_init_process_group_writes_nothing(mori_env):
    """Writes happen only inside a sync, where the target guards its region: building the topology must not write
    (a write that failed there would leave a target that is still usable)."""
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node=f"r{i % 2}") for i in range(4)]
    _wire(actors, rollouts, fab)
    assert not fab["writes"] and fab["inflight"] == 0


def test_nothing_is_written_to_a_worker_before_it_receives(mori_env):
    """A sender may start before its target has entered the sync (Ray does not order an async actor's tasks): it must
    not write until the target grants its slots. If the target never comes (the sync is given up), the target's region
    was never touched, so the target may serve the next sync."""
    mod, fab = mori_env
    actor = _new(mod, 1, node="trainer", is_master=True)
    rollout = _new(mod, 1, node="r0")

    async def run():
        _wire([actor], [rollout], fab)
        sending = asyncio.ensure_future(actor.send_weights(iter(_make_weights(0)), global_steps=0))
        await asyncio.sleep(0.1)
        assert not fab["writes"] and fab["inflight"] == 0  # waiting for the target
        sending.cancel()  # the sync is given up before the target ever started
        with pytest.raises(asyncio.CancelledError):
            await sending
        assert _unusable(actor) and not fab["writes"]
        fresh = _new(mod, 1, node="trainer", is_master=True)
        rollout.finalize()
        _wire([fresh], [rollout], fab)  # the untouched target is reusable
        ws = _make_weights(1)
        _, got = await asyncio.gather(fresh.send_weights(iter(ws), global_steps=1), _consume(rollout, 1))
        assert all(_same(got[n], w) for n, w in ws)

    asyncio.run(run())


_UNPICKLED = []


def _mark_unpickled():
    _UNPICKLED.append(True)


class _Payload:
    def __reduce__(self):
        return (_mark_unpickled, ())


def test_control_inbox_drops_foreign_messages(mori_env):
    """A control inbox listens on the worker's own address only and decodes JSON, never pickle: a pickle (whose
    loading would run code), garbage, and well-formed messages of another sync, including an abort, are dropped
    and the sync completes."""
    import zmq

    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node=f"r{i % 2}") for i in range(4)]
    _wire(actors, rollouts, fab)
    for e in actors + rollouts:
        assert e._inbox.getsockopt(zmq.LAST_ENDPOINT).decode().startswith("tcp://127.0.0.1:")
    foreign = [
        pickle.dumps({"t": "abort", "seq": (actors[0]._sync_id, 1), "reason": _Payload()}),
        b"\x00garbage",
        b'{"t": "abort", "seq": ["another-sync", 1], "reason": "not ours"}',
        b'{"t": "free", "seq": ["another-sync", 1], "c": 0, "slot": 0, "from": "x"}',
        b"[1, 2]",
    ]
    ctx = zmq.Context()
    try:
        for e in (actors[0], rollouts[0]):
            sock = ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 1000)
            sock.connect(e._inbox_addr)
            for raw in foreign:
                sock.send(raw)
            sock.close()  # lingers until delivered; the pump reads them first when the sync starts
    finally:
        ctx.term()

    async def run():
        ws = _make_weights(5)
        res = await asyncio.gather(
            *[a.send_weights(iter(ws), global_steps=5) for a in actors], *[_consume(r, 5) for r in rollouts]
        )
        for got in res[len(actors) :]:
            assert set(got) == {n for n, _ in ws} and all(_same(got[n], w) for n, w in ws)

    asyncio.run(run())
    assert not _UNPICKLED
    assert all(e._unusable is None for e in actors + rollouts)


def _push_raw(addresses, messages):
    """Send raw control messages to the given inboxes, as anyone on the network could; delivered before returning."""
    import zmq

    ctx = zmq.Context()
    try:
        for address in addresses:
            sock = ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 1000)
            sock.connect(address)
            for raw in messages:
                sock.send(raw)
            sock.close()
    finally:
        ctx.term()


def test_control_inbox_drops_invalid_messages_of_the_current_sync(mori_env):
    """Well-formed JSON with the current sync's id but a wrong structure or outside this topology (no type, a free
    of an unknown slot or peer, a bucket of an unknown stripe or with bad fields, an abort without a reason) is
    dropped: it must neither stop the inbox nor be taken for a real bucket, and the sync completes."""
    import json

    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node=f"r{i % 2}") for i in range(4)]
    _wire(actors, rollouts, fab)
    seq = [actors[0]._sync_id, actors[0]._round + 1]  # the id of the sync started below
    bucket = {"t": "bucket", "seq": seq, "k": 0, "slot": 0, "metas": [], "used": 0, "is_last": True, "uid": None}
    invalid = [
        {"seq": seq},
        {"t": "noop", "seq": seq},
        {"t": "abort", "seq": seq},
        {"t": "free", "seq": seq, "c": 0, "slot": 0, "from": "nobody"},
        {"t": "free", "seq": seq, "c": [0], "slot": {}, "from": None},
        {**bucket, "c": 99},
        {**bucket, "c": 0, "slot": 7},
        {**bucket, "c": 0, "used": "all"},
        {**bucket, "c": 0, "used": 8, "metas": [["w", [4], "float32", 0, 16, 0]]},  # entry beyond the bucket
        {**bucket, "c": 0, "is_last": "yes"},
        {**bucket, "c": 0, "uid": "x"},
    ]
    _push_raw([e._inbox_addr for e in actors + rollouts], [json.dumps(m).encode() for m in invalid])

    async def run():
        ws = _make_weights(6)
        res = await asyncio.gather(
            *[a.send_weights(iter(ws), global_steps=6) for a in actors], *[_consume(r, 6) for r in rollouts]
        )
        for got in res[len(actors) :]:
            assert set(got) == {n for n, _ in ws} and all(_same(got[n], w) for n, w in ws)

    t = time.time()
    asyncio.run(run())
    assert time.time() - t < 5
    assert all(e._unusable is None for e in actors + rollouts)


def test_a_failing_control_inbox_fails_the_sync_at_once(mori_env, monkeypatch):
    """If handling a control message ever raises, the sync fails at once on that worker (and so on its peers)
    instead of every waiter running into timeout_s."""
    import json

    mod, fab = mori_env
    actors = [_new(mod, 1, node="trainer", is_master=True)]
    rollouts = [_new(mod, 1, node="r0") for _ in range(2)]
    _wire(actors, rollouts, fab)
    monkeypatch.setattr(mod.MoriCheckpointEngine, "_invalid", lambda self, msg: None)  # let a bad message through
    seq = [actors[0]._sync_id, actors[0]._round + 1]
    _push_raw(
        [rollouts[0]._inbox_addr], [json.dumps({"t": "free", "seq": seq, "c": 5, "slot": 0, "from": "x"}).encode()]
    )

    async def run():
        return await asyncio.gather(
            *[a.send_weights(iter(_make_weights(0)), global_steps=0) for a in actors],
            *[_consume(r, 0) for r in rollouts],
            return_exceptions=True,
        )

    t = time.time()
    res = asyncio.run(run())
    assert time.time() - t < 5  # far below timeout_s
    assert any(isinstance(x, RuntimeError) and "control message handling failed" in str(x) for x in res), res
    _fail_stop_holds(actors + rollouts, res)


async def _update_weights_closing_on_failure(self, global_steps=None):
    """Like verl's CheckpointEngineWorker.update_weights, but closes the receive stream when the adapter fails."""
    weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
    try:
        await self.server_adapter.update_weights(weights, global_steps=global_steps)
    except BaseException:
        await weights.aclose()
        raise


@pytest.mark.parametrize("how", ["error", "aiter-then-error", "cancel"])
def test_adapter_failing_before_its_first_tensor_ends_the_sync(mori_env, how):
    """The server adapter fails (or the update is cancelled) before it iterates the weights, as an adapter setting up
    its IPC buffers can: once the caller closes the receive stream, the receiver must end the sync as failed, refuse
    reuse, and its peers must stop instead of waiting for it."""
    update_weights = _update_weights_closing_on_failure
    mod, fab = mori_env
    actors = [_new(mod, 2, node="trainer", is_master=(i == 0)) for i in range(2)]
    rollouts = [_new(mod, 2, node="r0") for _ in range(3)]

    async def broken_adapter(weights, **kwargs):
        if how == "aiter-then-error":  # takes the iterator, then fails before its first anext()
            aiter(weights)
        if how != "cancel":
            raise RuntimeError("IPC server initialization failed")
        await asyncio.sleep(30)  # cancelled while setting up

    async def adapter(weights, **kwargs):
        async for _name, w in weights:
            w.clone()

    def worker(engine, update):
        return types.SimpleNamespace(
            checkpoint_engine=engine, server_adapter=types.SimpleNamespace(update_weights=update)
        )

    async def run():
        _wire(actors, rollouts, fab)
        t = time.time()
        tasks = [asyncio.ensure_future(a.send_weights(iter(_make_weights(0)), global_steps=0)) for a in actors]
        tasks += [
            asyncio.ensure_future(update_weights(worker(r, broken_adapter if i == 0 else adapter), 0))
            for i, r in enumerate(rollouts)
        ]
        if how == "cancel":
            await asyncio.sleep(0.05)
            tasks[2].cancel()
        res = await asyncio.gather(*tasks, return_exceptions=True)
        assert time.time() - t < 5  # nobody waited for the timeout
        return res

    res = asyncio.run(run())
    assert isinstance(res[2], (RuntimeError, asyncio.CancelledError)) and _unusable(rollouts[0])
    _fail_stop_holds(actors + rollouts, res)
