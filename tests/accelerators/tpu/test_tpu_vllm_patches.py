# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for TPU rollout."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
import types
from unittest import mock

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def test_patch_vllm_for_tpu_applies_precision_patches_and_leaves_tpu_worker_untouched():
    """patch_vllm_for_tpu installs precision patches once vllm_torchtpu is present, without patching TPUWorker."""
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    def upstream_reset_encoder_cache(self) -> None:
        pass

    class TPUWorker:
        reset_encoder_cache = upstream_reset_encoder_cache

    worker_mod.TPUWorker = TPUWorker
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }

    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "_apply_tpu_precision_patches") as apply_precision,
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()
        apply_precision.assert_called_once_with()
        assert tpu_vllm_patches._PATCHES_APPLIED is True
        assert TPUWorker.reset_encoder_cache is upstream_reset_encoder_cache


def _fake_tpu_worker_module(configure):
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")
    worker_mod.TPUWorker = type("TPUWorker", (), {})
    worker_mod._configure_tpu_process_env = configure
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }
    return worker_mod, modules


def _patch_in_fake_worker_process(modules):
    """Run patch_vllm_for_tpu as a vLLM worker would, without its process-global multiprocessing patch."""
    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "patch_multiprocessing_for_tpu"),
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()


def test_subhost_worker_hook_is_not_installed_for_other_engines():
    """A TP=8 or multi-host engine (no SUBHOST_MEGASCALE_PORT_ENV) keeps vllm-torchtpu's function as is."""

    def configure(rank, local_rank, world_size, local_world_size):
        pass

    worker_mod, modules = _fake_tpu_worker_module(configure)
    with mock.patch.dict(os.environ):
        os.environ.pop(tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV, None)
        _patch_in_fake_worker_process(modules)
    assert worker_mod._configure_tpu_process_env is configure


def test_subhost_worker_hook_sets_per_worker_ports_and_drops_pod_env():
    calls = []

    def configure(rank, local_rank, world_size, local_world_size):
        calls.append((rank, local_rank, world_size, local_world_size))

    worker_mod, modules = _fake_tpu_worker_module(configure)
    pod_env = {name: "pod" for name in tpu_vllm_patches._POD_TPU_ENV_VARS}
    with (
        mock.patch.dict(os.environ, {**pod_env, "MEGASCALE_PORT": "8081"}),
        mock.patch.object(os, "makedirs") as makedirs,
    ):
        # The second TP=4 engine of a host: device offset 4, MegaScale block from 19080.
        os.environ[tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV] = "19080"
        _patch_in_fake_worker_process(modules)
        with mock.patch.dict("sys.modules", modules):
            tpu_vllm_patches.patch_tpu_worker_subhost_env()  # a second call does not wrap twice
        worker_mod._configure_tpu_process_env(1, 5, 4, 8)

        assert calls == [(1, 5, 4, 8)]  # vllm-torchtpu's own variables are still set first
        assert os.environ["MEGASCALE_PORT"] == "19082"
        assert os.environ["MEGASCALE_DEBUG_PORT"] == "19083"
        assert os.environ["TPU_VISIBLE_DEVICES"] == os.environ["TPU_VISIBLE_CHIPS"] == "5"
        assert os.environ["TPU_LOG_DIR"] == "/tmp/tpu_logs/rank_5"
        assert not set(tpu_vllm_patches._POD_TPU_ENV_VARS) & set(os.environ)
    makedirs.assert_called_once_with("/tmp/tpu_logs/rank_5", exist_ok=True)


def test_subhost_worker_hook_drops_torch_tpu_devices_per_host():
    """A worker of an engine on part of a host must not keep a host-wide TORCH_TPU_DEVICES_PER_HOST (here the 8 of a
    whole TPU 7x host, e.g. exported for the trainer): torch_tpu sizes its node-local store by it."""
    worker_mod, modules = _fake_tpu_worker_module(lambda rank, local_rank, world_size, local_world_size: None)
    env = {"TORCH_TPU_DEVICES_PER_HOST": "8", tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21080"}
    with mock.patch.dict(os.environ, env), mock.patch.object(os, "makedirs"):
        _patch_in_fake_worker_process(modules)
        worker_mod._configure_tpu_process_env(1, 5, 4, 8)  # worker 1 of the second TP=4 engine of the host
        assert "TORCH_TPU_DEVICES_PER_HOST" not in os.environ


# ---------------------------------------------------------------------------
# Sub-host StartSession serialization (TPUWorker.init_device)
# ---------------------------------------------------------------------------


def _fake_tpu_worker_module_with_init_device(init_device_fn):
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    class TPUWorker:
        def __init__(self, rank: int, world_size: int):
            self.rank = rank
            self.parallel_config = types.SimpleNamespace(world_size=world_size)

        init_device = init_device_fn

    worker_mod.TPUWorker = TPUWorker
    worker_mod._configure_tpu_process_env = lambda rank, local_rank, world_size, local_world_size: None
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }
    return worker_mod, modules


def test_subhost_start_session_hook_is_no_op_for_single_engine_hosts():
    """Single-engine hosts (TP=8, TP=1, multi-host, non-v7x: no SUBHOST_MEGASCALE_PORT_ENV) do not wrap init_device."""

    def orig_init(self):
        return "ok"

    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    with mock.patch.dict(os.environ):
        os.environ.pop(tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV, None)
        _patch_in_fake_worker_process(modules)
        assert not tpu_vllm_patches.subhost_serialize_start_session_enabled()
    assert worker_mod.TPUWorker.init_device is orig_init


@pytest.mark.parametrize("disabled_value", ["0", "false", "OFF", "no"])
def test_subhost_start_session_hook_can_be_disabled_via_env(disabled_value):
    def orig_init(self):
        return "ok"

    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    env = {
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21000",
        tpu_vllm_patches.SUBHOST_SERIALIZE_START_SESSION_ENV: disabled_value,
    }
    with mock.patch.dict(os.environ, env):
        _patch_in_fake_worker_process(modules)
        assert not tpu_vllm_patches.subhost_serialize_start_session_enabled()
    assert worker_mod.TPUWorker.init_device is orig_init


def test_subhost_start_session_hook_is_idempotent_and_patches_init_device(tmp_path, monkeypatch):
    calls = []

    def orig_init(self):
        calls.append((self.rank, self.parallel_config.world_size))
        return "initialized"

    lock_path = str(tmp_path / "start_session.lock")
    monkeypatch.setattr(tpu_vllm_patches, "SUBHOST_START_SESSION_LOCK_FILE", lock_path)
    worker_mod, modules = _fake_tpu_worker_module_with_init_device(orig_init)
    env = {
        tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV: "21000",
        tpu_vllm_patches.SUBHOST_ENGINE_ID_ENV: "engine-single",
    }
    with mock.patch.dict(os.environ, env):
        os.environ.pop(tpu_vllm_patches.SUBHOST_SERIALIZE_START_SESSION_ENV, None)
        _patch_in_fake_worker_process(modules)
        wrapped = worker_mod.TPUWorker.init_device
        assert wrapped is not orig_init
        with mock.patch.dict("sys.modules", modules):
            tpu_vllm_patches.patch_tpu_worker_subhost_start_session()
        assert worker_mod.TPUWorker.init_device is wrapped  # idempotent
        assert worker_mod.TPUWorker(rank=0, world_size=1).init_device() == "initialized"
    assert calls == [(0, 1)]
    assert os.path.exists(lock_path)


@pytest.mark.parametrize(
    "num_engines, world_size",
    [
        (2, 4),  # 2 x TP=4 on a TPU 7x host
        (4, 2),  # 4 x TP=2 on a TPU 7x host
        (2, 1),  # single-worker path
    ],
)
def test_subhost_start_session_serializes_engines_while_running_siblings_concurrently(
    tmp_path, num_engines, world_size
):
    lock_path = str(tmp_path / "start_session.lock")
    start_barrier = threading.Barrier(num_engines * world_size)
    intra_barriers = {f"eng-{e}": threading.Barrier(world_size) for e in range(num_engines)}
    state_lock = threading.Lock()
    active_engines: dict[str, int] = {}
    max_concurrent_engines = 0
    intervals: dict[str, list[tuple[float, float]]] = {f"eng-{e}": [] for e in range(num_engines)}
    errors: list[BaseException] = []

    def worker_fn(engine_id: str, rank: int):
        try:
            start_barrier.wait(timeout=10.0)
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=world_size,
                engine_id=engine_id,
                lock_path=lock_path,
                timeout_s=15.0,
                poll_interval_s=0.005,
            ):
                t_enter = time.monotonic()
                nonlocal max_concurrent_engines
                with state_lock:
                    active_engines[engine_id] = active_engines.get(engine_id, 0) + 1
                    max_concurrent_engines = max(max_concurrent_engines, len(active_engines))
                # Every rank of this engine must reach init_device concurrently (simulating tpu_dist rendezvous).
                intra_barriers[engine_id].wait(timeout=10.0)
                time.sleep(0.03)
                t_exit = time.monotonic()
                with state_lock:
                    intervals[engine_id].append((t_enter, t_exit))
                    active_engines[engine_id] -= 1
                    if active_engines[engine_id] == 0:
                        del active_engines[engine_id]
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker_fn, args=(f"eng-{e}", r)) for e in range(num_engines) for r in range(world_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)
        assert not t.is_alive()

    assert errors == []
    assert max_concurrent_engines == 1
    # Engine intervals [min_enter, max_exit] must be strictly non-overlapping across co-located engines.
    spans = sorted((min(s for s, _ in ivs), max(e for _, e in ivs), eid) for eid, ivs in intervals.items())
    for eid, ivs in intervals.items():
        assert len(ivs) == world_size, f"engine {eid} ran {len(ivs)} workers, expected {world_size}"
    for (_, prev_end, prev_id), (next_start, _, next_id) in zip(spans[:-1], spans[1:], strict=True):
        assert prev_end <= next_start, f"{prev_id} ({prev_end}) overlapped {next_id} ({next_start})"
    # Ephemeral per-engine coordination files are cleaned up; the persistent host lockfile stays.
    remaining = sorted(os.listdir(tmp_path))
    assert remaining == ["start_session.lock"]


def test_subhost_start_session_releases_lock_before_model_load(tmp_path):
    """After init_device finishes, co-located engines run subsequent stages (load_model / compile) in parallel."""
    lock_path = str(tmp_path / "start_session.lock")
    load_barrier = threading.Barrier(4)  # 2 engines x TP=2 must enter load_model concurrently
    errors: list[BaseException] = []

    def worker_fn(engine_id: str, rank: int):
        try:
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=2,
                engine_id=engine_id,
                lock_path=lock_path,
                timeout_s=10.0,
                poll_interval_s=0.005,
            ):
                time.sleep(0.02)
            # Only succeeds if both engines have released the StartSession lock before load_model.
            load_barrier.wait(timeout=5.0)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker_fn, args=(f"eng-{e}", r)) for e in range(2) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
        assert not t.is_alive()
    assert errors == []


def test_subhost_start_session_serializes_across_subprocesses(tmp_path):
    """Two real multi-worker sub-host engines (4 OS processes) serialize strictly via POSIX flock."""
    lock_path = str(tmp_path / "start_session.lock")
    log_path = str(tmp_path / "events.jsonl")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    script = (
        "import json, sys, time\n"
        "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (\n"
        "    serialize_subhost_start_session,\n"
        ")\n"
        "lock_path, log_path, engine_id = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "rank, world_size = int(sys.argv[4]), int(sys.argv[5])\n"
        "with serialize_subhost_start_session(\n"
        "    rank=rank, world_size=world_size, engine_id=engine_id,\n"
        "    lock_path=lock_path, timeout_s=15.0, poll_interval_s=0.005,\n"
        "):\n"
        "    t0 = time.time()\n"
        "    time.sleep(0.08)\n"
        "    t1 = time.time()\n"
        "    with open(log_path, 'a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps({'engine': engine_id, 'rank': rank, 't0': t0, 't1': t1}) + '\\n')\n"
    )
    env = {**os.environ, "PYTHONPATH": repo_root}
    procs = [
        subprocess.Popen([sys.executable, "-c", script, lock_path, log_path, f"proc-eng-{e}", str(r), "2"], env=env)
        for e in range(2)
        for r in range(2)
    ]
    for p in procs:
        assert p.wait(timeout=20.0) == 0

    with open(log_path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 4
    by_engine: dict[str, list[dict]] = {}
    for rec in records:
        by_engine.setdefault(rec["engine"], []).append(rec)
    spans = sorted((min(r["t0"] for r in rs), max(r["t1"] for r in rs), eid) for eid, rs in by_engine.items())
    assert spans[0][1] <= spans[1][0], f"Subprocess engines overlapped: {spans}"


@pytest.mark.parametrize("failing_rank", [0, 2])
def test_subhost_start_session_releases_lock_on_exception(tmp_path, failing_rank):
    """If an engine raises during init_device (either rank 0 or a sibling), the lock is released and peers proceed."""
    lock_path = str(tmp_path / "start_session.lock")
    intra = threading.Barrier(4)
    seen_errors: list[str] = []

    def failing_worker(rank: int):
        try:
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=rank,
                world_size=4,
                engine_id="eng-fail",
                lock_path=lock_path,
                timeout_s=5.0,
                poll_interval_s=0.005,
            ):
                intra.wait(timeout=5.0)
                if rank == failing_rank:
                    raise RuntimeError(f"rank-{rank} init boom")
        except RuntimeError as exc:
            seen_errors.append(str(exc))

    threads = [threading.Thread(target=failing_worker, args=(r,)) for r in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive()
    assert seen_errors == [f"rank-{failing_rank} init boom"]

    # Peer engine acquires the lock immediately without hanging.
    peer_ran = []
    with tpu_vllm_patches.serialize_subhost_start_session(
        rank=0,
        world_size=1,
        engine_id="eng-peer",
        lock_path=lock_path,
        timeout_s=5.0,
        poll_interval_s=0.005,
    ):
        peer_ran.append(True)
    assert peer_ran == [True]


def test_subhost_start_session_unblocks_siblings_when_rank0_aborts_before_go(tmp_path):
    """If rank 0 raises before GO, siblings see ABORT and unblock promptly instead of waiting for timeout."""
    lock_path = str(tmp_path / "start_session.lock")
    sibling_done = threading.Event()
    real_open = os.open

    def fail_on_host_lock(path, flags, *args, **kwargs):
        if path == lock_path:
            raise OSError("simulated lock open failure")
        return real_open(path, flags, *args, **kwargs)

    def sibling_fn():
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=1,
            world_size=2,
            engine_id="eng-abort",
            lock_path=lock_path,
            timeout_s=30.0,
            poll_interval_s=0.005,
        ):
            sibling_done.set()

    sibling = threading.Thread(target=sibling_fn)
    sibling.start()
    try:
        with mock.patch.object(tpu_vllm_patches.os, "open", side_effect=fail_on_host_lock):
            with pytest.raises(OSError, match="simulated lock open failure"):
                with tpu_vllm_patches.serialize_subhost_start_session(
                    rank=0,
                    world_size=2,
                    engine_id="eng-abort",
                    lock_path=lock_path,
                    timeout_s=30.0,
                    poll_interval_s=0.005,
                ):
                    pass
        assert sibling_done.wait(timeout=3.0), "Sibling did not unblock on rank-0 ABORT"
    finally:
        sibling.join(timeout=5.0)
        assert not sibling.is_alive()


def test_subhost_start_session_recovers_after_hard_process_crash(tmp_path):
    """If a process holding the host lock dies abruptly (os._exit), the OS drops flock and peer engines proceed."""
    lock_path = str(tmp_path / "start_session.lock")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    crash_script = (
        "import os, sys\n"
        "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import serialize_subhost_start_session\n"
        "with serialize_subhost_start_session(rank=0, world_size=1, engine_id='eng-crash', lock_path=sys.argv[1]):\n"
        "    os._exit(42)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", crash_script, lock_path],
        env={**os.environ, "PYTHONPATH": repo_root},
        timeout=15.0,
    )
    assert proc.returncode == 42

    ran = False
    with tpu_vllm_patches.serialize_subhost_start_session(
        rank=0,
        world_size=1,
        engine_id="eng-after-crash",
        lock_path=lock_path,
        timeout_s=5.0,
        poll_interval_s=0.005,
    ):
        ran = True
    assert ran is True


def test_subhost_start_session_sibling_detects_rank0_hard_crash_promptly(tmp_path):
    """If rank 0 dies abruptly (os._exit) while waiting for the host lock, sibling rank 1 unblocks immediately."""
    lock_path = str(tmp_path / "start_session.lock")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    # Hold the host lock from the test process so rank 0 enters the leader section and blocks on host_fd.
    blocker_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(blocker_fd, fcntl.LOCK_EX)
    sibling_unblocked = threading.Event()
    try:
        rank0_script = (
            "import os, sys, threading\n"
            "from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (\n"
            "    serialize_subhost_start_session,\n"
            ")\n"
            "threading.Timer(0.25, lambda: os._exit(99)).start()\n"
            "with serialize_subhost_start_session(\n"
            "    rank=0, world_size=2, engine_id='eng-r0-crash',\n"
            "    lock_path=sys.argv[1], timeout_s=30.0, poll_interval_s=0.005,\n"
            "):\n"
            "    pass\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", rank0_script, lock_path],
            env={**os.environ, "PYTHONPATH": repo_root},
        )

        def sibling_fn():
            with tpu_vllm_patches.serialize_subhost_start_session(
                rank=1,
                world_size=2,
                engine_id="eng-r0-crash",
                lock_path=lock_path,
                timeout_s=30.0,
                poll_interval_s=0.005,
            ):
                sibling_unblocked.set()

        sibling = threading.Thread(target=sibling_fn)
        sibling.start()
        assert proc.wait(timeout=10.0) == 99
        # Sibling detects leader_path flock drop well before timeout_s=30.0.
        assert sibling_unblocked.wait(timeout=3.0), "Sibling did not detect rank-0 crash promptly"
        sibling.join(timeout=5.0)
        assert not sibling.is_alive()
    finally:
        fcntl.flock(blocker_fd, fcntl.LOCK_UN)
        os.close(blocker_fd)


@pytest.mark.parametrize("rank, world_size", [(-1, 2), (2, 2), (0, 0)])
def test_subhost_start_session_rejects_invalid_rank_or_world_size(tmp_path, rank, world_size):
    with pytest.raises(ValueError, match="Expected 0 <= rank < world_size"):
        with tpu_vllm_patches.serialize_subhost_start_session(
            rank=rank,
            world_size=world_size,
            engine_id="bad",
            lock_path=str(tmp_path / "start_session.lock"),
        ):
            pass
