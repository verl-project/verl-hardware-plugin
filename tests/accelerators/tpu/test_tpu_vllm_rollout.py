# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU vLLM rollout glue.

vLLM, vllm-torchtpu and torch_tpu are not installed on CI, so ``vllm_async_server`` is replaced
by a minimal stand-in and every Ray actor handle is faked. The tests pin the contract with
upstream verl (which hooks are overridden and what they return) and the launch arguments.
"""

import asyncio
import importlib
import multiprocessing.process
import os
import pickle
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import ray
import ray.util.scheduling_strategies  # noqa: F401  (tpu_vllm reaches it through ``ray.util``)

# Import before any test patches sys.modules, so mock.patch.dict never evicts it.
from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches

VAS = "verl.workers.rollout.vllm_rollout.vllm_async_server"


class _FakeHttpServer:
    """Stands in for upstream ``vLLMHttpServer``: records the hooks the TPU subclass chains to."""

    def __init__(self):
        self.calls = []

    def _preprocess_engine_kwargs(self, engine_kwargs):
        self.calls.append("preprocess")
        engine_kwargs.setdefault("from_upstream", True)

    async def run_server(self, args):
        self.calls.append(("run_server", args))
        return "ran"

    def _get_worker_extension_cls(self):
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"


class _FakeReplica:
    """Stands in for upstream ``vLLMReplica``."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.server_class = None

    def _get_server_name_prefix(self):
        return "vllm_"


@pytest.fixture
def tpu_vllm():
    """Import ``tpu_vllm`` against a stubbed ``vllm_async_server``, with ``patch_vllm_for_tpu`` mocked."""
    vas = ModuleType(VAS)
    vas.vLLMHttpServer = _FakeHttpServer
    vas.vLLMReplica = _FakeReplica
    with (
        mock.patch.dict(sys.modules, {VAS: vas}),
        mock.patch.object(tpu_vllm_patches, "patch_vllm_for_tpu"),
    ):
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm", None)
        module = importlib.import_module("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
        yield module
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm", None)


def _done(value):
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


# ---------------------------------------------------------------------------
# Replica loader registration
# ---------------------------------------------------------------------------


def test_vllm_loader_defers_off_tpu_and_picks_tpu_replica_on_tpu():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from verl_hardware_plugin.accelerators.tpu.registration import register_rollout

    sentinel_upstream = type("UpstreamReplica", (), {})
    sentinel_tpu = type("TPUvLLMReplica", (), {})
    fake_tpu_module = ModuleType("verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm")
    fake_tpu_module.TPUvLLMReplica = sentinel_tpu

    with mock.patch.dict(RolloutReplicaRegistry._registry, {"vllm": lambda: sentinel_upstream}):
        register_rollout()

        with mock.patch("verl.utils.device.get_resource_name", return_value="GPU"):
            assert RolloutReplicaRegistry.get("vllm") is sentinel_upstream

        with (
            mock.patch("verl.utils.device.get_resource_name", return_value="TPU"),
            mock.patch.dict(sys.modules, {"verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm": fake_tpu_module}),
        ):
            assert RolloutReplicaRegistry.get("vllm") is sentinel_tpu


# ---------------------------------------------------------------------------
# TPUvLLMHttpServer / TPUvLLMReplica hooks
# ---------------------------------------------------------------------------


def test_server_patches_vllm_before_upstream_init(tpu_vllm):
    assert not tpu_vllm.patch_vllm_for_tpu.called  # importing the module patches nothing
    order = []
    tpu_vllm.patch_vllm_for_tpu.side_effect = lambda: order.append("patch")
    with mock.patch.object(_FakeHttpServer, "__init__", lambda self: order.append("upstream init")):
        tpu_vllm.TPUvLLMHttpServer()
    assert order == ["patch", "upstream init"]


def test_server_engine_kwargs_disable_sleep_mode_and_keep_mp_executor(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    engine_kwargs = {"enable_sleep_mode": True}
    server._preprocess_engine_kwargs(engine_kwargs)

    assert server.calls == ["preprocess"]  # upstream validation still runs first
    # No distributed_executor_backend: upstream's "mp" stays, and vllm-torchtpu picks its mp multi-host executor.
    assert engine_kwargs == {"enable_sleep_mode": False, "from_upstream": True}


def test_server_worker_extension_defers_to_upstream(tpu_vllm):
    upstream_cls = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
    server = tpu_vllm.TPUvLLMHttpServer()
    assert server._get_worker_extension_cls() == upstream_cls  # no rollout config
    for backend in ("ray-plasma-store", "tpu"):  # "tpu": former name of ray-plasma-store
        server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend=backend))
        assert server._get_worker_extension_cls() == upstream_cls
    # No run_server override: vLLM v0.29 has no V0 engine to switch off.
    assert "run_server" not in tpu_vllm.TPUvLLMHttpServer.__dict__


@pytest.mark.parametrize("backend", ["tpu-sync"])
def test_server_worker_extension_for_tpu_sync(tpu_vllm, backend):
    server = tpu_vllm.TPUvLLMHttpServer()
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend=backend))
    assert server._get_worker_extension_cls() == tpu_vllm.TPU_SYNC_WORKER_EXTENSION_CLS
    module_name, cls_name = tpu_vllm.TPU_SYNC_WORKER_EXTENSION_CLS.rsplit(".", 1)
    assert (module_name, cls_name) == (
        "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension",
        "vLLMTPUSyncWorkerExtension",
    )


def test_replica_uses_tpu_server(tpu_vllm):
    replica = tpu_vllm.TPUvLLMReplica(0, "config", "model_config")
    assert replica.server_class is not None
    assert replica.args == (0, "config", "model_config")


# ---------------------------------------------------------------------------
# launch_tpu_vllm_servers
# ---------------------------------------------------------------------------


class _FakeRemoteMethod:
    def __init__(self, result):
        self._result = result

    def remote(self, *args, **kwargs):
        return _done(self._result(*args, **kwargs) if callable(self._result) else self._result)


class _FakeWorker:
    def __init__(self, node_id):
        self._node_id = node_id
        self.__ray_call__ = _FakeRemoteMethod(self._call)

    def _call(self, fn):
        # launch_tpu_vllm_servers asks the first worker for its node id.
        with mock.patch("ray.get_runtime_context", return_value=SimpleNamespace(get_node_id=lambda: self._node_id)):
            return fn(self)


class _FakeServer:
    get_master_address = _FakeRemoteMethod(("10.0.0.1", 1234, 5678))
    get_server_address = _FakeRemoteMethod(("10.0.0.1", 8000))

    def __init__(self):
        self.launch_kwargs = None
        self.launch_server = _FakeRemoteMethod(self._launch)

    def _launch(self, **kwargs):
        self.launch_kwargs = kwargs


class _FakeServerClass:
    def __init__(self):
        self.options_kwargs = []
        self.init_kwargs = []
        self.servers = []

    def options(self, **kwargs):
        self.options_kwargs.append(kwargs)
        return self

    def remote(self, **kwargs):
        self.init_kwargs.append(kwargs)
        self.servers.append(_FakeServer())
        return self.servers[-1]


NODE_A = "a" * 56
NODE_B = "b" * 56


def _fake_replica(data_parallel_size=1):
    workers = [_FakeWorker(node) for node in (NODE_A, NODE_B) for _ in range(4)]
    return SimpleNamespace(
        config=SimpleNamespace(
            data_parallel_size=data_parallel_size, tensor_model_parallel_size=8, ray_actor_max_concurrency=1234
        ),
        model_config="model_config",
        rollout_mode="standalone",
        workers=workers,
        replica_rank=3,
        name_suffix="",
        is_reward_model=False,
        is_teacher_model=False,
        gpus_per_replica_node=4,
        nnodes=2,
        servers=[],
        server_class=_FakeServerClass(),
        _get_server_name_prefix=lambda: "vllm_",
    )


def test_launch_tpu_vllm_servers_one_server_per_host(tpu_vllm):
    replica = _fake_replica()
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET"], rollout_env_vars=lambda: {})

    async def run():
        with mock.patch.object(tpu_vllm, "get_platform", return_value=platform):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    server_class = replica.server_class
    assert [o["name"] for o in server_class.options_kwargs] == ["vllm_server_3_0", "vllm_server_3_1"]
    assert [o["scheduling_strategy"].node_id for o in server_class.options_kwargs] == [NODE_A, NODE_B]
    for options in server_class.options_kwargs:
        assert options["max_concurrency"] == 1234
        assert options["runtime_env"] == {"env_vars": {"RAY_NOSET": "1", "TPU_MULTIHOST_BACKEND": "mp"}}

    for node_rank, init in enumerate(server_class.init_kwargs):
        assert init["workers"] == replica.workers[node_rank * 4 : (node_rank + 1) * 4]
        assert init["node_rank"] == node_rank
        assert init["nnodes"] == 2
        assert init["gpus_per_node"] == 4
        assert init["cuda_visible_devices"] == ""

    # Every host joins the engine at the master address of node 0.
    for server in server_class.servers:
        assert server.launch_kwargs == {"master_address": "10.0.0.1", "master_port": 1234, "dp_rpc_port": 5678}
    assert replica.servers == server_class.servers
    assert replica._server_handle is server_class.servers[0]
    assert replica._server_address == "10.0.0.1:8000"


def test_launch_tpu_vllm_servers_rejects_data_parallel(tpu_vllm):
    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(tpu_vllm.launch_tpu_vllm_servers(_fake_replica(data_parallel_size=2)))


@pytest.mark.parametrize(
    "pod_type, chips, topology",
    [
        ("v7x-8", 8, "2,2,1,2"),
        ("v6e-8", 8, "2,4,1"),
        ("v5p-16", 4, "2,2,1"),
        # TP=2 / TP=4 on TPU 7x: the mesh of the replica's one or two chips.
        ("v7x-8", 2, "1,1,1,2"),
        ("v7x-8", 4, "1,2,1,2"),
        # No known mesh: the replica keeps the inherited value instead of failing.
        ("v7x-8", 6, None),
    ],
)
def test_launch_tpu_vllm_servers_single_host_topology_from_ray(tpu_vllm, monkeypatch, pod_type, chips, topology):
    # The launcher may run on a CPU Ray head: no TPU_ACCELERATOR_TYPE, but the slice's TORCH_TPU_TOPOLOGY may leak.
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,2,4")
    replica = _fake_replica()
    replica.workers = [_FakeWorker(NODE_A) for _ in range(chips)]
    replica.gpus_per_replica_node = chips
    replica.nnodes = 1
    nodes = [
        {"Alive": True, "Resources": {"TPU": 4.0}, "NodeID": NODE_B, "Labels": {"ray.io/tpu-pod-type": "v6e-4"}},
        {"Alive": True, "Resources": {"TPU": 4.0}, "NodeID": NODE_A, "Labels": {"ray.io/tpu-pod-type": pod_type}},
    ]
    platform = SimpleNamespace(ray_noset_envvars=lambda: [], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_vllm, "get_platform", return_value=platform),
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=nodes),
        ):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    [options] = replica.server_class.options_kwargs
    assert options["runtime_env"]["env_vars"].get("TORCH_TPU_TOPOLOGY") == topology


# ---------------------------------------------------------------------------
# Several engines on one TPU 7x host (8 x TP=4, 16 x TP=2)
# ---------------------------------------------------------------------------


def _single_host_replica(replica_rank, chips, node=NODE_A):
    replica = _fake_replica()
    replica.workers = [_FakeWorker(node) for _ in range(chips)]
    replica.gpus_per_replica_node = chips
    replica.nnodes = 1
    replica.replica_rank = replica_rank
    return replica


def _launch_on_v7x_hosts(tpu_vllm, replicas):
    """Launch the replicas concurrently, as verl does, on single-host TPU 7x nodes; return their server env."""
    nodes = [
        {"Alive": True, "Resources": {"TPU": 8.0}, "NodeID": node, "Labels": {"ray.io/tpu-pod-type": "v7x-8"}}
        for node in (NODE_A, NODE_B)
    ]
    platform = SimpleNamespace(ray_noset_envvars=lambda: [], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_vllm, "get_platform", return_value=platform),
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=nodes),
        ):
            await asyncio.gather(*[tpu_vllm.launch_tpu_vllm_servers(replica) for replica in replicas])

    asyncio.run(run())
    envs = []
    for replica in replicas:
        [options] = replica.server_class.options_kwargs
        envs.append(options["runtime_env"]["env_vars"])
    return envs


def _engine_ports(env, chips):
    """Every port an engine and its workers listen on."""
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import (
        SUBHOST_MEGASCALE_PORT_ENV,
        subhost_worker_env,
    )

    megascale = int(env[SUBHOST_MEGASCALE_PORT_ENV])
    ports = {int(env["MASTER_PORT"]), int(env["MEGASCALE_COORDINATOR_ADDRESS"].rsplit(":", 1)[1])}
    ports |= {int(address.rsplit(":", 1)[1]) for address in env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")}
    for rank in range(chips):
        worker = subhost_worker_env(rank, int(env["TPU_LOCAL_RANK_OFFSET"]) + rank, megascale)
        ports |= {int(worker["MEGASCALE_PORT"]), int(worker["MEGASCALE_DEBUG_PORT"])}
    return ports


@pytest.mark.parametrize("chips, topology", [(4, "1,2,1,2"), (2, "1,1,1,2")])
def test_launch_tpu_vllm_servers_gives_engines_on_one_host_disjoint_devices_and_ports(tpu_vllm, chips, topology):
    # Replicas arrive in any order (asyncio.gather): here the highest rank first.
    per_host = 8 // chips
    replicas = [_single_host_replica(rank, chips) for rank in reversed(range(per_host))]
    envs = _launch_on_v7x_hosts(tpu_vllm, replicas)

    assert [int(env["TPU_LOCAL_RANK_OFFSET"]) for env in envs] == [slot * chips for slot in range(per_host)]
    seen_ports = set()
    for replica, env in zip(replicas, envs, strict=True):
        assert env["TORCH_TPU_TOPOLOGY"] == topology
        assert env["DEBUG_TPU_LOCAL_RANK_OFFSET"] == env["TPU_LOCAL_RANK_OFFSET"]
        assert env["TPU_MULTIHOST_BACKEND"] == "mp"
        assert env["MASTER_ADDR"] == "127.0.0.1"
        assert len(env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == chips
        for name, prefix in (
            ("VLLM_CACHE_ROOT", "/tmp/vllm_cache"),
            ("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor_cache"),
            ("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", "/tmp/tpu_tier3"),
        ):
            assert env[name] == f"{prefix}_tp{chips}_r{replica.replica_rank}"
        ports = _engine_ports(env, chips)
        assert all(1024 <= port < 32768 for port in ports)  # below Linux's ephemeral range, which portpicker uses
        assert seen_ports.isdisjoint(ports)
        seen_ports |= ports


def test_launch_tpu_vllm_servers_rejects_more_engines_than_a_host_holds(tpu_vllm):
    with pytest.raises(ValueError, match="no free slot"):
        _launch_on_v7x_hosts(tpu_vllm, [_single_host_replica(rank, 4) for rank in range(3)])


def test_launch_tpu_vllm_servers_slots_are_per_host(tpu_vllm):
    replicas = [_single_host_replica(rank, 4, node) for rank, node in enumerate((NODE_A, NODE_B, NODE_A, NODE_B))]
    envs = _launch_on_v7x_hosts(tpu_vllm, replicas)
    assert [env["TPU_LOCAL_RANK_OFFSET"] for env in envs] == ["0", "0", "4", "4"]


@pytest.mark.parametrize("pod_type, chips", [("v7x-8", 8), ("v7x-8", 1), ("v6e-8", 4)])
def test_launch_tpu_vllm_servers_whole_host_and_other_replicas_are_unchanged(tpu_vllm, monkeypatch, pod_type, chips):
    # A TP=8 replica owns its TPU 7x host; TP=1 (a standalone process opens only a chip's first core) and other
    # generations are not sub-host engines either. Their server env is what it was before.
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    replica = _single_host_replica(0, chips)
    nodes = [{"Alive": True, "Resources": {"TPU": 8.0}, "NodeID": NODE_A, "Labels": {"ray.io/tpu-pod-type": pod_type}}]
    platform = SimpleNamespace(ray_noset_envvars=lambda: [], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_vllm, "get_platform", return_value=platform),
            mock.patch("ray.is_initialized", return_value=True),
            mock.patch("ray.nodes", return_value=nodes),
        ):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())
    [options] = replica.server_class.options_kwargs
    topology = tpu_vllm.single_host_tpu_topology(chips, pod_type=pod_type)
    assert options["runtime_env"] == {"env_vars": {"TPU_MULTIHOST_BACKEND": "mp", "TORCH_TPU_TOPOLOGY": topology}}
    assert tpu_vllm._HOST_SLOTS == {}


def test_host_slots_never_move_and_survive_restarts(tpu_vllm):
    alloc = tpu_vllm.allocate_host_slot
    # Forward and reverse arrival of two TP=4 engines: neither ever shares a slot.
    assert [alloc("a", engine, 4, 8) for engine in ("r0", "r4")] == [0, 1]
    assert [alloc("b", engine, 4, 8) for engine in ("r4", "r0")] == [0, 1]
    # Out-of-order arrival of four TP=2 engines.
    assert [alloc("c", engine, 2, 8) for engine in ("r12", "r4", "r8", "r0")] == [0, 1, 2, 3]
    # A restarted engine (same name) gets its slot back without disturbing the others.
    assert alloc("b", "r4", 4, 8) == 0 and alloc("b", "r0", 4, 8) == 1
    # A full host rejects a third TP=4 engine; releasing one frees its slot for a replacement.
    with pytest.raises(ValueError, match="no free slot"):
        alloc("b", "r8", 4, 8)
    tpu_vllm.release_host_slot("b", "r4")
    assert alloc("b", "r8", 4, 8) == 0 and alloc("b", "r0", 4, 8) == 1
    # Engines of different sizes cannot share a host, and a size must tile the host.
    with pytest.raises(ValueError, match="already runs engines of 4 devices"):
        alloc("b", "r9", 2, 8)
    with pytest.raises(ValueError, match="do not tile"):
        alloc("d", "r0", 3, 8)
    tpu_vllm.reset_host_slots("b")
    assert alloc("b", "r9", 2, 8) == 0
    tpu_vllm.reset_host_slots()
    assert tpu_vllm._HOST_SLOTS == {} and tpu_vllm._HOST_ENGINE_DEVICES == {}


def test_subhost_port_base_is_configurable(tpu_vllm, monkeypatch):
    monkeypatch.setenv(tpu_vllm.SUBHOST_PORT_BASE_ENV, "21000")
    env = tpu_vllm.subhost_engine_env(slot=1, devices_per_engine=4, replica_rank=5)
    assert env["MEGASCALE_COORDINATOR_ADDRESS"] == "127.0.0.1:21080"
    assert env["MASTER_PORT"] == "21280"
    assert env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] == "localhost:21380,localhost:21381,localhost:21382,localhost:21383"


# ---------------------------------------------------------------------------
# patch_vllm_for_tpu (without vllm-torchtpu installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_patches():
    """Run patch_vllm_for_tpu with its process-global side effects rolled back afterwards."""

    base_process = multiprocessing.process.BaseProcess
    saved_init = base_process.__init__
    had_flag = "_tpu_patched" in base_process.__dict__
    with mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False):
        yield tpu_vllm_patches
    base_process.__init__ = saved_init
    if not had_flag and "_tpu_patched" in base_process.__dict__:
        del base_process._tpu_patched


def test_patch_vllm_for_tpu_side_effects_without_torchtpu(isolated_patches):
    with mock.patch.dict(os.environ):
        os.environ.pop("VLLM_DISABLE_COMPILE_CACHE", None)
        isolated_patches.patch_vllm_for_tpu()
        # vLLM's compile cache is left on (the plugin no longer sets VLLM_DISABLE_COMPILE_CACHE).
        assert "VLLM_DISABLE_COMPILE_CACHE" not in os.environ
    assert multiprocessing.process.BaseProcess._tpu_patched is True
    # The executor patches need vllm-torchtpu, so the guard stays open for a later retry.
    assert isolated_patches._PATCHES_APPLIED is False


def test_patch_vllm_for_tpu_is_a_no_op_once_applied(isolated_patches):
    isolated_patches._PATCHES_APPLIED = True
    # vLLM is made unimportable, so getting past the guard would log "Skipping vLLM TPU executor patches".
    with mock.patch.dict(sys.modules, {"vllm": None}), mock.patch.object(isolated_patches.logger, "debug") as debug:
        isolated_patches.patch_vllm_for_tpu()
    debug.assert_not_called()


def test_pickleable_process_wrapper_applies_patches_before_target():
    wrapper = pickle.loads(pickle.dumps(tpu_vllm_patches.PickleableProcessWrapper(len)))
    with mock.patch.object(tpu_vllm_patches, "patch_vllm_for_tpu") as patch:
        assert wrapper([1, 2, 3]) == 3
    patch.assert_called_once_with()


def test_multiprocessing_patch_wraps_keyword_and_positional_targets(isolated_patches):
    isolated_patches.patch_multiprocessing_for_tpu()
    for process in (multiprocessing.Process(target=len), multiprocessing.Process(None, len)):
        assert isinstance(process._target, isolated_patches.PickleableProcessWrapper)
        assert process._target.target is len
    assert multiprocessing.Process()._target is None
