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
        self.replica_rank = args[0] if args else 0
        self.is_reward_model = kwargs.get("is_reward_model", False)
        self.is_teacher_model = kwargs.get("is_teacher_model", False)
        self.name_suffix = kwargs.get("name_suffix", "")
        self.workers = []
        self.server_class = None
        self.launched = False

    def _get_server_name_prefix(self):
        return "vllm_"

    async def init_standalone(self, resource_pool=None):
        self.standalone_pool = resource_pool

    def _get_server_env_vars(self, node_ids):
        return {"FROM_UPSTREAM": "1"}

    async def launch_servers(self):
        if self.workers:
            worker_node_ids = await asyncio.gather(
                *[
                    worker.__ray_call__.remote(lambda w: ray.get_runtime_context().get_node_id())
                    for worker in self.workers
                ]
            )
            self.server_env_vars = self._get_server_env_vars(worker_node_ids[:: self.gpus_per_replica_node])
        self.launched = True


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


class _FakePool:
    def __init__(self, name, process_on_nodes):
        self.name = name
        self.process_on_nodes = process_on_nodes
        self.created = []

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        self.created.append((strategy, device_name))
        return ["pg"]


class _FakePoolManager:
    def __init__(self, resource_pool_spec, mapping, max_colocate_count):
        self.resource_pool_dict = {name: _FakePool(name, nodes) for name, nodes in resource_pool_spec.items()}

    def create_resource_pool(self):
        pass


def _standalone_replica(tpu_vllm, nnodes, **attrs):
    replica = tpu_vllm.TPUvLLMReplica(3, "config", "model_config")
    replica.__dict__.update(replica_rank=3, nnodes=nnodes, gpus_per_replica_node=4, name_suffix="")
    replica.__dict__.update({"is_reward_model": False, "is_teacher_model": False, **attrs})
    return replica


@pytest.mark.parametrize(
    "kind, pool_name",
    [
        ({}, "rollout_pool_3"),
        ({"is_reward_model": True}, "rollout_pool_reward_3"),
        ({"is_teacher_model": True, "name_suffix": "_org_teacher"}, "rollout_pool_teacher_3_org_teacher"),
    ],
)
def test_single_host_replica_gets_a_strict_pack_placement_group(tpu_vllm, kind, pool_name):
    # verl builds a standalone replica's workers with bin_pack=False (Ray's best-effort PACK), which can split a
    # replica over hosts. A replica on one host hands verl its pool, under verl's name, with the placement group
    # already created by STRICT_PACK; verl's worker group reuses it.
    replica = _standalone_replica(tpu_vllm, nnodes=1, **kind)
    with (
        mock.patch.object(tpu_vllm, "ResourcePoolManager", _FakePoolManager),
        mock.patch.object(tpu_vllm, "get_device_name", return_value="tpu"),
    ):
        asyncio.run(replica.init_standalone())
    pool = replica.standalone_pool
    assert (pool.name, pool.process_on_nodes, pool.created) == (pool_name, [4], [("STRICT_PACK", "tpu")])


def test_other_replicas_keep_verl_init_standalone(tpu_vllm):
    with mock.patch.object(tpu_vllm, "ResourcePoolManager") as manager:
        # A replica over several hosts: verl creates its pool.
        replica = _standalone_replica(tpu_vllm, nnodes=2)
        asyncio.run(replica.init_standalone())
        assert replica.standalone_pool is None
        # A pool the caller passes is used as it is.
        replica = _standalone_replica(tpu_vllm, nnodes=1)
        asyncio.run(replica.init_standalone(resource_pool="caller's pool"))
        assert replica.standalone_pool == "caller's pool"
        # A verl whose init_standalone takes no pool creates its own.
        calls = []

        async def init_standalone_without_pool(self):
            calls.append(self)

        with mock.patch.object(_FakeReplica, "init_standalone", init_standalone_without_pool):
            asyncio.run(replica.init_standalone())
        assert calls == [replica]
    manager.assert_not_called()


# ---------------------------------------------------------------------------
# TPUvLLMReplica._get_server_env_vars / launch_servers
# ---------------------------------------------------------------------------


class _FakeRemoteMethod:
    def __init__(self, result):
        self._result = result

    def remote(self, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        future.set_result(self._result(*args, **kwargs) if callable(self._result) else self._result)
        return future


class _FakeWorker:
    def __init__(self, node_id):
        self._node_id = node_id
        self.__ray_call__ = _FakeRemoteMethod(self._call)

    def _call(self, fn):
        with mock.patch("ray.get_runtime_context", return_value=SimpleNamespace(get_node_id=lambda: self._node_id)):
            return fn(self)


NODE_A = "a" * 56
NODE_B = "b" * 56


def _make_tpu_replica(tpu_vllm, *, replica_rank=0, data_parallel_size=1, nnodes=2, chips_per_host=4):
    replica = tpu_vllm.TPUvLLMReplica(replica_rank, "config", "model_config")
    replica.config = SimpleNamespace(
        data_parallel_size=data_parallel_size,
        tensor_model_parallel_size=chips_per_host * nnodes,
    )
    replica.nnodes = nnodes
    replica.gpus_per_replica_node = chips_per_host
    return replica


def test_replica_server_env_vars_chains_upstream_and_sets_mp_backend(tpu_vllm):
    replica = _make_tpu_replica(tpu_vllm, nnodes=2, chips_per_host=4)
    assert replica._get_server_env_vars([NODE_A, NODE_B]) == {
        "FROM_UPSTREAM": "1",
        "TPU_MULTIHOST_BACKEND": "mp",
    }


def test_replica_launch_servers_delegates_to_upstream_and_rejects_data_parallel(tpu_vllm):
    replica = _make_tpu_replica(tpu_vllm, data_parallel_size=1)
    asyncio.run(replica.launch_servers())
    assert replica.launched is True

    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(_make_tpu_replica(tpu_vllm, data_parallel_size=2).launch_servers())


@pytest.mark.parametrize("chips", [4, 8])
def test_replica_launch_servers_rejects_a_single_host_replica_split_across_hosts(tpu_vllm, chips):
    # Ray's best-effort PACK put half of a single-host replica's workers on another host. Upstream launch_servers
    # inspects only worker 0's host when nnodes == 1, so TPUvLLMReplica checks every worker before starting a server.
    replica = _make_tpu_replica(tpu_vllm, replica_rank=3, nnodes=1, chips_per_host=chips)
    replica.workers = [_FakeWorker(node) for node in (NODE_A, NODE_B) for _ in range(chips // 2)]
    with pytest.raises(RuntimeError, match=f"replica 3 must run on one host, but Ray placed its {chips} workers on 2"):
        asyncio.run(replica.launch_servers())
    assert replica.launched is False


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
def test_replica_single_host_topology_from_ray(tpu_vllm, monkeypatch, pod_type, chips, topology):
    # The launcher may run on a CPU Ray head: no TPU_ACCELERATOR_TYPE, but the slice's TORCH_TPU_TOPOLOGY may leak.
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "2,2,4")
    replica = _make_tpu_replica(tpu_vllm, nnodes=1, chips_per_host=chips)
    nodes = [
        {"Alive": True, "Resources": {"TPU": 4.0}, "NodeID": NODE_B, "Labels": {"ray.io/tpu-pod-type": "v6e-4"}},
        {"Alive": True, "Resources": {"TPU": 4.0}, "NodeID": NODE_A, "Labels": {"ray.io/tpu-pod-type": pod_type}},
    ]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        env_vars = replica._get_server_env_vars([NODE_A])
    assert env_vars.get("TORCH_TPU_TOPOLOGY") == topology
    assert env_vars["FROM_UPSTREAM"] == "1"
    assert env_vars["TPU_MULTIHOST_BACKEND"] == "mp"


# ---------------------------------------------------------------------------
# Several engines on one TPU 7x host (8 x TP=4, 16 x TP=2)
# ---------------------------------------------------------------------------


def _single_host_replica(tpu_vllm, replica_rank, chips, node=NODE_A):
    replica = _make_tpu_replica(tpu_vllm, replica_rank=replica_rank, nnodes=1, chips_per_host=chips)
    replica.workers = [_FakeWorker(node) for _ in range(chips)]
    return replica


def _launch_on_v7x_hosts(replicas):
    """Launch the replicas concurrently, as verl does, on single-host TPU 7x nodes; return their server env."""
    nodes = [
        {"Alive": True, "Resources": {"TPU": 8.0}, "NodeID": node, "Labels": {"ray.io/tpu-pod-type": "v7x-8"}}
        for node in (NODE_A, NODE_B)
    ]

    async def run():
        with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
            await asyncio.gather(*[replica.launch_servers() for replica in replicas])

    asyncio.run(run())
    return [replica.server_env_vars for replica in replicas]


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
def test_replica_gives_engines_on_one_host_disjoint_devices_and_ports(tpu_vllm, chips, topology):
    # Replicas arrive in any order (asyncio.gather): here the highest rank first.
    per_host = 8 // chips
    replicas = [_single_host_replica(tpu_vllm, rank, chips) for rank in reversed(range(per_host))]
    envs = _launch_on_v7x_hosts(replicas)

    assert [int(env["TPU_LOCAL_RANK_OFFSET"]) for env in envs] == [slot * chips for slot in range(per_host)]
    seen_ports = set()
    for replica, env in zip(replicas, envs, strict=True):
        assert env["FROM_UPSTREAM"] == "1"
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


def test_replica_rejects_more_engines_than_a_host_holds(tpu_vllm):
    with pytest.raises(ValueError, match="no free slot"):
        _launch_on_v7x_hosts([_single_host_replica(tpu_vllm, rank, 4) for rank in range(3)])


def test_replica_slots_are_per_host(tpu_vllm):
    replicas = [
        _single_host_replica(tpu_vllm, rank, 4, node) for rank, node in enumerate((NODE_A, NODE_B, NODE_A, NODE_B))
    ]
    envs = _launch_on_v7x_hosts(replicas)
    assert [env["TPU_LOCAL_RANK_OFFSET"] for env in envs] == ["0", "0", "4", "4"]


@pytest.mark.parametrize("pod_type, chips", [("v7x-8", 8), ("v7x-8", 1), ("v6e-8", 4)])
def test_replica_whole_host_and_other_replicas_are_unchanged(tpu_vllm, monkeypatch, pod_type, chips):
    # A TP=8 replica owns its TPU 7x host; TP=1 (a standalone process opens only a chip's first core) and other
    # generations are not sub-host engines either. Their server env is what it was before.
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    replica = _single_host_replica(tpu_vllm, 0, chips)
    nodes = [{"Alive": True, "Resources": {"TPU": 8.0}, "NodeID": NODE_A, "Labels": {"ray.io/tpu-pod-type": pod_type}}]
    with mock.patch("ray.is_initialized", return_value=True), mock.patch("ray.nodes", return_value=nodes):
        asyncio.run(replica.launch_servers())
    topology = tpu_vllm.single_host_tpu_topology(chips, pod_type=pod_type)
    assert replica.server_env_vars == {
        "FROM_UPSTREAM": "1",
        "TPU_MULTIHOST_BACKEND": "mp",
        "TORCH_TPU_TOPOLOGY": topology,
    }
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
