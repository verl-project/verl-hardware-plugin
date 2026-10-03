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
from verl_hardware_plugin.rollout import tpu_vllm_patches

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
        sys.modules.pop("verl_hardware_plugin.rollout.tpu_vllm", None)
        module = importlib.import_module("verl_hardware_plugin.rollout.tpu_vllm")
        yield module
        sys.modules.pop("verl_hardware_plugin.rollout.tpu_vllm", None)


def _done(value):
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


# ---------------------------------------------------------------------------
# Replica loader registration
# ---------------------------------------------------------------------------


def test_vllm_loader_defers_off_tpu_and_picks_tpu_replica_on_tpu():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from verl_hardware_plugin.rollout import register_all_rollouts

    sentinel_upstream = type("UpstreamReplica", (), {})
    sentinel_tpu = type("TPUvLLMReplica", (), {})
    fake_tpu_module = ModuleType("verl_hardware_plugin.rollout.tpu_vllm")
    fake_tpu_module.TPUvLLMReplica = sentinel_tpu

    with mock.patch.dict(RolloutReplicaRegistry._registry, {"vllm": lambda: sentinel_upstream}):
        register_all_rollouts()

        with mock.patch("verl.utils.device.get_resource_name", return_value="GPU"):
            assert RolloutReplicaRegistry.get("vllm") is sentinel_upstream

        with (
            mock.patch("verl.utils.device.get_resource_name", return_value="TPU"),
            mock.patch.dict(sys.modules, {"verl_hardware_plugin.rollout.tpu_vllm": fake_tpu_module}),
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


def test_server_engine_kwargs_force_tpu_executor(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    engine_kwargs = {"distributed_executor_backend": "mp", "enable_sleep_mode": True}
    server._preprocess_engine_kwargs(engine_kwargs)

    assert server.calls == ["preprocess"]  # upstream validation still runs first
    assert engine_kwargs == {
        "distributed_executor_backend": "external_launcher",
        "enable_sleep_mode": False,
        "from_upstream": True,
    }


def test_server_worker_extension_defers_to_upstream(tpu_vllm):
    upstream_cls = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
    server = tpu_vllm.TPUvLLMHttpServer()
    assert server._get_worker_extension_cls() == upstream_cls
    assert "_get_worker_extension_cls" not in tpu_vllm.TPUvLLMHttpServer.__dict__
    # No run_server override: vLLM v0.29 has no V0 engine to switch off.
    assert "run_server" not in tpu_vllm.TPUvLLMHttpServer.__dict__


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
        self.options_kwargs = None
        self.init_kwargs = None
        self.server = _FakeServer()

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, **kwargs):
        self.init_kwargs = kwargs
        return self.server


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
        resource_pool=SimpleNamespace(
            get_placement_groups=lambda device_name=None: [
                SimpleNamespace(id=SimpleNamespace(hex=lambda: "pg0hex")),
                SimpleNamespace(id=SimpleNamespace(hex=lambda: "pg1hex")),
            ]
        ),
        servers=[],
        server_class=_FakeServerClass(),
        _get_server_name_prefix=lambda: "vllm_",
    )


def test_launch_tpu_vllm_servers_single_server_spanning_all_workers(tpu_vllm):
    replica = _fake_replica()
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET"], rollout_env_vars=lambda: {})

    async def run():
        with mock.patch.object(tpu_vllm, "get_platform", return_value=platform):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    server_class = replica.server_class
    options = server_class.options_kwargs
    assert options["name"] == "vllm_server_3_0"
    assert options["max_concurrency"] == 1234
    assert options["scheduling_strategy"].node_id == NODE_A
    assert options["runtime_env"] == {
        "env_vars": {
            "RAY_NOSET": "1",
            "TPU_MULTIHOST_BACKEND": "ray",
            "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "0",
            "VERL_TPU_PG_IDS": "pg0hex,pg1hex",
        }
    }

    init = server_class.init_kwargs
    assert init["workers"] is replica.workers
    assert init["node_rank"] == 0
    assert init["nnodes"] == 2
    assert init["cuda_visible_devices"] == ""

    assert server_class.server.launch_kwargs == {"master_address": "10.0.0.1", "master_port": 1234, "dp_rpc_port": 5678}
    assert replica.servers == [server_class.server]
    assert replica._server_address == "10.0.0.1:8000"


def test_launch_tpu_vllm_servers_rejects_data_parallel(tpu_vllm):
    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(tpu_vllm.launch_tpu_vllm_servers(_fake_replica(data_parallel_size=2)))


# ---------------------------------------------------------------------------
# patch_vllm_for_tpu (without vllm-torchtpu installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_patches():
    """Run patch_vllm_for_tpu with its process-global side effects rolled back afterwards."""

    base_process = multiprocessing.process.BaseProcess
    saved_init = base_process.__init__
    had_flag = "_tpu_patched" in base_process.__dict__
    env = {k: v for k, v in os.environ.items() if k != "VLLM_DISABLE_COMPILE_CACHE"}
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
    ):
        yield tpu_vllm_patches
    base_process.__init__ = saved_init
    if not had_flag and "_tpu_patched" in base_process.__dict__:
        del base_process._tpu_patched


def test_patch_vllm_for_tpu_env_side_effects_without_torchtpu(isolated_patches):
    isolated_patches.patch_vllm_for_tpu()

    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert multiprocessing.process.BaseProcess._tpu_patched is True
    # The executor patches need vllm-torchtpu, so the guard stays open for a later retry.
    assert isolated_patches._PATCHES_APPLIED is False


def test_patch_vllm_for_tpu_is_a_no_op_once_applied(isolated_patches):
    isolated_patches._PATCHES_APPLIED = True
    # vLLM is made unimportable, so getting past the guard would log "Skipping vLLM TPU executor patches".
    with mock.patch.dict(sys.modules, {"vllm": None}), mock.patch.object(isolated_patches.logger, "debug") as debug:
        isolated_patches.patch_vllm_for_tpu()
    debug.assert_not_called()
    assert "VLLM_DISABLE_COMPILE_CACHE" not in os.environ


def test_patch_vllm_for_tpu_keeps_explicit_compile_cache_setting(isolated_patches):
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
    isolated_patches.patch_vllm_for_tpu()
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "0"


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


def test_local_ranks_count_workers_per_host_in_rank_order():
    assert tpu_vllm_patches._local_ranks(["b", "b", "a", "a", "a"]) == [0, 1, 0, 1, 2]


def test_tpu_worker_envs_two_hosts_of_four_chips():
    ips = ["10.0.0.2"] * 4 + ["10.0.0.1"] * 4  # rank order: the driver's host first
    env = {k: v for k, v in os.environ.items() if k != "TORCH_TPU_TOPOLOGY"}
    with mock.patch.dict(os.environ, env, clear=True):
        envs = tpu_vllm_patches._tpu_worker_envs(ips, base_port=8070)

    addresses = ",".join(f"{ip}:{8070 + chip}" for ip in ("10.0.0.2", "10.0.0.1") for chip in range(4))
    assert envs[5] == {
        "TPU_VISIBLE_CHIPS": "1",
        "TPU_PROCESS_PORT": "8071",
        "TPU_PROCESS_ADDRESSES": addresses,
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
        "CLOUD_TPU_TASK_ID": "1",
        "TPU_WORKER_HOSTNAMES": "10.0.0.2,10.0.0.1",
        # One process per chip: libtpu sees 8 single-chip "hosts" laid out as the 2x4 slice.
        "TORCH_TPU_TOPOLOGY": "2,4,1",
        "TPU_HOST_BOUNDS": "2,4,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
    }
    assert [e["TPU_VISIBLE_CHIPS"] for e in envs] == ["0", "1", "2", "3"] * 2
    assert [e["CLOUD_TPU_TASK_ID"] for e in envs] == ["0"] * 4 + ["1"] * 4
