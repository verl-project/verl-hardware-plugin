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
    """Import ``tpu_vllm`` against a stubbed ``vllm_async_server`` on a non-TPU host."""
    vas = ModuleType(VAS)
    vas.vLLMHttpServer = _FakeHttpServer
    vas.vLLMReplica = _FakeReplica
    env = {k: v for k, v in os.environ.items() if k not in ("VERL_PLATFORM", "VLLM_DISABLE_COMPILE_CACHE")}
    with mock.patch.dict(sys.modules, {VAS: vas}), mock.patch.dict(os.environ, env, clear=True):
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
        loader = RolloutReplicaRegistry._registry["vllm"]
        register_all_rollouts()  # idempotent: must not wrap the plugin loader again
        assert RolloutReplicaRegistry._registry["vllm"] is loader

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


def test_module_import_does_not_patch_off_tpu(tpu_vllm):

    assert not tpu_vllm.is_tpu_vllm_run()
    assert not tpu_vllm_patches._PATCHES_APPLIED


def test_server_engine_kwargs_force_tpu_executor(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()
    engine_kwargs = {"distributed_executor_backend": "mp", "enable_sleep_mode": True}
    with mock.patch.object(tpu_vllm, "prepare_tpu_server_env") as prepare_env:
        server._preprocess_engine_kwargs(engine_kwargs)

    assert server.calls == ["preprocess"]  # upstream validation still runs first
    assert engine_kwargs == {
        "distributed_executor_backend": "external_launcher",
        "enable_sleep_mode": False,
        "from_upstream": True,
    }
    prepare_env.assert_called_once_with()


def test_prepare_tpu_server_env_selects_ray_multihost_backend(tpu_vllm):
    tpu_vllm.prepare_tpu_server_env()
    assert os.environ["TPU_MULTIHOST_BACKEND"] == "ray"
    assert os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] == "0"
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"


def test_server_worker_extension_defers_to_upstream_unless_raiden(tpu_vllm):
    upstream_cls = "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"
    server = tpu_vllm.TPUvLLMHttpServer()
    assert server._get_worker_extension_cls() == upstream_cls
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="tpu"))
    assert server._get_worker_extension_cls() == upstream_cls
    server.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="raiden"))
    assert server._get_worker_extension_cls() == tpu_vllm.RAIDEN_WORKER_EXTENSION_CLS
    # No run_server override: vLLM v0.29 has no V0 engine to switch off.
    assert "run_server" not in tpu_vllm.TPUvLLMHttpServer.__dict__


def test_server_collective_rpc_returns_engine_result(tpu_vllm):
    server = tpu_vllm.TPUvLLMHttpServer()

    async def fake_collective_rpc(**kwargs):
        return [kwargs["method"], kwargs["args"]]

    server.engine = SimpleNamespace(collective_rpc=fake_collective_rpc)
    assert asyncio.run(server.collective_rpc("probe", args=(1,))) == ["probe", (1,)]


def test_replica_uses_tpu_server_and_no_gpu(tpu_vllm):
    replica = tpu_vllm.TPUvLLMReplica(0, "config", "model_config")
    assert replica.rollout_worker_use_gpu() is False
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
    def __init__(self, node_id, chip, env):
        self._node_id, self._chip, self._env = node_id, chip, env
        self.__ray_call__ = _FakeRemoteMethod(self._call)

    def _call(self, fn):
        if getattr(fn, "__name__", "") == "probe_stale_tpu_engines":
            return {"hostname": f"host-{self._node_id}", "zombies": 0, "engines": 0}
        # get_tpu_server_launch_config sends two lambdas: (node, chips) and the env filter.
        with (
            mock.patch("ray.get_runtime_context", return_value=SimpleNamespace(get_node_id=lambda: self._node_id)),
            mock.patch.dict(os.environ, self._env, clear=True),
        ):
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
    worker_env = {
        "TPU_VISIBLE_CHIPS": "0",
        "TPU_WORKER_ID": "0",
        "LIBTPU_INIT_ARGS": "--base",
        "UNRELATED": "dropped",
    }
    workers = [_FakeWorker(NODE_A, str(i), {**worker_env, "TPU_VISIBLE_CHIPS": str(i)}) for i in range(4)]
    workers += [_FakeWorker(NODE_B, str(i), {**worker_env, "TPU_VISIBLE_CHIPS": str(i)}) for i in range(4)]
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


def test_launch_tpu_vllm_servers_single_server_spanning_all_workers(tpu_vllm):
    replica = _fake_replica()
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET"], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_vllm, "get_platform", return_value=platform),
            mock.patch.object(tpu_vllm, "is_tpu_vllm_run", return_value=True),
            mock.patch.dict(os.environ, {"VERL_TPU_EXTRA_LIBTPU_INIT_ARGS": "--extra"}),
        ):
            await tpu_vllm.launch_tpu_vllm_servers(replica)

    asyncio.run(run())

    server_class = replica.server_class
    options = server_class.options_kwargs
    assert options["name"] == "vllm_server_3_0"
    assert options["max_concurrency"] == 1234
    assert options["scheduling_strategy"].node_id == NODE_A
    env_vars = options["runtime_env"]["env_vars"]
    assert env_vars["RAY_NOSET"] == "1"
    assert env_vars["TPU_WORKER_ID"] == "0"
    assert "UNRELATED" not in env_vars
    assert env_vars["LIBTPU_INIT_ARGS"] == "--base --extra"
    assert env_vars["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] == "LIBTPU_INIT_ARGS"

    init = server_class.init_kwargs
    assert init["workers"] is replica.workers
    assert init["node_rank"] == 0
    assert init["nnodes"] == 2
    assert init["cuda_visible_devices"] == "0,1,2,3,0,1,2,3"

    assert server_class.server.launch_kwargs == {"master_address": "10.0.0.1", "master_port": 1234, "dp_rpc_port": 5678}
    assert replica.servers == [server_class.server]
    assert replica._server_address == "10.0.0.1:8000"


def test_launch_tpu_vllm_servers_rejects_data_parallel(tpu_vllm):
    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(tpu_vllm.launch_tpu_vllm_servers(_fake_replica(data_parallel_size=2)))


def test_report_stale_tpu_engines_enforces_limit(tpu_vllm):
    workers = [SimpleNamespace(__ray_call__=_FakeRemoteMethod({"hostname": "h0", "zombies": 7, "engines": 5}))]

    async def run(limit):
        with (
            mock.patch.object(tpu_vllm, "is_tpu_vllm_run", return_value=True),
            mock.patch.dict(os.environ, {"VERL_TPU_MAX_STALE_ENGINES": limit}),
        ):
            await tpu_vllm.report_stale_tpu_engines(workers)

    asyncio.run(run("-1"))  # report only
    asyncio.run(run("5"))  # at the limit
    with pytest.raises(RuntimeError, match="VERL_TPU_MAX_STALE_ENGINES=4"):
        asyncio.run(run("4"))


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
    env["RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK"] = "hook"
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(ray, "init", ray.init),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
    ):
        yield tpu_vllm_patches
    base_process.__init__ = saved_init
    if not had_flag and "_tpu_patched" in base_process.__dict__:
        del base_process._tpu_patched


def test_patch_vllm_for_tpu_env_side_effects_without_torchtpu(isolated_patches):
    assert isolated_patches.ray_distributed_executor is None
    isolated_patches.patch_vllm_for_tpu()

    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert "VLLM_USE_V1" not in os.environ  # V0 is gone in vLLM v0.29; the flag is not set anymore
    assert "RAY_RUNTIME_ENV_WORKER_PROCESS_SETUP_HOOK" not in os.environ
    assert multiprocessing.process.BaseProcess._tpu_patched is True
    # The executor patches need vllm-torchtpu, so the guard stays open for a later retry.
    assert isolated_patches._PATCHES_APPLIED is False


def test_patch_vllm_for_tpu_is_a_no_op_once_applied(isolated_patches):
    isolated_patches._PATCHES_APPLIED = True
    original_ray_init = ray.init
    isolated_patches.patch_vllm_for_tpu()
    assert ray.init is original_ray_init
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"  # env side effects still re-applied


def test_patch_vllm_for_tpu_keeps_explicit_compile_cache_setting(isolated_patches):
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "0"
    isolated_patches.patch_vllm_for_tpu()
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "0"


def test_patched_ray_init_strips_worker_process_setup_hook(isolated_patches):
    seen = {}

    def fake_init(*args, **kwargs):
        seen.update(kwargs)

    with mock.patch.object(ray, "init", fake_init):
        isolated_patches.patch_vllm_for_tpu()
        ray.init(runtime_env={"worker_process_setup_hook": "x", "env_vars": {"A": "1"}})
    assert seen["runtime_env"] == {"env_vars": {"A": "1"}}


def test_pickleable_process_wrapper_applies_patches_before_target():

    wrapper = pickle.loads(pickle.dumps(tpu_vllm_patches.PickleableProcessWrapper(len)))
    with mock.patch.object(tpu_vllm_patches, "patch_vllm_for_tpu") as patch:
        assert wrapper([1, 2, 3]) == 3
    patch.assert_called_once_with()


@pytest.mark.parametrize(
    "total_chips,num_nodes,expected",
    [
        (8, 2, ("2,4,1", "2,4,1", "1,1,1", "4")),
        (4, 1, ("2,2,1", "1,1,1", "2,2,1", "4")),
        (4, 2, ("2,2,1", "1,1,1", "1,1,1", "2")),
        (32, 8, ("4,8,1", "4,8,1", "1,1,1", "4")),
    ],
)
def test_resolve_tpu_topology_bounds(total_chips, num_nodes, expected):
    from verl_hardware_plugin.rollout.tpu_vllm_patches import _resolve_tpu_topology_bounds

    env = {k: v for k, v in os.environ.items() if k not in ("TORCH_TPU_TOPOLOGY", "VLLM_TPU_CHIPS_PER_HOST")}
    with mock.patch.dict(os.environ, env, clear=True):
        assert _resolve_tpu_topology_bounds(total_chips, num_nodes) == expected


def test_resolve_tpu_topology_bounds_env_override():
    from verl_hardware_plugin.rollout.tpu_vllm_patches import _resolve_tpu_topology_bounds

    with mock.patch.dict(os.environ, {"TORCH_TPU_TOPOLOGY": "4,4,1", "VLLM_TPU_CHIPS_PER_HOST": "8"}):
        assert _resolve_tpu_topology_bounds(16, 2) == ("4,4,1", "4,4,1", "1,1,1", "8")


def test_server_max_concurrency_prefers_config_and_falls_back_to_replica(tpu_vllm):
    _server_max_concurrency = tpu_vllm._server_max_concurrency

    upstream = SimpleNamespace(config=SimpleNamespace(ray_actor_max_concurrency=1234), max_concurrency=1)
    assert _server_max_concurrency(upstream) == 1234
    older = SimpleNamespace(config=SimpleNamespace(), max_concurrency=1100)
    assert _server_max_concurrency(older) == 1100
