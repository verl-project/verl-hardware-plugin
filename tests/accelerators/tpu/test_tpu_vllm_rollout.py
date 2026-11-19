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
        self.launched = False

    def _get_server_name_prefix(self):
        return "vllm_"

    def _get_server_env_vars(self, node_ids):
        return {"FROM_UPSTREAM": "1"}

    async def launch_servers(self):
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
# TPUvLLMReplica._get_server_env_vars / launch_servers
# ---------------------------------------------------------------------------

NODE_A = "a" * 56
NODE_B = "b" * 56


def _make_tpu_replica(tpu_vllm, *, data_parallel_size=1, nnodes=2, chips_per_host=4):
    replica = tpu_vllm.TPUvLLMReplica(0, "config", "model_config")
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


@pytest.mark.parametrize(
    "pod_type, chips, topology",
    [
        ("v7x-8", 8, "2,2,1,2"),
        ("v6e-8", 8, "2,4,1"),
        ("v5p-16", 4, "2,2,1"),
        # TP=2 / TP=4 on TPU 7x: no known sub-host mesh, so the replica keeps the inherited value instead of failing.
        ("v7x-8", 2, None),
        ("v7x-8", 4, None),
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
