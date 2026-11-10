# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU SGLang rollout glue and SGLang tpu-sync worker adapter.

SGLang, ``sglang-torchtpu`` and ``torch_tpu`` are not installed on CI, so ``async_sglang_server`` and
``sglang.srt`` modules are replaced by minimal stand-ins and Ray actor handles are faked.
"""

import asyncio
import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import ray
import ray.util.scheduling_strategies  # noqa: F401
import torch

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

from verl.workers.rollout.replica import RolloutMode  # noqa: E402
from verl_hardware_plugin.accelerators.tpu import device_api  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.weight_sync import (  # noqa: E402
    tpu_sync_checkpoint_engine as tpu_sync_engine,
)
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_parity as parity  # noqa: E402

ASS = "verl.workers.rollout.sglang_rollout.async_sglang_server"


class _ConfigObj(dict):
    """Dict with attribute access, mimicking OmegaConf / RolloutConfig in SGLangHttpServer."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeSGLangHttpServer:
    """Stands in for upstream ``SGLangHttpServer``."""

    def __init__(
        self,
        config,
        model_config,
        rollout_mode,
        workers,
        replica_rank,
        node_rank,
        nnodes,
        cuda_visible_devices="",
        base_gpu_id=0,
        disaggregation_role="null",
        disaggregation_bootstrap_port=None,
    ):
        self.config = config
        self.model_config = model_config
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.cuda_visible_devices = cuda_visible_devices
        self.base_gpu_id = base_gpu_id
        self._disaggregation_role = disaggregation_role
        self._disaggregation_bootstrap_port = disaggregation_bootstrap_port
        self._server_address = "127.0.0.1"
        self._server_port = None
        self._master_address = "127.0.0.1" if nnodes > 1 and node_rank == 0 else None
        self._master_port = 29500 if nnodes > 1 and node_rank == 0 else None
        self._master_sock = mock.Mock() if nnodes > 1 and node_rank == 0 else None
        self.global_steps = None

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps


class _FakeSGLangReplica:
    """Stands in for upstream ``SGLangReplica``."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.server_class = None


@pytest.fixture
def tpu_sglang():
    """Import ``tpu_sglang`` against a stubbed ``async_sglang_server``."""
    ass = ModuleType(ASS)
    ass.SGLangHttpServer = _FakeSGLangHttpServer
    ass.SGLangReplica = _FakeSGLangReplica
    ass.run_uvicorn = mock.AsyncMock(return_value=(30000, "uvicorn_task"))
    with mock.patch.dict(sys.modules, {ASS: ass}):
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang", None)
        module = importlib.import_module("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang")
        yield module
        sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang", None)


def _done(value):
    future = asyncio.get_event_loop().create_future()
    future.set_result(value)
    return future


class _FakeRemoteMethod:
    def __init__(self, result):
        self._result = result

    def remote(self, *args, **kwargs):
        return _done(self._result(*args, **kwargs) if callable(self._result) else self._result)


class _FakeWorker:
    def __init__(self, node_id, node_ip="10.0.0.1", env=None):
        self._node_id = node_id
        self._node_ip = node_ip
        self._env = dict(env or {})
        self.__ray_call__ = _FakeRemoteMethod(self._call)

    def _call(self, fn):
        with (
            mock.patch("ray.get_runtime_context", return_value=SimpleNamespace(get_node_id=lambda: self._node_id)),
            mock.patch("ray.util.get_node_ip_address", return_value=self._node_ip),
            mock.patch.dict("os.environ", self._env, clear=True),
        ):
            return fn(self)


class _FakeLaunchedServer:
    def __init__(self, address="10.0.0.1", port=30000, leader=True):
        self._address = address
        self._port = port
        self.leader = leader
        self.launch_kwargs = None
        self.get_tpu_rendezvous_ports = _FakeRemoteMethod((address, 29500, 29501))
        self.get_server_address = _FakeRemoteMethod(lambda: (self._address, self._port))
        self.is_leader = _FakeRemoteMethod(lambda: self.leader)
        self.launch_server = _FakeRemoteMethod(self._launch)

    def _launch(self, **kwargs):
        self.launch_kwargs = kwargs


class _FakeServerClass:
    def __init__(self, server_configs=None):
        self.options_kwargs = []
        self.init_kwargs = []
        self.servers = []
        self._server_configs = list(server_configs or [])

    def options(self, **kwargs):
        self.options_kwargs.append(kwargs)
        return self

    def remote(self, **kwargs):
        idx = len(self.servers)
        self.init_kwargs.append(kwargs)
        cfg = self._server_configs[idx] if idx < len(self._server_configs) else {}
        server = _FakeLaunchedServer(
            address=cfg.get("address", f"10.0.0.{idx + 1}"),
            port=cfg.get("port", 30000 + idx),
            leader=cfg.get("leader", idx == 0),
        )
        self.servers.append(server)
        return server


NODE_A = "a" * 56
NODE_B = "b" * 56


def _default_rollout_config(**overrides):
    base = {
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.6,
        "enforce_eager": False,
        "tensor_model_parallel_size": 8,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "load_format": "safetensors",
        "max_num_seqs": 256,
        "max_model_len": 1536,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": False,
        "skip_tokenizer_init": True,
        "seed": 7,
        "ray_actor_max_concurrency": 512,
        "prometheus": SimpleNamespace(enable=False, served_model_name=None),
        "enable_rollout_routing_replay": False,
        "mtp": None,
        "engine_kwargs": {},
    }
    base.update(overrides)
    return _ConfigObj(base)


def _default_model_config():
    return SimpleNamespace(local_path="/models/Qwen3-4B", trust_remote_code=False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_sglang_loader_defers_off_tpu_and_calls_upstream_before_tpu_replica():
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from verl_hardware_plugin.accelerators.tpu.registration import register_rollout

    sentinel_upstream = type("UpstreamSGLangReplica", (), {})
    sentinel_tpu = type("TPUSGLangReplica", (), {})
    upstream_loader = mock.Mock(return_value=sentinel_upstream)
    fake_tpu_module = ModuleType("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang")
    fake_tpu_module.TPUSGLangReplica = sentinel_tpu

    with mock.patch.dict(RolloutReplicaRegistry._registry, {"sglang": upstream_loader}):
        register_rollout()
        upstream_loader.assert_not_called()

        with mock.patch("verl.utils.device.get_resource_name", return_value="GPU"):
            assert RolloutReplicaRegistry.get("sglang") is sentinel_upstream
        upstream_loader.assert_called_once_with()

        with (
            mock.patch("verl.utils.device.get_resource_name", return_value="TPU"),
            mock.patch.dict(sys.modules, {"verl_hardware_plugin.accelerators.tpu.rollout.tpu_sglang": fake_tpu_module}),
        ):
            assert RolloutReplicaRegistry.get("sglang") is sentinel_tpu
        assert upstream_loader.call_count == 2


# ---------------------------------------------------------------------------
# launch_tpu_sglang_servers
# ---------------------------------------------------------------------------


def test_launch_tpu_sglang_servers_multihost_gke_sorts_by_worker_id_and_promotes_leader(tpu_sglang):
    # Host B has TPU_WORKER_ID=0 and Host A has TPU_WORKER_ID=1; normalized leader ends up on index 1.
    env_b = {"TPU_WORKER_ID": "0", "TPU_HOST_BOUNDS": "1,2,1", "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1"}
    env_a = {"TPU_WORKER_ID": "1", "TPU_HOST_BOUNDS": "1,2,1", "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1"}
    workers = [_FakeWorker(NODE_A, "10.0.0.11", env_a) for _ in range(4)] + [
        _FakeWorker(NODE_B, "10.0.0.10", env_b) for _ in range(4)
    ]
    server_class = _FakeServerClass(
        server_configs=[
            {"address": "10.0.0.10", "port": 30000, "leader": False},
            {"address": "10.0.0.11", "port": 30001, "leader": True},
        ]
    )
    replica = SimpleNamespace(
        config=_default_rollout_config(),
        model_config=_default_model_config(),
        rollout_mode=RolloutMode.STANDALONE,
        workers=workers,
        replica_rank=2,
        name_suffix="",
        is_reward_model=False,
        is_teacher_model=False,
        gpus_per_replica_node=4,
        nnodes=2,
        servers=[],
        server_class=server_class,
    )
    platform = SimpleNamespace(ray_noset_envvars=lambda: ["RAY_NOSET_TPU"], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_sglang, "get_platform", return_value=platform),
            mock.patch.object(tpu_sglang, "ray_tpu_accelerator_name", return_value="v6e-8"),
        ):
            await tpu_sglang.launch_tpu_sglang_servers(replica)

    asyncio.run(run())

    # Sorted by TPU_WORKER_ID: NODE_B (id=0) comes first, NODE_A (id=1) second.
    assert [o["scheduling_strategy"].node_id for o in server_class.options_kwargs] == [NODE_B, NODE_A]
    env_vars = server_class.options_kwargs[0]["runtime_env"]["env_vars"]
    assert env_vars == {
        "RAY_NOSET_TPU": "1",
        "SGLANG_PLATFORM": "tpu",
        "TPU_WORKER_HOSTNAMES": "10.0.0.10,10.0.0.11",
        "TORCH_TPU_TOPOLOGY": "2,4,1",
        "TPU_TOPOLOGY": "2x4x1",
        "TORCH_TPU_DEVICES_PER_HOST": "4",
    }
    for srv in server_class.servers:
        assert srv.launch_kwargs == {
            "master_address": "10.0.0.10",
            "master_port": 29500,
            "bootstrap_port": 29501,
            "tpu_node_normalization": True,
        }
    # Normalized leader (server 1 at 10.0.0.11:30001) promoted to replica.servers[0] and _server_handle.
    assert replica.servers[0] is server_class.servers[1]
    assert replica._server_handle is server_class.servers[1]
    assert replica._server_address == "10.0.0.11:30001"


@pytest.mark.parametrize(
    ("pod_type", "chips", "expected_topology"),
    [
        ("v6e-1", 1, "1,1,1"),
        ("v6e-8", 4, "2,2,1"),
        ("v7x-8", 8, "2,2,1,2"),
    ],
)
def test_launch_tpu_sglang_servers_single_host_topology(tpu_sglang, pod_type, chips, expected_topology):
    workers = [_FakeWorker(NODE_A, "10.0.0.1") for _ in range(chips)]
    server_class = _FakeServerClass()
    replica = SimpleNamespace(
        config=_default_rollout_config(tensor_model_parallel_size=chips),
        model_config=_default_model_config(),
        rollout_mode=RolloutMode.STANDALONE,
        workers=workers,
        replica_rank=0,
        name_suffix="",
        is_reward_model=False,
        is_teacher_model=False,
        gpus_per_replica_node=chips,
        nnodes=1,
        servers=[],
        server_class=server_class,
    )
    platform = SimpleNamespace(ray_noset_envvars=lambda: [], rollout_env_vars=lambda: {})

    async def run():
        with (
            mock.patch.object(tpu_sglang, "get_platform", return_value=platform),
            mock.patch.object(tpu_sglang, "ray_tpu_accelerator_name", return_value=pod_type),
        ):
            await tpu_sglang.launch_tpu_sglang_servers(replica)

    asyncio.run(run())

    env_vars = server_class.options_kwargs[0]["runtime_env"]["env_vars"]
    assert env_vars["SGLANG_PLATFORM"] == "tpu"
    assert env_vars["TPU_WORKER_HOSTNAMES"] == ""
    assert env_vars["TORCH_TPU_TOPOLOGY"] == expected_topology
    if chips == 1:
        assert env_vars["TPU_VISIBLE_CHIPS"] == "0"


def test_launch_tpu_sglang_servers_rejects_data_parallel(tpu_sglang):
    replica = SimpleNamespace(config=_default_rollout_config(data_parallel_size=2))
    with pytest.raises(NotImplementedError, match="data_parallel_size"):
        asyncio.run(tpu_sglang.launch_tpu_sglang_servers(replica))


# ---------------------------------------------------------------------------
# TPUSGLangHttpServer ServerArgs, lifecycle & collective_rpc
# ---------------------------------------------------------------------------


def test_server_args_defaults_and_overrides(tpu_sglang):
    cfg = _default_rollout_config(
        engine_kwargs={"sglang": {"attention_backend": "torch_native", "torch_compile_max_bs": 128}}
    )
    server = tpu_sglang.TPUSGLangHttpServer(
        config=cfg,
        model_config=_default_model_config(),
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        nnodes=2,
    )
    args = server._build_tpu_server_args_dict(tpu_node_normalization=True)
    assert args["device"] == "tpu"
    assert args["kv_cache_dtype"] == "bfloat16"
    assert args["enable_torch_compile"] is True
    assert args["torch_compile_max_bs"] == 128
    assert args["enable_memory_saver"] is False
    assert args["enable_weights_cpu_backup"] is False
    assert args["enable_draft_weights_cpu_backup"] is False
    assert args["context_length"] == 1536
    assert args["chunked_prefill_size"] == 4096
    assert args["max_prefill_tokens"] == 4096
    assert args["disable_radix_cache"] is True
    assert args["grammar_backend"] == "none"
    assert args["random_seed"] == 7
    assert args["attention_backend"] == "torch_native"
    assert args["tpu_node_normalization"] is True
    assert args["dist_init_addr"] == "127.0.0.1:29500"


def test_scheduler_rpc_hook_and_collective_rpc_roundtrip(tpu_sglang):
    class FakeRpcReqInput:
        def __init__(self, method, parameters=None):
            self.method = method
            self.parameters = parameters or {}

    class FakeRpcReqOutput:
        def __init__(self, success, message=""):
            self.success = success
            self.message = message

    class FakeScheduler:
        # SGLang >= 0.5.12 stores tp_group (GroupCoordinator) and tp_size on Scheduler without setting tp_rank.
        tp_size = 1
        tp_group = SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None)
        tp_worker = SimpleNamespace(model_runner=SimpleNamespace(model=None, model_config=None))

        def handle_rpc_request(self, recv_req):
            return FakeRpcReqOutput(success=True, message="passthrough")

    sglang_pkg = ModuleType("sglang")
    srt_pkg = ModuleType("sglang.srt")
    managers_pkg = ModuleType("sglang.srt.managers")
    io_struct_mod = ModuleType("sglang.srt.managers.io_struct")
    scheduler_mod = ModuleType("sglang.srt.managers.scheduler")
    utils_pkg = ModuleType("sglang.srt.utils")
    network_mod = ModuleType("sglang.srt.utils.network")
    zmq_mod = ModuleType("zmq")
    zmq_mod.DEALER = 5
    zmq_mod.BLOCKY = 0
    zmq_mod.RCVTIMEO = 27
    zmq_mod.Context = lambda n=1: SimpleNamespace()

    sent = []
    scheduler_inst = FakeScheduler()

    io_struct_mod.RpcReqInput = FakeRpcReqInput
    io_struct_mod.RpcReqOutput = FakeRpcReqOutput
    io_struct_mod.sock_send = lambda sock, req: sent.append(req)
    io_struct_mod.sock_recv = lambda sock, flags=0: scheduler_inst.handle_rpc_request(sent[-1])
    scheduler_mod.Scheduler = FakeScheduler
    network_mod.get_zmq_socket = lambda ctx, stype, ipc, server: SimpleNamespace(setsockopt=lambda *a: None)

    mods = {
        "zmq": zmq_mod,
        "sglang": sglang_pkg,
        "sglang.srt": srt_pkg,
        "sglang.srt.managers": managers_pkg,
        "sglang.srt.managers.io_struct": io_struct_mod,
        "sglang.srt.managers.scheduler": scheduler_mod,
        "sglang.srt.utils": utils_pkg,
        "sglang.srt.utils.network": network_mod,
    }
    with mock.patch.dict(sys.modules, mods):
        tpu_sglang.install_sglang_scheduler_rpc_hook()
        # Non-verl RPC still delegates to original handle_rpc_request.
        other_out = scheduler_inst.handle_rpc_request(FakeRpcReqInput("save_remote_model"))
        assert other_out.success is True and other_out.message == "passthrough"

        server = tpu_sglang.TPUSGLangHttpServer(
            config=_default_rollout_config(),
            model_config=_default_model_config(),
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=0,
            node_rank=0,
            nnodes=1,
        )
        server.tokenizer_manager = object()
        server.port_args = SimpleNamespace(rpc_ipc_name="ipc:///tmp/fake_rpc")

        # get_model_weights_stats returns {"error": "No model found on worker"} when model is None.
        res = asyncio.run(server.collective_rpc("get_model_weights_stats"))
        assert res == [{"error": "No model found on worker"}]

        # Unknown method propagates a clean RuntimeError with TP rank context.
        with pytest.raises(RuntimeError, match="TP rank 0: AttributeError"):
            asyncio.run(server.collective_rpc("nonexistent_rpc_method"))


def test_abort_and_resume_generation_disables_cuda_empty_cache(tpu_sglang):
    paused = []
    continued = []
    flushed = []

    class FakePauseGenerationReqInput:
        def __init__(self, mode="abort"):
            self.mode = mode

    class FakeContinueGenerationReqInput:
        def __init__(self, torch_empty_cache: bool = True):
            self.torch_empty_cache = torch_empty_cache

    io_struct_mod = ModuleType("sglang.srt.managers.io_struct")
    io_struct_mod.PauseGenerationReqInput = FakePauseGenerationReqInput
    io_struct_mod.ContinueGenerationReqInput = FakeContinueGenerationReqInput

    mods = {
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.managers": ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.io_struct": io_struct_mod,
    }
    server = tpu_sglang.TPUSGLangHttpServer(
        config=_default_rollout_config(),
        model_config=_default_model_config(),
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        nnodes=1,
    )
    server.tokenizer_manager = SimpleNamespace(
        pause_generation=mock.AsyncMock(side_effect=lambda req: paused.append(req)),
        continue_generation=mock.AsyncMock(side_effect=lambda req: continued.append(req)),
        flush_cache=mock.AsyncMock(side_effect=lambda: flushed.append(True)),
    )

    with mock.patch.dict(sys.modules, mods):
        asyncio.run(server.abort_all_requests())
        asyncio.run(server.resume_generation())

    assert len(paused) == 1 and paused[0].mode == "abort"
    assert flushed == [True]
    assert len(continued) == 1 and continued[0].torch_empty_cache is False


def test_launch_server_loads_plugins_before_server_args_and_marks_server_up(tpu_sglang):
    call_order = []

    class FakeServerArgs:
        def __init__(self, **kwargs):
            call_order.append("ServerArgs")
            self.kwargs = kwargs
            self.enable_metrics = False

    class FakeEngine:
        @classmethod
        def _launch_subprocesses(cls, **kwargs):
            call_order.append("_launch_subprocesses")
            tok_mgr = SimpleNamespace(server_status=None, _subprocess_watchdog=None)
            tmpl_mgr = SimpleNamespace()
            port_args = SimpleNamespace(rpc_ipc_name="ipc:///tmp/fake_launch_rpc")
            sched_res = SimpleNamespace(scheduler_infos=[{"max_req_input_len": 1536}])
            watchdog = SimpleNamespace()
            return tok_mgr, tmpl_mgr, port_args, sched_res, watchdog, []

    sglang_pkg = ModuleType("sglang")
    srt_pkg = ModuleType("sglang.srt")
    ep_pkg = ModuleType("sglang.srt.entrypoints")
    engine_mod = ModuleType("sglang.srt.entrypoints.engine")
    http_mod = ModuleType("sglang.srt.entrypoints.http_server")
    managers_pkg = ModuleType("sglang.srt.managers")
    tok_mod = ModuleType("sglang.srt.managers.tokenizer_manager")
    plugins_mod = ModuleType("sglang.srt.plugins")
    server_args_mod = ModuleType("sglang.srt.server_args")
    utils_pkg = ModuleType("sglang.srt.utils")
    network_mod = ModuleType("sglang.srt.utils.network")
    zmq_mod = ModuleType("zmq")
    zmq_mod.DEALER = 5
    zmq_mod.Context = lambda n=1: SimpleNamespace()

    engine_mod.Engine = FakeEngine
    engine_mod.init_tokenizer_manager = lambda *a, **k: None
    engine_mod.run_detokenizer_process = lambda *a, **k: None
    ep_pkg.engine = engine_mod
    srt_pkg.entrypoints = ep_pkg
    sglang_pkg.srt = srt_pkg

    global_states = []
    http_mod._GlobalState = lambda **kw: SimpleNamespace(**kw)
    http_mod.app = SimpleNamespace()
    http_mod.set_global_state = lambda gs: global_states.append(gs)
    tok_mod.ServerStatus = SimpleNamespace(Up="UP")
    plugins_mod.load_plugins = lambda: call_order.append("load_plugins")
    server_args_mod.ServerArgs = FakeServerArgs
    network_mod.get_zmq_socket = lambda ctx, stype, ipc, bind: SimpleNamespace()

    mods = {
        "zmq": zmq_mod,
        "sglang": sglang_pkg,
        "sglang.srt": srt_pkg,
        "sglang.srt.entrypoints": ep_pkg,
        "sglang.srt.entrypoints.engine": engine_mod,
        "sglang.srt.entrypoints.http_server": http_mod,
        "sglang.srt.managers": managers_pkg,
        "sglang.srt.managers.tokenizer_manager": tok_mod,
        "sglang.srt.plugins": plugins_mod,
        "sglang.srt.server_args": server_args_mod,
        "sglang.srt.utils": utils_pkg,
        "sglang.srt.utils.network": network_mod,
    }
    with (
        mock.patch.dict(sys.modules, mods),
        mock.patch(
            "verl.workers.rollout.utils.run_uvicorn",
            new=mock.AsyncMock(return_value=(30123, "uvicorn_task")),
        ),
    ):
        server = tpu_sglang.TPUSGLangHttpServer(
            config=_default_rollout_config(),
            model_config=_default_model_config(),
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=0,
            node_rank=0,
            nnodes=1,
        )
        asyncio.run(server.launch_server())

    assert call_order == ["load_plugins", "ServerArgs", "_launch_subprocesses"]
    assert server.is_leader() is True
    assert server._server_port == 30123
    assert server.tokenizer_manager.server_status == "UP"
    assert server.scheduler_info == {"max_req_input_len": 1536}
    assert len(global_states) == 1


# ---------------------------------------------------------------------------
# SGLangTPUSyncWorker end-to-end weight sync & parity without importing vLLM
# ---------------------------------------------------------------------------


def test_sglang_tpu_sync_worker_installs_weights_and_passes_parity_without_vllm(monkeypatch):
    from tests.accelerators.tpu.test_tpu_sync_checkpoint_engine import (
        KV_HEADS,
        TRAINER_SHAPES,
        _assert_state,
        _expected_vllm_state,
        _fake_ray_get,
        _FakeTpuSync,
        _FakeVllmModel,
        _make_manager,
        _trainer_engine,
        _trainer_weights,
    )

    world = _FakeTpuSync()
    for name, module in world.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(device_api, "_accelerator_is_tpu_cache", None)
    monkeypatch.setattr(device_api, "_tpu_runtime_up_pid", None)
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(is_initialized=lambda: True), raising=False)
    monkeypatch.setattr(torch, "accelerator", world.accelerator())
    monkeypatch.setattr(tpu_sync_engine, "TPU_SYNC_DEVICE", "cpu")
    monkeypatch.setattr(tpu_sync_engine, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr(tpu_sync_engine, "create_tpu_sync_rendezvous", world.create_rendezvous)
    monkeypatch.setattr(parity, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "10.0.0.1")
    monkeypatch.setattr(ray, "get", _fake_ray_get)

    # Ensure vLLM is NOT in sys.modules and guard against any accidental import of vLLM.
    for k in [k for k in list(sys.modules) if k == "vllm" or k.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, k, raising=False)

    sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension", None)
    ext_mod = importlib.import_module("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension")
    assert "vllm" not in sys.modules

    tp = 4  # TP=4 > KV_HEADS=2 exercises GQA KV-head replication too!
    weights_step0 = _trainer_weights(seed=10)
    models = [_FakeVllmModel(tp=tp, flip_o_proj=False) for _ in range(tp)]
    # Pre-populate step-0 model parameters with the exact expected shards so step-0 exact_parity=True succeeds!
    for rank, model in enumerate(models):
        expected0 = _expected_vllm_state(weights_step0, rank, tp)
        with torch.no_grad():
            for k, param in model.state_dict().items():
                param.copy_(expected0[k])

    samplers = []
    for rank, model in enumerate(models):
        # Match SGLang >= 0.5.12 Scheduler layout: no scheduler.tp_rank attribute; rank lives on tp_group.rank_in_group.
        fake_scheduler = SimpleNamespace(
            tp_size=tp,
            tp_group=SimpleNamespace(rank_in_group=rank, world_size=tp, cpu_group=None),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=model,
                    model_config=SimpleNamespace(hf_text_config=None, get_total_num_kv_heads=lambda: KV_HEADS),
                )
            ),
        )
        samplers.append(ext_mod.SGLangTPUSyncWorker(fake_scheduler))

    events = []
    engines = [_trainer_engine(0, verify_parity="all")]
    current_weights = {"w": weights_step0}
    manager, server = _make_manager(
        engines, lambda _r: iter(current_weights["w"].items()), samplers, events, verify_parity="all"
    )

    # Step 0: verifies bit-exact parity (ExactParityCheck) + L1/L2 norm parity!
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=0))
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights_step0, rank, tp))
    assert server.global_steps == 0

    # Step 1: update weights with fresh random bf16 tensors and verify all shards + norm parity!
    weights_step1 = _trainer_weights(seed=99)
    current_weights["w"] = weights_step1
    assert set(weights_step1) == set(TRAINER_SHAPES)
    asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=1))
    for rank, model in enumerate(models):
        _assert_state(model, _expected_vllm_state(weights_step1, rank, tp))
    assert server.global_steps == 1
    assert "vllm" not in sys.modules


def test_sglang_tpu_sync_worker_checks_the_tied_lm_head(monkeypatch, caplog):
    """With tie_word_embeddings, the SGLang worker copies lm_head from the embedding and the driver checks it every sync
    (the worker shares TPUSyncWorkerMixin.install_tpu_sync_weights with the vLLM one)."""
    from tests.accelerators.tpu.test_tpu_sync_checkpoint_engine import (
        KV_HEADS,
        _expected_vllm_state,
        _fake_ray_get,
        _FakeTpuSync,
        _FakeVllmModel,
        _make_manager,
        _NoCopyParameter,
        _tied_trainer_weights,
        _trainer_engine,
    )

    world = _FakeTpuSync()
    for name, module in world.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(device_api, "_accelerator_is_tpu_cache", None)
    monkeypatch.setattr(device_api, "_tpu_runtime_up_pid", None)
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(is_initialized=lambda: True), raising=False)
    monkeypatch.setattr(torch, "accelerator", world.accelerator())
    monkeypatch.setattr(tpu_sync_engine, "TPU_SYNC_DEVICE", "cpu")
    monkeypatch.setattr(tpu_sync_engine, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr(tpu_sync_engine, "create_tpu_sync_rendezvous", world.create_rendezvous)
    monkeypatch.setattr(parity, "get_tpu_sync_rendezvous", world.get_rendezvous)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(ray, "get", _fake_ray_get)
    sys.modules.pop("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension", None)
    ext_mod = importlib.import_module("verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension")

    tp = 2
    weights = {"w": _tied_trainer_weights(seed=20)}
    models = [_FakeVllmModel(tp=tp) for _ in range(tp)]
    for rank, model in enumerate(models):
        model.load_state_dict(_expected_vllm_state(weights["w"], rank, tp))
    samplers = [
        ext_mod.SGLangTPUSyncWorker(
            SimpleNamespace(
                tp_size=tp,
                tp_group=SimpleNamespace(rank_in_group=rank, world_size=tp, cpu_group=None),
                tp_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        model=model,
                        model_config=SimpleNamespace(hf_text_config=None, get_total_num_kv_heads=lambda: KV_HEADS),
                    )
                ),
            )
        )
        for rank, model in enumerate(models)
    ]
    events: list = []
    manager, _ = _make_manager(
        [_trainer_engine(0, tie_word_embeddings=True)],
        lambda _r: iter(weights["w"].items()),
        samplers,
        events,
        tie_word_embeddings=True,
    )

    for step in (0, 1):
        weights["w"] = _tied_trainer_weights(seed=20 + step)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=tpu_sync_engine.__name__):
            asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=step))
        prefix = f"[TPU-SYNC TIED LM_HEAD | Step {step} | replica 0]"
        lines = [line for line in caplog.text.splitlines() if prefix in line]
        assert len(lines) == tp
        assert all("lm_head copied; max|lm_head - embed|=0 (bitwise equal)" in line for line in lines)
        for rank, model in enumerate(models):
            assert torch.equal(model.lm_head.weight, weights["w"]["model.embed_tokens.weight"].chunk(tp, dim=0)[rank])

    # A copy into one worker's lm_head that does not land: the next sync fails before generation resumes.
    models[1].lm_head.weight = _NoCopyParameter(models[1].lm_head.weight.data, requires_grad=False)
    weights["w"] = _tied_trainer_weights(seed=30)
    events.clear()
    with pytest.raises(RuntimeError, match=r"Sampler Rank 1: lm_head copied from embed_tokens but not bitwise equal"):
        asyncio.run(tpu_sync_engine.update_tpu_sync_weights(manager, global_steps=2))
    assert "install_tpu_sync_weights" in events and "resume" not in events
