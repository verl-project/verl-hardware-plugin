# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SGLang rollout replica and HTTP server for Google TPU (``sglang-torchtpu``).

Upstream verl builds the SGLang rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers an ``sglang`` loader that returns :class:`TPUSGLangReplica`, mirroring :class:`TPUvLLMReplica`
while reusing upstream's :class:`SGLangHttpServer` request / tokenization / HTTP serving path:

* One server actor per host of the replica, holding zero TPU chips on the Ray actor itself: SGLang's
  ``Engine._launch_subprocesses`` spawns one scheduler process per local chip (or TensorCore on TPU 7x),
  joining multi-host slices via ``dist_init_addr`` and ``sglang-torchtpu``'s TPU topology normalization.
* TPU ``ServerArgs`` defaults (``device="tpu"``, ``enable_torch_compile`` unless ``enforce_eager=True``,
  ``enable_memory_saver=False``, ``enable_weights_cpu_backup=False``, ``kv_cache_dtype="bfloat16"`` when
  rollout ``dtype`` is bf16, and ``pallas_tpu_rpa_v3`` attention selected by ``TpuSRTPlatform``).
* A lightweight ``collective_rpc`` bridge over SGLang's built-in ``rpc_ipc_name`` ZMQ channel and
  ``Scheduler.tp_group.cpu_group`` Gloo broadcast, exposing ``init_tpu_sync_on_worker``,
  ``install_tpu_sync_weights``, ``get_model_weights_stats`` (``tpu-sync``) and
  ``load_weights_from_ray_registry`` (``ray-plasma-store``) on every TP worker.
"""

import asyncio
import json
import logging
import math
import os
from typing import Any, Callable, Optional

os.environ.setdefault("SGLANG_PLATFORM", "tpu")

import ray
import torch
from omegaconf import OmegaConf

from verl.plugin.platform import get_platform
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer, SGLangReplica
from verl_hardware_plugin.accelerators.tpu.platform_tpu import (
    TPU_V7X_CORES_PER_CHIP,
    gke_tpu_slice_topology,
    is_tpu_v7x,
    ray_tpu_accelerator_name,
    resolve_tpu_topology_bounds,
    single_host_tpu_topology,
)
from verl_hardware_plugin.accelerators.tpu.profilers.tpu_sglang_timeline import SGLangServerTimelineMixin

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VERL_COLLECTIVE_RPC_METHOD = "_verl_collective_rpc"
_SGLANG_TPU_SLICEBUILDER_BASE_PORT = 8070


def _parse_bounds(value: str) -> list[int]:
    return [int(d) for d in value.replace("x", ",").split(",") if d]


def _pick_free_port(address: str, excluded_ports: set[int]) -> int:
    """Pick an ephemeral TCP port on ``address`` outside ``excluded_ports``."""
    held_socks = []
    try:
        for _ in range(32):
            port, sock = get_free_port(address, with_alive_sock=True)
            if port not in excluded_ports:
                if sock is not None:
                    sock.close()
                return port
            if sock is not None:
                held_socks.append(sock)
        port, _ = get_free_port(address, with_alive_sock=False)
        return port
    finally:
        for sock in held_socks:
            sock.close()


def _dispatch_sglang_collective_rpc(scheduler: Any, payload_json: str) -> Any:
    """Execute one ``collective_rpc`` call on ``scheduler``'s :class:`SGLangTPUSyncWorker`."""
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension import SGLangTPUSyncWorker

    adapter = getattr(scheduler, "_verl_tpu_sync_worker", None)
    if adapter is None:
        adapter = SGLangTPUSyncWorker(scheduler)
        scheduler._verl_tpu_sync_worker = adapter

    payload = json.loads(payload_json)
    method = payload["method"]
    rpc_args = tuple(payload.get("args") or ())
    rpc_kwargs = dict(payload.get("kwargs") or {})
    target = getattr(adapter, method, None)
    if target is None:
        raise AttributeError(f"SGLangTPUSyncWorker has no RPC method {method!r}")
    return target(*rpc_args, **rpc_kwargs)


def _scheduler_tp_rank_and_size(scheduler: Any) -> tuple[Any, int]:
    """Best-effort ``(tp_rank, tp_size)`` lookup for scheduler RPC dispatch and error reporting."""
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension import (
        resolve_sglang_tp_rank_and_size,
    )

    try:
        return resolve_sglang_tp_rank_and_size(scheduler)
    except Exception:
        tp_size = getattr(scheduler, "tp_size", None) or getattr(getattr(scheduler, "tp_group", None), "world_size", 1)
        return getattr(scheduler, "tp_rank", "?"), int(tp_size or 1)


def install_sglang_scheduler_rpc_hook() -> None:
    """Wrap ``Scheduler.handle_rpc_request`` so ``collective_rpc`` gathers per-rank return values."""
    from sglang.srt.managers.io_struct import RpcReqOutput
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_verl_tpu_rpc_hooked", False):
        return

    orig_handle_rpc_request = Scheduler.handle_rpc_request

    def _handle_rpc_request(self: Any, recv_req: Any):
        if getattr(recv_req, "method", None) != _VERL_COLLECTIVE_RPC_METHOD:
            return orig_handle_rpc_request(self, recv_req)

        tp_rank, tp_size = _scheduler_tp_rank_and_size(self)
        params = getattr(recv_req, "parameters", None) or {}
        try:
            res = _dispatch_sglang_collective_rpc(self, params["payload"])
            local_outcome = (True, res)
        except Exception as exc:
            logger.exception("SGLang TPU collective_rpc failed on TP rank %s", tp_rank)
            local_outcome = (False, f"TP rank {tp_rank}: {type(exc).__name__}: {exc}")

        cpu_group = getattr(getattr(self, "tp_group", None), "cpu_group", None)
        if cpu_group is not None and tp_size > 1:
            outcomes: list[Any] = [None] * tp_size
            torch.distributed.all_gather_object(outcomes, local_outcome, group=cpu_group)
            torch.distributed.barrier(group=cpu_group)
        else:
            outcomes = [local_outcome]

        errors = [val for ok, val in outcomes if not ok]
        if errors:
            return RpcReqOutput(success=False, message="; ".join(str(err) for err in errors))
        return RpcReqOutput(success=True, message=json.dumps([val for _, val in outcomes]))

    Scheduler.handle_rpc_request = _handle_rpc_request
    Scheduler._verl_tpu_rpc_hooked = True


def run_tpu_sglang_scheduler_process(*args: Any, **kwargs: Any) -> None:
    """Entrypoint passed to ``Engine._launch_subprocesses`` so spawned TP schedulers install the RPC hook."""
    os.environ.setdefault("SGLANG_PLATFORM", "tpu")
    install_sglang_scheduler_rpc_hook()
    from sglang.srt.entrypoints.engine import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


async def launch_tpu_sglang_servers(replica: SGLangReplica) -> None:
    """Launch one SGLang rollout server actor per host of a TPU replica."""
    if replica.config.data_parallel_size > 1:
        raise NotImplementedError(
            "actor_rollout_ref.rollout.data_parallel_size="
            f"{replica.config.data_parallel_size} is not supported on TPU. The TPU "
            "distributed runtime requires each compiled program to span the whole "
            "TPU mesh, but a single DP group only spans "
            f"tensor_model_parallel_size={replica.config.tensor_model_parallel_size} "
            "chips. Set actor_rollout_ref.rollout.data_parallel_size=1 instead: verl "
            "then creates one rollout replica per tensor_model_parallel_size chips, "
            "which is equivalent to engine-internal data parallelism."
        )

    chips_per_host = replica.gpus_per_replica_node
    host_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    ray.util.get_node_ip_address().strip("[]"),
                    os.environ.get("TPU_WORKER_ID", ""),
                    os.environ.get("TPU_HOST_BOUNDS", ""),
                    os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", ""),
                    os.environ.get("TPU_ACCELERATOR_TYPE", ""),
                )
            )
            for worker in replica.workers[::chips_per_host]
        ]
    )

    host_entries = [
        (host_infos[idx], replica.workers[idx * chips_per_host : (idx + 1) * chips_per_host])
        for idx in range(replica.nnodes)
    ]
    if replica.nnodes > 1:
        worker_ids = [entry[0][2] for entry in host_entries]
        if all(wid.isdigit() for wid in worker_ids) and {int(wid) for wid in worker_ids} == set(range(replica.nnodes)):
            host_entries.sort(key=lambda entry: int(entry[0][2]))

    node_ids = [entry[0][0] for entry in host_entries]
    node_ips = [entry[0][1] for entry in host_entries]
    host_bounds = next((entry[0][3] for entry in host_entries if entry[0][3]), "")
    chip_bounds = next((entry[0][4] for entry in host_entries if entry[0][4]), "")
    worker_accel_type = next((entry[0][5] for entry in host_entries if entry[0][5]), "")
    pod_type = ray_tpu_accelerator_name(node_ids=list(node_ids)) or worker_accel_type

    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **get_platform().rollout_env_vars(),
        "SGLANG_PLATFORM": "tpu",
    }
    tpu_node_normalization: Optional[bool] = None

    if replica.nnodes == 1:
        # Clear any slice-wide multi-host hostlist inherited from the pod so single-host replicas stay local.
        env_vars["TPU_WORKER_HOSTNAMES"] = ""
        if chips_per_host == 1:
            env_vars.update(
                {
                    "TPU_VISIBLE_CHIPS": "0",
                    "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
                    "TPU_HOST_BOUNDS": "1,1,1",
                    "TORCH_TPU_TOPOLOGY": "1,1,1",
                }
            )
        else:
            topology = single_host_tpu_topology(chips_per_host, pod_type=pod_type)
            if topology:
                env_vars["TORCH_TPU_TOPOLOGY"] = topology
            else:
                logger.warning(
                    "No single-host TPU mesh of %d chips (TPU type %r); the SGLang replica keeps the inherited "
                    "TORCH_TPU_TOPOLOGY",
                    chips_per_host,
                    pod_type,
                )
    else:
        total_chips = replica.nnodes * chips_per_host
        slice_topology, slice_hosts = gke_tpu_slice_topology(
            {"TPU_HOST_BOUNDS": host_bounds, "TPU_CHIPS_PER_HOST_BOUNDS": chip_bounds}
        )
        slice_devices = (
            math.prod(_parse_bounds(slice_topology)) * (TPU_V7X_CORES_PER_CHIP if is_tpu_v7x(pod_type) else 1)
            if slice_topology
            else 0
        )
        if slice_topology and slice_hosts == replica.nnodes and slice_devices == total_chips:
            topology = f"{slice_topology},{TPU_V7X_CORES_PER_CHIP}" if is_tpu_v7x(pod_type) else slice_topology
            tpu_node_normalization = True
        else:
            topology, _, _, _ = resolve_tpu_topology_bounds(
                total_chips=total_chips, num_nodes=replica.nnodes, pod_type=pod_type
            )
            # Partial-slice and non-GKE pools lack full-slice GKE coordinate bounds, so skip host-rank normalization.
            tpu_node_normalization = False

        env_vars.update(
            {
                "TPU_WORKER_HOSTNAMES": ",".join(node_ips),
                "TORCH_TPU_TOPOLOGY": topology,
                "TPU_TOPOLOGY": "x".join(topology.split(",")[:3]),
                "TORCH_TPU_DEVICES_PER_HOST": str(chips_per_host),
            }
        )

    for node_rank, ((node_id, *_), host_workers) in enumerate(host_entries):
        if replica.is_reward_model:
            name = f"sglang_server_reward_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        elif replica.is_teacher_model:
            name = f"sglang_server_teacher_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        else:
            name = f"sglang_server_{replica.replica_rank}_{node_rank}{replica.name_suffix}"
        server = replica.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
            name=name,
            max_concurrency=replica.config.ray_actor_max_concurrency,
        ).remote(
            config=replica.config,
            model_config=replica.model_config,
            rollout_mode=replica.rollout_mode,
            workers=host_workers,
            replica_rank=replica.replica_rank,
            node_rank=node_rank,
            nnodes=replica.nnodes,
            cuda_visible_devices="",
            base_gpu_id=0,
        )
        replica.servers.append(server)

    master_address, master_port, bootstrap_port = None, None, None
    if replica.nnodes > 1:
        master_address, master_port, bootstrap_port = await replica.servers[0].get_tpu_rendezvous_ports.remote(
            chips_per_host=chips_per_host
        )
    await asyncio.gather(
        *[
            server.launch_server.remote(
                master_address=master_address,
                master_port=master_port,
                bootstrap_port=bootstrap_port,
                tpu_node_normalization=tpu_node_normalization,
            )
            for server in replica.servers
        ]
    )

    # Serving and control RPCs follow normalized rank 0, which TPU node normalization may move to another host.
    leader_flags = await asyncio.gather(*[server.is_leader.remote() for server in replica.servers])
    if any(leader_flags):
        leader_idx = leader_flags.index(True)
        if leader_idx != 0:
            replica.servers[0], replica.servers[leader_idx] = replica.servers[leader_idx], replica.servers[0]

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class TPUSGLangHttpServer(SGLangServerTimelineMixin, SGLangHttpServer):
    """``SGLangHttpServer`` configured for ``sglang-torchtpu`` and TPU weight synchronization."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[Any],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str = "",
        base_gpu_id: int = 0,
        disaggregation_role: str = "null",
        disaggregation_bootstrap_port: Optional[int] = None,
    ):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        super().__init__(
            config=config,
            model_config=model_config,
            rollout_mode=rollout_mode,
            workers=workers,
            replica_rank=replica_rank,
            node_rank=node_rank,
            nnodes=nnodes,
            cuda_visible_devices=cuda_visible_devices,
            base_gpu_id=base_gpu_id,
            disaggregation_role=disaggregation_role,
            disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        )
        self.tokenizer_manager: Any = None
        self.template_manager: Any = None
        self.scheduler_info: Any = None
        self.port_args: Any = None
        self._rpc_context: Any = None
        self._rpc_socket: Any = None
        self._rpc_lock = asyncio.Lock()

    def is_leader(self) -> bool:
        """Whether this server owns ``TokenizerManager`` and the HTTP endpoint (normalized ``node_rank == 0``)."""
        return self.tokenizer_manager is not None

    def get_tpu_rendezvous_ports(self, chips_per_host: int = 4) -> tuple[Optional[str], Optional[int], Optional[int]]:
        """Return ``(master_address, master_port, bootstrap_port)`` on rank 0 for multi-host TPU rendezvous."""
        if self.nnodes <= 1:
            return None, None, None
        excluded = set(range(_SGLANG_TPU_SLICEBUILDER_BASE_PORT, _SGLANG_TPU_SLICEBUILDER_BASE_PORT + chips_per_host))
        if self._master_port is not None:
            excluded.add(self._master_port)
        bootstrap_port = _pick_free_port(self._server_address, excluded)
        return self._master_address, self._master_port, bootstrap_port

    def _build_tpu_server_args_dict(self, tpu_node_normalization: Optional[bool] = None) -> dict[str, Any]:
        """Construct the ``ServerArgs`` keyword dictionary for ``sglang-torchtpu``."""
        raw_engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        if OmegaConf.is_config(raw_engine_kwargs):
            engine_kwargs = dict(OmegaConf.to_container(raw_engine_kwargs, resolve=True) or {})
        else:
            engine_kwargs = dict(raw_engine_kwargs)

        attention_backend = engine_kwargs.pop("attention_backend", None)
        mm_attention_backend = engine_kwargs.pop("mm_attention_backend", None)
        custom_weight_loader = list(engine_kwargs.pop("custom_weight_loader", None) or [])

        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        max_num_seqs = self.config.get("max_num_seqs", None)
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", None)

        enable_torch_compile = engine_kwargs.pop("enable_torch_compile", not bool(self.config.enforce_eager))
        torch_compile_max_bs = engine_kwargs.pop("torch_compile_max_bs", None)
        if torch_compile_max_bs is None and enable_torch_compile and max_num_seqs is not None:
            torch_compile_max_bs = int(max_num_seqs)

        kv_cache_dtype = engine_kwargs.pop("kv_cache_dtype", None)
        if kv_cache_dtype is None:
            # TpuSRTPlatform defaults "auto" to fp8_e4m3; match rollout dtype when bf16 is requested.
            kv_cache_dtype = "bfloat16" if str(self.config.dtype).lower() in ("bfloat16", "bf16") else "auto"

        enable_prefix_caching = self.config.get("enable_prefix_caching", True)
        disable_radix_cache = engine_kwargs.pop(
            "disable_radix_cache", False if enable_prefix_caching is None else not bool(enable_prefix_caching)
        )
        grammar_backend = engine_kwargs.pop("grammar_backend", "none")

        args: dict[str, Any] = {
            "model_path": self.model_config.local_path,
            "device": "tpu",
            "dtype": self.config.dtype,
            "kv_cache_dtype": kv_cache_dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": True,
            "enable_torch_compile": enable_torch_compile,
            "enable_memory_saver": False,
            "enable_weights_cpu_backup": False,
            "enable_draft_weights_cpu_backup": False,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": infer_tp,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": max_num_seqs,
            "context_length": self.config.max_model_len,
            "disable_radix_cache": disable_radix_cache,
            "grammar_backend": grammar_backend,
            "log_level": "error",
            "mm_attention_backend": mm_attention_backend,
            "attention_backend": attention_backend,
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "custom_weight_loader": custom_weight_loader or None,
        }
        if torch_compile_max_bs is not None:
            args["torch_compile_max_bs"] = torch_compile_max_bs
        seed = self.config.get("seed", None)
        if seed is not None:
            args.setdefault("random_seed", int(seed))
        if max_num_batched_tokens is not None:
            args.setdefault("chunked_prefill_size", int(max_num_batched_tokens))
            args.setdefault("max_prefill_tokens", int(max_num_batched_tokens))
        if tpu_node_normalization is not None:
            args.setdefault("tpu_node_normalization", tpu_node_normalization)

        if self.nnodes > 1:
            dist_init_addr = (
                f"[{self._master_address}]:{self._master_port}"
                if is_valid_ipv6_address(self._master_address)
                else f"{self._master_address}:{self._master_port}"
            )
            args["dist_init_addr"] = dist_init_addr

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name
            args["enable_metrics"] = True

        if self.config.enable_rollout_routing_replay:
            args["enable_return_routed_experts"] = True

        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            args["speculative_algorithm"] = self.config.mtp.speculative_algorithm
            args["speculative_num_steps"] = self.config.mtp.speculative_num_steps
            args["speculative_eagle_topk"] = self.config.mtp.speculative_eagle_topk
            args["speculative_num_draft_tokens"] = self.config.mtp.speculative_num_draft_tokens

        args.update(engine_kwargs)
        return args

    def _init_rpc_socket(self) -> None:
        """Bind the leader's ZMQ ``DEALER`` socket on ``port_args.rpc_ipc_name`` for ``collective_rpc``."""
        if self._rpc_socket is not None or self.port_args is None:
            return
        import zmq
        from sglang.srt.utils.network import get_zmq_socket

        self._rpc_context = zmq.Context(1)
        self._rpc_socket = get_zmq_socket(self._rpc_context, zmq.DEALER, self.port_args.rpc_ipc_name, True)

    async def launch_server(
        self,
        master_address: Optional[str] = None,
        master_port: Optional[int] = None,
        bootstrap_port: Optional[int] = None,
        tpu_node_normalization: Optional[bool] = None,
    ):
        import sglang.srt.entrypoints.engine
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.entrypoints.http_server import _GlobalState, app, set_global_state
        from sglang.srt.managers.tokenizer_manager import ServerStatus
        from sglang.srt.plugins import load_plugins
        from sglang.srt.server_args import ServerArgs

        from verl.workers.rollout.utils import run_uvicorn

        if self.nnodes > 1:
            if self.node_rank != 0:
                assert master_address and master_port, "non-master node should provide master address and port"
                self._master_address = master_address
                self._master_port = master_port
            if bootstrap_port is not None:
                os.environ["SGLANG_TPU_TOPOLOGY_BOOTSTRAP_PORT"] = str(bootstrap_port)

        args = self._build_tpu_server_args_dict(tpu_node_normalization=tpu_node_normalization)

        # Release the port-reservation socket before SGLang's TCPStore / node normalization binds _master_port.
        master_sock = getattr(self, "_master_sock", None)
        if master_sock is not None:
            master_sock.close()
            self._master_sock = None

        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        load_plugins()
        server_args = ServerArgs(**args)
        launch_res = Engine._launch_subprocesses(
            server_args=server_args,
            init_tokenizer_manager_func=sglang.srt.entrypoints.engine.init_tokenizer_manager,
            run_scheduler_process_func=run_tpu_sglang_scheduler_process,
            run_detokenizer_process_func=sglang.srt.entrypoints.engine.run_detokenizer_process,
        )
        self.tokenizer_manager = launch_res[0]
        self.template_manager = launch_res[1]
        if len(launch_res) > 3 and hasattr(launch_res[2], "rpc_ipc_name"):
            self.port_args = launch_res[2]
            scheduler_init_result = launch_res[3]
            subprocess_watchdog = launch_res[4] if len(launch_res) > 4 else None
            scheduler_infos = getattr(scheduler_init_result, "scheduler_infos", None) or []
            self.scheduler_info = scheduler_infos[0] if scheduler_infos else {}
        else:
            self.port_args = getattr(self.tokenizer_manager, "port_args", None)
            self.scheduler_info = launch_res[2]
            subprocess_watchdog = None

        if self.tokenizer_manager is None:
            return

        if subprocess_watchdog is not None:
            self.tokenizer_manager._subprocess_watchdog = subprocess_watchdog
        self._init_rpc_socket()

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = {"server_args": server_args}
        app.warmup_thread_args = (server_args, None, None)

        if server_args.enable_metrics:
            try:
                from sglang.srt.utils import add_prometheus_track_response_middleware

                add_prometheus_track_response_middleware(app)
            except ImportError:
                from sglang.srt.utils.common import add_prometheus_middleware

                add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_uvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Invoke ``method(*args, **kwargs)`` on every SGLang TP worker and return the per-rank results."""
        if not isinstance(method, str):
            raise TypeError(f"TPUSGLangHttpServer.collective_rpc requires a method name string, got {method!r}")
        if self.tokenizer_manager is None:
            raise RuntimeError("collective_rpc must be called on the leader TPUSGLangHttpServer")

        import zmq
        from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput, sock_recv, sock_send

        self._init_rpc_socket()
        payload = json.dumps({"method": method, "args": list(args), "kwargs": kwargs or {}})
        req = RpcReqInput(method=_VERL_COLLECTIVE_RPC_METHOD, parameters={"payload": payload})

        def _send_and_recv() -> Any:
            self._rpc_socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000) if timeout is not None else -1)
            sock_send(self._rpc_socket, req)
            return sock_recv(self._rpc_socket, flags=zmq.BLOCKY)

        async with self._rpc_lock:
            reply = await asyncio.to_thread(_send_and_recv)

        if not isinstance(reply, RpcReqOutput):
            raise RuntimeError(f"Unexpected SGLang RPC reply for {method}: {reply!r}")
        if not reply.success:
            raise RuntimeError(f"SGLang collective_rpc({method}) failed: {reply.message}")
        return json.loads(reply.message) if reply.message else []

    async def wake_up(self):
        """TPU keeps weights and KV cache resident in HBM (no CUDA VMM sleep mode); flush any stale radix cache."""
        if self.tokenizer_manager is None:
            return
        if self.rollout_mode == RolloutMode.HYBRID:
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        await self.tokenizer_manager.flush_cache()

    async def sleep(self):
        """No-op on TPU: ``sglang-torchtpu`` has no memory-saver sleep mode."""
        return

    async def clear_kv_cache(self):
        if self.tokenizer_manager is not None:
            await self.tokenizer_manager.flush_cache()

    async def release_kv_cache(self):
        """No-op on TPU."""
        return

    async def resume_kv_cache(self):
        """No-op on TPU."""
        return

    async def abort_all_requests(
        self, reset_prefix_cache: bool = True, reject_request: bool = False, abort_only: bool = False
    ):
        if self.tokenizer_manager is None:
            return
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        await self.tokenizer_manager.pause_generation(PauseGenerationReqInput(mode="abort"))
        if reset_prefix_cache and not abort_only:
            await self.tokenizer_manager.flush_cache()

    async def resume_generation(self):
        if self.tokenizer_manager is None:
            return
        from sglang.srt.managers.io_struct import ContinueGenerationReqInput

        try:
            # ContinueGenerationReqInput defaults torch_empty_cache=True in SGLang >= 0.5.12, which calls
            # torch.cuda.memory_reserved() inside Scheduler.continue_generation and fails on CPU-only torch + torch_tpu.
            req = ContinueGenerationReqInput(torch_empty_cache=False)
        except TypeError:
            req = ContinueGenerationReqInput()
        await self.tokenizer_manager.continue_generation(req)


class TPUSGLangReplica(SGLangReplica):
    """SGLang rollout replica for TPU: one server actor per host, none of which holds a chip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUSGLangHttpServer)

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_sglang_servers(self)
