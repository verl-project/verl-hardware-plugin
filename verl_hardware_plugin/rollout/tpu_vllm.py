# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM rollout replica and HTTP server for Google TPU.

Upstream verl builds the vLLM rollout through ``RolloutReplicaRegistry``. On TPU the plugin
registers a ``vllm`` loader that returns :class:`TPUvLLMReplica`. It keeps upstream's HTTP server
and changes only what TPU needs:

* One server actor per replica, on the first worker's node, with every worker's TPU
  environment forwarded (``launch_tpu_vllm_servers``). vLLM then spans the hosts through its
  Ray executor instead of verl's per-node ``mp`` launch.
* ``distributed_executor_backend=external_launcher`` and ``enable_sleep_mode=False``: the
  multi-host executor is selected by ``patch_vllm_for_tpu``, and vllm-torchtpu has no sleep
  mode.
* Standalone rollout workers do not claim a GPU resource.

The same logic previously lived behind ``get_resource_name() == "TPU"`` branches in verl's
``vllm_async_server.py`` and ``replica.py``.
"""

import asyncio
import logging
import os

import ray

from verl.plugin.platform import get_platform
from verl.utils.device import get_device_name, get_resource_name
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from verl_hardware_plugin.rollout.tpu_vllm_patches import patch_vllm_for_tpu

logger = logging.getLogger(__name__)


def is_tpu_vllm_run() -> bool:
    """Returns True if executing on a Google TPU resource."""
    return get_resource_name() == "TPU"


def prepare_tpu_server_env() -> None:
    """Routes vllm-torchtpu to its Ray multi-host backend in the server process."""
    os.environ["TPU_MULTIHOST_BACKEND"] = "ray"
    os.environ["VLLM_USE_RAY_V2_EXECUTOR_BACKEND"] = "0"
    # See patch_vllm_for_tpu: a reloaded empty AOT artifact runs the model eagerly (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    try:
        import vllm_torchtpu.envs as tpu_envs

        tpu_envs.TPU_MULTIHOST_BACKEND = "ray"
        if hasattr(tpu_envs, "__getattr__") and hasattr(tpu_envs.__getattr__, "cache_clear"):
            tpu_envs.__getattr__.cache_clear()
    except Exception as env_err:
        logger.warning(f"Failed to force TPU_MULTIHOST_BACKEND to ray: {env_err}")


async def get_tpu_server_launch_config(workers):
    """
    Asynchronously queries node ID, visible chips, and TPU specific environment
    variables from all TPU workers for launching the server actor on TPU.
    """
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (
                    ray.get_runtime_context().get_node_id(),
                    os.environ.get("TPU_VISIBLE_CHIPS", "0"),
                )
            )
            for worker in workers
        ]
    )

    worker_tpu_envs = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("TPU_")
                    or k.startswith("TORCH_TPU_")
                    or k
                    in (
                        "CLOUD_TPU_TASK_ID",
                        "CHIPS_PER_HOST",
                        "JAX_MEM_FRACTION",
                        "JAX_THREE_G_MEM_ALLOC_ON_FREE",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "LIBTPU_INIT_ARGS",
                        "TORCH_DYNAMO_RECOMPILE_LIMIT",
                        "SKIP_JAX_PRECOMPILE",
                        "VLLM_ENABLE_V1_MULTIPROCESSING",
                        # Keep the vLLM AOT compile cache disabled in the server
                        # process too: a reloaded artifact degrades the model to
                        # eager execution and reintroduces the unaligned-DUS
                        # crash (b/501165531). See patch_vllm_for_tpu().
                        "VLLM_DISABLE_COMPILE_CACHE",
                        "VERL_PLATFORM",
                        "XLA_FLAGS",
                    )
                }
            )
            for worker in workers
        ]
    )

    node_id = worker_infos[0][0]
    visible_chips = ",".join([info[1] for info in worker_infos])
    tpu_env_vars = worker_tpu_envs[0] if worker_tpu_envs else {}

    return node_id, visible_chips, tpu_env_vars


# =============================================================================
# Preflight census of leaked vLLM engine processes.
#
# Every rollout engine that shuts down leaves processes in state Z, reparented
# to PID 1, one of them VLLM::EngineCore. Nothing reaps them, so the count grows
# for the lifetime of the pod. Past some point the host stops handing out TPU
# devices and the next job dies ~8k log lines into engine init with an opaque
# "local device count is 0". Naming the host up front turns that into a lead.
#
# Enforcement is off by default. The count correlates with failures (0, 1 and 4
# passed; 5 and ~10 failed) but the threshold rests on too little data to fail a
# good run over, so the census reports unless asked otherwise:
#
#   VERL_TPU_MAX_STALE_ENGINES unset, or -1   (default) report only, never fail
#   VERL_TPU_MAX_STALE_ENGINES=0              fail if any host leaked even one
#   VERL_TPU_MAX_STALE_ENGINES=4              fail if a host leaked more than 4
#
# Negative is the "off" sentinel rather than 0, so 0 stays usable as a real
# threshold. There is deliberately no wait loop: zombies never clear on their
# own, so waiting could only delay the failure.
#
# NOTE: do not name the libtpu fusion CHECK in any message emitted from here. An
# earlier revision did, and log scrapers searching for the crash matched this
# check's output instead.
# =============================================================================


def probe_stale_tpu_engines(self=None):
    """Return a zombie-process census for this host.

    Executed remotely on each TPU worker actor via ``__ray_call__``, hence the
    unused ``self`` parameter and the function-local imports: the callable is
    pickled and must not depend on the caller's module state.

    Returns a dict with ``hostname``, ``zombies`` (every process in state ``Z``)
    and ``engines`` (those whose command name looks like a vLLM EngineCore).

    ``/proc/<pid>/cmdline`` is empty for a zombie -- that is why ``ps`` renders
    them as ``<defunct>`` -- so the name has to come from the ``comm`` field of
    ``/proc/<pid>/stat`` instead.
    """
    import os as _os
    import socket as _socket

    # /proc/<pid>/stat truncates comm to 15 characters, so "VLLM::EngineCore"
    # arrives as "VLLM::EngineCor". Match on the truncated form.
    engine_marker = "EngineCor"

    census = {"hostname": _socket.gethostname(), "zombies": 0, "engines": 0}

    try:
        entries = _os.listdir("/proc")
    except OSError:
        return census

    for entry in entries:
        if not entry.isdigit():
            continue

        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                stat_line = handle.read().decode(errors="replace")
        except OSError:
            # Process exited mid-scan, or /proc is not readable. Not actionable.
            continue

        # comm sits between the first '(' and the last ')' and may itself
        # contain spaces or parentheses, so anchor on the LAST ')'.
        close = stat_line.rfind(")")
        if close == -1:
            continue

        fields = stat_line[close + 1 :].split()
        if not fields or fields[0] != "Z":
            continue

        census["zombies"] += 1

        opened = stat_line.find("(")
        comm = stat_line[opened + 1 : close] if opened != -1 else ""
        if engine_marker in comm:
            census["engines"] += 1

    return census


async def report_stale_tpu_engines(workers) -> None:
    """Log a per-host census of leaked vLLM engine processes before launching servers.

    Reports only, unless ``VERL_TPU_MAX_STALE_ENGINES`` is set to a non-negative
    value, in which case a host holding more than that many leaked engines
    raises before any TPU work starts. Unset or negative disables enforcement;
    ``0`` means "fail if any host has leaked even one". Negative is the off
    sentinel rather than ``0`` so that ``0`` stays usable as a real threshold.

    This never blocks, and deliberately does not wait for the count to come
    down. Waiting would assume the stragglers eventually clear; these are
    zombies, so nothing short of recycling the pod removes them and a wait could
    only delay the failure.

    Args:
        workers: TPU worker actor handles. There is one per chip, so several of
            them report the same host; the census is deduplicated by hostname.

    Raises:
        RuntimeError: Only when enforcement is enabled and a host is over the limit.
    """
    if not is_tpu_vllm_run() or not workers:
        return

    raw_limit = os.environ.get("VERL_TPU_MAX_STALE_ENGINES", "-1")
    try:
        limit = int(raw_limit)
    except ValueError:
        _tpu_preflight_log(f"ignoring malformed VERL_TPU_MAX_STALE_ENGINES={raw_limit!r}, staying report-only")
        limit = -1

    try:
        per_actor = await asyncio.gather(*[worker.__ray_call__.remote(probe_stale_tpu_engines) for worker in workers])
    except Exception as probe_err:
        # The census is a diagnostic aid; never let it be the thing that breaks
        # a run that would otherwise have worked.
        _tpu_preflight_log(f"census failed, continuing without preflight check: {probe_err}")
        return

    by_host = {}
    for census in per_actor:
        if isinstance(census, dict) and census.get("hostname"):
            by_host[census["hostname"]] = census

    if not by_host:
        return

    summary = ", ".join(
        f"{host}: {by_host[host]['engines']} engine(s) / {by_host[host]['zombies']} zombie(s)"
        for host in sorted(by_host)
    )
    _tpu_preflight_log(f"leaked rollout engines across {len(by_host)} host(s) -- {summary}")

    if limit < 0:
        return

    over = sorted(host for host, census in by_host.items() if census["engines"] > limit)
    if not over:
        return

    raise RuntimeError(
        f"[TPU preflight] {len(over)}/{len(by_host)} TPU host(s) exceed "
        f"VERL_TPU_MAX_STALE_ENGINES={limit}: {', '.join(over)}. Leaked engine processes "
        "accumulate for the lifetime of the pod and are never reaped. Past some point the "
        "host stops handing out TPU devices and vLLM engine init dies with 'local device "
        "count is 0' about 8k log lines in. Recycle the workers and resubmit:\n"
        "    kubectl delete pod -l ray.io/cluster=<your-cluster>,ray.io/node-type=worker\n"
        f"Census: {summary}"
    )


def _tpu_preflight_log(message: str, tag: str = "TPU preflight") -> None:
    """Emit a preflight message via both logging and stdout."""
    logger.warning("[%s] %s", tag, message)
    print(f"[{tag}] {message}", flush=True)


def _server_max_concurrency(replica: vLLMReplica) -> int:
    """Ray ``max_concurrency`` for the server actor.

    Upstream verl reads ``RolloutConfig.ray_actor_max_concurrency``; older builds (including the
    TPU fork) only expose ``vLLMReplica.max_concurrency``.
    """
    value = getattr(replica.config, "ray_actor_max_concurrency", None)
    return value if value is not None else replica.max_concurrency


async def launch_tpu_vllm_servers(replica: vLLMReplica) -> None:
    """Launch the vLLM rollout server actor for a TPU replica."""
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

    await report_stale_tpu_engines(replica.workers)

    node_id, visible_chips, tpu_env_vars = await get_tpu_server_launch_config(replica.workers)

    prefix = replica._get_server_name_prefix()
    if replica.is_reward_model:
        name = f"{prefix}server_reward_{replica.replica_rank}_0{replica.name_suffix}"
    elif replica.is_teacher_model:
        name = f"{prefix}server_teacher_{replica.replica_rank}_0{replica.name_suffix}"
    else:
        name = f"{prefix}server_{replica.replica_rank}_0{replica.name_suffix}"

    platform_env_vars = get_platform().rollout_env_vars()
    env_vars = {
        **{var: "1" for var in get_platform().ray_noset_envvars()},
        **platform_env_vars,
        **tpu_env_vars,
    }
    if "VERL_PLATFORM" in os.environ:
        env_vars["VERL_PLATFORM"] = os.environ["VERL_PLATFORM"]

    flags_to_copy = set()
    resource_pool = getattr(replica, "resource_pool", None)
    if resource_pool is not None:
        pgs = resource_pool.get_placement_groups(device_name=get_device_name())
        if pgs:
            env_vars["VERL_TPU_PG_IDS"] = ",".join(pg.id.hex() for pg in pgs)
            flags_to_copy.add("VERL_TPU_PG_IDS")

    for flag_var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
        base_value = platform_env_vars.get(flag_var) or tpu_env_vars.get(flag_var)
        extra_value = os.environ.get(f"VERL_TPU_EXTRA_{flag_var}")
        resolved = " ".join(v for v in (base_value, extra_value) if v)
        _tpu_preflight_log(
            f"{flag_var}: base={base_value!r} extra={extra_value!r} -> engine={resolved or None!r}",
            tag="TPU env",
        )
        if resolved:
            env_vars[flag_var] = resolved
            flags_to_copy.add(flag_var)

    if flags_to_copy:
        copy_var = "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
        existing = env_vars.get(copy_var) or os.environ.get(copy_var) or ""
        names = {tok.strip() for tok in existing.split(",") if tok.strip()}
        env_vars[copy_var] = ",".join(sorted(names | flags_to_copy))
        _tpu_preflight_log(f"{copy_var}={env_vars[copy_var]}", tag="TPU env")

    server = replica.server_class.options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        ),
        runtime_env={"env_vars": env_vars},
        name=name,
        max_concurrency=_server_max_concurrency(replica),
    ).remote(
        config=replica.config,
        model_config=replica.model_config,
        rollout_mode=replica.rollout_mode,
        workers=replica.workers,
        replica_rank=replica.replica_rank,
        node_rank=0,
        gpus_per_node=replica.gpus_per_replica_node,
        nnodes=replica.nnodes,
        cuda_visible_devices=visible_chips,
    )
    replica.servers.append(server)

    master_address, master_port, dp_rpc_port = await replica.servers[0].get_master_address.remote()
    await replica.servers[0].launch_server.remote(
        master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
    )

    server_address, server_port = await replica.servers[0].get_server_address.remote()
    replica._server_handle = replica.servers[0]
    replica._server_address = (
        f"[{server_address}]:{server_port}"
        if is_valid_ipv6_address(server_address)
        else f"{server_address}:{server_port}"
    )


class TPUvLLMHttpServer(vLLMHttpServer):
    """``vLLMHttpServer`` with the TPU engine arguments."""

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        super()._preprocess_engine_kwargs(engine_kwargs)
        # engine_kwargs is merged last into the vLLM CLI args, so these win over the defaults.
        # patch_vllm_for_tpu switches a multi-host engine to the Ray executor at config time.
        engine_kwargs["distributed_executor_backend"] = "external_launcher"
        engine_kwargs["enable_sleep_mode"] = False
        prepare_tpu_server_env()


class TPUvLLMReplica(vLLMReplica):
    """vLLM rollout replica for TPU: one server actor per replica, no GPU resource claims."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(TPUvLLMHttpServer)

    def rollout_worker_use_gpu(self) -> bool:
        return False

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )
        await launch_tpu_vllm_servers(self)


# This module is imported by the driver (through the ``vllm`` replica loader) and by every
# TPUvLLMHttpServer actor, which is the process that spawns the vLLM EngineCore.
if is_tpu_vllm_run():
    patch_vllm_for_tpu()
