# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for multi-host TPU rollout.

``patch_vllm_for_tpu`` lets the vLLM engine of a ``TPUvLLMHttpServer`` run on all hosts of a TPU
slice through Ray. Each patch fills a gap in the pinned vLLM (v0.29.0) and vllm-torchtpu
(9faafb17) and can be dropped once upstream closes it:

* ``EngineArgs.create_engine_config``, ``initialize_ray_cluster`` and
  ``RayDistributedExecutor._init_workers_ray``: run the engine on vLLM's generic Ray executor,
  on the placement groups verl reserved for the replica (one per host), with one worker per chip
  that gets the TPU multi-host environment. vllm-torchtpu's own Ray executor can use only a
  single placement group. Upstream fix: let it use several.
* ``RayDistributedExecutor._execute_dag``: run each step with plain Ray calls instead of a
  compiled Ray graph. The graph runs the model on a background thread, where vllm-torchtpu's
  raised torch.compile recompile limit does not apply, so the first request fails. Upstream fix:
  raise the limit on every thread.
* ``TPUWorker.reset_encoder_cache``: no-op. verl resets vLLM's caches after every weight update,
  and vllm-torchtpu's worker does not implement this call. Upstream fix: implement it.
* ``TPUModelRunner.gather_logprobs`` and ``sample_from_logits``: normalize log probabilities
  and draw sampling noise in FP32, avoiding BF16 cancellation and sampling bias.
* ``VLLM_DISABLE_COMPILE_CACHE=1``: a reloaded compile-cache artifact can make vLLM run the model
  eagerly, which crashes libtpu (b/501165531). Upstream fix: that bug.
* ``multiprocessing.process.BaseProcess.__init__`` and ``RayWorkerWrapper.__init__``: install
  these patches in the EngineCore process and in vLLM's worker actors, neither of which imports
  the plugin.
"""

import importlib.util
import inspect
import logging
import multiprocessing.process
import os
from collections import Counter

import ray
import torch
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl_hardware_plugin.accelerators.tpu.platform_tpu import resolve_tpu_topology_bounds

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def patch_tpu_logprobs() -> None:
    """Use FP32 log_softmax when reporting vllm-torchtpu token log probabilities."""
    if importlib.util.find_spec("vllm_torchtpu") is None:
        return

    from vllm.v1.outputs import LogprobsTensors
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if getattr(TPUModelRunner, "_verl_fp32_logprobs_patched", False):
        return

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor, sampled_tokens: torch.Tensor) -> LogprobsTensors:
        token_ids = sampled_tokens.to(torch.int64)
        token_logits = logits.gather(-1, token_ids)
        token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
        # Match the trainer's log_softmax + gather calculation. Upcast BEFORE
        # normalization: BF16 (token_logit - logsumexp(logits)).float() loses
        # small log probabilities to cancellation before the final cast.
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
        token_logprobs = log_probs.gather(-1, token_ids)

        max_logprobs = self.model_config.max_logprobs
        if max_logprobs > 0:
            topk_indices = torch.topk(logits, max_logprobs, dim=-1).indices
            topk_logprobs = log_probs.gather(-1, topk_indices)
            logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
            logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        else:
            logprob_token_ids = token_ids
            logprobs = token_logprobs

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )

    TPUModelRunner.gather_logprobs = gather_logprobs
    TPUModelRunner._verl_fp32_logprobs_patched = True
    logger.info("Applied TPU generator FP32 log_softmax log-probability patch.")


# Read only by the plain-Python wrapper, never by compiled code.
_fp32_sampling_noise_logged = False


# TODO: Remove once vllm-torchtpu draws its sampling uniforms in FP32.
def patch_tpu_sampler() -> None:
    """Draw TPU sampling noise from FP32 random values to reduce BF16 rounding bias.

    Use the runner's sampling generator when available.
    Apply this patch before the TPU model runner is created.
    """
    if importlib.util.find_spec("vllm_torchtpu") is None:
        return

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if getattr(TPUModelRunner, "_verl_fp32_sampler_patched", False):
        return

    original = getattr(TPUModelRunner, "sample_from_logits", None)
    expected = ["self", "logits", "temperatures", "u", "top_k", "top_p", "all_greedy"]
    try:
        parameters = list(inspect.signature(original).parameters) if callable(original) else None
    except (TypeError, ValueError):
        # A compile wrapper that hides its signature: nothing to check, assume the known layout.
        parameters = expected
    if parameters in (["args", "kwargs"], ["self", "args", "kwargs"]):
        parameters = expected
    if parameters != expected:
        logger.error(
            "TPU sampler FP32 patch NOT applied: TPUModelRunner.sample_from_logits has parameters %s, "
            "expected %s. Sampling noise stays BF16, which biases policy-gradient training.",
            parameters,
            expected,
        )
        return

    def sample_from_logits(self, logits, temperatures, u, top_k, top_p, all_greedy=False):
        # The all-greedy path never reads ``u``; preserve its compiled graph.
        if not all_greedy and u.dtype != torch.float32:
            global _fp32_sampling_noise_logged
            if not _fp32_sampling_noise_logged:
                _fp32_sampling_noise_logged = True
                logger.warning(
                    "verl TPU sampler patch active: sampling noise is drawn in float32 instead of %s "
                    "(logged once per process).",
                    str(u.dtype).removeprefix("torch."),
                )
            u = torch.rand(
                u.shape,
                dtype=torch.float32,
                device=u.device,
                # None during precompile warm-up, where the call site also uses the global RNG.
                generator=getattr(self, "_sampling_generator", None),
            )
        return original(self, logits, temperatures, u, top_k, top_p, all_greedy=all_greedy)

    # functools.wraps copies torch.compile's bookkeeping attributes, which can let
    # Dynamo unwrap straight to the compiled function and skip the redraw.
    sample_from_logits.__doc__ = getattr(original, "__doc__", None)
    sample_from_logits.__wrapped__ = original  # type: ignore[attr-defined]
    TPUModelRunner.sample_from_logits = sample_from_logits
    TPUModelRunner._verl_fp32_sampler_patched = True
    logger.info("Applied TPU generator FP32 sampling-noise patch.")


def patch_vllm_for_tpu() -> None:
    """Install the patches in this process. Later calls return at once."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    patch_multiprocessing_for_tpu()
    # vLLM can reload a degenerate AOT compile artifact (num_artifacts=0) that silently runs the
    # model eagerly; eager execution then hits the XLA:TPU unaligned-DUS CHECK (b/501165531).
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    # Imported here: CPU tests and processes outside the rollout may lack vLLM or vllm-torchtpu.
    try:
        import vllm.v1.executor.ray_executor as ray_executor
        from vllm.engine.arg_utils import EngineArgs
        from vllm_torchtpu.executors.ray_distributed_executor import RayWorkerWrapper
        from vllm_torchtpu.worker.tpu_worker import TPUWorker
    except ImportError as exc:
        logger.debug("Skipping vLLM TPU executor patches (vllm / vllm_torchtpu not installed): %s", exc)
        return

    patch_tpu_logprobs()
    patch_tpu_sampler()

    original_create_engine_config = EngineArgs.create_engine_config

    def create_engine_config(self, *args, **kwargs):
        # vllm-torchtpu picks its own Ray executor, which takes a single placement group, but verl
        # reserves one per host. Use vLLM's generic Ray executor, which the patches below adapt to
        # TPU. It does not support async scheduling, which vLLM turned on for the
        # external_launcher backend that TPUvLLMHttpServer passes in.
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        vllm_config.parallel_config.distributed_executor_backend = "ray"
        vllm_config.scheduler_config.async_scheduling = False
        return vllm_config

    original_worker_init = RayWorkerWrapper.__init__

    def worker_init(self, *args, **kwargs):
        # vLLM's worker actors never import the plugin. Ray sends this method to them with the
        # actor class, so it installs the patches there.
        patch_vllm_for_tpu()
        original_worker_init(self, *args, **kwargs)

    EngineArgs.create_engine_config = create_engine_config
    RayWorkerWrapper.__init__ = worker_init
    TPUWorker.reset_encoder_cache = _reset_encoder_cache
    ray_executor.initialize_ray_cluster = _initialize_ray_cluster
    ray_executor.RayDistributedExecutor._init_workers_ray = _init_workers_ray
    ray_executor.RayDistributedExecutor._execute_dag = _execute_dag
    _PATCHES_APPLIED = True
    logger.info("Successfully applied all TPU patches to vLLM and vllm-torchtpu.")


def _initialize_ray_cluster(parallel_config, ray_address=None) -> None:
    """Connect to Ray as vLLM does, but create no placement group: ``_init_workers_ray`` uses verl's."""
    if not ray.is_initialized():
        ray.init(address=ray_address, runtime_env=parallel_config.ray_runtime_env)


def _init_workers_ray(self, placement_group, **ray_remote_kwargs) -> None:
    """Start and initialize one vLLM worker per TPU chip of the replica.

    vLLM's version uses the single ``placement_group`` (unset here) and leaves the TPU environment
    to vllm-torchtpu's executor. This one places the workers on the placement groups verl reserved
    for the replica (``VERL_TPU_PG_IDS``, one per host) and gives each worker its chip's TPU
    environment.
    """
    from vllm.platforms import current_platform
    from vllm.ray.ray_env import get_env_vars_to_copy
    from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
    from vllm.v1.executor.ray_utils import WORKER_SPECIFIC_ENV_VARS
    from vllm_torchtpu import envs as vllm_torchtpu_envs
    from vllm_torchtpu.executors.ray_distributed_executor import RayWorkerWrapper

    pg_ids = os.environ["VERL_TPU_PG_IDS"].split(",")
    logger.info("Reusing %d verl placement group(s) for TPU rollout: %s", len(pg_ids), pg_ids)
    bundles = [
        (pg, bundle_index)
        for pg in (PlacementGroup(ray.PlacementGroupID.from_hex(pg_id)) for pg_id in pg_ids)
        for bundle_index, bundle in enumerate(pg.bundle_specs)
        if bundle.get(current_platform.ray_device_key, 0)
    ][: self.parallel_config.world_size]
    self.workers = [
        ray.remote(
            num_cpus=0,
            num_gpus=0,
            resources={current_platform.ray_device_key: 1},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            ),
            **ray_remote_kwargs,
        )(RayWorkerWrapper).remote(rpc_rank=rank)
        for rank, (pg, bundle_index) in enumerate(bundles)
    ]

    # Rank the workers as vLLM does: the driver's host first, because rank 0 serves the
    # torch.distributed store at the driver's address, then each host's workers together.
    driver_ip = get_ip()
    ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])
    workers_per_ip = Counter(ips)
    order = sorted(range(len(ips)), key=lambda i: (ips[i] != driver_ip, workers_per_ip[ips[i]], ips[i]))
    self.workers = [self.workers[i] for i in order]
    ips = [ips[i] for i in order]
    self.collective_rpc("adjust_rank", args=({created: rank for rank, created in enumerate(order)},))

    driver_env = {
        name: os.environ[name]
        for name in get_env_vars_to_copy(
            exclude_vars=WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars),
            destination="workers",
        )
        if name in os.environ
    }
    # torch_tpu's own rendezvous, separate from the torch.distributed one below.
    torch_tpu_master = {"MASTER_ADDR": ips[0], "MASTER_PORT": str(get_open_port())}
    self._env_vars_for_all_workers = [
        {**driver_env, **torch_tpu_master, **tpu_env}
        for tpu_env in _tpu_worker_envs(ips, vllm_torchtpu_envs.TORCH_TPU_BASE_PORT)
    ]
    self.collective_rpc("update_environment_variables", args=(self._get_env_vars_to_be_updated(),))
    logger.info("TPU rollout process addresses: %s", self._env_vars_for_all_workers[0]["TPU_PROCESS_ADDRESSES"])

    distributed_init_method = get_distributed_init_method(driver_ip, get_open_port())
    local_ranks = _local_ranks(ips)
    all_kwargs = [
        dict(
            vllm_config=self.vllm_config,
            local_rank=local_ranks[rank],
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=rank % self.parallel_config.tensor_parallel_size == 0,
        )
        for rank in range(len(self.workers))
    ]
    self.collective_rpc("init_worker", args=(all_kwargs,))
    self.collective_rpc("init_device")
    self.collective_rpc("load_model")


def _local_ranks(worker_ips: list[str]) -> list[int]:
    """Index of each worker among the workers on its host, for workers in rank order."""
    return [worker_ips[:rank].count(ip) for rank, ip in enumerate(worker_ips)]


def _tpu_worker_envs(worker_ips: list[str], base_port: int) -> list[dict[str, str]]:
    """The TPU multi-host environment of each vLLM worker, for workers in rank order.

    Ranks on one host must be contiguous. A worker drives chip ``local_rank`` of its host and serves
    the slice builder on ``base_port + local_rank``. These are the variables libtpu and torch_tpu
    read to join the workers into one TPU mesh.
    """
    hosts = list(dict.fromkeys(worker_ips))
    local_ranks = _local_ranks(worker_ips)
    addresses = ",".join(
        f"{ip}:{base_port + local_rank}" for ip, local_rank in zip(worker_ips, local_ranks, strict=True)
    )
    topology, host_bounds, chips_per_host_bounds, _ = resolve_tpu_topology_bounds(len(worker_ips), len(hosts))
    return [
        {
            "TPU_VISIBLE_CHIPS": str(local_rank),
            "TPU_PROCESS_PORT": str(base_port + local_rank),
            "TPU_PROCESS_ADDRESSES": addresses,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": addresses,
            "CLOUD_TPU_TASK_ID": str(hosts.index(ip)),
            "TPU_WORKER_HOSTNAMES": ",".join(hosts),
            "TORCH_TPU_TOPOLOGY": topology,
            "TPU_HOST_BOUNDS": host_bounds,
            "TPU_CHIPS_PER_HOST_BOUNDS": chips_per_host_bounds,
        }
        for ip, local_rank in zip(worker_ips, local_ranks, strict=True)
    ]


def _execute_dag(self, scheduler_output, grammar_output, non_block=False):
    """Run one step on every worker with plain Ray calls instead of vLLM's compiled Ray graph.

    The graph runs the model on a background thread, where torch.compile's recompile limit is
    still the default 8: vllm-torchtpu raises it to 1024 on the main thread only (dynamo config
    overrides are per thread), and the first request fails with FailOnRecompileLimitHit. verl uses
    no KV connector, so rank 0's output is the step's result.
    """
    from vllm.v1.executor.ray_utils import FutureWrapper, detach_zero_copy_from_model_runner_output

    refs = [worker.execute_model_ray.remote((scheduler_output, grammar_output)) for worker in self.workers]
    if non_block:
        return FutureWrapper(refs[0])
    output = ray.get(refs)[0]
    detach_zero_copy_from_model_runner_output(output)
    return output


def _reset_encoder_cache(self) -> None:
    """``TPUWorker.reset_encoder_cache``, which verl calls after every weight update: a no-op.

    The encoder cache holds multimodal encoder outputs; text models have none to reset.
    """


class PickleableProcessWrapper:
    """Process target that runs ``patch_vllm_for_tpu`` in the child before ``target``.

    This is how the patches reach vLLM's EngineCore. Inside a Ray actor vLLM always uses the
    ``spawn`` start method, and the EngineCore target is ``EngineCoreProc.run_engine_core``, which
    vllm-torchtpu replaces with its own wrapper. A spawned child imports only the modules its pickled
    target names, so without this wrapper it would never import the plugin.
    """

    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        patch_vllm_for_tpu()
        return self.target(*args, **kwargs)


def patch_multiprocessing_for_tpu() -> None:
    """Wrap the target of every ``multiprocessing`` process started from now on in ``PickleableProcessWrapper``."""
    base_process = multiprocessing.process.BaseProcess
    if getattr(base_process, "_tpu_patched", False):
        return
    original_init = base_process.__init__

    def patched_init(self, group=None, target=None, *args, **kwargs):
        if target is not None:
            target = PickleableProcessWrapper(target)
        original_init(self, group, target, *args, **kwargs)

    base_process.__init__ = patched_init  # type: ignore[method-assign]
    base_process._tpu_patched = True  # type: ignore[attr-defined]
