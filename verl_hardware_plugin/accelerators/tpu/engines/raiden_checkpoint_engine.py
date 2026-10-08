# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Raiden (tpu-sync) checkpoint engine: peer-to-peer weight sync between TPU slices.

The ``tpu`` checkpoint engine stages the whole model through Ray's object store. ``raiden`` instead
moves the weights over the network from the trainer chips to the rollout chips with the Raiden transfer
library (``tpu-sync-torch``, https://github.com/google/tpu-sync). Every trainer rank registers the full
weights that verl gathers for it, and each rollout chip receives only its vLLM tensor-parallel shard::

     [Trainer slice: FSDP ranks]                       [Rollout slice: vLLM TP workers]
    +-------------------------------------+           +--------------------------------------+
    | RaidenCheckpointEngine.send_weights |           | vLLMRaidenWorkerExtension            |
    |  - bind device tensors to a         |           |  - init_raiden_sync_on_worker:       |
    |    WeightSynchronizer               |           |    allocate TP-sharded receive       |
    |  - register "trainer/<rank>"        |           |    buffers, register "sampler/<rank>"|
    +------------------+------------------+           |  - install_raiden_weights: H2D and   |
                       |                              |    fuse into vLLM's parameters       |
                       +-- host-to-host, resharded -->+--------------------------------------+
                                     ^
                     [Driver: update_raiden_weights + RaidenController]

``update_raiden_weights`` runs on the driver in place of ``CheckpointEngineManager.update_weights``
(see ``apply_tpu_checkpoint_engine_hooks``). Trainer and rollout processes find the controller, and
the tensor shapes, through the ``RayWeightRegistry`` actor.

Security: the controller and the WeightSynchronizer endpoints accept unauthenticated connections and
carry model weights in the clear. Run them only on a trusted cluster network.
"""

import asyncio
import gc
import logging
import os
import time
from collections.abc import Iterable
from typing import Any, Optional

import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import get_ray_weight_registry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

RAIDEN_BACKEND = "raiden"
# Parallel transfer streams per worker, unless engine_kwargs.raiden.parallelism overrides it.
RAIDEN_DEFAULT_PARALLELISM = 8
# How long the driver waits for every trainer and rollout worker to register with the controller.
RAIDEN_REGISTRATION_TIMEOUT_S = 60.0
# Raiden reads and writes device buffers in place, so the tensors it is given must live on this device.
RAIDEN_DEVICE = "tpu"


def tpu_synchronize(strict: bool = False) -> None:
    """Waits for pending TPU work, so Raiden reads final device buffers.

    Args:
        strict: Raise if ``torch_tpu`` cannot synchronize instead of logging a warning.
    """
    try:
        from torch_tpu._internal import sync as torch_tpu_sync

        torch_tpu_sync.synchronize(wait=True)
    except Exception as e:
        if strict:
            raise RuntimeError(f"TPU synchronization failed: {e}") from e
        logger.warning(f"Could not synchronize via torch_tpu: {e}")


def compute_tensor_stats(items: list) -> dict:
    """Compute deterministic L1/L2 norms and parameter counts across tensors.

    Reductions run on the device holding the tensors, to avoid copying the weights to the host.
    ``items`` holds ``(name, tensor)`` pairs (named tensors get a ``per_tensor`` entry) or bare tensors.
    """
    total_numel = 0
    total_l1 = 0.0
    total_l2_sq = 0.0
    per_tensor = {}

    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            name, p = item
        else:
            name = None
            p = item
        p_local = p.to_local() if hasattr(p, "to_local") else p

        # Reduce in float32 for precision.
        p_float = p_local.float()
        t_l1 = float(p_float.abs().sum().item())
        t_l2_sq = float(p_float.pow(2).sum().item())
        numel = p_local.numel()

        if name is not None:
            per_tensor[name] = {
                "l1": t_l1,
                "l2": float(t_l2_sq**0.5),
                "l2_sq": t_l2_sq,
                "numel": numel,
                "shape": list(p_local.shape),
                "dtype": str(p_local.dtype),
            }
        total_numel += numel
        total_l1 += t_l1
        total_l2_sq += t_l2_sq

    return {
        "total_numel": total_numel,
        "num_tensors": len(items),
        "l1_norm": total_l1,
        "l2_norm": float(total_l2_sq**0.5),
        "l2_sq": total_l2_sq,
        "per_tensor": per_tensor,
    }


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = RAIDEN_DEFAULT_PARALLELISM,
    listener_port: int = 0,
    bind_ip: str = "127.0.0.1",
) -> Any:
    """Creates a tpu_sync ``WeightSynchronizer`` over ``device_tensors``."""
    from tpu_sync.api.torch.weight_synchronizer import WeightSynchronizer

    return WeightSynchronizer(
        device_tensors,
        local_port=local_port,
        parallelism=parallelism,
        listener_port=listener_port,
        bind_ip=bind_ip,
        unsafe_skip_buffer_lock=True,
        auto_h2d=False,
    )


def _unwrap_tensor(t: Any) -> Any:
    """Extract the raw underlying local tensor from nn.Parameter or DTensor wrappers."""
    t = t.to_local().data if hasattr(t, "to_local") else (t.data if hasattr(t, "data") else t)
    return t


def filter_tied_embeddings(
    named_items: Iterable[tuple[str, Any]], tie_word_embeddings: bool = True
) -> list[tuple[str, Any]]:
    """Drop ``lm_head.weight`` from the weights to send when it is tied to the input embedding.

    For tied models (``tie_word_embeddings=True``, e.g. Qwen3-0.6B / 4B), ``lm_head.weight`` is the same
    tensor as ``embed_tokens.weight``, so it is not sent; the sampler copies ``embed_tokens`` into its
    ``lm_head`` after receiving the weights.

    For untied models (``tie_word_embeddings=False``, e.g. Qwen3-8B and larger), ``lm_head.weight`` is a
    separate trained weight and is kept. Dropping it would leave the sampler with a wrong ``lm_head``
    (stale, or overwritten with ``embed_tokens``).

    Args:
        named_items: ``(name, tensor)`` pairs in HF naming.
        tie_word_embeddings: ``hf_config.tie_word_embeddings`` of the trained model.

    Returns:
        The ``(name, tensor)`` pairs to send.
    """
    items = list(named_items)
    has_embed = any("embed_tokens" in k or "tok_embeddings" in k for k, _ in items)
    if has_embed and tie_word_embeddings:
        items = [(k, v) for k, v in items if not (k == "lm_head.weight" or k.endswith(".lm_head.weight"))]
    return items


def validate_and_sanitize_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    device: Optional[torch.device] = None,
) -> list[tuple[str, torch.Tensor]]:
    """Validate and sanitize tensors for zero-copy DMA registration with Raiden.

    Performs physical memory validation:
    1. Drops None, non-tensor objects, zero-element tensors, and unallocated meta tensors.
    2. Unwraps DTensor/nn.Parameter to local tensor buffers.
    3. Ensures tensors physically reside on ``device`` (TPU HBM by default).
    4. Guarantees memory contiguity (contiguous buffers) required for direct DMA.
    """
    if device is None:
        device = torch.device(RAIDEN_DEVICE)

    sanitized = []
    for item in named_tensors:
        name, p = item[0], item[1]
        if p is None:
            continue
        t = _unwrap_tensor(p)
        if not isinstance(t, torch.Tensor):
            continue

        # Skip 0-element tensors and unallocated meta tensors (prevents C++ nullptr faults)
        if t.numel() == 0 or getattr(t, "is_meta", False):
            continue

        # Ensure the tensor physically resides in device memory
        if t.device.type != device.type:
            try:
                t = t.to(device)
            except Exception as e:
                logger.warning(f"Could not move {name} to {device}: {e}")
                continue

        # Ensure contiguous memory layout (required for direct DMA pointer calculation)
        if not t.is_contiguous():
            t = t.contiguous()

        sanitized.append((name, t))

    return sanitized


def setup_raiden_controller() -> tuple[Any, Any, str]:
    """Start a Raiden controller server in this process and publish its address in ``RayWeightRegistry``.

    Returns:
        ``(controller, server, "host:port")``. The caller must keep ``server`` referenced while syncing.
    """
    from tpu_sync.rpc import raiden_controller

    # TODO(security): the controller (like the WeightSynchronizer endpoints) accepts unauthenticated
    # connections. Add authentication once tpu_sync supports it; until then it must only be reachable
    # from the cluster's trusted network.
    controller = raiden_controller.RaidenController(port=0)
    server = raiden_controller.RaidenControllerServer(controller)
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info(f"RaidenControllerServer started on the driver: {address}")

    try:
        registry = get_ray_weight_registry()
        ray.get(registry.set_controller_address.remote(address))
        logger.info(f"Stored the RaidenController address ({address}) in RayWeightRegistry")
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store RaidenController address ({address}) in RayWeightRegistry: {reg_err}"
        ) from reg_err
    return controller, server, address


def raiden_is_tile_aligned(local_shape: list) -> bool:
    """True when a 2D+ shard matches the TPU (8, 128) tile, so it can skip the CPU (de)tiling pass."""
    return (len(local_shape) >= 2) and (local_shape[-1] % 128 == 0) and (local_shape[-2] % 8 == 0)


def apply_raiden_skip_tiling(ws: Any, skip_tiling_plan: list) -> None:
    """Set the per-tensor skip_tiling plan on a WeightSynchronizer (API name differs across tpu_sync builds)."""
    if hasattr(ws, "test_only_set_skip_tiling"):
        ws.test_only_set_skip_tiling(skip_tiling_plan)
    elif hasattr(ws, "set_skip_tiling"):
        ws.set_skip_tiling(skip_tiling_plan)


def _as_bool(value: Any) -> bool:
    """Parse a bool flag that may arrive as a string (env var, CLI override) as well as a bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


@CheckpointEngineRegistry.register(RAIDEN_BACKEND)
class RaidenCheckpointEngine(CheckpointEngine):
    """Trainer-side Raiden engine: exposes this rank's weights to the controller-driven P2P transfer.

    Engine kwargs (``actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden``):

    * ``parallelism`` (default 8): parallel transfer streams per worker.
    * ``verify_parity`` (default False): post weight norms after every sync so the driver can check
      that the rollout received the trainer's weights.
    * ``tie_word_embeddings`` (default unset): whether ``lm_head`` is tied to the input embedding, so it
      need not be sent. Unset sends ``lm_head``, which is correct for tied and untied models alike.
    """

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs: Any) -> None:
        # CheckpointEngine defines no __init__ (object.__init__ takes no arguments).
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.parallelism = int(kwargs.get("parallelism", RAIDEN_DEFAULT_PARALLELISM))
        self.verify_parity = _as_bool(kwargs.get("verify_parity", False))
        tie_word_embeddings = kwargs.get("tie_word_embeddings")
        self.tie_word_embeddings = None if tie_word_embeddings is None else _as_bool(tie_word_embeddings)
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        self._registry: Any = None
        self._controller_addr: Optional[str] = None
        self._trainer_raiden_ws: Any = None

    @property
    def registry(self) -> Any:
        """The ``RayWeightRegistry`` actor, looked up on first use."""
        if self._registry is None:
            self._registry = get_ray_weight_registry()
        return self._registry

    def prepare(self) -> dict:
        return {}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def finalize(self) -> None:
        self._close_synchronizer()

    def _close_synchronizer(self) -> None:
        ws, self._trainer_raiden_ws = self._trainer_raiden_ws, None
        close = getattr(ws, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                logger.debug(f"Trainer Rank {self.rank}: closing the WeightSynchronizer failed: {e}")

    async def send_weights(self, weights: Any, global_steps: Optional[int] = None) -> None:
        """Bind this rank's weights to a new WeightSynchronizer, register them with the controller and stage them
        in host memory.

        Args:
            weights: Iterable (or dict) of ``(hf_name, tensor)`` pairs: the full weights, which verl gathers on
                every trainer rank.
            global_steps: Trainer step of these weights.

        The transfer itself is driven by the controller after this returns, so the synchronizer, which pins the
        bound tensors, is kept until ``release_sync_buffers``.
        """
        step_key = global_steps if global_steps is not None else 0
        logger.info(f"RaidenCheckpointEngine: [Step {step_key}] Start send_weights...")
        tpu_synchronize(strict=True)

        # Materialize weights in a single pass and filter tied embeddings
        named_weights = filter_tied_embeddings(
            weights.items() if hasattr(weights, "items") else weights,
            tie_word_embeddings=bool(self.tie_word_embeddings),
        )

        # Trainer sends pure canonical un-fused model weights directly
        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}

        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights, device=torch.device(RAIDEN_DEVICE))

        tpu_synchronize(strict=True)

        # A new synchronizer every sync: release_sync_buffers destroys the previous one after its transfer.
        self._close_synchronizer()

        logger.info(f"Trainer Rank {self.rank}: binding {len(valid_weights)} tensors to WeightSynchronizer")
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_raiden_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=self.parallelism,
            bind_ip=bind_ip,
        )

        # Record global shapes in RayWeightRegistry so the sampler can size its receive buffers.
        # TODO(tpu): Move global_shapes registration to a one-time setup step during initialization
        # (e.g., in prepare or build_process_group/model_init) instead of per weight-sync step,
        # since tensor shapes are static across training iterations and only need to be communicated once.
        try:
            global_shapes = {name: list(p.shape) for name, p in valid_weights}
            await self.registry.set_global_shapes.remote(global_shapes)
        except Exception as e:
            logger.warning(f"Could not record global shapes in RayWeightRegistry: {e}")

        # Build variable metadata protos for each dynamic tensor
        # For Trainer (full unsharded model gathered on every rank):
        # sharding_spec is empty strings (unsharded) and mesh_shape is [1] * rank
        from tpu_sync.rpc import raiden_controller, raiden_service_pb2

        variable_protos = []
        for idx, (name, p) in enumerate(valid_weights):
            shape = list(p.shape)
            variable_protos.append(
                raiden_service_pb2.VariableMetadataProto(
                    name=name,
                    shape=shape,
                    mesh_shape=[1] * len(shape),
                    layout=list(range(len(shape) - 1, -1, -1)),
                    item_size=p.element_size(),
                    layer_idx=idx,
                    sharding_spec=[""] * len(shape),
                )
            )

        # TODO(tpu): Consider passing controller_address directly during orchestration (e.g., via
        # CheckpointEngineManager / actor_wg.update_weights or init_process_group) rather than querying
        # the RayWeightRegistry actor, eliminating cross-process registry lookups altogether.
        if self._controller_addr is None:
            for _ in range(30):
                try:
                    self._controller_addr = await self.registry.get_controller_address.remote()
                    if self._controller_addr:
                        break
                except Exception as e:
                    logger.debug(f"Trainer Rank {self.rank}: controller address lookup failed: {e}")
                await asyncio.sleep(0.1)
        if not self._controller_addr:
            raise RuntimeError(
                f"Trainer Rank {self.rank}: No RaidenController address found in RayWeightRegistry after timeout"
            )

        try:
            ctrl_client = raiden_controller.RaidenControllerClientFacade(self._controller_addr)
            unit_id = raiden_controller.RaidenId("trainer", str(self.rank), "weights")
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._trainer_raiden_ws.local_port}"],
                f"{bind_ip}:{self._trainer_raiden_ws.listener_port}",
                mesh_shape=[1, 1],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as reg_err:
            logger.error(
                f"Trainer Rank {self.rank} failed to register with RaidenController "
                f"({self._controller_addr}): {reg_err}"
            )
            raise
        logger.info(
            f"Trainer Rank {self.rank} bound {len(valid_weights)} dynamic tensors and registered "
            f"directly with RaidenController ({self._controller_addr}): "
            f"data_port={self._trainer_raiden_ws.local_port}, "
            f"listener_port={self._trainer_raiden_ws.listener_port}"
        )

        # Stage weights to host buffer via D2H DMA
        t_d2h_start = time.perf_counter()
        self._trainer_raiden_ws.d2h()
        t_d2h = time.perf_counter() - t_d2h_start
        logger.info(
            f"[RAIDEN TELEMETRY | Trainer Worker] Trainer Rank {self.rank}: D2H DMA transfer completed in {t_d2h:.4f}s"
        )

        # Compute deterministic stats & norms across all sent tensors on rank 0 only: every rank holds the
        # full model.
        if not (self.verify_parity and self.is_master):
            return
        try:
            trainer_stats = compute_tensor_stats(valid_weights)
            trainer_stats["rank"] = self.rank
            # Post master stats to RayWeightRegistry
            self.registry.set_stats.remote(step_key, trainer_stats)
            logger.info(
                f"[RAIDEN PARITY] Successfully stored trainer rank {self.rank} stats for step {step_key}: "
                f"L1={trainer_stats['l1_norm']:.4f}, numel={trainer_stats['total_numel']}"
            )
        except Exception as e:
            logger.warning(f"Failed to record trainer rank {self.rank} stats in RayWeightRegistry: {e}")

    def release_sync_buffers(self) -> dict:
        """Free this rank's WeightSynchronizer, and with it the bound send tensors, after the transfer.

        Must only be called after the controller transfer that reads them has completed. The synchronizer pins
        the device buffers of every bound tensor (a bf16 copy of the full model on every trainer chip), and
        tpu_sync has no unbind, so destroying it is the only way to give that HBM back to training before the
        next sync. The next send_weights creates and registers a new synchronizer.

        TODO(tpu): Reuse the synchronizer across syncs instead of creating it again every sync: allocating and
        pinning its host buffers and registering with the controller cost sync latency that grows with model
        size. Freeing the HBM between syncs while keeping the rest needs a tpu_sync API to unbind or drop the
        device buffers.
        """
        if self._trainer_raiden_ws is None:
            return {}
        # The torch WeightSynchronizer has no close(); the C++ object (which holds references to the
        # device tensors and the host staging memory) is destroyed when the last Python reference goes.
        ws, self._trainer_raiden_ws = self._trainer_raiden_ws, None
        del ws
        gc.collect()
        # No empty_cache(): on TPU it clears the eager-op compilation cache (forcing recompiles) and the
        # TPU runtime reuses freed HBM without it.
        tpu_synchronize()
        return {}

    def receive_weights(self, global_steps: Optional[int] = None, **kwargs: Any):
        raise NotImplementedError(
            "Rollout workers receive Raiden weights through vLLMRaidenWorkerExtension via collective_rpc."
        )


def _raiden_engine_kwargs(config: Any) -> dict[str, Any]:
    """Returns ``engine_kwargs.raiden`` of a ``CheckpointEngineConfig`` (empty when unset)."""
    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    return dict(engine_kwargs.get(RAIDEN_BACKEND) or {})


def _rollout_world_size(replica: Any) -> int:
    """Number of vLLM workers (one per chip) of a rollout replica."""
    if getattr(replica, "world_size", None):
        return int(replica.world_size)
    if getattr(replica, "workers", None):
        return len(replica.workers)
    return 1


def _get_or_start_controller(manager: Any) -> Any:
    """Returns the Raiden controller of ``manager``, starting it on the first sync."""
    controller = getattr(manager, "_raiden_controller", None)
    if controller is None:
        try:
            controller, server, address = setup_raiden_controller()
        except Exception as e:
            raise RuntimeError(f"Failed to start the RaidenControllerServer on the driver: {e}") from e
        # The server must stay referenced while the trainer and rollout workers use it.
        manager._raiden_controller, manager._raiden_server, manager._raiden_address = controller, server, address
    return controller


async def _wait_for_registration(controller: Any, src_units: list, dst_units: list, timeout_s: float) -> None:
    """Waits until every trainer (``src_units``) and rollout (``dst_units``) unit registered with the controller."""
    deadline = time.perf_counter() + timeout_s
    while True:
        # TODO(tpu): Use a public tpu_sync API to list the registered units once there is one.
        with controller._lock:
            registered = set(controller._registered_shards.keys())
        missing_src = [u for u in src_units if u not in registered]
        missing_dst = [u for u in dst_units if u not in registered]
        if not missing_src and not missing_dst:
            logger.info(
                f"[RAIDEN CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units "
                "verified and registered."
            )
            return
        if time.perf_counter() > deadline:
            raise RuntimeError(
                f"Timeout ({timeout_s}s) waiting for workers to register with RaidenController! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)


async def update_raiden_weights(manager: Any, global_steps: Optional[int] = None) -> dict[str, float]:
    """Sync the trainer's weights to the vLLM rollout over Raiden, coordinated by a controller on the driver.

    Replaces ``CheckpointEngineManager.update_weights`` for ``backend=raiden``.

    Args:
        manager: The ``CheckpointEngineManager`` (trainer worker group, rollout replicas and config).
        global_steps: Trainer step of the weights; the rollout tags new generations with it.

    Returns:
        Per-phase timings in seconds, logged as step metrics by trainers that record sync metrics.
    """
    raiden_kwargs = _raiden_engine_kwargs(manager.config)
    verify_parity = _as_bool(raiden_kwargs.get("verify_parity", False))
    parallelism = int(raiden_kwargs.get("parallelism", RAIDEN_DEFAULT_PARALLELISM))
    if len(manager.replicas) != 1:
        # Each rollout replica is a separate vLLM engine whose workers register as "sampler/<rank>"
        # with rank restarting at 0, so the units of several replicas would collide on the controller.
        raise NotImplementedError(
            f"The raiden checkpoint engine supports a single rollout replica, got {len(manager.replicas)}. "
            "Set actor_rollout_ref.rollout.tensor_model_parallel_size to the number of rollout chips."
        )
    controller = _get_or_start_controller(manager)

    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()

    # 1. Every trainer rank binds its tensors to a new synchronizer, registers them with the controller, records
    #    the global shapes and stages the tensors in host memory. ray.get blocks until all ranks finish, so run
    #    it in a thread to keep the event loop free.
    t_init_trainer_start = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    if actor_refs is not None:
        await asyncio.to_thread(ray.get, actor_refs)
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    # 2. Every rollout worker allocates its receive buffers and registers them with the controller
    #    (a no-op after the first sync).
    t_init_sampler_start = time.perf_counter()
    await asyncio.gather(
        *[
            replica.server_handle.collective_rpc.remote(
                method="init_raiden_sync_on_worker", kwargs={"parallelism": parallelism}
            )
            for replica in manager.replicas
        ]
    )
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Registration barrier on the controller.
    # TODO(tpu): Move worker registration and barrier verification to a one-time setup step during
    # initialization (e.g. in prepare/build_process_group), since shard registration is persistent on the
    # controller and does not need to be repeated on every weight sync iteration.
    from tpu_sync.api.common import RaidenId
    from tpu_sync.rpc.raiden_controller import RaidenMemoryType

    t_barrier_start = time.perf_counter()
    src_units = [
        RaidenId(job_name="trainer", job_replica_id=str(rank), data_name="weights")
        for rank in range(manager.actor_wg.world_size)
    ]
    dst_units = [
        RaidenId(job_name="sampler", job_replica_id=str(rank), data_name="weights")
        for rank in range(sum(_rollout_world_size(replica) for replica in manager.replicas))
    ]
    await _wait_for_registration(controller, src_units, dst_units, timeout_s=RAIDEN_REGISTRATION_TIMEOUT_S)
    t_barrier = time.perf_counter() - t_barrier_start

    # 4. Controller-driven P2P transfer: trainer device -> trainer host -> rollout host, resharded.
    t_transfer_start = time.perf_counter()
    transfer_future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=RaidenMemoryType.DRAM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=0,
        parallelism=parallelism,
        req_id=f"verl_step_{global_steps or 0}",
    )
    await transfer_future.wait()
    t_transfer = time.perf_counter() - t_transfer_start

    # The trainer buffers are no longer read once the transfer is done; free them while the samplers install,
    # so the HBM is back before the next training step.
    release_refs = manager.actor_wg.execute_checkpoint_engine(["release_sync_buffers"] * manager.actor_wg.world_size)

    # 5. Rollout workers copy the received weights to TPU HBM and fuse them into vLLM's parameters.
    t_install_start = time.perf_counter()
    install_results = await asyncio.gather(
        *[replica.server_handle.collective_rpc.remote(method="install_raiden_weights") for replica in manager.replicas]
    )
    t_install = time.perf_counter() - t_install_start
    # collective_rpc returns one result per TP worker of each replica (when the server passes worker results
    # through); each is the timing dict returned by vLLMRaidenWorkerExtension.install_raiden_weights. The
    # slowest worker gates the sync, so report the max.
    worker_install_stats = [
        r for per_replica in install_results for r in (per_replica or []) if isinstance(r, dict) and "h2d" in r
    ]
    t_h2d_pure = max((r["h2d"] for r in worker_install_stats), default=None)
    # Tag new generations with this weight version (the trajectory staleness metrics read it), as the other
    # checkpoint backends do once the new weights are loaded. abort_replicas() already reset the prefix
    # cache, and no request has run since.
    if global_steps is not None:
        await asyncio.gather(
            *[replica.server_handle.set_global_steps.remote(global_steps) for replica in manager.replicas]
        )
    if release_refs is not None:
        await asyncio.to_thread(ray.get, release_refs)

    t_total = time.perf_counter() - t_total_start

    # 6. Parity verification (optional, default off).
    if verify_parity:
        try:
            await _verify_parity_async(manager, global_steps)
        except Exception as e:
            logger.warning(f"Failed to execute parity verification: {e}")

    logger.info(
        f"[RAIDEN TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s\n"
        f"  * Trainer Raiden Init    : {t_init_trainer:.4f}s\n"
        f"  * Sampler Raiden Init    : {t_init_sampler:.4f}s\n"
        f"  * Raiden Barrier Check   : {t_barrier:.4f}s\n"
        f"  * RaidenController P2P   : {t_transfer:.4f}s\n"
        f"  * Sampler H2D DMA        : {t_install:.4f}s\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    # 7. Resume generation immediately
    await manager.resume_generation_replicas()

    # Surface the phase timers as step metrics. Trainers that record sync metrics (e.g. the separate-async
    # trainer's _pending_sync_metrics) log them next to timing_s/update_weights with every configured
    # backend (console / tensorboard / wandb). timing_s/update_weights ~= quiesce + total_sync.
    metrics = {
        "timing_s/tpu-sync/quiesce": t_abort,
        "timing_s/tpu-sync/trainer_init": t_init_trainer,
        "timing_s/tpu-sync/sampler_init": t_init_sampler,
        "timing_s/tpu-sync/barrier": t_barrier,
        "timing_s/tpu-sync/p2p_transfer": t_transfer,
        "timing_s/tpu-sync/sampler_h2d": t_install,
        "timing_s/tpu-sync/total_sync": t_total,
    }
    if t_h2d_pure is not None:
        # Pure _raiden_ws.h2d() time on the slowest sampler worker (sampler_h2d above also includes the
        # fuse/transpose into vLLM params, the TPU sync barrier and the RPC round trip).
        metrics["timing_s/tpu-sync/sampler_h2d_pure"] = t_h2d_pure
    return metrics


async def _verify_parity_async(manager: Any, global_steps: Optional[int] = None) -> None:
    """Compare distributed norms between Trainer Rank 0 and Sampler TP workers.

    TODO(tpu): Refactor parity verification to use rank-local scalar partitioning.
    Instead of collecting per-tensor dictionaries across all Sampler workers and checking
    replicated vs sharded heuristics over 200+ tensors, each worker can reduce its model
    locally into scalar metrics (numel, l1, l2_sq) before RPC return:
      - Rank 0 accumulates both sharded tensors and replicated 1D tensors (e.g. RMSNorms).
      - Ranks 1..N-1 accumulate only sharded tensors.
    The orchestrator can then perform verification in O(ranks) pure scalar arithmetic
    rather than O(tensors * ranks) loop aggregation.
    """
    step_key = global_steps if global_steps is not None else 0
    if step_key <= 0:
        return

    try:
        registry = get_ray_weight_registry()
        trainer_entry = None
        for _ in range(25):
            trainer_entry = await registry.get_stats.remote(step_key)
            if trainer_entry is not None:
                break
            await asyncio.sleep(0.5)

        sampler_entries = await asyncio.gather(
            *[
                replica.server_handle.collective_rpc.remote(method="get_model_weights_stats")
                for replica in manager.replicas
            ]
        )

        if not trainer_entry or not sampler_entries:
            logger.warning(f"[RAIDEN PARITY] Incomplete stats data for step {step_key}")
            return

        # Unpacks and flattens the results collected from all Sampler rollout replicas into a single flat list
        # of worker dictionary objects.
        sampler_workers = [
            w
            for res in sampler_entries
            for w in (res if isinstance(res, list | tuple) else [res])
            if isinstance(w, dict)
        ]
        if not sampler_workers:
            logger.warning(
                f"[RAIDEN PARITY] No sampler worker stats for step {step_key}: the rollout server's "
                "collective_rpc must return the worker results for the parity check."
            )
            return

        trainer_master = trainer_entry.get("master", trainer_entry)
        trainer_per_tensor = trainer_master.get("per_tensor", {})
        # Gets master list of all model tensor names (e.g., "model.layers.0.self_attn.qkv_proj.weight").
        all_param_names = list(sampler_workers[0].get("per_tensor", {}).keys()) if sampler_workers else []

        total_trainer_numel = trainer_master.get("total_numel", 0)
        total_trainer_l1 = trainer_master.get("l1_norm", 0.0)
        global_trainer_l2 = trainer_master.get("l2_norm", 0.0)

        total_sampler_l1, total_sampler_l2_sq, total_sampler_numel = 0.0, 0.0, 0
        mismatches = []

        for name in all_param_names:
            s_numels = [w.get("per_tensor", {}).get(name, {}).get("numel", 0) for w in sampler_workers]
            s_l1s = [w.get("per_tensor", {}).get(name, {}).get("l1", 0.0) for w in sampler_workers]
            s_l2_sqs = [
                w.get("per_tensor", {})
                .get(name, {})
                .get("l2_sq", w.get("per_tensor", {}).get(name, {}).get("l2", 0.0) ** 2)
                for w in sampler_workers
            ]

            is_replicated = (
                len(set(s_numels)) == 1
                and (name.endswith("layernorm.weight") or "norm" in name)
                and len(s_numels[0:1]) > 0
                and s_numels[0] < 10000
            )
            if is_replicated:
                s_agg_numel = s_numels[0]
                s_agg_l1 = s_l1s[0]
                s_agg_l2 = s_l2_sqs[0] ** 0.5
            else:
                s_agg_numel = sum(s_numels)
                s_agg_l1 = sum(s_l1s)
                s_agg_l2 = sum(s_l2_sqs) ** 0.5

            t_data = trainer_per_tensor.get(name, {})
            t_agg_numel = t_data.get("numel", 0)
            t_agg_l1 = t_data.get("l1", 0.0)

            total_sampler_numel += s_agg_numel
            total_sampler_l1 += s_agg_l1
            total_sampler_l2_sq += s_agg_l2**2

            delta_numel = abs(s_agg_numel - t_agg_numel)
            delta_l1 = abs(s_agg_l1 - t_agg_l1)
            rel_tol = 1e-3 * max(abs(t_agg_l1), 1.0)

            if delta_numel != 0 or delta_l1 > rel_tol:
                mismatches.append(f"  * {name}: Trainer(L1={t_agg_l1:.4f}) vs Sampler(L1={s_agg_l1:.4f})")

        global_sampler_l2 = total_sampler_l2_sq**0.5
        total_l1_delta = abs(total_sampler_l1 - total_trainer_l1)
        total_l2_delta = abs(global_sampler_l2 - global_trainer_l2)
        total_numel_delta = abs(total_sampler_numel - total_trainer_numel)
        total_rel_tol = 1e-3 * max(abs(total_trainer_l1), 1.0)

        if not mismatches and total_numel_delta == 0 and total_l1_delta <= total_rel_tol:
            logger.info(
                f"[RAIDEN PARITY VERIFIED | Step {step_key}] 100% DISTRIBUTED NORM PARITY CONFIRMED!\n"
                f"  * Global L1 Norm: {total_sampler_l1:.6f} "
                f"(Trainer={total_trainer_l1:.6f}, delta={total_l1_delta:.6f})\n"
                f"  * Global L2 Norm: {global_sampler_l2:.6f} "
                f"(Trainer={global_trainer_l2:.6f}, delta={total_l2_delta:.6f})\n"
                f"  * Total Parameters: {total_sampler_numel} across {len(all_param_names)} tensors"
            )
        else:
            mismatches_summary = "\n".join(mismatches[:10])
            logger.error(
                f"[RAIDEN PARITY MISMATCH | Step {step_key}] Norms do NOT match!\n"
                f"  * Trainer: numel={total_trainer_numel}, L1={total_trainer_l1:.6f}, L2={global_trainer_l2:.6f}\n"
                f"  * Sampler: numel={total_sampler_numel}, L1={total_sampler_l1:.6f}, L2={global_sampler_l2:.6f}\n"
                f"  * Mismatched Tensors ({len(mismatches)} / {len(all_param_names)}):\n"
                f"{mismatches_summary}"
            )
    except Exception as e:
        logger.warning(f"Error during parity verification for step {step_key}: {e}")
