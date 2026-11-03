# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""``tpu-sync`` checkpoint engine: peer-to-peer weight sync between TPU slices with tpu-sync.

The ``ray-plasma-store`` checkpoint engine stages the whole model through Ray's object store. ``tpu-sync`` instead
moves the weights over the network from the trainer chips to the rollout chips with the tpu-sync transfer
library (``tpu-sync-torch``, https://github.com/google/tpu-sync). Every trainer rank registers its local FSDP
shard of each weight (tensors that are not split evenly on dim 0 are gathered and registered full), and each
rollout chip receives only its vLLM tensor-parallel shard; the controller reshards in between::

     [Trainer slice: FSDP ranks]                       [Rollout slice: vLLM TP workers]
    +--------------------------------------+           +--------------------------------------+
    | TPUSyncCheckpointEngine.send_weights |           | vLLMTPUSyncWorkerExtension           |
    |  - bind local FSDP shards to a       |           |  - init_tpu_sync_on_worker:          |
    |    WeightSynchronizer                |           |    allocate TP-sharded receive       |
    |  - register "trainer/<rank>"         |           |    buffers, register "sampler/<rank>"|
    +------------------+-------------------+           |  - install_tpu_sync_weights: H2D and |
                       |                               |    fuse into vLLM's parameters       |
                       +-- host-to-host, resharded --->+--------------------------------------+
                                     ^
                     [Driver: update_tpu_sync_weights + tpu-sync controller]

``update_tpu_sync_weights`` runs on the driver in place of ``CheckpointEngineManager.update_weights``
(see ``apply_checkpoint_engine_manager_hooks`` in ``weight_sync_backends.py``). Trainer and rollout
processes find the controller, and the tensor shapes, through the job's ``TpuSyncRendezvous`` actor
(``tpu_sync_rendezvous.py``).

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
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import tpu_synchronize
from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights
from verl_hardware_plugin.accelerators.tpu.options import TPU_SYNC_DEFAULT_PARALLELISM, get_tpu_sync_options
from verl_hardware_plugin.accelerators.tpu.profilers import timeline
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_api
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import (
    post_trainer_stats,
    report_exact_parity,
    verify_rollout_parity,
)
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_rendezvous import (
    create_tpu_sync_rendezvous,
    get_tpu_sync_rendezvous,
)
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import (
    TPU_SYNC_BACKEND,
    apply_checkpoint_engine_manager_hooks,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How long the driver waits for every trainer and rollout worker to register with the controller.
TPU_SYNC_REGISTRATION_TIMEOUT_S = 60.0
# tpu-sync reads and writes device buffers in place, so the tensors it is given must live on this device.
TPU_SYNC_DEVICE = "tpu"


def create_torch_weight_synchronizer(
    device_tensors: list[list[torch.Tensor]],
    local_port: int = 0,
    parallelism: int = TPU_SYNC_DEFAULT_PARALLELISM,
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
    """Validate and sanitize tensors for zero-copy DMA registration with tpu-sync.

    Performs physical memory validation:
    1. Drops None, non-tensor objects, zero-element tensors, and unallocated meta tensors.
    2. Unwraps DTensor/nn.Parameter to local tensor buffers.
    3. Ensures tensors physically reside on ``device`` (TPU HBM by default).
    4. Guarantees memory contiguity (contiguous buffers) required for direct DMA.
    """
    if device is None:
        device = torch.device(TPU_SYNC_DEVICE)

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
        # tpu-sync transfers whole device buffers, so a view into a larger buffer (e.g. a slice of a fused or
        # all-gathered tensor) needs its own storage.
        if t._base is not None or t.storage_offset() != 0:
            t = t.clone()

        sanitized.append((name, t))

    return sanitized


def setup_tpu_sync_controller() -> tuple[Any, Any, str, Any, str]:
    """Start a tpu-sync controller server in this process and publish its address in the job's rendezvous actor.

    Returns:
        ``(controller, server, "host:port", rendezvous, rendezvous_name)``. The caller must keep ``server`` and
        ``rendezvous`` referenced while syncing (the rendezvous actor is not detached: Ray destroys it once no
        handle to it is left), and passes the address to the rollout workers.
    """
    # TODO(security): the controller (like the WeightSynchronizer endpoints) accepts unauthenticated
    # connections. Add authentication once tpu_sync supports it; until then it must only be reachable
    # from the cluster's trusted network.
    controller, server = tpu_sync_api.new_controller_server()
    port = server.start()
    ip = ray.util.get_node_ip_address().strip("[]")
    address = f"{ip}:{port}"
    logger.info(f"tpu-sync controller server started on the driver: {address}")

    try:
        # This job's own rendezvous actor: other jobs on the cluster have theirs, and it ends with this job.
        rendezvous, rendezvous_name = create_tpu_sync_rendezvous()
        ray.get(rendezvous.set_controller_address.remote(address))
        logger.info(f"Stored the tpu-sync controller address ({address}) in {rendezvous_name}")
    except Exception as reg_err:
        raise RuntimeError(
            f"Failed to store the tpu-sync controller address ({address}) in the tpu-sync rendezvous: {reg_err}"
        ) from reg_err
    return controller, server, address, rendezvous, rendezvous_name


@CheckpointEngineRegistry.register(TPU_SYNC_BACKEND)
class TPUSyncCheckpointEngine(CheckpointEngine):
    """Trainer-side ``tpu-sync`` engine: exposes this rank's weights to the controller-driven tpu-sync P2P transfer.

    Engine kwargs (``actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync``):

    * ``parallelism`` (default 8): parallel transfer streams per worker.
    * ``verify_parity`` (default ``off``): ``norm`` posts weight norms after every sync so the driver can check
      that the rollout received the trainer's weights; ``exact`` has the rollout workers compare the step-0 sync
      bit for bit with the checkpoint vLLM loaded (trainer ranks do nothing for it); ``all`` runs both. A failed
      check raises ``TPUSyncParityError`` on the driver, before generation resumes.
    * ``tie_word_embeddings`` (default unset): whether ``lm_head`` is tied to the input embedding, so it
      need not be sent. Unset sends ``lm_head``, which is correct for tied and untied models alike.
    * ``release_buffers_after_sync`` (default True): free the bound send tensors after every transfer
      (``release_sync_buffers``) instead of keeping them resident until the next sync. The deprecated
      environment variable ``VERL_TPU_SYNC_RELEASE_BUFFERS`` still overrides it, with a warning.

    The options are parsed by ``options.get_tpu_sync_options`` (``TPUSyncOptions``); unknown keys are an error.
    """

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs: Any) -> None:
        # CheckpointEngine defines no __init__ (object.__init__ takes no arguments).
        self.bucket_size = bucket_size
        self.is_master = is_master
        # verl passes checkpoint_engine.engine_kwargs.<backend> here; parsed (and validated) in one place.
        options = get_tpu_sync_options(kwargs)
        self.parallelism = options.parallelism
        # Trainer ranks only post norm stats; the exact check runs on the rollout workers.
        self.verify_parity = options.verify_norms
        self.tie_word_embeddings = options.tie_word_embeddings
        # The synchronizer pins the device buffers of every bound tensor (a bf16 copy of this rank's weights), so
        # dropping the Python references alone frees nothing. On by default so the trainer keeps that HBM headroom
        # between syncs; opt out with engine_kwargs.tpu-sync.release_buffers_after_sync=False.
        self.release_buffers_after_sync = options.release_buffers_after_sync
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = int(os.environ.get("RANK", "0"))

        self._registry: Any = None
        self._controller_addr: Optional[str] = None
        self._trainer_tpu_sync_ws: Any = None
        # (name, shape, dtype, shard) of the tensors the live synchronizer was created and registered with; a later
        # send_weights with the same signature rebinds instead of re-creating it.
        self._registered_signature: Optional[list[tuple]] = None
        # name -> (num_shards, shard_index) for the tensors of the last send_weights sent as a local dim-0 shard.
        self._shard_info: dict[str, tuple[int, int]] = {}
        # Keeps the bound device buffers alive until release_sync_buffers (the controller-driven push reads
        # them after send_weights returns).
        self._bound_tensors: Optional[list[torch.Tensor]] = None
        self._warned_no_unbind = False

    @property
    def registry(self) -> Any:
        """This job's ``TpuSyncRendezvous`` actor, looked up on first use (the driver creates it before any sync)."""
        if self._registry is None:
            self._registry = get_tpu_sync_rendezvous()
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
        ws, self._trainer_tpu_sync_ws = self._trainer_tpu_sync_ws, None
        self._registered_signature = None
        self._bound_tensors = None
        close = getattr(ws, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                logger.debug(f"Trainer Rank {self.rank}: closing the WeightSynchronizer failed: {e}")

    async def send_weights(self, weights: Any, global_steps: Optional[int] = None) -> None:
        """Bind this rank's weights to its WeightSynchronizer and register them with the controller.

        Args:
            weights: What the training engine's ``get_per_tensor_param`` returned. When it has ``local_shards()``
                (``TrainerWeights``), every FSDP shard that method recognizes is registered as this rank's block of
                the full tensor and tpu-sync reshards the blocks for the rollout, so the trainer never gathers them;
                the remaining tensors arrive full. Otherwise an iterable (or dict) of ``(hf_name, tensor)`` pairs
                holding the full weights on every rank.
            global_steps: Trainer step of these weights.

        The transfer itself, including the device-to-host copy, is driven by the controller after this returns,
        so the bound tensors are kept until ``release_sync_buffers``. The synchronizer is created and registered
        on the first sync and reused (rebound through ``bind_weights``) afterwards as long as the tensor names,
        shapes, dtypes and shard layout do not change.
        """
        step_key = global_steps if global_steps is not None else 0
        timeline_lane = timeline.trainer_lane()
        t_start = time.time()
        logger.info(f"TPUSyncCheckpointEngine: [Step {step_key}] Start send_weights...")
        tpu_synchronize(strict=True)

        # Materialize weights in a single pass and filter tied embeddings
        if isinstance(weights, TrainerWeights):
            exported = list(weights.local_shards())
            shard_info = {name: shard for name, _, shard in exported if shard is not None}
            named_weights = [(name, tensor) for name, tensor, _ in exported]
            del exported
        else:
            shard_info = {}
            named_weights = weights.items() if hasattr(weights, "items") else weights
        named_weights = filter_tied_embeddings(named_weights, tie_word_embeddings=bool(self.tie_word_embeddings))

        # Trainer sends pure canonical un-fused model weights directly
        unfused_weights = {k: _unwrap_tensor(v) for k, v in named_weights}

        sorted_weights = sorted(unfused_weights.items(), key=lambda x: x[0])
        valid_weights = validate_and_sanitize_tensors(sorted_weights, device=torch.device(TPU_SYNC_DEVICE))
        self._shard_info = {name: shard_info[name] for name, _ in valid_weights if name in shard_info}

        tpu_synchronize(strict=True)
        t_gathered = time.time()

        # Reuse the synchronizer when the tensor signature is unchanged: rebinding keeps the pinned host staging
        # buffers, listener threads and controller registration, and only swaps the device buffers the D2H reads
        # from (release_sync_buffers unbinds them between syncs). Re-creating it every sync re-allocates, pins
        # and first-touches a model-sized host buffer per rank and re-registers with the controller.
        signature = [(name, tuple(t.shape), t.dtype, self._shard_info.get(name)) for name, t in valid_weights]
        if self._trainer_tpu_sync_ws is not None and signature == self._registered_signature:
            logger.info(f"Trainer Rank {self.rank}: rebinding {len(valid_weights)} tensors to the WeightSynchronizer")
            self._trainer_tpu_sync_ws.bind_weights([[t] for _, t in valid_weights])
        else:
            await self._create_and_register(valid_weights)
            self._registered_signature = signature
        self._bound_tensors = [t for _, t in valid_weights]
        t_bound = time.time()

        if timeline_lane is not None:
            timeline.emit(
                "update_weights",
                t_start,
                t_bound - t_start,
                group=timeline.TRAINER_WORKERS,
                lane=timeline_lane,
                step=step_key,
                tensors=len(valid_weights),
            )
            timeline.emit(
                "gather",
                t_start,
                t_gathered - t_start,
                group=timeline.TRAINER_WORKERS,
                lane=timeline_lane,
                step=step_key,
            )
            timeline.emit(
                "bind",
                t_gathered,
                t_bound - t_gathered,
                group=timeline.TRAINER_WORKERS,
                lane=timeline_lane,
                step=step_key,
            )

        # No explicit d2h() here: the controller-driven push (PushWeightsResharded, started by start_transfer)
        # performs a pipelined D2H overlapped with the host-to-host transfer, using the per-transfer skip_tiling
        # plan. An explicit d2h() before the first transfer has no plan yet and stages the wrong (de-tiled)
        # layout, and on every sync it copies the whole model to host memory a second time; it was only ever
        # masked by the push's own D2H.

        if not self.verify_parity:
            return
        post_trainer_stats(self.registry, step_key, self.rank, self.is_master, valid_weights, self._shard_info)

    async def _create_and_register(self, valid_weights: list[tuple[str, torch.Tensor]]) -> None:
        """(Re)creates the WeightSynchronizer over ``valid_weights`` and registers it with the controller."""
        self._close_synchronizer()

        logger.info(f"Trainer Rank {self.rank}: binding {len(valid_weights)} tensors to WeightSynchronizer")
        bind_ip = ray.util.get_node_ip_address().strip("[]")
        self._trainer_tpu_sync_ws = create_torch_weight_synchronizer(
            [[t] for _, t in valid_weights],
            local_port=0,
            listener_port=0,
            parallelism=self.parallelism,
            bind_ip=bind_ip,
        )

        def full_shape(name: str, p: torch.Tensor) -> list[int]:
            shape = list(p.shape)
            if name in self._shard_info:
                shape[0] *= self._shard_info[name][0]
            return shape

        # Record global shapes in the rendezvous so the sampler can size its receive buffers.
        # TODO(tpu): Move global_shapes registration to a one-time setup step during initialization
        # (e.g., in prepare or build_process_group/model_init) instead of per weight-sync step,
        # since tensor shapes are static across training iterations and only need to be communicated once.
        try:
            global_shapes = {name: full_shape(name, p) for name, p in valid_weights}
            await self.registry.set_global_shapes.remote(global_shapes)
        except Exception as e:
            logger.warning(f"Could not record global shapes in the tpu-sync rendezvous: {e}")

        # Build variable metadata protos for each dynamic tensor. Every variable carries its full shape:
        # - a local FSDP shard is block `shard_index` of `num_shards` dim-0 blocks: mesh_shape=[num_shards, 1, ...]
        #   and global_shard_indices=[shard_index], and tpu-sync reshards the blocks of all ranks for the rollout;
        # - a full tensor (the same on every rank) is unsharded: mesh_shape=[1] * rank.
        variable_protos = []
        max_shards = 1
        for idx, (name, p) in enumerate(valid_weights):
            shape = full_shape(name, p)
            if name in self._shard_info:
                num_shards, shard_index = self._shard_info[name]
                max_shards = max(max_shards, num_shards)
                sharding: dict[str, Any] = {
                    "mesh_shape": [num_shards] + [1] * (len(shape) - 1),
                    "sharding_spec": ["fsdp"] + [""] * (len(shape) - 1),
                    "global_shard_indices": [shard_index],
                }
            else:
                sharding = {"mesh_shape": [1] * len(shape), "sharding_spec": [""] * len(shape)}
            variable_protos.append(
                tpu_sync_api.variable_metadata(
                    name=name,
                    shape=shape,
                    layout=list(range(len(shape) - 1, -1, -1)),
                    item_size=p.element_size(),
                    layer_idx=idx,
                    **sharding,
                )
            )
        full_names = [name for name, _ in valid_weights if name not in self._shard_info]
        logger.info(
            f"Trainer Rank {self.rank}: registering {len(self._shard_info)} local shards and "
            f"{len(full_names)} full tensors (e.g. {full_names[:4]})"
        )

        # TODO(tpu): Consider passing controller_address directly during orchestration (e.g., via
        # CheckpointEngineManager / actor_wg.update_weights or init_process_group) rather than querying
        # the rendezvous actor, eliminating cross-process registry lookups altogether.
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
                f"Trainer Rank {self.rank}: No tpu-sync controller address found in the rendezvous after timeout"
            )

        try:
            ctrl_client = tpu_sync_api.controller_client(self._controller_addr)
            unit_id = tpu_sync_api.unit_id("trainer", str(self.rank))
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._trainer_tpu_sync_ws.local_port}"],
                f"{bind_ip}:{self._trainer_tpu_sync_ws.listener_port}",
                mesh_shape=[max_shards, 1],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as reg_err:
            logger.error(
                f"Trainer Rank {self.rank} failed to register with the tpu-sync controller "
                f"({self._controller_addr}): {reg_err}"
            )
            raise
        logger.info(
            f"Trainer Rank {self.rank} bound {len(valid_weights)} dynamic tensors and registered "
            f"directly with the tpu-sync controller ({self._controller_addr}): "
            f"data_port={self._trainer_tpu_sync_ws.local_port}, "
            f"listener_port={self._trainer_tpu_sync_ws.listener_port}"
        )

    def release_sync_buffers(self) -> dict:
        """Free this rank's bound send tensors after the transfer, keeping the WeightSynchronizer when possible.

        Must only be called after the controller transfer that reads them has completed. A no-op when
        ``release_buffers_after_sync`` is disabled, in which case the bound tensors stay resident between syncs.

        With a ``tpu_sync`` build that has ``WeightSynchronizer.unbind_weights()`` the synchronizer's holds on the
        bound device buffers are dropped, so dropping our own references returns the HBM. The synchronizer itself,
        its pinned host staging buffers and its controller registration survive, and the next ``send_weights``
        rebinds through ``bind_weights()``. Compared to destroying and re-creating the synchronizer each sync, this
        removes the per-step rebuild (host buffer allocation, pinning and first touch, plus controller
        re-registration: ~2 s "Trainer init" + ~2 s release on Qwen3-8B, v6e-8) from the sync critical path.

        Older ``tpu_sync`` builds without ``unbind_weights`` fall back to destroying the synchronizer; the next
        ``send_weights`` then re-creates and re-registers it.
        """
        if not self.release_buffers_after_sync or self._trainer_tpu_sync_ws is None:
            return {}
        with timeline.span("release_tpu_sync", group=timeline.TRAINER_WORKERS, lane=timeline.trainer_lane()):
            if hasattr(self._trainer_tpu_sync_ws, "unbind_weights"):
                self._trainer_tpu_sync_ws.unbind_weights()
                self._bound_tensors = None
            else:
                if not self._warned_no_unbind:
                    logger.warning(
                        f"Trainer Rank {self.rank}: tpu_sync WeightSynchronizer has no unbind_weights(); destroying "
                        "it to free the send buffers each sync (upgrade tpu-sync-torch to keep it across syncs)."
                    )
                    self._warned_no_unbind = True
                # The torch WeightSynchronizer has no close(); the C++ object (which holds references to the
                # device tensors and the host staging memory) is destroyed when the last Python reference goes.
                self._close_synchronizer()
                gc.collect()
            # No empty_cache(): the TPU runtime reuses freed HBM by itself (the platform's empty_cache() does
            # nothing, and torch.tpu._clear_cache() would evict the compiled executables, forcing recompiles).
            tpu_synchronize()
        return {}

    def receive_weights(self, global_steps: Optional[int] = None, **kwargs: Any):
        raise NotImplementedError(
            "Rollout workers receive tpu-sync weights through vLLMTPUSyncWorkerExtension via collective_rpc."
        )


def _tpu_sync_engine_kwargs(config: Any, backend: Optional[str] = None) -> dict[str, Any]:
    """Returns the engine kwargs of ``backend`` in a ``CheckpointEngineConfig`` (empty when unset).

    verl hands the trainer-side engine ``engine_kwargs[<configured backend name>]``, so the driver reads the
    same entry: ``engine_kwargs.tpu-sync``.
    """
    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    backend = backend or getattr(config, "backend", None) or TPU_SYNC_BACKEND
    return dict(engine_kwargs.get(backend) or {})


def _rollout_world_size(replica: Any) -> int:
    """Number of vLLM workers (one per chip) of a rollout replica."""
    if getattr(replica, "world_size", None):
        return int(replica.world_size)
    if getattr(replica, "workers", None):
        return len(replica.workers)
    return 1


def _replica_rank(replica: Any, index: int) -> int:
    """``replica.replica_rank`` (verl's stable id of a rollout replica), or its list position without one."""
    rank = getattr(replica, "replica_rank", None)
    return int(rank) if rank is not None else index


def _sampler_job_name(replica: Any, index: int) -> str:
    """tpu-sync job name of a rollout replica: ``sampler<replica_rank>``.

    A rollout worker registers as (job name, rank within its replica). With several replicas the ranks
    ``0..TP-1`` would all collide under one name, and the controller would wait for ranks ``TP..N*TP-1``
    that nothing registers. The name is keyed on ``replica_rank``, not on the replica's position in
    ``manager.replicas``, so a replica keeps its name (and the registration its workers memoize on the
    first sync) when replicas are added or removed.
    """
    return f"sampler{_replica_rank(replica, index)}"


def _get_or_start_controller(manager: Any) -> Any:
    """Returns the tpu-sync controller of ``manager``, starting it on the first sync."""
    controller = getattr(manager, "_tpu_sync_controller", None)
    if controller is None:
        try:
            controller, server, address, rendezvous, rendezvous_name = setup_tpu_sync_controller()
        except Exception as e:
            raise RuntimeError(f"Failed to start the tpu-sync controller server on the driver: {e}") from e
        # The server must stay referenced while the trainer and rollout workers use it.
        manager._tpu_sync_controller, manager._tpu_sync_server, manager._tpu_sync_address = controller, server, address
        # Holding the handle keeps the job's rendezvous actor alive for as long as the manager.
        manager._tpu_sync_rendezvous, manager._tpu_sync_rendezvous_name = rendezvous, rendezvous_name
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
                f"[TPU-SYNC CONTROLLER] All {len(src_units)} Trainer and {len(dst_units)} Sampler units "
                "verified and registered."
            )
            return
        if time.perf_counter() > deadline:
            raise RuntimeError(
                f"Timeout ({timeout_s}s) waiting for workers to register with the tpu-sync controller! "
                f"Missing Trainer: {missing_src}, Missing Sampler: {missing_dst}"
            )
        await asyncio.sleep(0.1)


async def update_tpu_sync_weights(manager: Any, global_steps: Optional[int] = None) -> dict[str, float]:
    """Sync the trainer's weights to the vLLM rollout over tpu-sync, coordinated by a controller on the driver.

    Replaces ``CheckpointEngineManager.update_weights`` for ``backend=tpu-sync``. The rollout may hold several
    replicas (``tensor_model_parallel_size`` smaller than the number of rollout chips): every replica registers
    under its own tpu-sync job name, all of them receive in one transfer, and the parity check runs per replica.

    Args:
        manager: The ``CheckpointEngineManager`` (trainer worker group, rollout replicas and config).
        global_steps: Trainer step of the weights; the rollout tags new generations with it.

    Returns:
        Per-phase timings in seconds, logged as step metrics by trainers that record sync metrics.

    Raises:
        TPUSyncParityError: An enabled parity check (``verify_parity``) failed; generation is not resumed.
    """
    backend = manager.backend or TPU_SYNC_BACKEND
    options = get_tpu_sync_options(_tpu_sync_engine_kwargs(manager.config, backend), backend)
    verify_parity = options.verify_norms
    # The exact check needs the rollout to still hold the checkpoint vLLM loaded, which is only true for the step-0
    # sync of a fresh run: later syncs (and the first one after resuming from a checkpoint) carry trained weights.
    exact_install_kwargs = {"exact_parity": True} if options.verify_exact and not global_steps else {}
    parallelism = options.parallelism
    controller = _get_or_start_controller(manager)

    t_sync_wall = time.time()

    # 0. Pause the rollout concurrently with the trainer-side export below. The trainer only exports and binds its
    #    own tensors and registers with the controller; nothing reads or writes rollout memory before step 2
    #    (rollout init), so the pause need not run in front of it.
    async def _quiesce_rollout() -> float:
        t0 = time.perf_counter()
        if global_steps and global_steps > 0:
            try:
                await manager.abort_replicas()
            except Exception as e:
                logger.warning(f"Failed to abort replicas at step {global_steps}: {e}")
        return time.perf_counter() - t0

    quiesce_task = asyncio.create_task(_quiesce_rollout())

    t_total_start = time.perf_counter()

    # 1. Every trainer rank binds its tensors to its synchronizer, registers them with the controller and records
    #    the global shapes. ray.get blocks until all ranks finish, so run it in a thread to keep the event loop free.
    t_init_trainer_start = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    try:
        if actor_refs is not None:
            await asyncio.to_thread(ray.get, actor_refs)
    except BaseException:
        quiesce_task.cancel()
        raise
    t_init_trainer = time.perf_counter() - t_init_trainer_start

    # The rollout must be paused before anything below touches it.
    t_abort = await quiesce_task

    # 2. Every rollout worker allocates its receive buffers and registers them with the controller under
    #    its replica's job name (a no-op after the first sync).
    t_init_sampler_start = time.perf_counter()
    # vLLM's workers are not Ray processes, so the driver hands them what the trainer published in step 1.
    global_shapes = await manager._tpu_sync_rendezvous.get_global_shapes.remote()
    await asyncio.gather(
        *[
            replica.server_handle.collective_rpc.remote(
                method="init_tpu_sync_on_worker",
                kwargs={
                    "controller_address": manager._tpu_sync_address,
                    "global_shapes": global_shapes,
                    "parallelism": parallelism,
                    "job_name": _sampler_job_name(replica, replica_idx),
                },
            )
            for replica_idx, replica in enumerate(manager.replicas)
        ]
    )
    t_init_sampler = time.perf_counter() - t_init_sampler_start

    # 3. Registration barrier on the controller.
    # TODO(tpu): Move worker registration and barrier verification to a one-time setup step during
    # initialization (e.g. in prepare/build_process_group), since shard registration is persistent on the
    # controller and does not need to be repeated on every weight sync iteration.
    t_barrier_start = time.perf_counter()
    src_units = [tpu_sync_api.unit_id("trainer", str(rank)) for rank in range(manager.actor_wg.world_size)]
    # Every replica registers its TP workers as ranks '0'..'TP-1' under its own job name, and all of them go
    # into ONE transfer: tpu-sync shards a transfer across the units that share a job name and broadcasts across
    # job names, so its planner puts rank k of every replica into one broadcast tree, seeds a few of them from
    # the trainer and relays between the rollout hosts. One transfer per replica would make the trainer push
    # every byte once per replica instead.
    dst_units = [
        tpu_sync_api.unit_id(_sampler_job_name(replica, replica_idx), str(rank))
        for replica_idx, replica in enumerate(manager.replicas)
        for rank in range(_rollout_world_size(replica))
    ]
    await _wait_for_registration(controller, src_units, dst_units, timeout_s=TPU_SYNC_REGISTRATION_TIMEOUT_S)
    t_barrier = time.perf_counter() - t_barrier_start

    # 4. Controller-driven P2P transfer: trainer device -> trainer host -> rollout host(s), resharded; one
    #    transfer whose destination holds the units of every replica (see dst_units).
    t_transfer_start = time.perf_counter()
    transfer_future = controller.start_transfer(
        src_units=src_units,
        dst_units=dst_units,
        dst_mem_type=tpu_sync_api.dram_memory_type(),
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
        *[
            replica.server_handle.collective_rpc.remote(method="install_tpu_sync_weights", kwargs=exact_install_kwargs)
            for replica in manager.replicas
        ]
    )
    t_install = time.perf_counter() - t_install_start
    # collective_rpc returns one result per TP worker of each replica (when the server passes worker results
    # through); each is the timing dict returned by vLLMTPUSyncWorkerExtension.install_tpu_sync_weights. The
    # slowest worker gates the sync, so report the max.
    worker_install_stats = [
        r for per_replica in install_results for r in (per_replica or []) if isinstance(r, dict) and "h2d" in r
    ]
    t_h2d_pure = max((r["h2d"] for r in worker_install_stats), default=None)
    # Tag new generations with this weight version (the trajectory staleness metrics read it), as the other
    # checkpoint backends do once the new weights are loaded. abort_replicas() already reset the prefix
    # cache, and no request has run since.
    t_finalize_start = time.perf_counter()
    if global_steps is not None:
        await asyncio.gather(
            *[replica.server_handle.set_global_steps.remote(global_steps) for replica in manager.replicas]
        )
    t_finalize = time.perf_counter() - t_finalize_start

    # 6. Parity verification (optional, default off). An enabled check raises TPUSyncParityError on a mismatch or on
    #    missing stats/results; it runs before resume_generation_replicas, so a failed sync stops the run instead
    #    of resuming generation on wrong weights.
    t_parity_start = time.perf_counter()
    if exact_install_kwargs:
        report_exact_parity(global_steps, install_results, _replica_labels(manager))
    if verify_parity:
        await _verify_parity(manager, global_steps)
    t_parity = time.perf_counter() - t_parity_start

    # 7. Resume generation as soon as the rollout holds the new weights and their global_steps tag, without waiting
    #    for the trainer to free its send buffers below: that only gives trainer HBM back and must not keep the
    #    rollout idle.
    t_resume_start = time.perf_counter()
    await manager.resume_generation_replicas()
    t_resume = time.perf_counter() - t_resume_start

    # 8. Wait for the trainer to free its send buffers before the next training step.
    t_release_start = time.perf_counter()
    if release_refs is not None:
        await asyncio.to_thread(ray.get, release_refs)
    t_release = time.perf_counter() - t_release_start

    t_total = time.perf_counter() - t_total_start

    logger.info(
        f"[TPU-SYNC TELEMETRY | Orchestrator] Step {global_steps} Completed in {t_total:.4f}s:\n"
        f"  * Sampler Quiesce/Pause  : {t_abort:.4f}s (overlaps Trainer Init)\n"
        f"  * Trainer Init           : {t_init_trainer:.4f}s\n"
        f"  * Sampler Init           : {t_init_sampler:.4f}s\n"
        f"  * Barrier Check          : {t_barrier:.4f}s\n"
        f"  * Controller P2P         : {t_transfer:.4f}s\n"
        f"  * Sampler H2D DMA        : {t_install:.4f}s\n"
        f"  * Parity Check           : {t_parity:.4f}s\n"
        f"  * Sampler Resume         : {t_resume:.4f}s\n"
        f"  * Trainer Buffer Release : {t_release:.4f}s (rollout already generating)\n"
        f"  * Total End-to-End Sync  : {t_total:.4f}s"
    )

    if timeline.enabled():
        step = global_steps or 0
        # The quiesce runs concurrently with the trainer send; every later phase follows the longer of the two.
        sync_phases = [
            ("sync.sampler_init", t_init_sampler),
            ("sync.barrier", t_barrier),
            ("sync.p2p_transfer", t_transfer),
            ("sync.sampler_install", t_install),
            ("sync.finalize", t_finalize),
            ("sync.parity_check", t_parity if verify_parity or exact_install_kwargs else 0.0),
            ("sync.resume", t_resume),
            ("sync.trainer_release", t_release),
        ]
        t_concurrent = max(t_abort, t_init_trainer)
        timeline.emit(
            "update_weights" if step > 0 else "initial_weight_sync",
            t_sync_wall,
            t_concurrent + sum(d for _, d in sync_phases),
            step=step,
            quiesce=round(t_abort, 3),
            trainer_send=round(t_init_trainer, 3),
            p2p=round(t_transfer, 3),
            install=round(t_install, 3),
        )
        for phase_name, phase_dur in (("sync.quiesce", t_abort), ("sync.trainer_send", t_init_trainer)):
            if phase_dur >= 1e-4:
                timeline.emit(phase_name, t_sync_wall, phase_dur, step=step)
        t_cursor = t_sync_wall + t_concurrent
        for phase_name, phase_dur in sync_phases:
            if phase_dur >= 1e-4:
                timeline.emit(phase_name, t_cursor, phase_dur, step=step)
            t_cursor += phase_dur

    # Surface the phase timers as step metrics. Trainers that record sync metrics (e.g. the separate-async
    # trainer's _pending_sync_metrics) log them next to timing_s/update_weights with every configured
    # backend (console / tensorboard / wandb). total_sync spans the whole sync: the quiesce overlaps its trainer_init.
    metrics = {
        "timing_s/tpu-sync/quiesce": t_abort,
        "timing_s/tpu-sync/trainer_init": t_init_trainer,
        "timing_s/tpu-sync/sampler_init": t_init_sampler,
        "timing_s/tpu-sync/barrier": t_barrier,
        "timing_s/tpu-sync/p2p_transfer": t_transfer,
        "timing_s/tpu-sync/sampler_h2d": t_install,
        "timing_s/tpu-sync/sampler_resume": t_resume,
        "timing_s/tpu-sync/trainer_release": t_release,
        "timing_s/tpu-sync/total_sync": t_total,
    }
    if t_h2d_pure is not None:
        # Pure _tpu_sync_ws.h2d() time on the slowest sampler worker (sampler_h2d above also includes the
        # fuse/transpose into vLLM params, the TPU sync barrier and the RPC round trip).
        metrics["timing_s/tpu-sync/sampler_h2d_pure"] = t_h2d_pure
    return metrics


def _replica_labels(manager: Any) -> list[str]:
    """Names each of ``manager.replicas`` in the parity log lines (empty strings with one replica)."""
    if len(manager.replicas) == 1:
        return [""]
    return [f" | replica {_replica_rank(replica, index)}" for index, replica in enumerate(manager.replicas)]


async def _verify_parity(manager: Any, global_steps: Optional[int]) -> None:
    """Runs the optional norm parity check (``tpu_sync_parity``), naming the replicas when there are several."""
    await verify_rollout_parity(manager, global_steps, _replica_labels(manager))


apply_checkpoint_engine_manager_hooks()
