# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM worker extension that receives the trainer's weights over tpu-sync on TPU.

``TPUvLLMHttpServer`` passes this class to vLLM as ``worker_extension_cls`` when
``actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync``. The driver
(``update_tpu_sync_weights`` in ``verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_checkpoint_engine``) calls
its methods on every vLLM worker through ``collective_rpc``:

1. ``init_tpu_sync_on_worker``: allocate TP-sharded receive buffers for the tensors the trainer
   published and register them with the tpu-sync controller (first sync only).
2. ``install_tpu_sync_weights``: after the transfer, copy the received buffers to TPU HBM and fuse or
   transpose them into vLLM's parameters (every sync).
3. ``get_model_weights_stats``: weight norms for the optional parity check.

This module imports vLLM (through the upstream worker extension), so only vLLM workers import it.
"""

import logging
import math
import os
import time
from typing import Any, NamedTuple, Optional

import ray
import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from verl_hardware_plugin.accelerators.tpu.rollout import qwen3_5_layout
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_api
from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine as tpu_sync_engine
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck, compute_tensor_stats
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_rendezvous import get_tpu_sync_rendezvous
from verl_hardware_plugin.accelerators.tpu.weight_sync.weight_sync_backends import (
    get_parent_module,
    resolve_model_key,
    to_target_layout,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# How long a rollout worker waits for the controller address and the trainer's tensor shapes.
_REGISTRY_POLL_ATTEMPTS = 30
_REGISTRY_POLL_INTERVAL_S = 0.5
# Row-parallel layers: vLLM shards their input dimension (dim 1) across TP ranks; every other 2-D weight
# whose output dimension divides evenly is sharded on dim 0.
_ROW_PARALLEL_SUFFIXES = (".o_proj.weight", ".down_proj.weight")
# GQA key/value projections: sharded per KV head when vLLM replicates KV heads (tp_size > num_kv_heads).
_KV_PROJ_SUFFIXES = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")


class _SamplerSlice(NamedTuple):
    """How one vLLM TP rank holds a trainer tensor (see ``_sampler_sharding``)."""

    sharding_spec: list[str]
    local_shape: list[int]
    mesh_shape: list[int]
    global_shard_indices: list[int]
    replicas: int


# vLLM fuses these trainer tensors into one parameter: (fused suffix, source suffixes in concat order).
_FUSED_TARGETS = (
    (
        ".self_attn.qkv_proj.weight",
        (".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"),
    ),
    (
        ".self_attn.qkv_proj.bias",
        (".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"),
    ),
    (".mlp.gate_up_proj.weight", (".mlp.gate_proj.weight", ".mlp.up_proj.weight")),
    (".mlp.gate_up_proj.bias", (".mlp.gate_proj.bias", ".mlp.up_proj.bias")),
)


def _sampler_sharding(
    name: str,
    global_shape: list[int],
    tp_size: int,
    tp_rank: int = 0,
    num_kv_heads: Optional[int] = None,
    use_vllm_loaders: bool = False,
) -> _SamplerSlice:
    """Returns the slice of a trainer tensor that one vLLM TP rank holds.

    With GQA and ``tp_size > num_kv_heads``, vLLM replicates each KV head on ``tp_size // num_kv_heads`` ranks,
    so a dim-0 split of k_proj / v_proj over ``tp_size`` would hand every rank a fraction of a head. Those are
    registered as one slice per KV head instead, and ``global_shard_indices`` names the head this rank holds
    (ranks sharing a head receive the same slice). E.g. Qwen3-32B (8 KV heads) at TP=32: k_proj [1024, 5120]
    -> mesh [8, 1], local [128, 5120], rank 13 -> slice 3.

    When ``use_vllm_loaders=True`` (hybrid or MoE models), every >=2D tensor whose dim 0 divides evenly is sharded
    on dim 0 (and all-gathered at install time), while 1-D tensors are received whole.
    """
    spec_axes = [""] * len(global_shape)
    local_shape = list(global_shape)
    tp_axis_size = tp_size  # number of slices along the "tp" mesh axis
    global_shard_indices: list[int] = []  # empty: tpu-sync derives this rank's slice from the mesh geometry
    if use_vllm_loaders:
        if len(global_shape) >= 2 and global_shape[0] % tp_size == 0 and tp_size > 1:
            spec_axes[0] = "tp"
            local_shape[0] = global_shape[0] // tp_size
    elif len(global_shape) == 2:
        if any(name.endswith(s) for s in _ROW_PARALLEL_SUFFIXES):
            # Row parallel: partition input dimension (dim 1) across TP ranks
            spec_axes = ["", "tp"]
            local_shape[1] = global_shape[1] // tp_size
        elif num_kv_heads is not None and num_kv_heads < tp_size and name.endswith(_KV_PROJ_SUFFIXES):
            # GQA with replicated KV heads (see above): one slice per KV head.
            spec_axes = ["tp", ""]
            tp_axis_size = num_kv_heads
            local_shape[0] = global_shape[0] // num_kv_heads
            global_shard_indices = [tp_rank // (tp_size // num_kv_heads)]
        elif global_shape[0] % tp_size == 0:
            # Column parallel or vocab parallel: partition output dimension (dim 0) across TP ranks
            spec_axes = ["tp", ""]
            local_shape[0] = global_shape[0] // tp_size
    mesh_shape = [tp_axis_size if axis == "tp" else 1 for axis in spec_axes]
    # Number of TP ranks holding this same slice: 1 for a plain split, tp_size // num_kv_heads for GQA k/v,
    # tp_size for unsharded tensors. The norm parity check divides by it when summing over ranks.
    replicas = tp_size // math.prod(mesh_shape)
    return _SamplerSlice(spec_axes, local_shape, mesh_shape, global_shard_indices, replicas)


def _slice_from_spec(spec: list[str], local_shape: list[int], tp_size: int) -> _SamplerSlice:
    """``_SamplerSlice`` of a tensor split evenly over the TP ranks along the ``"tp"`` dimension of ``spec``."""
    mesh_shape = [tp_size if axis == "tp" else 1 for axis in spec]
    return _SamplerSlice(spec, local_shape, mesh_shape, [], tp_size // math.prod(mesh_shape))


def _uses_vllm_weight_loader(vllm_model: Any, global_shapes: dict[str, list[int]]) -> bool:
    """Whether to install received weights via vLLM's own checkpoint loaders (hybrid linear-attn or MoE models)."""
    return (
        hasattr(type(vllm_model), "get_mamba_state_shape_from_config")
        or any(len(shape) not in (1, 2) for shape in global_shapes.values())
        or any(".linear_attn." in name or ".experts." in name for name in global_shapes)
    )


class vLLMTPUSyncWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM Worker Extension for Google Cloud TPU with tpu-sync P2P weight synchronization.

    Workloads using the 'tpu-sync' checkpoint engine for high-speed TPU weight sync
    must use this worker extension to bind TPU model tensors with WeightSynchronizer
    and orchestrate dynamic zero-copy H2D DMA transfers during rollout.
    """

    def _get_vllm_model(self) -> Any:
        """Extract the underlying nn.Module from the vLLM V1 worker."""
        worker: Any = getattr(self, "worker", self)
        return worker.model_runner.model

    def _wait_for_tpu_sync_rendezvous(
        self, rank_val: int, rendezvous_name: Optional[str] = None
    ) -> tuple[str, dict[str, list[int]]]:
        """Returns the controller address and the trainer's global tensor shapes from the job's rendezvous actor."""
        for _ in range(_REGISTRY_POLL_ATTEMPTS):
            try:
                rendezvous = get_tpu_sync_rendezvous(rendezvous_name)
                addr, shapes = ray.get(
                    [rendezvous.get_controller_address.remote(), rendezvous.get_global_shapes.remote()]
                )
                if addr and shapes:
                    return addr, shapes
            except Exception as e:
                logger.debug(f"tpu-sync Sampler Rank {rank_val}: registry lookup failed: {e}")
            time.sleep(_REGISTRY_POLL_INTERVAL_S)
        raise RuntimeError(
            f"tpu-sync Sampler Rank {rank_val}: No tpu-sync controller address or "
            f"global shapes found in the tpu-sync rendezvous {rendezvous_name!r} after timeout"
        )

    def init_tpu_sync_on_worker(
        self,
        parallelism: int = tpu_sync_engine.TPU_SYNC_DEFAULT_PARALLELISM,
        job_name: str = "sampler0",
        rendezvous_name: Optional[str] = None,
    ) -> bool:
        """Initialize tpu-sync WeightSynchronizer listener and register with the central tpu-sync controller.

        ``job_name`` is the tpu-sync job this worker's replica registers under (the driver names every rollout
        replica after its ``replica_rank``), so the ranks ``0..TP-1`` of several replicas do not collide on
        the controller. ``rendezvous_name`` is the job's ``TpuSyncRendezvous`` actor, which the driver names.
        Only the first call does work: the receive buffers, the synchronizer and the
        registration are reused by every later sync.
        """
        if getattr(self, "_tpu_sync_ws", None) is not None:
            return True

        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            logger.warning("tpu-sync Sampler: could not locate vllm_model to bind parameters.")
            return False

        bind_ip = ray.util.get_node_ip_address().strip("[]")
        rank_val = getattr(self, "rank", 0)
        # The listener port is assigned by the OS, as on the trainer side, and registered with the controller
        # below; fixed ``base + rank`` ports collided when two replicas shared a host.
        listener_port = 0

        # 1. Fetch tpu-sync controller address and un-fused global shapes from the job's rendezvous
        controller_addr, global_shapes_map = self._wait_for_tpu_sync_rendezvous(rank_val, rendezvous_name)

        # 2. Determine tensor parallel size from vLLM V1 config
        worker: Any = getattr(self, "worker", self)
        parallel_config = getattr(getattr(worker, "vllm_config", None), "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not tp_size or tp_size <= 0:
            raise RuntimeError(
                f"tpu-sync Sampler Rank {rank_val}: Unable to determine tensor_parallel_size (tp_size) "
                f"from worker.vllm_config.parallel_config!"
            )

        # Qwen3.5 installs each rank's slices straight into vLLM's parameters (qwen3_5_layout); other hybrid or
        # MoE models go through vLLM's loaders, which needs every tensor whole on every rank.
        hf_text_config = getattr(worker.vllm_config.model_config, "hf_text_config", None)
        qwen3_5_plan, no_plan_reason = qwen3_5_layout.plan_direct_install(
            vllm_model, global_shapes_map, hf_text_config, tp_size, rank_val
        )
        use_vllm_loaders = qwen3_5_plan is None and _uses_vllm_weight_loader(vllm_model, global_shapes_map)
        self._tpu_sync_qwen3_5_plan = qwen3_5_plan
        self._tpu_sync_use_vllm_loaders = use_vllm_loaders
        install_mode = "qwen3_5 direct" if qwen3_5_plan else "vllm loaders" if use_vllm_loaders else "direct"
        logger.info(f"tpu-sync Sampler Rank {rank_val}: install mode {install_mode}")
        if use_vllm_loaders:
            # Every sync then all-gathers the whole model on every rank and re-runs vLLM's loaders (slow).
            logger.warning(
                f"tpu-sync Sampler Rank {rank_val}: installing through vLLM's checkpoint loaders (TP all-gather of "
                f"every tensor); no direct install: {no_plan_reason}"
            )

        # 3. Allocate local TPU staging buffers matching Trainer un-fused layout and sharding specs
        # TODO(tpu): Copy the received weights straight into vLLM's parameters, with no staging copy. The
        # staging tensors are a second full copy of this rank's TP shard (~1.9 GiB per chip for Qwen3-8B,
        # ~7.6 GiB for Qwen3-32B at TP=8) that must fit next to vLLM's preallocated weights and KV cache on
        # every sync. Unfused tensors (o_proj, down_proj, norms, embed_tokens, lm_head) could bind the vLLM
        # parameter buffers directly; fused qkv_proj / gate_up_proj and transposed (_tpu_weight_flipped)
        # weights need tpu_sync to write into a slice or layout of the target tensor.
        staging_tensors = {}
        staging_sharded = {}
        variable_protos = []
        valid_params = []
        replicas = {}
        # Total KV heads of the model, to slice k_proj / v_proj per KV head when vLLM replicates them. rank_val is
        # this worker's TP rank: a single rollout replica without pipeline parallelism, as the registration below
        # ("sampler/<rank>" on mesh [1, tp_size]) already assumes.
        num_kv_heads = worker.vllm_config.model_config.get_total_num_kv_heads()
        # Tensors received straight into vLLM's weights instead of a staging buffer (Qwen3.5 experts under vLLM
        # expert parallelism, see qwen3_5_layout.zero_copy_targets).
        zero_copy: dict[str, torch.Tensor] = {}
        if qwen3_5_plan is not None:
            init_sd = vllm_model.state_dict()

            def _find_init_key(k: str) -> Optional[str]:
                resolved = resolve_model_key(k, init_sd)
                return resolved if resolved in init_sd else None

            zero_copy = qwen3_5_layout.zero_copy_targets(qwen3_5_plan, init_sd, _find_init_key, global_shapes_map)

        for idx, (name, g_shape) in enumerate(sorted(global_shapes_map.items(), key=lambda x: x[0])):
            g_shape = list(g_shape)
            rule = qwen3_5_layout.sampler_sharding_rule(name, g_shape, tp_size) if qwen3_5_plan else None
            if rule is not None:
                shard = _slice_from_spec(*rule, tp_size)
            else:
                shard = _sampler_sharding(
                    name,
                    g_shape,
                    tp_size,
                    tp_rank=rank_val,
                    num_kv_heads=num_kv_heads,
                    use_vllm_loaders=use_vllm_loaders,
                )
            replicas[name] = shard.replicas

            # Allocate a local staging buffer with matching layout (the trainer sends bf16), or receive straight
            # into the vLLM weight.
            t = zero_copy.get(name)
            if t is not None:
                if list(t.shape) != list(shard.local_shape):
                    raise RuntimeError(
                        f"tpu-sync Sampler Rank {rank_val}: {name} slice {shard.local_shape} != vLLM weight "
                        f"{list(t.shape)}"
                    )
            else:
                device = torch.device(tpu_sync_engine.TPU_SYNC_DEVICE)
                t = torch.empty(shard.local_shape, dtype=torch.bfloat16, device=device)
            staging_tensors[name] = t
            staging_sharded[name] = "tp" in shard.sharding_spec
            valid_params.append((name, t))

            variable_protos.append(
                tpu_sync_api.variable_metadata(
                    name=name,
                    shape=g_shape,
                    mesh_shape=shard.mesh_shape,
                    layout=list(range(len(shard.local_shape) - 1, -1, -1)),
                    item_size=t.element_size(),
                    layer_idx=idx,
                    sharding_spec=shard.sharding_spec,
                    global_shard_indices=shard.global_shard_indices,
                )
            )

        self._tpu_sync_staging = staging_tensors
        self._tpu_sync_staging_sharded = staging_sharded
        self._sorted_vllm_params = valid_params
        self._tpu_sync_replicas = replicas
        if zero_copy:
            logger.info(
                f"tpu-sync Sampler Rank {rank_val}: {len(zero_copy)} tensors "
                f"({sum(t.numel() * t.element_size() for t in zero_copy.values()) / 2**30:.2f} GiB) are received "
                "straight into vLLM's weights (no staging copy)"
            )

        tpu_sync_engine.tpu_synchronize()

        logger.info(
            f"tpu-sync Sampler Rank {rank_val}: binding {len(valid_params)} staging tensors to WeightSynchronizer "
            f"(sample: name={valid_params[0][0]}, shape={valid_params[0][1].shape}, "
            f"total numel={sum(t.numel() for _, t in valid_params)})"
        )

        self._tpu_sync_ws = tpu_sync_engine.create_torch_weight_synchronizer(
            [[t] for _, t in valid_params],
            local_port=0,
            parallelism=parallelism,
            listener_port=listener_port,
            bind_ip=bind_ip,
        )

        # Whether a tensor can skip the CPU (de)tiling pass is decided by tpu-sync's planner per transfer, from the
        # source and destination slices together, and delivered to this listener with the transfer. Do not set it
        # here from the local shape alone: both ends must make the same choice, and when they disagree (the
        # sender de-tiles to row-major, h2d() copies those bytes raw into tiled HBM) the weights come out permuted
        # while every norm stays unchanged. With full trainer tensors the local guess usually matches the planner,
        # but not once the trainer sends shards that are not 8-row aligned (e.g. Qwen3's embedding at 32+ FSDP
        # ranks).

        try:
            ctrl_client = tpu_sync_api.controller_client(controller_addr)
            unit_id = tpu_sync_api.unit_id(job_name, str(rank_val))
            ctrl_client.register_work_unit(
                unit_id,
                [f"{bind_ip}:{self._tpu_sync_ws.local_port}"],
                f"{bind_ip}:{self._tpu_sync_ws.listener_port}",
                mesh_shape=[1, tp_size],
                variables=variable_protos,
                mesh_axes=["fsdp", "tp"],
            )
        except Exception as e:
            logger.error(f"tpu-sync Sampler Rank {rank_val} failed to register with the tpu-sync controller: {e}")
            raise
        logger.info(
            f"tpu-sync Sampler Rank {rank_val} bound {len(valid_params)} dynamic staging tensors and registered "
            f"directly with the tpu-sync controller ({controller_addr}) as {job_name}/{rank_val}: "
            f"mesh_shape=[1, {tp_size}], "
            f"data_port={self._tpu_sync_ws.local_port}, listener_port={self._tpu_sync_ws.listener_port}"
        )

        return True

    @torch.no_grad()
    def install_tpu_sync_weights(self, exact_parity: bool = False) -> dict:
        """Install received weights from host staging buffer into TPU HBM via zero-copy H2D DMA
        and fuse/transpose them directly into vLLM model parameters.

        Args:
            exact_parity: Set by the driver on the step-0 sync with ``verify_parity=exact|all``: compare every
                received tensor bit for bit with the vLLM parameter it overwrites (``ExactParityCheck``).

        Returns:
            Per-worker timings in seconds: ``total`` (whole install), ``h2d`` (pure ``_tpu_sync_ws.h2d()``)
            and ``sync`` (final TPU sync barrier), plus ``exact_parity`` (this rank's result) when requested.
            Empty if the synchronizer is not initialized.
        """
        if getattr(self, "_tpu_sync_ws", None) is None:
            logger.warning("tpu-sync Sampler: install_tpu_sync_weights called before _tpu_sync_ws was initialized.")
            return {}

        rank = getattr(self, "rank", 0)
        t_start = time.perf_counter()
        t_h2d_start = time.perf_counter()
        # h2d() applies the skip_tiling plan the listener received with this transfer (see
        # init_tpu_sync_on_worker); nothing must override it here.
        self._tpu_sync_ws.h2d()
        t_h2d = time.perf_counter() - t_h2d_start

        if getattr(self, "_tpu_sync_use_vllm_loaders", False):
            self._install_with_vllm_loaders()
            t_sync_start = time.perf_counter()
            tpu_sync_engine.tpu_synchronize()
            t_sync = time.perf_counter() - t_sync_start
            t_total = time.perf_counter() - t_start
            logger.info(
                f"[TPU-SYNC TELEMETRY | Sampler Worker] Sampler Rank {rank}: "
                f"install_tpu_sync_weights (load_weights) completed in {t_total:.4f}s "
                f"(H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
            )
            result: dict[str, Any] = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
            if exact_parity:
                # vLLM's loaders re-shard and re-layout the tensors themselves; there is no 1:1 target to compare.
                result["exact_parity"] = {"rank": rank, "skipped": "not supported with the vLLM-loader install"}
            return result

        # Unpack, fuse, and transpose staging tensors into vLLM model parameters
        vllm_model = self._get_vllm_model()
        model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
        module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}
        parity = ExactParityCheck() if exact_parity else None

        def find_model_key(k: str) -> Optional[str]:
            resolved = resolve_model_key(k, model_sd)
            return resolved if resolved in model_sd else None

        def copy_into(resolved_key: str, src: torch.Tensor) -> None:
            target_param = model_sd[resolved_key]
            target_local = target_param.to_local() if hasattr(target_param, "to_local") else target_param
            parent_mod = get_parent_module(resolved_key, module_dict)
            is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))
            adapted = to_target_layout(src, target_local, is_flipped)
            if parity is not None:
                parity.check(resolved_key, target_local, adapted)
            target_local.copy_(adapted)

        consumed_staging = set()
        qwen3_5_plan = getattr(self, "_tpu_sync_qwen3_5_plan", None)
        fused_targets: tuple[tuple[str, tuple[str, ...]], ...] = _FUSED_TARGETS
        if qwen3_5_plan is not None:
            # GatedDeltaNet projections, conv, A_log / dt_bias and the stacked experts (see qwen3_5_layout).
            consumed_staging |= qwen3_5_layout.install(
                self._tpu_sync_staging, model_sd, find_model_key, copy_into, qwen3_5_plan
            )
            fused_targets = _FUSED_TARGETS + qwen3_5_layout.FUSED_TARGETS

        # 1. Handle fused projections (QKV and Gate-Up)
        for target_suffix, src_suffixes in fused_targets:
            primary_src = src_suffixes[0]
            for name in list(self._tpu_sync_staging.keys()):
                if name.endswith(primary_src):
                    layer_prefix = name[: -len(primary_src)]
                    layer_src_keys = [layer_prefix + s for s in src_suffixes]
                    if all(sk in self._tpu_sync_staging for sk in layer_src_keys):
                        resolved_key = find_model_key(layer_prefix + target_suffix)
                        if resolved_key is not None:
                            fused = torch.cat([self._tpu_sync_staging[sk] for sk in layer_src_keys], dim=0)
                            copy_into(resolved_key, fused)
                            consumed_staging.update(layer_src_keys)

        # 2. Handle all remaining non-fused parameters (o_proj, down_proj, layernorms, embeddings)
        unresolved = []
        for name, src_t in self._tpu_sync_staging.items():
            if name in consumed_staging:
                continue
            resolved_key = find_model_key(name)
            if resolved_key is not None:
                copy_into(resolved_key, src_t)
            else:
                unresolved.append(name)
        if qwen3_5_plan is not None and unresolved:
            raise RuntimeError(f"tpu-sync direct install: no vLLM parameter for {len(unresolved)}: {unresolved[:8]}")

        # 3. Handle tied word embeddings. Only when the trainer did not send lm_head itself: for untied
        #    models (e.g. Qwen3-8B / 32B) the received lm_head must not be overwritten by embed_tokens.
        received_lm_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in self._tpu_sync_staging)
        if received_lm_head:
            pass
        elif (
            hasattr(vllm_model, "lm_head")
            and hasattr(vllm_model, "model")
            and hasattr(vllm_model.model, "embed_tokens")
        ):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.model.embed_tokens.weight)
        elif hasattr(vllm_model, "lm_head") and hasattr(vllm_model, "embed_tokens"):
            if vllm_model.lm_head.weight.data_ptr() != vllm_model.embed_tokens.weight.data_ptr():
                vllm_model.lm_head.weight.copy_(vllm_model.embed_tokens.weight)

        t_sync_start = time.perf_counter()
        tpu_sync_engine.tpu_synchronize()
        t_sync = time.perf_counter() - t_sync_start
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[TPU-SYNC TELEMETRY | Sampler Worker] Sampler Rank {rank}: install_tpu_sync_weights "
            f"completed in {t_total:.4f}s (H2D={t_h2d:.4f}s, TPUSyncBarrier={t_sync:.4f}s)"
        )
        # Returned through collective_rpc so the orchestrator can log them as step metrics.
        result = {"total": t_total, "h2d": t_h2d, "sync": t_sync}
        if parity is not None:
            # Covers every tensor written through copy_into; Qwen3.5's stacked experts are written directly
            # (qwen3_5_layout._write_w13 / _write_w2) and are not compared.
            result["exact_parity"] = parity.result(rank, unresolved)
        return result

    @torch.no_grad()
    def _install_with_vllm_loaders(self) -> None:
        """Load the received checkpoint-format (HF-named) tensors through vLLM's own weight loaders.

        Why: the default install path copies each received tensor straight into vLLM's parameter buffers and
        knows the dense layouts of Qwen3 / Llama (fused ``qkv_proj`` / ``gate_up_proj``, row- vs column-parallel
        splits, the transposed TPU weights). Hybrid and MoE models need much more: Qwen3.5's GatedDeltaNet fuses
        several HF tensors into one vLLM parameter and has 1-D / conv tensors with their own layouts, and the MoE
        experts arrive as HF's stacked ``gate_up_proj`` / ``down_proj`` while vLLM stores TP-sharded, padded,
        kernel-layout expert weights. Instead of re-implementing those layouts here, this path hands vLLM exactly
        what a checkpoint load would and lets the model's ``load_weights`` place everything
        (``_uses_vllm_weight_loader`` selects it).

        How: each staging tensor holds this rank's dim-0 slice (or the whole tensor); it is all-gathered over the
        vLLM TP group one tensor at a time, in the same (sorted) order on every rank, and streamed into
        ``model_runner.reload_weights``. vLLM's layerwise reload restores each layer's checkpoint-format
        parameters, runs the model's ``load_weights`` (fusion, TP sharding) and re-runs
        ``process_weights_after_loading`` (TPU kernel layouts). This install takes most of ``update_weights`` for
        Qwen3.5-35B-A3B (~55 s): every rank gathers every tensor, and vLLM's MoE loaders loop over the experts.
        """
        from vllm.distributed import get_tp_group

        tp_group = get_tp_group()
        staging = self._tpu_sync_staging
        sharded = self._tpu_sync_staging_sharded
        t_gather = 0.0

        def _weights():
            nonlocal t_gather
            for name in sorted(staging):
                t = staging[name]
                if sharded[name]:
                    t0 = time.perf_counter()
                    t = tp_group.all_gather(t, dim=0)
                    # Run every collective on its own: deferred (fused) collective programs can be cut at different
                    # points on different ranks and then never match up, which stalls the TP group.
                    tpu_sync_engine.tpu_synchronize(t)
                    t_gather += time.perf_counter() - t0
                yield name, t

        worker: Any = getattr(self, "worker", self)
        model_runner = worker.model_runner
        if hasattr(model_runner, "reload_weights"):
            # vllm-torchtpu's dense linear method transposes weights to (k, n) in process_weights_after_loading,
            # guarded by a per-module "already flipped" flag that layerwise reload does not reset, so the fresh
            # weights would stay [n_out, n_in] and fail to copy into the live [n_in, n_out] tensors. Clear it first.
            for module in self._get_vllm_model().modules():
                if getattr(module, "_tpu_weight_flipped", False):
                    module._tpu_weight_flipped = False
            model_runner.reload_weights(
                weights_iterator=qwen3_5_layout.refuse_split_tensors(_weights()), is_checkpoint_format=True
            )
        else:
            self._get_vllm_model().load_weights(qwen3_5_layout.refuse_split_tensors(_weights()))
        logger.info(f"Sampler Rank {getattr(self, 'rank', 0)}: TP all-gathers of the received tensors: {t_gather:.2f}s")

    def get_model_weights_stats(self) -> dict:
        """Computes deterministic parameter count, L1 norm, and L2 norm of this worker's received weights."""
        vllm_model = self._get_vllm_model()
        if vllm_model is None:
            return {"error": "No model found on worker"}

        if not getattr(self, "_sorted_vllm_params", None):
            raise RuntimeError(
                f"get_model_weights_stats is only supported for tpu-sync after initialization, "
                f"but self._sorted_vllm_params is not initialized on worker rank {getattr(self, 'rank', 'unknown')}!"
            )

        res = compute_tensor_stats(self._sorted_vllm_params)
        res["rank"] = getattr(self, "rank", 0)
        # How many TP ranks hold each tensor's slice, so the parity check counts every slice once.
        res["replicas"] = dict(getattr(self, "_tpu_sync_replicas", None) or {})
        return res
