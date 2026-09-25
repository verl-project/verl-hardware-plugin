# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TPU CheckpointEngine for weight synchronization between TorchTitan trainer and vLLM rollout."""

import asyncio
import gc
import inspect
import logging
import os
import re
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import ray
import torch
from torch.distributed.tensor import DTensor

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl_hardware_plugin.engines.ray_weight_registry import RayWeightRegistry

logger = logging.getLogger(__name__)

TPU_COPY_CHUNK_SIZE_PARAMETERS = 30
RAY_WEIGHT_REGISTRY_ACTOR_NAME = "RayWeightRegistry"
RAY_WEIGHT_REGISTRY_NAMESPACE = "verl"

# vLLM stores q/k/v and gate/up as single fused parameters, while TorchTitan
# exports them under their original HuggingFace names.
_FUSED_PROJECTIONS: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _synchronize_tpu() -> None:
    """Synchronizes pending device operations on TPU when ``torch_tpu`` is active."""
    try:
        import torch_tpu

        torch_tpu._internal.sync.synchronize(wait=True)
    except Exception:
        pass


def get_clean_name(name: str) -> str:
    """Strips FSDP/DCP wrapper prefixes from state dict keys to match standard model namespaces."""
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("module.", "")


def get_layer_group(key: str) -> str:
    """Returns the layer group identifier (e.g. ``'layers.0'``, ``'embeddings'``, ``'output'``) for a parameter key."""
    clean_k = get_clean_name(key)
    match = re.search(r"layers\.(\d+)\.", clean_k)
    if match:
        return f"layers.{match.group(1)}"
    if "tok_embeddings" in clean_k:
        return "embeddings"
    return "output"


def _resolve_model_key(key: str, model_sd: dict[str, torch.Tensor]) -> str:
    """Resolves a cleaned trainer key against vLLM's ``model_sd`` (handling optional ``model.`` prefixes)."""
    if key in model_sd:
        return key
    if key.startswith("model.") and key[6:] in model_sd:
        return key[6:]
    prefixed = f"model.{key}"
    if prefixed in model_sd:
        return prefixed
    return key


def _get_parent_module(target_key: str, module_dict: dict[str, torch.nn.Module]) -> torch.nn.Module | None:
    """Returns the owning ``torch.nn.Module`` for ``target_key``, or ``None`` if not found."""
    parent_name = target_key.rsplit(".", 1)[0] if "." in target_key else ""
    if parent_name in module_dict:
        return module_dict[parent_name]
    if parent_name.startswith("model.") and parent_name[6:] in module_dict:
        return module_dict[parent_name[6:]]
    prefixed = f"model.{parent_name}"
    return module_dict.get(prefixed)


def _to_target_layout(tensor: torch.Tensor, target_local: torch.Tensor, flipped: bool) -> torch.Tensor:
    """Matches ``vllm-torchtpu``'s ``(n_in, n_out)`` 2D weight layout when required."""
    if tensor.ndim == 2 and (flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)):
        return tensor.transpose(0, 1).contiguous()
    return tensor.contiguous()


def _build_fused_shard(
    fused_key: str,
    parts: list[torch.Tensor],
    rank: int,
    model_sd: dict[str, torch.Tensor],
    module_dict: dict[str, torch.nn.Module],
) -> tuple[str, torch.Size, int, torch.Tensor]:
    """Shards each source projection for ``rank`` and concatenates them into a fused vLLM parameter."""
    target_v = model_sd[fused_key]
    target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
    module = _get_parent_module(fused_key, module_dict)
    flipped = bool(getattr(module, "_tpu_weight_flipped", False))

    out_dim = target_local.shape[1] if (flipped and target_local.ndim == 2) else target_local.shape[0]
    tp_size = getattr(module, "tp_size", max(1, sum(p.shape[0] for p in parts) // out_dim))
    kv_replicas = getattr(module, "num_kv_head_replicas", 1)
    kv_tp, kv_rank = max(1, tp_size // kv_replicas), rank // kv_replicas

    shards = []
    for idx, part in enumerate(parts):
        n_shards, shard_rank = (tp_size, rank) if idx == 0 else (kv_tp, kv_rank)
        size = part.shape[0] // n_shards
        shards.append(part[shard_rank * size : (shard_rank + 1) * size])

    fused = _to_target_layout(torch.cat(shards, dim=0), target_local, flipped)
    return (fused_key, target_local.shape, target_local.numel(), fused.reshape(-1))


def _slice_parameter_for_rank(
    key: str,
    raw_tensors: dict[str, torch.Tensor],
    rank: int,
    model_sd: dict[str, torch.Tensor],
    module_dict: dict[str, torch.nn.Module],
) -> tuple[str, torch.Size, int, torch.Tensor] | None:
    """Slices or fuses a trainer parameter into the local shard expected by vLLM rank ``rank``."""
    target_key = _resolve_model_key(key, model_sd)

    if target_key not in model_sd:
        for suffix in (".weight", ".bias"):
            if not key.endswith(suffix):
                continue
            base = key[: -len(suffix)]
            for fused_name, sources in _FUSED_PROJECTIONS.items():
                if not base.endswith(sources[0]):
                    continue
                prefix = base[: -len(sources[0])]
                fused_key = _resolve_model_key(f"{prefix}{fused_name}{suffix}", model_sd)
                parts = [raw_tensors.get(f"{prefix}{s}{suffix}") for s in sources]
                if fused_key not in model_sd or any(p is None for p in parts):
                    return None
                return _build_fused_shard(fused_key, parts, rank, model_sd, module_dict)  # type: ignore[arg-type]
        return None

    target_v = model_sd[target_key]
    target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
    parent_mod = _get_parent_module(target_key, module_dict)
    is_flipped = bool(getattr(parent_mod, "_tpu_weight_flipped", False))

    param_cpu_global = raw_tensors[key]
    if param_cpu_global.ndim == 2 and is_flipped:
        param_cpu_global = param_cpu_global.transpose(0, 1)

    eff_shape = param_cpu_global.shape
    if target_local.shape == eff_shape:
        param_cpu_local = param_cpu_global.contiguous()
    else:
        param_cpu_local = param_cpu_global.contiguous()
        for dim, (global_dim, local_dim) in enumerate(zip(eff_shape, target_local.shape, strict=False)):
            if global_dim != local_dim:
                rank_offset = local_dim * rank
                indices = [slice(None)] * len(eff_shape)
                indices[dim] = slice(rank_offset, rank_offset + local_dim)
                param_cpu_local = param_cpu_global[tuple(indices)].contiguous()
                break

    return (target_key, target_local.shape, target_local.numel(), param_cpu_local.reshape(-1))


def _load_single_group_on_worker(
    vllm_model: Any,
    group_sd: dict[str, Any],
    rank: int,
    executor: ThreadPoolExecutor,
    temp_tpu_tensors: list[torch.Tensor],
    target_device: str = "tpu",
) -> int:
    """Slices and copies a single layer group's tensors into ``vllm_model`` on ``target_device``."""
    flat_tensors = group_sd["flat_tensors"]
    metadata = group_sd["metadata"]

    clean_metadata: dict[torch.dtype, list[tuple[str, torch.Size, int, int]]] = {}
    num_keys = 0
    for dtype, items in metadata.items():
        clean_items = []
        offset = 0
        for key, shape, numel in items:
            clean_k = get_clean_name(key)
            clean_items.append((clean_k, shape, numel, offset))
            num_keys += 1
            if "tok_embeddings.weight" in clean_k:
                lm_k = clean_k.replace("tok_embeddings", "lm_head")
                clean_items.append((lm_k, shape, numel, offset))
                num_keys += 1
            offset += numel
        clean_metadata[dtype] = clean_items

    model_sd = vllm_model.state_dict() if hasattr(vllm_model, "state_dict") else vllm_model.model.state_dict()
    module_dict = dict(vllm_model.named_modules()) if hasattr(vllm_model, "named_modules") else {}
    effective_device = target_device if (target_device != "tpu" or hasattr(torch, "tpu")) else "cpu"

    for dtype, flat_data in flat_tensors.items():
        items = clean_metadata.get(dtype, [])
        if not items:
            continue

        flat_cpu = torch.from_numpy(flat_data) if not isinstance(flat_data, torch.Tensor) else flat_data
        if dtype == torch.bfloat16 and flat_cpu.dtype == torch.int16:
            flat_cpu = flat_cpu.view(torch.bfloat16)

        raw_tensors: dict[str, torch.Tensor] = {}
        dedup_keys: list[str] = []
        seen_clean_keys: set[str] = set()
        for key, shape, numel, offset in items:
            raw_tensors[key] = flat_cpu[offset : offset + numel].view(shape)
            if key not in seen_clean_keys:
                seen_clean_keys.add(key)
                dedup_keys.append(key)

        sliced_results = list(
            executor.map(
                lambda k, rt=raw_tensors: _slice_parameter_for_rank(k, rt, rank, model_sd, module_dict),
                dedup_keys,
            )
        )

        local_items: list[tuple[str, torch.Size, int, int]] = []
        local_tensors_to_cat: list[torch.Tensor] = []
        local_offset = 0
        for res in sliced_results:
            if res is None:
                continue
            target_key, target_shape, target_numel, param_cpu_local_flat = res
            local_tensors_to_cat.append(param_cpu_local_flat)
            local_items.append((target_key, target_shape, target_numel, local_offset))
            local_offset += target_numel

        if not local_tensors_to_cat:
            continue

        flat_local_cpu = torch.cat(local_tensors_to_cat)
        for i in range(0, len(local_items), TPU_COPY_CHUNK_SIZE_PARAMETERS):
            chunk = local_items[i : i + TPU_COPY_CHUNK_SIZE_PARAMETERS]
            chunk_start_offset = chunk[0][3]
            chunk_end_offset = chunk[-1][3] + chunk[-1][2]
            flat_chunk_dev = flat_local_cpu[chunk_start_offset:chunk_end_offset].to(effective_device)
            temp_tpu_tensors.append(flat_chunk_dev)

            for target_key, local_shape, local_numel, offset in chunk:
                rel_offset = offset - chunk_start_offset
                slice_dev = flat_chunk_dev[rel_offset : rel_offset + local_numel].view(local_shape)
                target_v = model_sd[target_key]
                target_local = target_v.to_local() if isinstance(target_v, DTensor) else target_v
                target_local.copy_(slice_dev)

    return num_keys


def load_weights_on_worker(
    vllm_model: Any,
    state_dict: dict[str, Any] | None,
    rank: int,
    target_device: str = "tpu",
) -> int:
    """Loads packed trainer weights into a vLLM worker model via CPU sharding and chunked TPU copies."""
    if state_dict is None:
        return 0

    t_start = time.perf_counter()
    grouped_dict = state_dict["grouped"] if "grouped" in state_dict else {"all": state_dict}

    total_keys = 0
    temp_tpu_tensors: list[torch.Tensor] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for group_sd in grouped_dict.values():
            total_keys += _load_single_group_on_worker(
                vllm_model,
                group_sd,
                rank,
                executor=executor,
                temp_tpu_tensors=temp_tpu_tensors,
                target_device=target_device,
            )

    _synchronize_tpu()
    del temp_tpu_tensors
    gc.collect()

    if rank == 0:
        logger.info("Worker 0: Loaded %d keys in %.3fs", total_keys, time.perf_counter() - t_start)
    return total_keys


def load_weights_from_ray_registry(self: Any, step_key: int) -> int:
    """vLLM TPUWorker RPC entrypoint that pulls ``step_key`` weights from ``TPUWeightRegistry`` via host mmap."""
    rank_val = getattr(
        self, "rank", getattr(self, "adjusted_rank", getattr(self, "rpc_rank", int(os.environ.get("RANK", "0"))))
    )

    shm_dir = "/tmp/verl_weight_cache/shared"
    os.makedirs(shm_dir, exist_ok=True)
    shm_file_path = f"{shm_dir}/state_dict_{step_key}.pt"
    shm_tmp_path = f"{shm_dir}/state_dict_{step_key}.tmp"
    shm_ready_path = f"{shm_dir}/state_dict_{step_key}.ready"
    lock_path = f"{shm_dir}/state_dict_{step_key}.lock"

    is_master = False
    lock_fd = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        is_master = True
    except FileExistsError:
        is_master = False

    if is_master:
        try:
            for file_name in os.listdir(shm_dir):
                if file_name.startswith("state_dict_") and f"_{step_key}." not in file_name:
                    try:
                        os.remove(os.path.join(shm_dir, file_name))
                    except OSError:
                        pass

            registry = ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
            state_dict_ref = ray.get(registry.get_weights.remote(step_key))
            if state_dict_ref is None:
                return 0

            if isinstance(state_dict_ref, list) and len(state_dict_ref) == 1:
                state_dict_ref = state_dict_ref[0]
            state_dict_data = ray.get(state_dict_ref) if isinstance(state_dict_ref, ray.ObjectRef) else state_dict_ref

            if isinstance(state_dict_data, dict) and "grouped" in state_dict_data:
                for group_sd in state_dict_data["grouped"].values():
                    flat_tensors = group_sd["flat_tensors"]
                    for dtype, arr in list(flat_tensors.items()):
                        if isinstance(arr, np.ndarray):
                            flat_tensors[dtype] = torch.from_numpy(arr)

            torch.save(state_dict_data, shm_tmp_path)
            os.replace(shm_tmp_path, shm_file_path)
            del state_dict_data
            gc.collect()

            with open(shm_ready_path, "w") as ready_file:
                ready_file.write("ready")
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
    else:
        t_wait_start = time.time()
        while not os.path.exists(shm_ready_path):
            time.sleep(0.05)
            if time.time() - t_wait_start > 30:
                break

    time.sleep((rank_val % 4) * 0.4)
    gc.collect()
    try:
        state_dict_data = torch.load(shm_file_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:
        logger.warning("Shared memory state dict %s not available: %s", shm_file_path, e)
        return 0

    worker_inst = self.worker if getattr(self, "worker", None) is not None else self
    vllm_model = worker_inst.get_model() if hasattr(worker_inst, "get_model") else worker_inst.model_runner.model
    num_keys = load_weights_on_worker(vllm_model, state_dict_data, rank_val)
    del state_dict_data
    gc.collect()
    return num_keys


@CheckpointEngineRegistry.register("tpu")
class TPUCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for transferring model weights from TorchTitan trainer to vLLM rollout on TPU via Ray.

    Trainer rank 0 all-gathers sharded ``DTensor`` parameters, packs them into Ray Plasma
    (``ray.put``), and registers the ``ObjectRef`` in ``RayWeightRegistry``. Rollout ``TPUWorker``
    processes (invoked via ``collective_rpc``) then pull the state dict from Plasma, shard/fuse
    parameters for their tensor-parallel rank, and copy them into TPU HBM::

         [Trainer TPU 0]                                   [Rollout TPU k]
        +-------------------------------+                 +---------------------------------+
        | Process 1:                    |                 | Process 1 (CPU-ONLY STUB!):     |
        | ActorRolloutRefWorker         |                 | CheckpointEngineWorker          |
        |  - TorchTitanEngine           |                 |  - use_gpu=False (0 TPU chips)  |
        |  - TPUCheckpointEngine        |                 |  - checkpoint_engine = None     |
        |    (Offloads full model to    |                 +---------------------------------+
        |     CPU & ray.put(state_dict))|
        +---------------+---------------+                 +---------------------------------+
                        |                                 | Process 2 (EXCLUSIVE TPU OWNER):|
                        | set_weights(step, [ref])        | vLLM TPUWorker (owns libtpu)    |
                        v                                 |  - Patched with                 |
              +-------------------+     1. Lookup [ref]   |    load_weights_from_ray_       |
              | RayWeightRegistry |< - - - - - - - - - - -|    registry(step)               |
              | (Detached Actor)  |     2. ray.get(ref)   |  - Pulls full model from Plasma,|
              +-------------------+        from Plasma    |    slices TP shard, .to("tpu")  |
                        ^                                 +---------------------------------+
                        |                                                  ^
          [Ray Controller: CheckpointEngineManager]                        |
          Calls replica.server_handle.collective_rpc(                      |
              "load_weights_from_ray_registry", args=(step,) --------------+
          )
    """

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs: Any) -> None:
        self.is_master = is_master
        self.bucket_size = bucket_size
        self.registry: Any = None

        if ray.is_initialized():
            try:
                self.registry = ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
            except ValueError:
                try:
                    self.registry = RayWeightRegistry.options(
                        name=RAY_WEIGHT_REGISTRY_ACTOR_NAME,
                        namespace=RAY_WEIGHT_REGISTRY_NAMESPACE,
                        lifetime="detached",
                    ).remote()
                except Exception:
                    self.registry = ray.get_actor(
                        RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE
                    )

            if self.is_master and self.registry is not None:
                try:
                    ray.get(self.registry.clear.remote())
                except Exception as e:
                    logger.warning("Could not reset RayWeightRegistry left over from a previous job: %s", e)

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict, dict]:
        return {}, {}

    def init_process_group(self, **kwargs: Any) -> None:
        pass

    def finalize(self) -> None:
        pass

    @staticmethod
    def pack_weights_to_grouped_dict(weights: Generator[tuple[str, torch.Tensor], None, None]) -> dict[str, Any]:
        """Consumes a weight generator on CPU and flattens tensors by layer group and dtype."""
        grouped_weights: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for key, tensor in weights:
            cpu_v = tensor.detach().cpu()
            del tensor
            grouped_weights.setdefault(get_layer_group(key), []).append((key, cpu_v))

        grouped_dict: dict[str, dict[str, Any]] = {}
        for group_name in list(grouped_weights.keys()):
            group_items = grouped_weights.pop(group_name)
            by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
            for key, cpu_v in group_items:
                by_dtype.setdefault(cpu_v.dtype, []).append((key, cpu_v))
            del group_items

            flat_tensors: dict[torch.dtype, np.ndarray] = {}
            metadata: dict[torch.dtype, list[tuple[str, torch.Size, int]]] = {}
            for dtype, items in by_dtype.items():
                flat_cpu = torch.cat([v.view(-1) for _, v in items])
                flat_tensors[dtype] = (
                    flat_cpu.view(torch.int16).numpy() if dtype == torch.bfloat16 else flat_cpu.numpy()
                )
                metadata[dtype] = [(k, v.shape, v.numel()) for k, v in items]
                del items
            del by_dtype

            grouped_dict[group_name] = {"flat_tensors": flat_tensors, "metadata": metadata}

        return {"grouped": grouped_dict}

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ) -> None:
        """All-gathers and packs weights on trainer rank 0, publishing the ``ObjectRef`` to ``RayWeightRegistry``."""
        t_start = time.perf_counter()
        _synchronize_tpu()

        if not self.is_master:
            for _key, tensor in weights:
                del tensor
            _synchronize_tpu()
            gc.collect()
            return

        step_key = global_steps if global_steps is not None else 0
        logger.info("TPUCheckpointEngine: [Step %s] Start send_weights...", step_key)

        t_offload_start = time.perf_counter()
        state_dict = self.pack_weights_to_grouped_dict(weights)
        _synchronize_tpu()
        t_offload = time.perf_counter() - t_offload_start

        t_put_start = time.perf_counter()
        ref = ray.put(state_dict)
        del state_dict
        t_put = time.perf_counter() - t_put_start

        t_reg_start = time.perf_counter()
        await self.registry.set_weights.remote(step_key, [ref])
        del ref
        gc.collect()
        t_reg = time.perf_counter() - t_reg_start

        logger.debug(
            "TPUCheckpointEngine Phase A [Step %s]: Total=%.3fs, OffloadFlatten=%.3fs, RayPut=%.3fs, Registry=%.3fs",
            step_key,
            time.perf_counter() - t_start,
            t_offload,
            t_put,
            t_reg,
        )

    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError("Rollout on TPU uses direct load_weights_from_ray_registry via collective_rpc.")


async def update_tpu_weights(manager: Any, global_steps: int | None = None) -> dict[str, Any]:
    """Synchronizes weights from the trainer worker group to vLLM rollout replicas on TPU."""
    t_abort_start = time.perf_counter()
    if global_steps and global_steps > 0:
        try:
            await manager.abort_replicas()
        except Exception as e:
            logger.warning("Failed to abort replicas at step %s: %s", global_steps, e)
    t_abort = time.perf_counter() - t_abort_start

    t_total_start = time.perf_counter()
    actor_refs = manager.actor_wg.update_weights(global_steps=global_steps, mode=manager.backend)
    if actor_refs is not None:
        ray.get(actor_refs)

    step_key = global_steps if global_steps is not None else 0
    registry = ray.get_actor(RAY_WEIGHT_REGISTRY_ACTOR_NAME, namespace=RAY_WEIGHT_REGISTRY_NAMESPACE)
    published = ray.get(registry.get_weights.remote(step_key))
    if published is None:
        raise RuntimeError(
            f"TPU weight sync failed: no weights published under step_key={step_key}. "
            "The trainer-side send_weights never reached the registry."
        )
    del published

    futures = [
        replica.server_handle.collective_rpc.remote(method="load_weights_from_ray_registry", args=(step_key,))
        for replica in manager.replicas
    ]
    results = await asyncio.gather(*futures)

    try:
        await registry.clear.remote()
    except Exception:
        pass
    gc.collect()

    flat_counts: list[Any] = []
    for replica_result in results:
        if isinstance(replica_result, list | tuple):
            flat_counts.extend(replica_result)
        elif replica_result is not None:
            flat_counts.append(replica_result)
    if flat_counts and not any(isinstance(n, int) and n > 0 for n in flat_counts):
        raise RuntimeError(
            f"TPU weight sync failed: no rollout worker loaded any tensor for "
            f"step_key={step_key} (per-worker key counts: {results})."
        )

    logger.info(
        "TPU weight sync for step %s completed in %.3fs", global_steps, time.perf_counter() - t_total_start + t_abort
    )
    await manager.resume_generation_replicas()
    return {}


def apply_tpu_checkpoint_engine_hooks() -> None:
    """Registers TPU weight-sync hooks on ``CheckpointEngineWorker``, ``CheckpointEngineManager``, and ``TPUWorker``."""
    try:
        for mod_path, cls_name in (
            ("vllm_torchtpu.worker.tpu_worker", "TPUWorker"),
            ("tpu_inference.worker.tpu_worker", "TPUWorker"),
            ("vllm.executor.ray_utils", "RayWorkerWrapper"),
        ):
            try:
                mod = __import__(mod_path, fromlist=[cls_name])
                worker_cls = getattr(mod, cls_name, None)
                if worker_cls is not None and not hasattr(worker_cls, "load_weights_from_ray_registry"):
                    worker_cls.load_weights_from_ray_registry = load_weights_from_ray_registry
            except ImportError as e:
                missing_pkg = getattr(e, "name", None) or mod_path
                logger.debug(
                    "Skipping load_weights_from_ray_registry hook for %s.%s: package %r caused ImportError (%s)",
                    mod_path,
                    cls_name,
                    missing_pkg,
                    e,
                )
                continue
    except Exception as e:
        logger.debug("Failed to attach load_weights_from_ray_registry to TPUWorker: %s", e)

    try:
        import verl.checkpoint_engine.base as ckpt_base
        from verl.plugin.platform import get_platform

        if not getattr(ckpt_base, "_verl_tpu_ckpt_patched", False):
            _orig_worker_init = ckpt_base.CheckpointEngineWorker.__init__
            _orig_mgr_update = ckpt_base.CheckpointEngineManager.update_weights

            def _patched_worker_init(self, rollout_config, model_config, server_adapter=None, *args, **kwargs):
                if get_platform().device_name == "tpu":
                    super(ckpt_base.CheckpointEngineWorker, self).__init__()
                    self.rollout_config = rollout_config
                    self.model_config = model_config
                    self.checkpoint_engine = None
                    self.server_adapter = None
                    self.replica_rank = kwargs.get("replica_rank", 0)
                    self.extra_rollout_args = args
                    self.extra_rollout_kwargs = kwargs
                    return
                _orig_worker_init(self, rollout_config, model_config, server_adapter, *args, **kwargs)

            @ckpt_base.auto_await
            async def _patched_mgr_update(self, global_steps: int | None = None):
                if self.backend == "tpu":
                    return await update_tpu_weights(self, global_steps=global_steps)
                res = _orig_mgr_update(self, global_steps=global_steps)
                if inspect.isawaitable(res):
                    return await res
                return res

            ckpt_base.CheckpointEngineWorker.__init__ = _patched_worker_init
            ckpt_base.CheckpointEngineManager.update_weights = _patched_mgr_update
            ckpt_base._verl_tpu_ckpt_patched = True
    except Exception as e:
        logger.debug("Failed to patch CheckpointEngineWorker/Manager for TPU: %s", e)


apply_tpu_checkpoint_engine_hooks()
