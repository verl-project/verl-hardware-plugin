# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Names and shared hooks/helpers of the TPU weight sync backends
(``actor_rollout_ref.rollout.checkpoint_engine.backend``).

* ``ray-plasma-store``: trainer rank 0 puts the full weights into the Ray object store (plasma) and every
  rollout worker pulls them from there (``RayPlasmaCheckpointEngine`` in ``ray_checkpoint_engine.py``).
* ``tpu-sync``: the trainer chips send the weights peer to peer to the rollout chips with tpu-sync
 ; each rollout chip receives only its tensor-parallel shard (``TPUSyncCheckpointEngine`` in
  ``tpu_sync_checkpoint_engine.py``).

The former name ``tpu`` is still accepted and emit a ``FutureWarning``.
"""

import inspect
import logging
import warnings
from typing import Optional

import torch

logger = logging.getLogger(__name__)

RAY_PLASMA_STORE_BACKEND = "ray-plasma-store"
TPU_SYNC_BACKEND = "tpu-sync"

# Former backend names, mapped to the current ones.
DEPRECATED_BACKEND_NAMES = {
    "tpu": RAY_PLASMA_STORE_BACKEND,
}

_warned_deprecated_names: set[str] = set()


def canonical_backend(backend: Optional[str]) -> Optional[str]:
    """Returns the current name of a weight sync backend.

    Args:
        backend: The configured ``checkpoint_engine.backend``.

    Returns:
        ``backend``, or its current name if it is a deprecated one (warned once per process).
    """
    if backend is None or backend not in DEPRECATED_BACKEND_NAMES:
        return backend
    current = DEPRECATED_BACKEND_NAMES[backend]
    if backend not in _warned_deprecated_names:
        _warned_deprecated_names.add(backend)
        # A warning rather than a log message: in verl's driver process the root logger is set to ERROR
        # (by model_hosting_container_standards, see docs/accelerators/tpu/weight_sync.md), which would hide it.
        warnings.warn(
            f"actor_rollout_ref.rollout.checkpoint_engine.backend={backend} is deprecated, use {current} instead.",
            FutureWarning,
            stacklevel=2,
        )
    return current


def resolve_model_key(key: str, model_sd: dict[str, torch.Tensor]) -> str:
    """Resolves a cleaned trainer key against vLLM's ``model_sd`` (handling optional ``model.`` prefixes)."""
    if key in model_sd:
        return key
    if key.startswith("model.") and key[6:] in model_sd:
        return key[6:]
    prefixed = f"model.{key}"
    if prefixed in model_sd:
        return prefixed
    return key


def get_parent_module(target_key: str, module_dict: dict[str, torch.nn.Module]) -> Optional[torch.nn.Module]:
    """Returns the owning ``torch.nn.Module`` for ``target_key``, or ``None`` if not found."""
    parent_name = target_key.rsplit(".", 1)[0] if "." in target_key else ""
    if parent_name in module_dict:
        return module_dict[parent_name]
    if parent_name.startswith("model.") and parent_name[6:] in module_dict:
        return module_dict[parent_name[6:]]
    prefixed = f"model.{parent_name}"
    return module_dict.get(prefixed)


def to_target_layout(tensor: torch.Tensor, target_local: torch.Tensor, flipped: bool) -> torch.Tensor:
    """Matches ``vllm-torchtpu``'s ``(n_in, n_out)`` 2D weight layout when required."""
    if tensor.ndim == 2 and (flipped or (tensor.shape != target_local.shape and tensor.T.shape == target_local.shape)):
        return tensor.transpose(0, 1).contiguous()
    return tensor.contiguous()


_resolve_model_key = resolve_model_key
_get_parent_module = get_parent_module
_to_target_layout = to_target_layout


def apply_checkpoint_engine_manager_hooks() -> None:
    """Patches ``CheckpointEngineWorker`` and ``CheckpointEngineManager`` to route TPU weight-sync backends."""
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
                backend = canonical_backend(self.backend)
                if backend == RAY_PLASMA_STORE_BACKEND:
                    from verl_hardware_plugin.accelerators.tpu.weight_sync import ray_checkpoint_engine

                    return await ray_checkpoint_engine.update_ray_weights(self, global_steps=global_steps)
                if backend == TPU_SYNC_BACKEND:
                    from verl_hardware_plugin.accelerators.tpu.weight_sync import tpu_sync_checkpoint_engine

                    return await tpu_sync_checkpoint_engine.update_tpu_sync_weights(self, global_steps=global_steps)
                res = _orig_mgr_update(self, global_steps=global_steps)
                if inspect.isawaitable(res):
                    return await res
                return res

            ckpt_base.CheckpointEngineWorker.__init__ = _patched_worker_init
            ckpt_base.CheckpointEngineManager.update_weights = _patched_mgr_update
            ckpt_base._verl_tpu_ckpt_patched = True
    except Exception as e:
        logger.debug("Failed to patch CheckpointEngineWorker/Manager for TPU: %s", e)
