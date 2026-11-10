# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl engine for the ``maxtext`` strategy.

verl starts one worker per GPU. The workers form one multi-process JAX job, so every worker
receives the whole batch (data-parallel rank 0 of size 1), runs the same step, and only
process 0 returns outputs. Data parallelism lives inside the MaxText mesh.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry

from .batch import arrays_from_batch, maxtext_batch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _MaxTextEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.session = None
        self._own_group = None

    def initialize(self):
        from .runtime import MaxTextSession

        self._own_group = _single_rank_group()
        self.session = MaxTextSession(self.engine_config, self.model_config)
        install_detach_handlers()

    @property
    def module(self):
        """The JAX state is not a torch module."""
        return None

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def train_mode(self, **kwargs):
        return BaseEngineCtx(self, "train", **kwargs)

    def eval_mode(self, **kwargs):
        return BaseEngineCtx(self, "eval", **kwargs)

    def optimizer_zero_grad(self):
        return None

    def optimizer_step(self):
        return None

    def lr_scheduler_step(self):
        return None

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        return None

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        # Each worker already holds the whole batch; reductions over it must not add copies.
        return self._own_group

    def is_mp_src_rank_with_outputs(self):
        return torch.distributed.get_rank() == 0

    def train_batch(self, data, loss_function):
        from .runtime import actor_config_of

        self.session.set_actor_config(actor_config_of(loss_function))
        metrics = self.session.train(self._batch(data, with_targets=True))
        return {"loss": [metrics["loss"]], "metrics": {"grad_norm": metrics["grad_norm"]}, "model_output": {}}

    def infer_batch(self, data, loss_function=None):
        del loss_function
        arrays = arrays_from_batch(data)
        log_probs, entropy = self.session.log_probs_and_entropy(self._batch(data, with_targets=False, arrays=arrays))
        width = arrays["token_ids"].shape[1]
        mask = arrays["token_mask"]
        output = {
            "log_probs": _as_nested(torch.as_tensor(log_probs[:, :width], dtype=torch.float32), mask),
            "entropy": _as_nested(torch.as_tensor(entropy[:, :width], dtype=torch.float32), mask),
        }
        return {"loss": [0.0], "metrics": {}, "model_output": output}

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if forward_only:
            return self.infer_batch(data, loss_function)
        return self.train_batch(data, loss_function)

    def get_per_tensor_param(self, *args, **kwargs):
        """Every actor rank walks this generator; rank 0's tensors reach the rollout."""
        del args, kwargs
        return self._hf_tensors(), None

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        """Save the fp32 master weights as HF safetensors from process 0. Optimizer state is not saved."""
        del hdfs_path, kwargs
        import jax.numpy as jnp

        from .checkpoint import prune_checkpoints, write_hf_checkpoint

        started = time.time()
        tensors = self._hf_tensors(dtype=jnp.float32)
        if not self.is_mp_src_rank_with_outputs():
            for _ in tensors:
                pass
            return
        stats = write_hf_checkpoint(tensors, local_path, source_dir=self._hf_source_dir())
        # local_path is <root>/global_step_N/actor.
        root = os.path.dirname(os.path.dirname(local_path.rstrip("/")))
        removed = prune_checkpoints(root, int(global_step), max_ckpt_to_keep)
        logger.warning(
            "maxtext checkpoint step %s: %s tensors, %.1f GiB in %.0fs; removed %s",
            global_step,
            stats["tensors"],
            stats["bytes"] / 2**30,
            time.time() - started,
            removed,
        )

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        """Load weights saved by ``save_checkpoint``. Adam moments start from zero."""
        del hdfs_path, kwargs
        self.session.load_hf_weights(local_path)
        if del_local_after_load and self.is_mp_src_rank_with_outputs():
            import shutil

            shutil.rmtree(local_path, ignore_errors=True)

    def _hf_tensors(self, dtype=None):
        from .export import check_layout, iter_hf_from_global

        weights = self.session.stacked_weights()
        heads, kv_heads, head_dim = _attention_dims(self.model_config)
        try:
            check_layout(weights, heads, head_dim)
        except (KeyError, ValueError) as exc:
            sample = ", ".join(list(_paths(weights))[:12])
            raise RuntimeError(
                f"MaxText parameters are not in the scanned Qwen3-MoE layout ({exc!r}). First paths: {sample}"
            ) from exc
        return iter_hf_from_global(weights, heads, kv_heads, head_dim, self.session.mesh, dtype=dtype)

    def _hf_source_dir(self):
        for candidate in (
            getattr(self.engine_config, "hf_weights_path", None),
            getattr(self.model_config, "local_path", None),
            getattr(self.model_config, "path", None),
        ):
            if candidate and os.path.isdir(str(candidate)):
                return str(candidate)
        return None

    def _batch(self, data, with_targets: bool, arrays=None) -> dict:
        arrays = arrays_from_batch(data) if arrays is None else arrays
        batch_size, seq_len = self.session.batch_shape
        return maxtext_batch(
            arrays["token_ids"],
            arrays["token_mask"],
            arrays["response_mask"],
            old_log_probs=arrays["old_log_probs"] if with_targets else None,
            advantages=arrays["advantages"] if with_targets else None,
            batch_size=batch_size,
            seq_len=seq_len,
        )


def _attention_dims(model_config) -> tuple[int, int, int]:
    hf = model_config.hf_config
    heads = int(hf.num_attention_heads)
    kv_heads = int(getattr(hf, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(hf, "head_dim", None) or hf.hidden_size // heads)
    return heads, kv_heads, head_dim


def _single_rank_group():
    import torch.distributed as dist

    rank = dist.get_rank()
    groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
    return groups[rank]


def _as_nested(values: torch.Tensor, token_mask: np.ndarray) -> torch.Tensor:
    lengths = token_mask.sum(axis=1).astype(int)
    return torch.nested.as_nested_tensor([values[row, :n] for row, n in enumerate(lengths)], layout=torch.jagged)


def _paths(tree, prefix=""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _paths(value, f"{prefix}/{key}" if prefix else str(key))
    else:
        yield prefix


def install_detach_handlers() -> None:
    """Give the separated actor worker CPU-snapshot handlers for ``maxtext``.

    Separate-async snapshots pi_old every step, and restores it only when
    ``parameter_sync_step > 1``. With one sync per step pi_old is the live state, so the
    snapshot keeps nothing.
    """
    try:
        from verl.experimental.separation.engine_workers import DetachActorWorker
    except ImportError:
        return
    original = DetachActorWorker._get_strategy_handlers
    if getattr(original, "_maxtext_aware", False):
        return

    def save_nothing(_module):
        return None

    def restore_unsupported(_module, _saved):
        raise NotImplementedError(
            "The maxtext strategy keeps one policy copy; set trainer.v1.separate_async.parameter_sync_step=1."
        )

    def _get_strategy_handlers(self):
        if getattr(self, "_strategy_handlers", None) is None and self.config.actor.strategy == "maxtext":
            self._strategy_handlers = (save_nothing, restore_unsupported)
        return original(self)

    _get_strategy_handlers._maxtext_aware = True  # type: ignore[attr-defined]
    DetachActorWorker._get_strategy_handlers = _get_strategy_handlers


@EngineRegistry.register(model_type="language_model", backend="maxtext", device="cuda", vendor="amd")
class MaxTextEngineWithLMHead(_MaxTextEngine):
    """MaxText language-model engine."""


@EngineRegistry.register(model_type="value_model", backend="maxtext", device="cuda", vendor="amd")
class MaxTextEngineWithValueHead(_MaxTextEngine):
    """MaxText has no value head; GRPO does not build this worker."""

    def initialize(self):
        raise NotImplementedError("The maxtext strategy has no value head; run GRPO with the critic disabled")
