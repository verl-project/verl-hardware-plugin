# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Standalone TorchTitan training engine for Google TPU (``torch_tpu``) devices.

Registers ``TorchTitanTPUEngineWithLMHead`` with verl's ``EngineRegistry`` under
``(model_type="language_model", backend="torchtitan", device="tpu", vendor="google")``.

This module implements the TPU TorchTitan engine directly on top of ``verl.workers.engine.base.BaseEngine``
without importing ``verl.workers.engine.torchtitan``, keeping the TPU TorchTitan backend completely
decoupled from core verl's GPU TorchTitan engine:
- ``spmd_backend="default"`` and ``CompileConfig(backend="tpu")`` with static-shape Dynamo settings
- ``TPUSplashAttention`` Pallas kernel for packed-document causal attention without dense 4D masks
- ``SimpleFSDP`` DTensor parameter sharding with ``spmd_safe`` compiled TransformerBlocks and cross-rank shape alignment
- ``DEFER_AND_FUSE`` eager execution mode for ops outside compiled blocks
- ``OptimizersContainer.Config(implementation="fused")`` (``VERL_TPU_OPTIM_IMPL`` overrides) and ``foreach=False``
  grad norm clipping
- Host-side global token counting (avoiding eager device collectives inside the XLA SPMD trace)
- Graph-break loss synchronization between forward and backward
- Sequence-length bucketing for both packed (``use_remove_padding=True``) and padded inputs
- Plugin-local adapters preserving gradients for bucket-padded SFT and RL loss inputs
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_optimizer,
    offload_fsdp_optimizer,
)
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import (
    detach_tree,
    enable_full_determinism,
    postprocess_batch_func,
    prepare_micro_batches,
)
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    align_micro_batch_shapes_across_ranks,
    bucket_length,
    compute_global_batch_num_tokens,
    configure_torch_compile_for_tpu,
    create_noop_dataloader_config,
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    fused_linear_logprobs_tpu,
    get_tpu_seq_bucket_size,
    get_tpu_simple_fsdp_parallelize_fn,
    install_tpu_attention,
    move_tpu_params_,
    pad_packed_inputs_for_tpu,
    prepare_tpu_loss_inputs,
    resolve_tpu_torchtitan_options,
    resolve_tpu_use_fused_linear_cross_entropy,
    tpu_eager_mode_context,
    tpu_synchronize,
    unwrap_metadata,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _to_jagged(padded: torch.Tensor, cu_seqlens: torch.Tensor, orig_seq_len: Optional[int]) -> torch.Tensor:
    """Jagged view of a per-token output; for bucket-padded outputs a CPU view that carries the padded device tensor.

    The loss reads the fixed-shape device tensor (``TPU_PADDED_VALUES_ATTR``), so no new XLA shape per micro-batch.
    """
    if orig_seq_len is None:
        return torch.nested.nested_tensor_from_jagged(padded, cu_seqlens)
    jagged = torch.nested.nested_tensor_from_jagged(padded.detach().cpu()[:orig_seq_len], cu_seqlens.detach().cpu())
    setattr(jagged, TPU_PADDED_VALUES_ATTR, padded)
    return jagged


class TPUEngineEvalModeCtx(BaseEngineCtx):
    """Eval-mode context manager that skips ``reshard()`` on non-``FSDPModule`` modules (e.g. ``SimpleFSDP``)."""

    def __init__(self, engine: TorchTitanTPUEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanTPUEngine)

        # Reshard the FSDP2 modules. SimpleFSDP modules are not FSDPModule and keep no unsharded copy.
        if self.engine.engine_config.data_parallel_shard_size > 1:
            for module in self.engine.module:
                for submodule in module.modules():
                    if isinstance(submodule, FSDPModule):
                        submodule.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class TPUEngineTrainModeCtx(BaseEngineCtx):
    """Train-mode context manager for ``TorchTitanTPUEngine``."""

    def __init__(self, engine: TorchTitanTPUEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanTPUEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


class TorchTitanTPUEngine(BaseEngine):
    """Standalone TorchTitan engine for Google TPU with ``tpu_dist`` / PJRT SPMD execution."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanTPUEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2 or SimpleFSDP).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        from torchtitan.components.checkpointer import CheckpointManager
        from torchtitan.components.loss import CrossEntropyLoss
        from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer, ParamGroupConfig
        from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
        from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
        from torchtitan.trainer import Trainer

        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        if engine_config.context_parallel_size > 1:
            raise NotImplementedError("context parallelism is not supported by the TPU TorchTitan engine")
        if engine_config.tensor_parallel_size > 1:
            logger.warning(
                "tensor_parallel_size=%d is not properly tested on TPU; try "
                "tensor_parallel_size=1 for better performance on TPU.",
                engine_config.tensor_parallel_size,
            )

        (
            self._use_splash_attention,
            self._use_simple_fsdp,
            self._tpu_eager_mode,
        ) = resolve_tpu_torchtitan_options(engine_config)
        self._use_fused_linear_cross_entropy = resolve_tpu_use_fused_linear_cross_entropy(engine_config)

        if engine_config.use_torch_compile:
            configure_torch_compile_for_tpu()

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)

        # Get ModelSpec from model registry
        model_module = importlib.import_module(f"torchtitan.models.{torchtitan_name}")
        model_spec = model_module.model_registry(
            torchtitan_flavor,
            attn_backend=self.engine_config.attn_type,
            seq_len=self.engine_config.max_seq_len,
        )
        if self._use_simple_fsdp:
            model_spec = dataclasses.replace(
                model_spec,
                parallelize_fn=get_tpu_simple_fsdp_parallelize_fn(torchtitan_name),
            )

        # torch_tpu's aten::_fused_adamw_ updates params, exp_avg and exp_avg_sq in place (buffer donation): no
        # param-sized temporary as in the foreach path (+4.2 GB/chip for Qwen3-4B at FSDP=4) and ~2x faster.
        optim_impl = os.environ.get("VERL_TPU_OPTIM_IMPL", "fused")
        optimizer = OptimizersContainer.Config(
            implementation=optim_impl,
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=self.optimizer_config.name,
                    optimizer_kwargs={
                        "lr": self.optimizer_config.lr,
                        "eps": self.optimizer_config.eps,
                        "betas": (self.optimizer_config.betas[0], self.optimizer_config.betas[1]),
                        "weight_decay": self.optimizer_config.weight_decay,
                    },
                )
            ],
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            spmd_backend=self.engine_config.spmd_backend,
            use_simple_fsdp=self._use_simple_fsdp,
        )
        # interval=1: save whenever verl calls save_checkpoint (TorchTitan's default 500 skips the other steps).
        checkpoint = CheckpointManager.Config(
            enable=True, interval=1, initial_load_in_hf=True, initial_load_path=model_config.path
        )
        compile_config = CompileConfig(enable=self.engine_config.use_torch_compile, backend="tpu")

        # On TPU, FSDP2 CPUOffloadPolicy is incompatible with torch_tpu SPMD / PJRT sharding,
        # and CUDA graphs are not applicable.
        training_kwargs: dict[str, Any] = {
            "enable_cpu_offload": False,
            "disable_cuda_graphs": True,
            "max_norm": self.optimizer_config.clip_grad,
        }
        if self.engine_config.forward_only:
            engine_dtype = getattr(self.engine_config, "dtype", "bfloat16")
            if engine_dtype in ("bfloat16", "float32"):
                training_kwargs["dtype"] = engine_dtype
        if self.engine_config.max_seq_len is not None:
            training_kwargs["max_context_length"] = self.engine_config.max_seq_len
        training = TrainingConfig(**training_kwargs)

        # Pad positions run 0..pad_size-1 (< bucket) and index the RoPE cache sized by max_context_length.
        if self.engine_config.pad_to_length and self.engine_config.pad_to_length_bucket > training.max_context_length:
            raise ValueError(
                f"pad_to_length_bucket ({self.engine_config.pad_to_length_bucket}) must not exceed the RoPE "
                f"cache length ({training.max_context_length}); raise engine.max_seq_len or lower the bucket"
            )

        # Activation checkpointing mode. Note: under spmd_backend="spmd_types" with
        # eager execution (use_torch_compile=False), selective/full AC recompute runs
        # on the autograd backward thread where the thread-local SPMD mesh is inactive,
        # so spmd.assert_type() raises "no current mesh". Set activation_checkpoint="none"
        # (or enable torch.compile, which recomputes in-graph) in that configuration.
        activation_checkpoint = {
            "selective": SelectiveAC.Config,
            "full": FullAC.Config,
            "none": lambda: None,
        }[self.engine_config.activation_checkpoint]()

        # Construct Torchtitan's Trainer.Config
        self.config = Trainer.Config(
            model_spec=model_spec,
            hf_assets_path=self.model_config.path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile_config,
            training=training,
            activation_checkpoint=activation_checkpoint,
            # Use a no-op dataloader since verl has its own data loading
            dataloader=create_noop_dataloader_config(),
            # Provide a concrete loss so Trainer.__init__ can build it;
            # verl uses its own loss function and ignores this one.
            loss=CrossEntropyLoss.Config(),
        )
        if self._use_fused_linear_cross_entropy:
            from torchtitan.experiments.tpu.qwen3.config_registry import _FUSED_LOSS_OVERRIDE

            if _FUSED_LOSS_OVERRIDE not in self.config.override.imports:
                self.config.override.imports.append(_FUSED_LOSS_OVERRIDE)
        self.trainer = Trainer(self.config)
        self.parallel_dims = self.trainer.parallel_dims
        # Like torchtitan's init_distributed (which verl bypasses): no autograd multithreading, so backward-thread
        # activation-checkpoint recompute can access the thread-local SPMD mesh / process groups.
        torch.autograd.set_multithreading_enabled(False)

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        model_parts = self.trainer.model_parts
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        install_tpu_attention(model_parts, use_splash=self._use_splash_attention)

        if engine_config.use_torch_compile:
            n_compiled = sum(
                getattr(m, "_compiled_call_impl", None) is not None
                for part in model_parts
                if isinstance(part, torch.nn.Module)
                for m in part.modules()
            )
            logger.warning("torch.compile(backend='tpu') enabled on %d modules", n_compiled)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params. ``ref.torchtitan.param_offload=True`` keeps the frozen reference model in host
        # memory outside compute_ref_log_prob: frees its bf16 shard of HBM, costs one host-to-TPU copy per step.
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._simple_fsdp_loss_scale = 1.0 / self.get_data_parallel_size() if self._use_simple_fsdp else None

        entropy_fn = (
            verl_F.entropy_from_logits_with_chunking
            if self.engine_config.entropy_from_logits_with_chunking
            else verl_F.entropy_from_logits
        )
        self.compute_entropy_from_logits = (
            torch.compile(entropy_fn, backend="tpu", dynamic=False)
            if self.engine_config.use_torch_compile
            else entropy_fn
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logger.info(
            "TorchTitanTPUEngine initialized (compile=%s, splash=%s, simple_fsdp=%s, eager_mode=%s)",
            engine_config.use_torch_compile,
            self._use_splash_attention,
            self._use_simple_fsdp,
            self._tpu_eager_mode,
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self) -> bool:
        """Whether the current rank is the first rank in the model-parallel group that holds model outputs."""
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def _drop_checkpoint_state_dict_cache(self) -> None:
        """Free torchtitan ``ModelWrapper.cached_state_dict`` after the initial HF load."""
        from torchtitan.components.checkpointer import MODEL

        self.checkpointer.states[MODEL].cached_state_dict = {}

    def initialize(self):
        """Initialize model/checkpointer and release unused optimizer state on forward-only reference workers."""
        self.module = self.trainer.model_parts
        self.checkpointer = self.trainer.checkpointer
        # load initial HF weights
        self.checkpointer.load()
        tpu_synchronize(wait=True)

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        self._drop_checkpoint_state_dict_cache()

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return TPUEngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return TPUEngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        mesh = self._get_data_parallel_mesh()
        if mesh is not None:
            return mesh.get_group()
        # If world_size == dp_size (e.g. single GPU, or all ranks are DP),
        # return WORLD so that collective ops in _postprocess_output
        # (allgather_dict_into_dict, all_reduce) still run and produce the
        # correct metric aggregation format.
        if torch.distributed.get_world_size() == self.get_data_parallel_size():
            return torch.distributed.group.WORLD
        return None

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        mesh = self.parallel_dims.get_optional_mesh("cp")
        return None if mesh is None else mesh.get_group()

    def _get_data_parallel_mesh(self):
        """Data-parallel mesh (dp_replicate x dp_shard). Excludes cp: CP ranks share the same samples."""
        return self.parallel_dims.get_optional_mesh("batch")

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False):
        """Perform forward and optionally backward pass with host-side token counting and XLA graph break."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        dp_group = self.get_data_parallel_group()
        batch_num_tokens = compute_global_batch_num_tokens(data, dp_group, self.engine_config.tensor_parallel_size)
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens)
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=dp_group,
            same_micro_num_in_dp=True,
        )
        if self._use_simple_fsdp:
            align_micro_batch_shapes_across_ranks(
                micro_batches, bucket_size=get_tpu_seq_bucket_size(self.engine_config)
            )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        with tpu_eager_mode_context(self._tpu_eager_mode):
            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                with self.trainer.train_context(), ctx, torch.profiler.record_function(f"micro_batch{micro_batch_idx}"):
                    loss, output = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only
                    )
                    if not forward_only:
                        if self._simple_fsdp_loss_scale is not None:
                            loss = loss * self._simple_fsdp_loss_scale
                        tpu_synchronize(loss, wait=False)
                        loss.backward()
                        del loss
                        tpu_synchronize(wait=True)
                    else:
                        del loss
                output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the trainer model without backward."""
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )
        # Non-PP forward. train_context (SPMD mesh) is set by the caller.
        assert len(model_parts) == 1
        pred = model_parts[0](inputs, **(extra_inputs or {}), **(extra_kwargs or {}))
        if isinstance(pred, tuple):
            return tuple(p.full_tensor() if isinstance(p, DTensor) else p for p in pred)
        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        return pred

    def forward_step(self, micro_batch: TensorDict, loss_function: Callable, forward_only: bool):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Clip gradients using the per-tensor norm path (``foreach=False``) and step the optimizer."""
        from torchtitan.distributed import utils as dist_utils

        with tpu_eager_mode_context(self._tpu_eager_mode):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.module for p in m.parameters()],
                self.config.training.max_norm,
                foreach=False,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )

            if not torch.isfinite(grad_norm):
                logger.warning("grad_norm is not finite (%s); skipping this optimizer step", grad_norm)
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

            return grad_norm.item()

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or accelerator."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        with tpu_eager_mode_context(self._tpu_eager_mode):
            if device == device_name:
                target_device = torch.device(f"{device_name}:{get_device_id()}")
                if model:
                    for module in self.module:
                        move_tpu_params_(module, target_device, include_grad=grad)
                if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                    load_fsdp_optimizer(self.optimizer, target_device)
            elif device == "cpu":
                if model:
                    for module in self.module:
                        move_tpu_params_(
                            module, "cpu", cache_cpu_copy=bool(self.engine_config.forward_only), include_grad=grad
                        )
                if not self.engine_config.forward_only and optimizer and self.optimizer is not None:
                    offload_fsdp_optimizer(self.optimizer)
                if model or optimizer:
                    gc.collect()
                    tpu_synchronize(wait=True)
            else:
                raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save checkpoint."""
        if self._is_offload_param:
            self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)

        torch.distributed.barrier()
        if self._is_offload_param:
            self.to(device="cpu", model=True, optimizer=False, grad=False)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        if self._is_offload_param:
            self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        # Extract step number from path (verl uses global_step_N format)
        match = re.search(r"global_step_(\d+)", local_path)
        if match:
            step = int(match.group(1))
            self.checkpointer.load(step=step)
        else:
            # Fallback to latest
            self.checkpointer.load(step=-1)
        for module in self.module:
            for p in module.parameters():
                if hasattr(p, "_tpu_cpu_offload_tensor"):
                    delattr(p, "_tpu_cpu_offload_tensor")
        tpu_synchronize(wait=True)

        torch.distributed.barrier()
        if self._is_offload_param:
            self.to(device="cpu", model=True, optimizer=False, grad=False)

        if self._is_offload_optimizer and self.optimizer is not None:
            offload_fsdp_optimizer(self.optimizer)

    def _to_hf_named_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Re-key a torchtitan state dict to HuggingFace names, values untouched."""
        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
        sd_adapter = self.checkpointer.sd_adapter
        if sd_adapter is not None:
            params = sd_adapter.to_hf(params)

        # When weight tying is enabled, the sd_adapter skips lm_head.weight during
        # to_hf() conversion (since it's the same tensor as embed_tokens.weight in
        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
        # add it back as a reference to embed_tokens.weight.
        if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
            params["lm_head.weight"] = params["model.embed_tokens.weight"]
        return params

    def _expert_stack_slots(self, name: str, param: Any) -> Optional[list[tuple[str, tuple]]]:
        """If ``name`` is a fused expert stack, enumerate ALL its experts' HF tensors, else ``None``."""
        sd_adapter = self.checkpointer.sd_adapter
        from_hf_map = getattr(sd_adapter, "from_hf_map", None) if sd_adapter is not None else None
        if not from_hf_map or param.ndim < 2:
            return None
        abstract = re.sub(r"(\d+)", "{}", name, count=1)
        hf_template = next((hf for hf, titan in from_hf_map.items() if titan == abstract), None)
        if hf_template is None or hf_template.count("{}") != 2:
            return None
        match = re.search(r"\d+", name)
        if match is None:
            return None
        layer_id = match.group(0)
        slot_shape = tuple(int(d) for d in param.shape[1:])
        return [(hf_template.format(layer_id, e), slot_shape) for e in range(int(param.shape[0]))]

    def get_per_tensor_param(self, **kwargs):
        """Yield HF-named parameters cast to ``bfloat16`` before ``DTensor.full_tensor()`` all-gather."""
        if self._is_offload_param:
            self.to(device=get_device_name(), model=True, optimizer=False, grad=False)

        params = {}
        for module in self.module:
            params.update(module.state_dict())

        if self._is_offload_param:
            self.to(device="cpu", model=True, optimizer=False, grad=False)

        stacks = {}
        for name, param in params.items():
            slots = self._expert_stack_slots(name, param)
            if slots is not None:
                stacks[name] = slots
        expert_stacks = [(params[name], slots) for name, slots in stacks.items()]
        dense = self._to_hf_named_params({k: v for k, v in params.items() if k not in stacks})
        params.clear()

        device = get_device_id()

        def _gen():
            gathered_tied: dict[int, torch.Tensor] = {}
            for name in list(dense.keys()):
                param = dense.pop(name)
                if isinstance(param, DTensor):
                    pid = id(param)
                    full = gathered_tied.get(pid)
                    if full is None:
                        full = param.to(device, dtype=torch.bfloat16, non_blocking=True).full_tensor()
                        if name in ("model.embed_tokens.weight", "lm_head.weight"):
                            gathered_tied[pid] = full
                    yield name, full
                    del full
                else:
                    yield name, param.to(torch.bfloat16, non_blocking=True)
                del param
            gathered_tied.clear()
            for stack, slots in expert_stacks:
                full = stack.to(device, non_blocking=True)
                if isinstance(full, DTensor):
                    full = full.full_tensor()
                full = full.to(torch.bfloat16, non_blocking=True)
                for e, (hf_name, _shape) in enumerate(slots):
                    yield hf_name, full[e].clone()
                del full

        return _gen(), None


@EngineRegistry.register(
    model_type="language_model",
    backend="torchtitan",
    device="tpu",
    vendor="google",
)
class TorchTitanTPUEngineWithLMHead(TorchTitanTPUEngine):
    """Standalone TorchTitan engine for language models with LM head on Google TPU."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Prepare bucket-padded, contiguous TPU inputs for TorchTitan forward execution."""
        use_remove_padding = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )
        pad_mode = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        bucket_size = get_tpu_seq_bucket_size(self.engine_config)
        if use_remove_padding:
            input_ids, position_ids, labels, orig_seq_len = pad_packed_inputs_for_tpu(
                input_ids=input_ids,
                position_ids=position_ids,
                micro_batch=micro_batch,
                device=get_device_id(),
                bucket_size=bucket_size,
            )
            output_args["orig_seq_len"] = orig_seq_len
        else:
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = bucket_length(int(max(input_ids.offsets().diff())), bucket_size=bucket_size)

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0).to(get_device_id())
            input_ids = (
                torch.nested.to_padded_tensor(input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len))
                .reshape(-1)
                .to(get_device_id())
            )

            if position_ids.dim() == 3:
                position_ids = (
                    torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, 4, max_seq_len))
                    .transpose(0, 1)
                    .reshape(4, -1)
                    .to(get_device_id())
                )
            else:
                position_ids = (
                    torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, max_seq_len))
                    .reshape(-1)
                    .to(get_device_id())
                )
            output_args["batch_size"] = batch_size
            output_args["max_seq_len"] = max_seq_len

        # Rows restart at position 0, so attention derives document (and row) boundaries from the positions.
        extra_inputs = {"positions": position_ids, **multi_modal_inputs}
        extra_inputs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in extra_inputs.items()}
        output_args["labels"] = labels
        return input_ids.contiguous(), extra_inputs, {}, output_args

    def prepare_model_outputs(self, logits: torch.Tensor, output_args: dict, micro_batch: TensorDict):
        """Convert TPU model logits into jagged log_probs/entropy while retaining bucket-padded device buffers."""
        use_remove_padding = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        )
        pad_mode = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = unwrap_metadata(micro_batch["temperature"])
        calculate_entropy = unwrap_metadata(
            tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        )
        labels = output_args["labels"]
        model_output: dict[str, torch.Tensor] = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        fused_log_probs = None
        if isinstance(logits, tuple):
            hidden_states, lm_head_weight_t = logits
            if use_remove_padding and not self.parallel_dims.tp_enabled:
                if labels.dim() > 1:
                    labels = labels.squeeze(0)
                hidden_rmpad = hidden_states.squeeze(0) if hidden_states.dim() > 2 else hidden_states
                fused_log_probs = fused_linear_logprobs_tpu(
                    hidden_rmpad,
                    labels,
                    lm_head_weight_t,
                    temperature=temperature,
                )
                if not calculate_entropy:
                    model_output["log_probs"] = _to_jagged(fused_log_probs, cu_seqlens, output_args.get("orig_seq_len"))
                    return model_output
            # Entropy needs the logits. The log-probs still come from the fused kernel, so the reference, old-log-prob
            # and update passes all compute them the same way (an on-policy PPO ratio is exactly 1).
            logits = hidden_states @ lm_head_weight_t

        if use_remove_padding:
            if labels.dim() > 1:
                labels = labels.squeeze(0)
            logits_rmpad = (logits.squeeze(0) if logits.dim() > 2 else logits) / temperature

            # Like verl's GPU path (flash-attn cross entropy, CUDA autocast), take the log-softmax and entropy of
            # the bf16 logits in fp32; torch_tpu's autocast would leave them in bf16.
            logits_rmpad = logits_rmpad.float()
            if fused_log_probs is not None:
                log_probs = fused_log_probs
            else:
                log_probs = logprobs_from_logits_naive(logits_rmpad, labels)

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    if self.engine_config.entropy_from_logits_with_chunking:
                        entropy_rmpad = self.compute_entropy_from_logits(
                            logits_rmpad,
                            chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                        )
                    else:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            orig_seq_len = output_args.get("orig_seq_len")
            log_probs = _to_jagged(log_probs.squeeze(0) if log_probs.dim() > 1 else log_probs, cu_seqlens, orig_seq_len)
            if calculate_entropy:
                entropy = _to_jagged(entropy_rmpad, cu_seqlens, orig_seq_len)
        else:
            if logits.dim() == 2 and "batch_size" in output_args and "max_seq_len" in output_args:
                logits = logits.view(output_args["batch_size"], output_args["max_seq_len"], -1)
            logits = (logits / temperature).float()  # fp32 log-softmax and entropy, as above
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits_naive(logits_rmpad, output_args["labels"])
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function: Callable, forward_only: bool):
        """Run one micro-batch forward step without eagerly transferring the full CPU TensorDict to TPU."""
        device_name = get_device_name()
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(inputs=input_ids, extra_inputs=extra_inputs, extra_kwargs=extra_kwargs)

            model_output = self.prepare_model_outputs(logits=logits, output_args=output_args, micro_batch=micro_batch)

            if loss_function is not None:
                loss_model_output, loss_data = prepare_tpu_loss_inputs(model_output, micro_batch)
                loss, metrics = loss_function(
                    model_output=loss_model_output, data=loss_data, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "loss_function must be provided when not forward_only"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            for value in model_output.values():
                if hasattr(value, TPU_PADDED_VALUES_ATTR):
                    delattr(value, TPU_PADDED_VALUES_ATTR)

            output = {
                "model_output": detach_tree(model_output),
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

        return loss, output
