# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Direct NeuronTorchTitan adapter for verl's actor and reference workers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from uuid import uuid4

import torch
from torch.distributed.tensor import DTensor

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig, TorchtitanOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.torchtitan import transformer_impl as titan_impl
from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngineWithLMHead
from verl.workers.engine.utils import detach_tree

from .batch import host_jagged, padded_inputs
from .checkpoint import NeuronCheckpointManager
from .compat import configure_torch_compile_for_neuron
from .config import NeuronTorchtitanEngineConfig
from .tensor_parallel import token_statistics

_actor_engine: TorchTitanNeuronEngineWithLMHead | None = None


def actor_engine() -> TorchTitanNeuronEngineWithLMHead:
    if _actor_engine is None:
        raise RuntimeError("The Neuron actor must initialize before its checkpoint engine")
    return _actor_engine


@contextmanager
def native_config_overrides(hf_config, optimizer_config=None, parallel_dims=None):
    """Scope Neuron defaults to construction of this worker's TorchTitan trainer."""
    from torchtitan.models import qwen3

    original_registry = qwen3.model_registry

    def model_registry(*args, **kwargs):
        spec = original_registry(*args, **kwargs)
        spec.model.enable_weight_tying = bool(hf_config.tie_word_embeddings)
        return spec

    training_config = titan_impl.TrainingConfig

    def make_training_config(*args, **kwargs):
        kwargs["enable_cpu_offload"] = False
        if optimizer_config is not None:
            kwargs["max_norm"] = optimizer_config.clip_grad
            if optimizer_config.total_training_steps > 0:
                kwargs["steps"] = optimizer_config.total_training_steps
        return training_config(*args, **kwargs)

    overrides = [
        (titan_impl, "CheckpointManager", NeuronCheckpointManager),
        (titan_impl, "CompileConfig", partial(titan_impl.CompileConfig, backend="neuron")),
        (titan_impl, "ParallelismConfig", partial(titan_impl.ParallelismConfig, enable_sequence_parallel=False)),
        (titan_impl, "TrainingConfig", make_training_config),
        (qwen3, "model_registry", model_registry),
        (
            titan_impl.OptimizersContainer,
            "Config",
            partial(titan_impl.OptimizersContainer.Config, implementation="for-loop"),
        ),
    ]
    if parallel_dims is not None:
        overrides.append((titan_impl.Trainer, "init_distributed", lambda trainer: parallel_dims))
    saved = []
    try:
        for owner, name, replacement in overrides:
            saved.append((owner, name, getattr(owner, name)))
            setattr(owner, name, replacement)
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


@EngineRegistry.register(model_type="language_model", backend="torchtitan", device="neuron", vendor="aws")
class TorchTitanNeuronEngineWithLMHead(TorchTitanEngineWithLMHead):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: NeuronTorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        if not isinstance(engine_config, NeuronTorchtitanEngineConfig):
            raise ValueError("Use the NeuronTorchtitanEngineConfig overlay for actor and reference engines")
        if model_config.hf_config.model_type != "qwen3":
            raise ValueError("Trainium v1 supports dense Qwen3 only")
        if model_config.use_remove_padding:
            raise ValueError("Trainium causal SDPA requires model.use_remove_padding=false")
        for field in ("num_attention_heads", "num_key_value_heads", "vocab_size", "intermediate_size"):
            if getattr(model_config.hf_config, field) % engine_config.tensor_parallel_size:
                raise ValueError(f"Trainium requires {field} divisible by tensor_parallel_size")
        if engine_config.use_torch_compile:
            configure_torch_compile_for_neuron(2 * model_config.hf_config.num_hidden_layers + 8)
        parallel_dims = None
        if engine_config.forward_only and _actor_engine is not None:
            dimensions = (
                "tensor_parallel_size",
                "data_parallel_shard_size",
                "data_parallel_replicate_size",
                "context_parallel_size",
                "expert_parallel_size",
                "pipeline_parallel_size",
                "spmd_backend",
            )
            if any(getattr(engine_config, name) != getattr(_actor_engine.engine_config, name) for name in dimensions):
                raise ValueError("A colocated Trainium reference must use the actor's parallel dimensions")
            parallel_dims = _actor_engine.parallel_dims
        with native_config_overrides(model_config.hf_config, optimizer_config, parallel_dims):
            super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        from .snapshots import install_snapshot_handlers

        install_snapshot_handlers()

    def _init_device_mesh(self):
        self.parallel_dims = self.trainer.parallel_dims
        self.device_mesh = self.parallel_dims.world_mesh
        torch.autograd.set_multithreading_enabled(False)
        if self.parallel_dims.tp == 1:
            self._data_parallel_group = torch.distributed.group.WORLD
        else:
            for tp_rank in range(self.parallel_dims.tp):
                ranks = list(range(tp_rank, torch.distributed.get_world_size(), self.parallel_dims.tp))
                group = torch.distributed.new_group(ranks, backend="cpu:gloo,neuron:neuron")
                if torch.distributed.get_rank() in ranks:
                    self._data_parallel_group = group

    def _get_data_parallel_mesh(self):
        return self.parallel_dims.get_optional_mesh("loss")

    def get_data_parallel_group(self):
        return self._data_parallel_group

    def initialize(self):
        global _actor_engine
        self.trainer.checkpointer.folder = str(Path(gettempdir()) / f"verl-neuron-initial-{uuid4().hex}")
        super().initialize()
        if self.engine_config.forward_only:
            self.trainer.optimizers = None
            self.trainer.lr_schedulers = None
            for module in self.module:
                module.requires_grad_(False)
        else:
            _actor_engine = self
        for state in self.checkpointer.states.values():
            if hasattr(state, "cached_state_dict"):
                state.cached_state_dict = {}

    def model_forward_step(self, *, inputs, extra_inputs=None, extra_kwargs=None):
        if len(self.module) != 1:
            raise ValueError("Trainium requires a single non-pipelined model part")
        return self.module[0](inputs, **(extra_inputs or {}), **(extra_kwargs or {}))

    def forward_step(self, micro_batch, loss_function, forward_only):
        pad_token_id = int(tu.get_non_tensor_data(micro_batch, "pad_token_id", 0))
        inputs, positions, labels, offsets = padded_inputs(
            micro_batch["input_ids"], self.engine_config.max_seq_len, pad_token_id
        )
        device = self.trainer.device
        inputs, positions, labels = inputs.to(device), positions.to(device), labels.to(device)
        with torch.autocast(device_type="neuron", dtype=torch.bfloat16):
            logits = self.model_forward_step(
                inputs=inputs, extra_inputs={"positions": positions}, extra_kwargs={"attention_masks": None}
            )
            temperature = micro_batch["temperature"]
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.to(device)
                if temperature.ndim == 1:
                    temperature = temperature[:, None, None]
            calculate_entropy = tu.get_non_tensor_data(micro_batch, "calculate_entropy", False)
            log_probs, entropy = token_statistics(logits, labels, temperature, calculate_entropy)
            model_output = {"log_probs": host_jagged(log_probs, offsets)}
            if calculate_entropy:
                model_output["entropy"] = host_jagged(entropy, offsets)
            if loss_function is None:
                if not forward_only:
                    raise ValueError("Training requires a loss function")
                loss, metrics = torch.zeros(()), {}
            else:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
        return loss, {"model_output": detach_tree(model_output), "loss": loss.detach().item(), "metrics": metrics}

    def hf_parameters(self):
        params = {}
        for module in self.module:
            params.update(module.state_dict())
        converted = self._to_hf_named_params(params)
        if self.model_config.hf_config.tie_word_embeddings:
            converted["lm_head.weight"] = converted["model.embed_tokens.weight"]
        return dict(sorted(converted.items()))

    def weight_sync_metadata(self):
        return [(name, tuple(value.shape), "bfloat16") for name, value in self.hf_parameters().items()]

    def get_per_tensor_param(self, **kwargs):
        cpu_threshold = self.engine_config.weight_sync_cpu_gather_threshold_bytes

        def tensors():
            for name, value in self.hf_parameters().items():
                # Rollout consumes BF16. Cast local shards before gathering so
                # large FP32 embeddings do not exceed Neuron collective limits.
                value = value.detach().to(dtype=torch.bfloat16)
                if isinstance(value, DTensor):
                    if cpu_threshold and value.numel() * value.element_size() >= cpu_threshold:
                        # Large Qwen3-14B exports hit a native collective DMA
                        # alignment failure. Keep the full gather on the host;
                        # iter_weight_buckets copies bounded slices to Neuron.
                        value = value.to("cpu").full_tensor()
                    else:
                        value = value.full_tensor()
                yield name, value.contiguous()

        return tensors(), None

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        if hdfs_path:
            raise NotImplementedError("Trainium checkpoints require a shared local filesystem")
        Path(local_path).mkdir(parents=True, exist_ok=True)
        self.checkpointer.folder = str(local_path)
        self.checkpointer.interval = 1
        self.trainer.step = global_step
        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep
        if not self.checkpointer.save(curr_step=global_step):
            raise RuntimeError("TorchTitan did not write the requested checkpoint")
        torch.distributed.barrier()
        # verl gives every update a different actor directory; TorchTitan's
        # retention only scans step-* children of its current directory.
        paths = getattr(self, "_saved_checkpoint_paths", [])
        path = str(Path(local_path).resolve())
        paths = [previous for previous in paths if previous != path] + [path]
        if max_ckpt_to_keep is not None and max_ckpt_to_keep > 0 and len(paths) > max_ckpt_to_keep:
            if torch.distributed.get_rank() == 0:
                for previous in paths[:-max_ckpt_to_keep]:
                    rmtree(previous)
            paths = paths[-max_ckpt_to_keep:]
        self._saved_checkpoint_paths = paths

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False, **kwargs):
        if hdfs_path:
            raise NotImplementedError("Trainium checkpoints require a shared local filesystem")
        if not Path(local_path).is_dir():
            raise FileNotFoundError(local_path)
        self.checkpointer.folder = str(local_path)
        if not self.checkpointer.load(step=-1):
            raise RuntimeError(f"No complete TorchTitan checkpoint in {local_path}")
        torch.distributed.barrier()

    def optimizer_step(self):
        from torchtitan.distributed import utils

        norm = utils.clip_grad_norm_(
            [parameter for module in self.module for parameter in module.parameters()],
            self.config.training.max_norm,
            foreach=False,
            pp_mesh=None,
            ep_enabled=False,
        )
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Trainium actor gradient norm is not finite")
        self.optimizer.step()
        return norm.item()
