# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Build a MaxText train state and run one GRPO step per call.

Primus' trainer calls ``run()`` and only returns when pretraining ends. This stops after
``setup_train_loop`` and drives MaxText's own ``train_step`` with the cross-entropy loss
replaced by the clipped policy loss; the optimizer update stays MaxText's.

Every verl worker is one JAX process with one GPU. All processes receive the same batch and
call each method in lockstep, so every array here is a global array over the MaxText mesh.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np

from .loss import clip_from_actor, clipped_policy_loss
from .params import initialize_jax_distributed, load_maxtext_params, prepare_jax_env


class MaxTextSession:
    _batch_shape: tuple[int, int]
    _train_step: Any
    _logprob_step: Any
    _state: Any
    _graph: Any
    _data_sharding: Any

    def __init__(self, engine_config, model_config):
        self._model_config = model_config
        prepare_jax_env(engine_config)
        initialize_jax_distributed()
        self._init_train_state(load_maxtext_params(engine_config))
        self._clip = (0.2, 0.2, 3.0)
        self._compile()
        hf_path = engine_config.hf_weights_path
        if hf_path is None and engine_config.load_hf_weights:
            hf_path = getattr(model_config, "local_path", None) or getattr(model_config, "path", None)
        if hf_path:
            self.load_hf_weights(hf_path)

    @property
    def batch_shape(self) -> tuple[int, int]:
        return self._batch_shape

    @property
    def mesh(self):
        return self._device_mesh

    def set_actor_config(self, actor_config) -> None:
        if actor_config is None:
            return
        clip = clip_from_actor(actor_config)
        if clip != self._clip:
            self._clip = clip
            self._compile()

    def train(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        with self._mesh_context():
            self._state, metrics = self._train_step(self._state, self._to_global(batch))
        scalar = metrics["scalar"]
        # learning/grad_norm is measured after clipping and saturates at the threshold.
        grad_norm = scalar.get("learning/raw_grad_norm", scalar.get("learning/grad_norm"))
        if grad_norm is None:
            raise RuntimeError("MaxText did not return learning/grad_norm; leave optimizer_memory_host_offload off.")
        return {"loss": _scalar(scalar["learning/loss"]), "grad_norm": _scalar(grad_norm)}

    def log_probs_and_entropy(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        with self._mesh_context():
            values, entropy = self._logprob_step(self._state, self._to_global(batch))
        return _local(values), _local(entropy)

    def stacked_weights(self) -> dict:
        """Parameter pytree of global device arrays."""
        import jax
        from flax import nnx

        merged = nnx.merge(self._graph, self._state)
        params = nnx.state(merged.model, nnx.Param)
        return jax.tree.map(lambda v: getattr(v, "value", v), params.to_pure_dict())

    def load_hf_weights(self, path: str) -> None:
        """Replace the parameters with a Hugging Face checkpoint; each process reads its own shards.

        Optimizer state stays as initialised. The compiled steps are reused because the new
        arrays keep the shapes, dtypes and shardings.
        """
        import jax
        from flax import nnx

        from .hf_import import HFReader, load_stacked_from_hf

        full = nnx.merge(self._graph, self._state)
        params = nnx.state(full.model, nnx.Param)
        like = jax.tree.map(lambda v: getattr(v, "value", v), params.to_pure_dict())

        def place(old, read_block):
            return jax.make_array_from_callback(old.shape, old.sharding, read_block)

        nnx.replace_by_pure_dict(params, load_stacked_from_hf(HFReader(path), like, place))
        nnx.update(full.model, params)
        self._graph, self._state = nnx.split(full)

    def _mesh_context(self):
        import jax
        from flax.core.spmd import logical_axis_rules

        stack = contextlib.ExitStack()
        stack.enter_context(jax.set_mesh(self._device_mesh))
        stack.enter_context(logical_axis_rules(self._config.logical_axis_rules))
        return stack

    def _to_global(self, batch: dict[str, np.ndarray]) -> dict:
        import jax

        shard = self._data_sharding
        return {
            key: jax.make_array_from_callback(value.shape, shard, lambda index, v=value: v[index])
            for key, value in batch.items()
        }

    def _init_train_state(self, params: dict) -> None:
        from absl import flags
        from flax import nnx
        from maxtext.trainers.pre_train import train as train_module
        from maxtext.utils import sharding, train_utils
        from primus.backends.maxtext.argument_builder import export_params_to_yaml

        self._train_module = train_module
        # Grain reads absl flags; this process never goes through absl.app.run.
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        yaml_path = export_params_to_yaml(params)
        try:
            initialized = train_module.initialize([train_module.__name__, yaml_path])
        finally:
            with contextlib.suppress(OSError):
                os.unlink(yaml_path)
        config, recorder = initialized[0], initialized[1]
        (
            _init_rng,
            _checkpoint_manager,
            self._state_shardings,
            _model,
            self._device_mesh,
            _learning_rate,
            _data_iterator,
            _data_loader,
            _rampup,
            _eval_iterator,
            state,
        ) = train_utils.setup_train_loop(config, recorder)
        self._config = config
        self._graph, self._state = nnx.split(state)
        self._data_sharding = sharding.get_input_data_sharding(config, self._device_mesh)
        self._batch_shape = (int(config.global_batch_size_to_load), int(config.max_target_length))

    def _install_loss(self) -> None:
        import jax.numpy as jnp

        from .batch import next_token_logprobs

        low, high, clip_c = self._clip

        def forward_logits(model, data):
            # Every row is used: with gradient accumulation MaxText already hands one micro-batch,
            # and the eval micro-batch size would drop rows when scoring.
            logits = model(
                decoder_input_tokens=data["inputs"],
                decoder_positions=data["inputs_position"],
                decoder_segment_ids=data["inputs_segmentation"],
                enable_dropout=False,
                decoder_target_tokens=data["targets"],
                decoder_target_mask=data["targets_segmentation"],
            )
            return logits[0] if isinstance(logits, tuple) else logits

        def loss_fn(model, config, data, dropout_rng, params, sparsity_state=None, is_train=True):
            del dropout_rng, params, sparsity_state
            token_log_probs = next_token_logprobs(forward_logits(model, data), data["inputs"])
            if is_train:
                loss, _ = clipped_policy_loss(
                    jnp,
                    token_log_probs,
                    data["old_log_probs"],
                    data["advantages"],
                    data["response_mask"],
                    low,
                    high,
                    clip_c,
                )
            else:
                loss = jnp.zeros((), dtype=jnp.float32)
            weight = jnp.sum(data["response_mask"])
            # With gradient accumulation MaxText sums micro-batch objectives and divides by the
            # summed weights once, so each micro-batch must contribute its token sum.
            accumulate = is_train and getattr(config, "gradient_accumulation_steps", 1) > 1
            zero = jnp.asarray(0.0, dtype=loss.dtype)
            aux = {
                "intermediate_outputs": {},
                "xent_sum": loss * weight,
                "total_weights": weight,
                "moe_lb_loss": zero,
                "indexer_loss": zero,
                "mtp_loss": zero,
            }
            return (loss * weight if accumulate else loss), aux

        self._forward_logits = forward_logits
        self._train_module.loss_fn = loss_fn

    def _compile(self) -> None:
        import jax
        from flax import nnx
        from jax.sharding import NamedSharding, PartitionSpec
        from maxtext.utils import sharding, train_utils

        from .batch import next_token_entropy, next_token_logprobs

        self._install_loss()
        config, graph = self._config, self._graph
        if getattr(config, "enable_diloco", False):
            params_shardings, state_shardings = self._state_shardings.params, self._state_shardings
        else:
            params_shardings, state_shardings = sharding.maybe_update_params_sharding_with_opt(
                config, self._state_shardings
            )
        self._train_step, _ = train_utils.jit_train_and_eval_step(
            config,
            graph,
            self._device_mesh,
            self._state,
            state_shardings,
            self._train_module.train_step,
            self._train_module.eval_step,
            params_shardings=params_shardings,
        )
        forward_logits = self._forward_logits

        def logprob_step(pure_state, batch):
            logits = forward_logits(nnx.merge(graph, pure_state).model, batch)
            return next_token_logprobs(logits, batch["inputs"]), next_token_entropy(logits)

        # Replicated outputs: every process returns the full batch from its own GPU.
        replicated = NamedSharding(self._device_mesh, PartitionSpec())
        self._logprob_step = jax.jit(
            logprob_step,
            in_shardings=(state_shardings, self._data_sharding),
            out_shardings=(replicated, replicated),
        )


def _local(array) -> np.ndarray:
    """Host copy of a fully replicated global array."""
    return np.asarray(array.addressable_data(0))


def _scalar(value) -> float:
    if getattr(value, "is_fully_replicated", True):
        return float(np.asarray(value.addressable_data(0) if hasattr(value, "addressable_data") else value))
    from jax.experimental import multihost_utils

    return float(multihost_utils.process_allgather(value, tiled=True).reshape(-1)[0])


def actor_config_of(loss_function) -> Any:
    return (getattr(loss_function, "keywords", None) or {}).get("config")
