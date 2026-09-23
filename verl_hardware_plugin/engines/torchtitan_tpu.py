# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""TorchTitan training engine for Google TPU (``torch_tpu``) devices.

Registers ``TorchTitanTPUEngineWithLMHead`` with verl's ``EngineRegistry`` under
``(model_type="language_model", backend="torchtitan", device="tpu", vendor="google")``.

This keeps all TPU TorchTitan training specializations out-of-tree in
``verl-hardware-plugin``:
- ``spmd_backend="default"`` and ``CompileConfig(backend="tpu")``
- ``OptimizersContainer.Config(implementation="foreach")`` and ``foreach=False`` grad norm clipping
- Host-side global token counting (avoiding eager device collectives inside the XLA SPMD trace)
- Graph-break loss synchronization between forward and backward
- Sequence-length bucketing for both packed (``use_remove_padding=True``) and padded inputs
- Shape-stable ``_tpu_padded_values`` attachment, which verl's TPU loss path consumes
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Callable

import torch
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngineWithLMHead
from verl.workers.engine.utils import (
    detach_tree,
    postprocess_batch_func,
    prepare_micro_batches,
)
from verl_hardware_plugin.engines.tpu_utils import (
    TPU_PADDED_VALUES_ATTR,
    bucket_length,
    compute_global_batch_num_tokens,
    pad_packed_inputs_for_tpu,
    replace_varlen_attention_with_tpu_attention,
    synchronize_tpu_loss,
    tpu_torchtitan_config_overrides,
    unwrap_metadata,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",
    backend="torchtitan",
    device="tpu",
    vendor="google",
)
class TorchTitanTPUEngineWithLMHead(TorchTitanEngineWithLMHead):
    """TorchTitan engine for Google TPU with ``tpu_dist`` / PJRT SPMD execution."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        if engine_config.tensor_parallel_size > 1:
            logger.warning(
                "tensor_parallel_size=%d is not properly tested on TPU; try "
                "tensor_parallel_size=1 for better performance on TPU.",
                engine_config.tensor_parallel_size,
            )

        with tpu_torchtitan_config_overrides():
            super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

        replace_varlen_attention_with_tpu_attention(getattr(self.trainer, "model_parts", None))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logger.info("TorchTitanTPUEngineWithLMHead initialized")

    def initialize(self):
        """Initialize model/checkpointer and release unused optimizer state on forward-only reference workers."""
        super().initialize()
        replace_varlen_attention_with_tpu_attention(self.module)
        if self.engine_config.forward_only:
            self.trainer.optimizers = None
            self.trainer.lr_schedulers = None

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

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self.trainer.train_context(), ctx, torch.profiler.record_function(f"micro_batch{micro_batch_idx}"):
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    synchronize_tpu_loss(loss)
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def optimizer_step(self):
        """Clip gradients using the per-tensor norm path (``foreach=False``) and step the optimizer."""
        from torchtitan.distributed import utils as dist_utils

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

        if use_remove_padding:
            input_ids, position_ids, labels, attention_mask, orig_seq_len = pad_packed_inputs_for_tpu(
                input_ids=input_ids,
                position_ids=position_ids,
                micro_batch=micro_batch,
                device=get_device_id(),
            )
            output_args["orig_seq_len"] = orig_seq_len
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = bucket_length(int(max(input_ids.offsets().diff())))

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0).to(get_device_id())
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            ).to(get_device_id())

            if position_ids.dim() == 3:
                position_ids = (
                    torch.nested.to_padded_tensor(position_ids, padding=0, output_size=(batch_size, 4, max_seq_len))
                    .transpose(0, 1)
                    .to(get_device_id())
                )
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                ).to(get_device_id())

            attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(
                attention_mask, padding=0, output_size=(batch_size, max_seq_len)
            ).to(get_device_id())

        extra_inputs = {"positions": position_ids}
        extra_kwargs = {"attention_masks": attention_mask}

        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import prepare_context_parallel_input

            input_ids, labels, extra_kwargs = prepare_context_parallel_input(
                input_ids,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                self.trainer.config.parallelism.context_parallel_load_balancer,
            )

        extra_inputs.update(multi_modal_inputs)
        output_args["labels"] = labels

        # Ensure all model inputs and kwargs are contiguous on TPU.
        input_ids = input_ids.contiguous()
        extra_inputs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in extra_inputs.items()}
        extra_kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in extra_kwargs.items()}

        return input_ids, extra_inputs, extra_kwargs, output_args

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
        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0) / temperature

            inplace_backward = not calculate_entropy
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

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

            padded_log_probs = log_probs.squeeze(0)
            orig_seq_len = output_args.get("orig_seq_len")
            if orig_seq_len is not None:
                cu_seqlens_cpu = cu_seqlens.detach().cpu()
                unpadded_log_probs = padded_log_probs.detach().cpu()[:orig_seq_len]
                log_probs = torch.nested.nested_tensor_from_jagged(unpadded_log_probs, cu_seqlens_cpu)
                setattr(log_probs, TPU_PADDED_VALUES_ATTR, padded_log_probs)
                if calculate_entropy:
                    unpadded_entropy = entropy_rmpad.detach().cpu()[:orig_seq_len]
                    entropy = torch.nested.nested_tensor_from_jagged(unpadded_entropy, cu_seqlens_cpu)
                    setattr(entropy, TPU_PADDED_VALUES_ATTR, entropy_rmpad)
            else:
                log_probs = torch.nested.nested_tensor_from_jagged(padded_log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits = logits / temperature
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=output_args["labels"])
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
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
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
