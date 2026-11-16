# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Synchronous TorchTitan persistence with CPU checkpoint control collectives."""

from dataclasses import dataclass

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint import MODEL, AsyncMode, CheckpointManager
from torchtitan.tools.utils import GarbageCollection


class NeuronCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def __init__(self, config, **kwargs):
        if config.async_mode != AsyncMode.DISABLED or config.last_save_in_hf:
            raise ValueError("Neuron RL checkpoints require synchronous DCP saves")
        if config.load_model_state_dict_with_streaming_planner:
            raise ValueError("Neuron RL uses the standard dense HF checkpoint loader")
        super().__init__(config=config, **kwargs)
        # Object collectives must use a CPU-only group even when the world group
        # supports both CPU and Neuron tensors. Model shards remain on device.
        self.pg = getattr(self, "pg", None)
        if self.pg is None and dist.is_initialized():
            self.pg = dist.new_group(backend="gloo")

    def dcp_save(self, state_dict, checkpoint_id, async_mode, enable_garbage_collection=False, to_hf=False):
        if async_mode != AsyncMode.DISABLED or to_hf:
            raise ValueError("Neuron RL checkpoints require synchronous DCP saves")
        dcp.save(state_dict, checkpoint_id=checkpoint_id, process_group=self.pg)
        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

    def dcp_load(self, state_dict, checkpoint_id, from_hf, from_quantized):
        if from_hf:
            if self.sd_adapter is None:
                raise ValueError("An HF state-dict adapter is required")
            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            reader = self.sd_adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
            dcp.load(hf_state_dict, storage_reader=reader, process_group=self.pg)
            state_dict = self.sd_adapter.from_hf(hf_state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id, process_group=self.pg)
        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)

    def close(self):
        super().close()
        group = getattr(self, "pg", None)
        if group is not None and dist.is_initialized():
            dist.destroy_process_group(group)
            self.pg = None
