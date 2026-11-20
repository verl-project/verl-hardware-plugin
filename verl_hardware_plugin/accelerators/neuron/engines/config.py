# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Configuration for the constrained native TorchTitan integration."""

from dataclasses import dataclass

from verl.workers.config import TorchtitanEngineConfig
from verl.workers.config.engine import EngineConfig


@dataclass
class NeuronTorchtitanEngineConfig(TorchtitanEngineConfig):
    use_remove_padding: bool = False
    attn_type: str = "sdpa"
    spmd_backend: str = "default"
    max_seq_len: int = 1024
    activation_checkpoint: str = "none"
    # Zero disables host gathering; otherwise applies to full BF16 export size.
    weight_sync_cpu_gather_threshold_bytes: int = 0

    def __post_init__(self):
        EngineConfig.__post_init__(self)
        if self.strategy != "torchtitan" or self.attn_type != "sdpa" or self.spmd_backend != "default":
            raise ValueError("Trainium requires strategy=torchtitan, attn_type=sdpa, spmd_backend=default")
        if not isinstance(self.tensor_parallel_size, int) or self.tensor_parallel_size < 1:
            raise ValueError("Trainium tensor_parallel_size must be a positive integer")
        dimensions = ("context_parallel_size", "expert_parallel_size", "pipeline_parallel_size")
        for field in dimensions:
            if getattr(self, field) != 1:
                raise ValueError(f"Trainium v1 requires {field}=1")
        if self.param_offload or self.optimizer_offload or self.offload_policy or self.use_remove_padding:
            raise ValueError("Trainium v1 does not support offload or packed inputs")
        if self.max_seq_len is None or self.max_seq_len <= 0 or self.max_seq_len % 512:
            raise ValueError("Trainium max_seq_len must be a positive multiple of 512")
        if self.dtype != "bfloat16":
            raise ValueError("Trainium v1 requires dtype=bfloat16")
        if (
            not isinstance(self.weight_sync_cpu_gather_threshold_bytes, int)
            or self.weight_sync_cpu_gather_threshold_bytes < 0
        ):
            raise ValueError("weight_sync_cpu_gather_threshold_bytes must be a nonnegative integer")
