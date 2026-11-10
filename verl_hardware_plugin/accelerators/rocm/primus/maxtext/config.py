# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Config dataclasses for the ``maxtext`` strategy, selected through Hydra ``_target_``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.trainer.config import CheckpointConfig
from verl.workers.config.actor import ActorConfig
from verl.workers.config.critic import CriticConfig
from verl.workers.config.engine import EngineConfig


@dataclass
class MaxTextEngineConfig(EngineConfig):
    """MaxText trainer configured by a Primus experiment YAML (``framework: maxtext``).

    Parallelism is the MaxText mesh from the YAML (``ici_*`` / ``dcn_*``) over all trainer GPUs.
    verl starts one worker per GPU; the workers form one multi-process JAX job.
    """

    strategy: str = "maxtext"
    primus_config_path: str = ""
    primus_module: str = "pre_trainer"
    primus_path: Optional[str] = None
    # MaxText checkout; defaults to $MAXTEXT_PATH, then <primus_path>/third_party/maxtext.
    maxtext_path: Optional[str] = None
    primus_overrides: list[str] = field(default_factory=list)
    # Hugging Face checkpoint to start from; null means the actor model path.
    hf_weights_path: Optional[str] = None
    load_hf_weights: bool = True
    # XLA_PYTHON_CLIENT_MEM_FRACTION: leave room for the torch buffers of the weight sync.
    mem_fraction: float = 0.85

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy != "maxtext":
            raise ValueError(f"MaxTextEngineConfig requires strategy='maxtext', got {self.strategy!r}")
        if not 0 < self.mem_fraction < 1:
            raise ValueError(f"mem_fraction must be in (0, 1), got {self.mem_fraction}")


@dataclass
class MaxTextActorConfig(ActorConfig):
    """Actor trained by MaxText. The optimizer comes from the Primus YAML, not ``actor.optim``."""

    strategy: str = "maxtext"
    maxtext: MaxTextEngineConfig = field(default_factory=MaxTextEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.maxtext


@dataclass
class MaxTextCriticConfig(CriticConfig):
    """Lets ``model_engine=maxtext`` compose; MaxText has no value-head engine."""

    strategy: str = "maxtext"
    maxtext: MaxTextEngineConfig = field(default_factory=MaxTextEngineConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.maxtext
