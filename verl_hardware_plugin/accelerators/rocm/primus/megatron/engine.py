# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl Megatron engine for AMD GPUs with optional Primus patches.

Registered as the ``(cuda, amd)`` variant of the ``megatron`` backend, so ``strategy: megatron``
keeps every verl Megatron code path. Primus activates only when the engine config is a
``PrimusMcoreEngineConfig`` with a ``primus_config_path``; otherwise this is plain verl Megatron.
"""

from __future__ import annotations

import logging
import os

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead, MegatronEngineWithValueHead

from ..yaml_config import ensure_primus_importable, load_module_params
from .config import PrimusMcoreEngineConfig, apply_primus_params, validate_primus_capabilities

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _check_export_name(name: str) -> str:
    if ".mlp.experts.linear_fc" in name and name.endswith(".weights"):
        raise RuntimeError(f"{name} is a Primus-Turbo packed expert parameter; export the per-expert views instead")
    return name


class _PrimusMixin:
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.primus_patch_runner = None
        primus = isinstance(engine_config, PrimusMcoreEngineConfig) and engine_config.primus_enabled
        if primus:
            ensure_primus_importable(engine_config.primus_path, engine_config.primus_turbo_path)
            params = load_module_params(
                engine_config.primus_config_path,
                module=engine_config.primus_module,
                overrides=engine_config.primus_overrides,
                framework="megatron",
            )
            apply_primus_params(engine_config, params)
            validate_primus_capabilities(engine_config, params, model_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        if primus:
            from .patches import PrimusPatchRunner

            runner = PrimusPatchRunner(engine_config)
            runner.run("setup")
            runner.run("build_args")
            runner.bind_megatron_globals()
            runner.run("before_train")
            self.primus_patch_runner = runner
            logger.info("Primus Megatron patches enabled: %s", ", ".join(runner.enabled_ids) or "none")

    def get_per_tensor_param(self, *args, **kwargs):
        tensors, peft_config = super().get_per_tensor_param(*args, **kwargs)
        if self.primus_patch_runner is None:
            return tensors, peft_config
        return ((_check_export_name(name), tensor) for name, tensor in tensors), peft_config


@EngineRegistry.register(model_type="language_model", backend="megatron", device="cuda", vendor="amd")
class PrimusMegatronEngineWithLMHead(_PrimusMixin, MegatronEngineWithLMHead):
    """Megatron language-model engine with Primus patches."""


@EngineRegistry.register(model_type="value_model", backend="megatron", device="cuda", vendor="amd")
class PrimusMegatronEngineWithValueHead(_PrimusMixin, MegatronEngineWithValueHead):
    """Megatron value-model engine with Primus patches."""
