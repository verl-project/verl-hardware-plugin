# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""``vllm.general_plugins`` entry point for the TPU rollout (``verl_tpu_rollout`` in pyproject.toml).

vLLM loads general plugins in every one of its processes (API server, EngineCore, workers) on any
machine where verl-hardware-plugin is installed, including processes unrelated to verl or to TPU.
``register`` patches vLLM only in processes that serve a verl TPU rollout replica.

Known cost: importing this module first imports the ``verl_hardware_plugin`` package, whose
``__init__`` loads every accelerator backend (and verl), so each vLLM process on a machine with the
plugin installed pays that import even when ``register`` then does nothing. The entry point exists
only to install the runtime patches in ``tpu_vllm_patches``; it goes away once vLLM /
vllm-torchtpu no longer need them.
"""

import os

# Set by ``launch_tpu_vllm_servers`` on the server actor. The EngineCore process inherits it, and
# vLLM's worker actors get it through the server actor's Ray runtime env, which vLLM reuses when
# it connects to Ray.
VERL_TPU_PG_IDS_ENV = "VERL_TPU_PG_IDS"


def register() -> None:
    """Patch vLLM for TPU rollout if this process serves a verl TPU replica; otherwise do nothing."""
    if not os.environ.get(VERL_TPU_PG_IDS_ENV):
        return
    from verl_hardware_plugin.accelerators.tpu.rollout.tpu_vllm_patches import patch_vllm_for_tpu

    patch_vllm_for_tpu()
