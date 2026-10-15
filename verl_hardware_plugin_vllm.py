# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""``vllm.general_plugins`` entry point of verl-hardware-plugin.

vLLM loads general plugins in every one of its processes (API server, EngineCore, workers) on any
machine where this distribution is installed, including processes unrelated to verl or to TPU. This
module therefore lives outside the ``verl_hardware_plugin`` package, whose import loads every
accelerator backend, and imports nothing until it knows the process serves a verl TPU rollout
replica.
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
