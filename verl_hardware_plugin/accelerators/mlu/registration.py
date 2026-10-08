# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cambricon MLU registration declaration and profiler hook."""

from verl_hardware_plugin.registration.backend import BackendRegistration


def apply_mlu_profiler_patches():
    """Apply all MLU profiler monkey-patches. Idempotent."""
    from verl_hardware_plugin.accelerators.mlu.profilers.torch_profile_mlu import (
        _patch_get_torch_profiler,
        _patch_tool_config,
    )

    _patch_tool_config()
    _patch_get_torch_profiler()


BACKEND = BackendRegistration(
    package=__package__,
    platform=".platform_mlu",
    engines=(
        (".engines.fsdp_mlu",),
        (".engines.megatron_mlu",),
        (".engines.cncl_checkpoint_engine", ".engines.cnixl_checkpoint_engine"),
    ),
    profiler=apply_mlu_profiler_patches,
)
