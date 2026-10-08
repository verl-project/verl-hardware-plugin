# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compatibility entry point for profiler registration."""

from verl_hardware_plugin.registration.registry import register_stage


def apply_mlu_profiler_patches():
    """Forward the legacy MLU hook to the accelerator-owned implementation."""
    from verl_hardware_plugin.accelerators.mlu.registration import apply_mlu_profiler_patches as apply

    apply()


def register_all_profiles() -> None:
    """Run the profiler stage declared by each backend."""
    register_stage("profiler")
