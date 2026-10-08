# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compatibility entry point for lazy rollout registration."""

from verl_hardware_plugin.registration.registry import register_stage


def register_all_rollouts() -> None:
    """Run the rollout stage declared by each backend."""
    register_stage("rollout")
