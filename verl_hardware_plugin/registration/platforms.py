# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Compatibility entry point for platform registration."""

from verl_hardware_plugin.registration.registry import register_stage


def register_all_platforms() -> None:
    """Run the platform stage declared by each backend."""
    register_stage("platform")
