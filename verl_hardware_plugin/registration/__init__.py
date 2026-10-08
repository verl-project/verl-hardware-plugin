# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unified registration entry point for accelerator and integration modules."""

from verl_hardware_plugin.registration.registry import register_all

__all__ = ["register_all"]
