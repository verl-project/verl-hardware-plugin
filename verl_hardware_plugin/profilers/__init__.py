# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Profiler integrations via monkey-patching verl's profiler utilities."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_mlu_profiler_patches():
    """Apply all MLU profiler monkey-patches. Idempotent."""
    from .torch_profile_mlu import _patch_get_torch_profiler, _patch_tool_config

    _patch_tool_config()
    _patch_get_torch_profiler()


def register_all_profiles():
    """Register all profiler extensions.

    Mirrors register_all_platforms() and register_all_engines() by importing
    profiler modules conditionally so missing hardware SDKs do not break
    other backends.
    """
    try:
        apply_mlu_profiler_patches()
        logger.info("Registered profiler: mlu")
    except Exception as e:
        logger.debug("MLU profiler patches not registered: %s", e)
