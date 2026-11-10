# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Register AMD ROCm components when imported."""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _is_rocm() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(torch.version, "hip", None))


try:
    from verl_hardware_plugin.accelerators.rocm.engines import mori_checkpoint_engine  # noqa: F401
except Exception as exc:
    logger.debug("ROCm MoRI checkpoint engine registration failed: %s", exc)
    try:  # say why when backend "mori" is requested, e.g. MoRI is not installed
        from verl.checkpoint_engine.base import CheckpointEngineRegistry

        CheckpointEngineRegistry.record_import_error("mori_checkpoint_engine", exc)
    except Exception:
        pass

try:
    from verl_hardware_plugin.accelerators.rocm.primus.resolver import register_resolver

    register_resolver()
except Exception as exc:
    logger.debug("ROCm ${primus:...} config resolver registration failed: %s", exc)

try:
    from verl_hardware_plugin.accelerators.rocm.hydra_searchpath import register_search_path

    register_search_path()
except Exception as exc:
    logger.debug("ROCm Hydra config search path registration failed: %s", exc)

if _is_rocm():
    try:
        from verl_hardware_plugin.accelerators.rocm.primus.megatron import engine  # noqa: F401
    except Exception as exc:
        logger.debug("ROCm Primus Megatron engine registration failed: %s", exc)
    try:
        from verl_hardware_plugin.accelerators.rocm.primus.maxtext import engine as _maxtext_engine  # noqa: F401
    except Exception as exc:
        logger.debug("ROCm Primus MaxText engine registration failed: %s", exc)
