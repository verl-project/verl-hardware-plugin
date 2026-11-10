# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Run Primus' Megatron kernel/model patches inside verl's Megatron engine.

verl keeps orchestration (process groups, data, optimizer loop); Primus contributes only
the patches selected by ``primus_patches``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
from types import SimpleNamespace

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LIFTED_VERSION_GATES = {
    "megatron.turbo.te_spec_provider": "TESpecProvider bindings still exist past Primus' <0.17 gate",
}


class _PrimusLoggerAdapter:
    """Sink Primus rank-0 logs into Python logging without Primus' print hijack."""

    def __init__(self, target: logging.Logger):
        self._target = target

    def info(self, message: str, *args, **kwargs) -> None:
        self._target.info(message)

    def debug(self, message: str, *args, **kwargs) -> None:
        self._target.debug(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        self._target.warning(message)

    def error(self, message: str, *args, **kwargs) -> None:
        self._target.error(message)


def _install_primus_logger() -> None:
    from primus.core.utils import logger as primus_logger

    if primus_logger._logger is None:
        primus_logger._logger = _PrimusLoggerAdapter(logger)
    try:
        from primus.core.utils.module_utils import set_logging_rank
    except ImportError:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        set_logging_rank(dist.get_rank(), dist.get_world_size())


def _detect_megatron_version() -> str | None:
    import ast
    from pathlib import Path

    candidates = []
    try:
        import megatron.core

        candidates.append(Path(megatron.core.__file__).resolve().parent / "package_info.py")
    except Exception:
        pass
    for path in candidates:
        if not path.exists():
            continue
        values = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                name = getattr(node.targets[0], "id", None)
                if name in {"MAJOR", "MINOR", "PATCH", "PRE_RELEASE"}:
                    values[name] = ast.literal_eval(node.value)
        pre = values.get("PRE_RELEASE")
        return f"{values['MAJOR']}.{values['MINOR']}.{values['PATCH']}" + (str(pre) if pre else "")
    try:
        from primus.backends.megatron.megatron_adapter import MegatronAdapter

        return MegatronAdapter().detect_backend_version()
    except Exception as error:
        logger.warning("Could not detect Megatron version for Primus patches: %s", error)
        return None


_KERNEL_PATCH_IDS = {
    "megatron.env.cuda_device_max_connections",
    "megatron.runtime.skip_compile_dependencies",
    "megatron.te.general_gemm_workspace_helper",
    "megatron.te.tp_overlap_te1",
    "megatron.te.tp_overlap_te2",
    "megatron.turbo.te_spec_provider",
    "megatron.turbo.moe_dispatcher",
    "megatron.turbo.rms_norm",
    "megatron.moe_alltoall_dtoh_turbo_grouped_gemm",
    "megatron.moe.permute_fusion",
    "megatron.moe.primus_topk_router",
    "megatron.transformer.custom_recompute_layer_ids",
    "megatron.gpt.decoder_layer_specs",
    "megatron.args.moe_layer_freq",
}

_BRIDGE_MODULE_TYPES = {
    "PrimusTopKRouter": "replicated",
    "PrimusTurboRMSNorm": "replicated",
    "PrimusTurboAttention": "column",
    "PrimusSharedExpertMLP": "column",
    "PrimusTurboColumnParallelLinear": "column",
    "PrimusTurboRowParallelLinear": "row",
    "PrimusTurboLayerNormColumnParallelLinear": "column",
    "PrimusTurboColumnParallelGroupedLinear": "column",
    "PrimusTurboRowParallelGroupedLinear": "row",
}


def _install_bridge_types() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    for name, parallelism in _BRIDGE_MODULE_TYPES.items():
        AutoMapping.register_module_type(name, parallelism)


def _install_name_kwarg_shims() -> None:
    import torch

    def subclasses(cls):
        for child in cls.__subclasses__():
            yield child
            yield from subclasses(child)

    for cls in set(subclasses(torch.nn.Module)):
        if not cls.__module__.startswith("primus.") or "__init__" not in cls.__dict__:
            continue
        original = cls.__dict__["__init__"]
        if getattr(original, "_verl_primus_name_shim", False):
            continue
        parameters = inspect.signature(original).parameters
        if "name" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue

        @functools.wraps(original)
        def init(self, *args, _original=original, name=None, **kwargs):
            _original(self, *args, **kwargs)

        init._verl_primus_name_shim = True  # type: ignore[attr-defined]
        cls.__init__ = init


class PrimusPatchRunner:
    """Run only Primus kernel/model patches; verl retains orchestration."""

    def __init__(self, engine_config):
        from .megatron_compat import install_megatron_compat_shims

        _install_primus_logger()
        install_megatron_compat_shims()
        import primus.backends.megatron.patches  # noqa: F401
        from primus.backends.megatron.training import global_vars as primus_globals
        from primus.core.patches import PatchRegistry

        params = dict(engine_config._primus_params)
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
            "sequence_parallel",
        ):
            params[name] = getattr(engine_config, name)
        self.args = SimpleNamespace(**params)
        self.module_name = engine_config.primus_module
        registered = {patch.id for patch in PatchRegistry._all_patches}
        selection = engine_config.primus_patches.strip()
        if selection == "none":
            self.enabled_ids = []
        elif selection == "all":
            self.enabled_ids = sorted(registered)
        elif selection == "kernels":
            self.enabled_ids = sorted(registered & _KERNEL_PATCH_IDS)
        else:
            requested = {item.strip() for item in selection.split(",") if item.strip()}
            unknown = requested - registered
            if unknown:
                raise ValueError(f"Unknown Primus patch ids: {', '.join(sorted(unknown))}")
            self.enabled_ids = sorted(requested)
        if getattr(primus_globals, "_GLOBAL_ARGS", None) is None:
            primus_globals.set_args(self.args)
        else:
            primus_globals._GLOBAL_ARGS = self.args
        _install_bridge_types()
        self.backend_version = _detect_megatron_version()
        for patch in PatchRegistry._all_patches:
            if patch.id in _LIFTED_VERSION_GATES and patch.id in self.enabled_ids and patch.backend_version_patterns:
                logger.info(
                    "Primus patch %s: ignoring backend version gate %s (%s)",
                    patch.id,
                    patch.backend_version_patterns,
                    _LIFTED_VERSION_GATES[patch.id],
                )
                patch.backend_version_patterns = []

    def bind_megatron_globals(self) -> None:
        from megatron.training import global_vars

        if getattr(global_vars, "_GLOBAL_ARGS", None) is None:
            global_vars.set_args(self.args)
        else:
            global_vars._GLOBAL_ARGS = self.args

    def run(self, phase: str) -> int:
        if not self.enabled_ids:
            return 0
        from primus.core.patches import run_patches

        module_config = SimpleNamespace(name=self.module_name, framework="megatron", params=self.args)
        applied = run_patches(
            backend="megatron",
            phase=phase,
            backend_version=self.backend_version,
            module_name=self.module_name,
            extra={"backend_args": self.args, "module_config": module_config},
            enabled_ids=self.enabled_ids,
            stop_on_error=True,
        )
        if phase == "before_train":
            _install_name_kwarg_shims()
        return applied
