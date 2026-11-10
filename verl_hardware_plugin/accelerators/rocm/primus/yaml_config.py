# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Resolve one module of a Primus experiment YAML into a flat parameter dict.

Primus' own loader handles ``extends``, the ``config``/``model`` presets and the
module ``overrides`` block, so the result matches what ``primus-cli train`` sees.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_META_KEYS = {"name", "framework", "config", "model", "modules", "overrides", "stage"}


def ensure_primus_importable(primus_path: Optional[str] = None, turbo_path: Optional[str] = None) -> None:
    """Put a Primus (and optional Primus-Turbo) checkout on ``sys.path`` and check it imports."""
    for candidate in (primus_path or os.environ.get("PRIMUS_PATH"), turbo_path or os.environ.get("PRIMUS_TURBO_PATH")):
        if not candidate:
            continue
        resolved = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Primus dependency not found: {resolved}")
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    importlib.invalidate_caches()
    if importlib.util.find_spec("primus") is None:
        raise ModuleNotFoundError("Install Primus or set primus_path / PRIMUS_PATH to a Primus checkout")


@functools.lru_cache(maxsize=16)
def _load_cached(path: str, mtime: float, module: str, overrides: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    from primus.core.config.merge_utils import deep_merge
    from primus.core.config.primus_config import get_module_config, get_module_names, load_primus_config
    from primus.core.utils.arg_utils import parse_cli_overrides
    from primus.core.utils.yaml_utils import nested_namespace_to_dict

    experiment = load_primus_config(Path(path))
    module_cfg = get_module_config(experiment, module)
    if module_cfg is None:
        names = ", ".join(get_module_names(experiment)) or "none"
        raise ValueError(f"{path} has no Primus module {module!r}; available: {names}")
    framework = getattr(module_cfg, "framework", None)
    if not framework:
        raise ValueError(f"Primus module {module!r} in {path} sets no framework")
    params = nested_namespace_to_dict(module_cfg.params)
    if overrides:
        params = deep_merge(params, parse_cli_overrides(list(overrides)))
    return framework, {key: value for key, value in params.items() if key not in _META_KEYS}


def load_module_params(
    config_path: str,
    module: str = "pre_trainer",
    overrides: Sequence[str] = (),
    framework: Optional[str] = None,
    primus_path: Optional[str] = None,
) -> dict[str, Any]:
    """Return the resolved flat params of ``module``; check its framework when ``framework`` is given."""
    ensure_primus_importable(primus_path)
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Primus config not found: {path}")
    found, params = _load_cached(str(path), path.stat().st_mtime, module, tuple(overrides or ()))
    if framework is not None and found != framework:
        raise ValueError(f"Primus module {module!r} in {path} uses framework={found!r}, expected {framework!r}")
    return dict(params)


def module_framework(config_path: str, module: str = "pre_trainer", primus_path: Optional[str] = None) -> str:
    """Return the ``framework`` of one module (``megatron``, ``maxtext``, ...)."""
    ensure_primus_importable(primus_path)
    path = Path(config_path).expanduser().resolve()
    found, _ = _load_cached(str(path), path.stat().st_mtime, module, ())
    return found
