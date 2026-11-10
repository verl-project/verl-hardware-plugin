# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""``${primus:<key>,<default>[,<engine node>]}``: read a value from the Primus YAML inside a verl config.

The plugin's Hydra groups (``model_engine=primus`` / ``model_engine=maxtext``) use it so that
the Primus YAML is the single source of truth for parallelism and optimizer settings, while
verl's driver-side code (batch validation, token budgets) still sees ordinary config values.

The engine node (default ``actor_rollout_ref.actor.megatron``) carries ``primus_config_path``,
``primus_module``, ``primus_overrides`` and ``primus_path``. With no ``primus_config_path``
the default is returned.
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

RESOLVER_NAME = "primus"
DEFAULT_ENGINE_NODE = "actor_rollout_ref.actor.megatron"

# Values verl spells differently from Megatron/Primus. Results must be scalars: OmegaConf.resolve
# cannot store a list returned by a resolver.
_DERIVED = {
    "dtype": lambda p: "bfloat16" if p.get("bf16") else ("float16" if p.get("fp16") else None),
    "lr_warmup_steps": lambda p: p.get("lr_warmup_iters"),
}


def _engine_params(root, engine_node: str) -> dict[str, Any] | None:
    from .yaml_config import load_module_params

    node = OmegaConf.select(root, engine_node, default=None)
    if node is None:
        return None
    path = node.get("primus_config_path")
    if not path:
        return None
    overrides = node.get("primus_overrides") or []
    return load_module_params(
        str(path),
        module=str(node.get("primus_module") or "pre_trainer"),
        overrides=[str(item) for item in overrides],
        primus_path=node.get("primus_path"),
    )


def primus_value(key: str, default: Any = None, engine_node: str = DEFAULT_ENGINE_NODE, *, _root_) -> Any:
    params = _engine_params(_root_, engine_node)
    if params is None:
        return default
    value = _DERIVED[key](params) if key in _DERIVED else params.get(key)
    return default if value is None else value


def register_resolver() -> None:
    if not OmegaConf.has_resolver(RESOLVER_NAME):
        OmegaConf.register_new_resolver(RESOLVER_NAME, primus_value)
