# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Context-parallel transforms of non-attention sequence mixers, looked up by the layers a model contains.

Core torchtitan ``ContextParallelTransform`` only makes the attention layers CP-aware. A hybrid model's other
sequence mixers (Qwen3.5's GatedDeltaNet, and later Mamba / KDA layers) carry state through the whole sequence, so
under ``context_parallel_degree > 1`` they need their own transform, or they silently run on each rank's token shard
and give wrong log-probs, losses and gradients.

The engine stays model-agnostic: ``context_parallel_layer_transforms`` walks the model config and, for every
registered ``CPLayerTransform`` whose layer config type the model contains, returns that layer's torchtitan
transform. A model whose layer needs a transform that the installed torchtitan does not ship fails loudly instead of
falling back. A layer is found by its config type, which can only appear in the model config once its module is
imported, so nothing is imported for models that do not use it.

Register another layer with ``register_cp_layer_transform``. Upstream follow-up: torchtitan's TPU CP has a validator
registry (``register_tpu_cp_validator``) but no transform registry; once it has one, this module should defer to it.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
from collections.abc import Mapping
from typing import Any, Optional

__all__ = [
    "CPLayerTransform",
    "context_parallel_layer_transforms",
    "module_available",
    "register_cp_layer_transform",
    "registered_cp_layer_transforms",
]


@dataclasses.dataclass(frozen=True)
class CPLayerTransform:
    """A torchtitan ``ModelConfigTransform`` that makes one kind of layer context-parallel.

    Attributes:
        name: Human-readable layer name for error messages, e.g. ``"Qwen3.5 GatedDeltaNet"``.
        layer_config: ``"module:Qualified.Name"`` of the layer's config class; the transform applies when
            ``model.traverse`` finds an instance of it.
        transform: ``"module:Qualified.Name"`` of the transform class. It is built with
            ``load_balancer=<parallelism.context_parallel_load_balancer>`` plus ``option_fields``.
        requirement: The torchtitan revision that ships ``transform``, quoted when it is missing.
        option_fields: ``TPUEngineOptions`` field -> keyword of ``transform``. An option that is set is passed; a
            transform without that field is an error (the installed torchtitan is too old for the option).
    """

    name: str
    layer_config: str
    transform: str
    requirement: str
    option_fields: Mapping[str, str] = dataclasses.field(default_factory=dict)


_REGISTRY: list[CPLayerTransform] = []


def register_cp_layer_transform(spec: CPLayerTransform) -> CPLayerTransform:
    """Registers ``spec`` (idempotent) and returns it."""
    if spec not in _REGISTRY:
        _REGISTRY.append(spec)
    return spec


def registered_cp_layer_transforms() -> tuple[CPLayerTransform, ...]:
    """The registered layer transforms, in the order they are applied."""
    return tuple(_REGISTRY)


def _split(path: str) -> tuple[str, str]:
    module, sep, qualname = path.partition(":")
    if not sep or not module or not qualname:
        raise ValueError(f"expected 'module:Qualified.Name', got {path!r}")
    return module, qualname


def _getattr_path(obj: Any, qualname: str) -> Any:
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _loaded_attr(path: str) -> Optional[Any]:
    """The object at ``path`` if its module is already imported, else None. Never imports anything."""
    module_name, qualname = _split(path)
    module = sys.modules.get(module_name)
    if module is None:
        return None
    try:
        return _getattr_path(module, qualname)
    except AttributeError:
        return None


def module_available(module_name: str) -> bool:
    """Whether ``module_name`` exists, without hiding errors raised inside a module that does exist.

    Only a missing ``module_name`` (or a missing parent package of it) counts as unavailable. Any other
    ``ImportError`` (e.g. a broken transitive dependency) propagates when the caller imports the module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError as e:
        if e.name and (module_name == e.name or module_name.startswith(e.name + ".")):
            return False
        raise


def _build_transform(spec: CPLayerTransform, load_balancer: Optional[str], options: Any, num_layers: int) -> Any:
    module_name, qualname = _split(spec.transform)
    missing = (
        f"The model has {num_layers} {spec.name} layer(s), which carry state through the whole sequence, and "
        f"context_parallel_degree > 1, but the installed torchtitan has no {spec.transform} to run them "
        "context-parallel. Without it they would run on each rank's token shard and silently give wrong log-probs "
        f"and gradients. Install {spec.requirement}, or turn context parallelism off."
    )
    if not module_available(module_name):
        raise RuntimeError(missing)
    module = importlib.import_module(module_name)  # a real ImportError inside the module propagates
    try:
        transform_cls = _getattr_path(module, qualname)
    except AttributeError as e:
        raise RuntimeError(missing) from e

    kwargs: dict[str, Any] = {"load_balancer": load_balancer}
    field_names = (
        {f.name for f in dataclasses.fields(transform_cls)} if dataclasses.is_dataclass(transform_cls) else None
    )
    for option, keyword in spec.option_fields.items():
        value = getattr(options, option, None)
        if value is None:
            continue  # keep the transform's own default
        if field_names is not None and keyword not in field_names:
            raise ValueError(
                f"engine_kwargs.tpu.{option}={value!r} is set, but the installed torchtitan's {spec.transform} has "
                f"no {keyword!r} field. Upgrade torchtitan or unset the option."
            )
        kwargs[keyword] = value
    return transform_cls(**kwargs)


def context_parallel_layer_transforms(model: Any, load_balancer: Optional[str], options: Any = None) -> list[Any]:
    """The CP transforms of the registered non-attention layers that ``model`` contains, in registry order.

    Args:
        model: The torchtitan model config (a ``Module.Config``, which has ``traverse``).
        load_balancer: ``parallelism.context_parallel_load_balancer`` of the CP strategy (``"headtail"`` for KV
            all-gather, ``None`` for Ulysses): the layers must read the same token layout as attention.
        options: ``TPUEngineOptions`` whose fields fill each transform's ``option_fields``.

    Raises:
        RuntimeError: if the model contains a registered layer whose transform the installed torchtitan lacks.
        ValueError: if an option is set that the installed transform does not accept.
    """
    transforms = []
    for spec in _REGISTRY:
        layer_config_cls = _loaded_attr(spec.layer_config)
        if layer_config_cls is None:
            continue  # its module is not imported, so the model cannot contain the layer
        num_layers = sum(1 for _ in model.traverse(layer_config_cls))
        if num_layers:
            transforms.append(_build_transform(spec, load_balancer, options, num_layers))
    return transforms


# Qwen3.5 (and other models built from torchtitan's qwen3_5 GatedDeltaNet). Minimum torchtitan:
# google-pytorch/torchtitan#201 (branch feat/tpu-qwen35-cp-on-ep; the transform landed in commit 36643b4c9,
# "feat(tpu): context parallelism for Qwen3.5 with a head-parallel GatedDeltaNet"). The
# context_parallel_gdn_strategy / context_parallel_gdn_reduce_dtype options need a later torchtitan whose transform
# has the ``strategy`` / ``reduce_dtype`` fields; with an older one they are rejected when set.
QWEN3_5_GATED_DELTA_NET = register_cp_layer_transform(
    CPLayerTransform(
        name="Qwen3.5 GatedDeltaNet",
        layer_config="torchtitan.models.qwen3_5.gdn:GatedDeltaNet.Config",
        transform="torchtitan.experiments.tpu.qwen3_5.cp_gated_delta:GatedDeltaNetContextParallelTransform",
        requirement="a torchtitan with google-pytorch/torchtitan#201 (experiments/tpu/qwen3_5/cp_gated_delta.py)",
        option_fields={
            "context_parallel_gdn_strategy": "strategy",
            "context_parallel_gdn_reduce_dtype": "reduce_dtype",
        },
    )
)
