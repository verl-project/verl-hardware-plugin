# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Flags of torchtitan's ``TPUSelectiveAC``, the TorchTitan engine's ``activation_checkpoint=selective`` policy.

``selective`` builds torchtitan's ``TPUSelectiveAC``: core ``SelectiveAC`` (saves the outputs of the compute-intensive
ops -- mm, bmm, ... -- except every second mm / linear, and recomputes everything else in backward) minus the
collectives it would save. Newer torchtitan versions give that policy flags that tune its save set. Each flag has an
``engine_kwargs.tpu`` option, off by default; any combination composes:

* ``selective_ac_recompute_bmm`` sets ``recompute_bmm``: recompute ``aten.bmm`` outputs in backward instead of saving
  them. Saving them hands one tensor per bmm from each compiled block's forward program to its backward program;
  Qwen3.5's XLA GatedDeltaNet chunk recurrence runs three small bmm per 64-token chunk, so that is three tensors per
  chunk of every GatedDeltaNet layer.
* ``selective_ac_save_pallas_attention`` sets ``save_pallas_attention``: also save the Pallas splash-attention
  forward's output and logsumexp, so backward runs the splash backward kernel on them instead of re-running the
  forward kernel first. Costs one bf16 ``[1, H, T, D]`` and one f32 ``[1, H, T]`` tensor per attention layer.
* ``selective_ac_save_kda_residuals`` sets ``save_kda_residuals``: also save the Tokamax KDA forward op of each
  GatedDeltaNet layer (its output and every residual of the KDA backward), so backward runs only the KDA backward
  kernels instead of replaying the forward kernel first. Costs those residuals for every GatedDeltaNet layer; it only
  acts when ``model_kernels`` turns on the Tokamax KDA kernels.

``recompute_bmm`` and ``save_pallas_attention`` come with google-pytorch/torchtitan#168, ``save_kda_residuals`` with
google-pytorch/torchtitan#170. A requested flag that the installed ``TPUSelectiveAC`` does not have stops the engine
build with an error instead of being dropped. With none of the options set, the engine's activation-checkpointing
config is left as it is. ``ac_save_splash_and_routing`` replaces this policy with its own save set (only the splash
forward and the MoE router's ``aten.topk``), so ``options.py`` rejects it together with these options.

The values do not change: a saved output and a recomputed one come from the same op on the same inputs. On TPU a
recomputed op is compiled into the backward program, where XLA may fuse the surrounding elementwise ops differently,
as for every op that selective AC recomputes.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import os
from typing import Any

from verl_hardware_plugin.accelerators.tpu.options import SELECTIVE_AC_FLAG_OPTIONS

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Modules that define torchtitan's TPUSelectiveAC, newest layout first: torchtitan moved the TPU experiment's
# activation-checkpoint policies from the experiment root into its components/ subpackage.
TPU_SELECTIVE_AC_MODULES = (
    "torchtitan.experiments.tpu.components.activation_checkpoint",
    "torchtitan.experiments.tpu.activation_checkpoint",
)


def tpu_selective_ac_class() -> Any:
    """Returns torchtitan's ``TPUSelectiveAC``, or core ``SelectiveAC`` for a torchtitan without it."""
    for module_name in TPU_SELECTIVE_AC_MODULES:
        try:
            return importlib.import_module(module_name).TPUSelectiveAC
        except (ImportError, AttributeError):
            continue
    from torchtitan.distributed.activation_checkpoint import SelectiveAC

    return SelectiveAC


def requested_flags(tpu_options: Any) -> dict[str, bool]:
    """The ``TPUSelectiveAC`` flags that ``tpu_options`` (``TPUEngineOptions``) turns on, by flag name."""
    return {flag: True for option, flag in SELECTIVE_AC_FLAG_OPTIONS.items() if getattr(tpu_options, option, None)}


def selective_ac_config(ac_config: Any, ac_mode: str, tpu_options: Any) -> Any:
    """Applies the ``engine_kwargs.tpu.selective_ac_*`` options to the engine's activation-checkpointing config.

    Args:
        ac_config: The config the engine built for ``ac_mode`` (``TPUSelectiveAC.Config()`` for ``selective``).
        ac_mode: The ``activation_checkpoint`` mode the engine runs with.
        tpu_options: The engine's ``TPUEngineOptions``.

    Returns:
        ``ac_config`` itself when none of the options is set. Otherwise a copy of ``ac_config`` (every other field
        kept) with the requested flags set.

    Raises:
        ValueError: an option is set but the engine does not run selective AC, or ``ac_config`` has no field for a
            requested flag (a torchtitan whose ``TPUSelectiveAC`` does not have it).
    """
    flags = requested_flags(tpu_options)
    if not flags:
        return ac_config
    options = [option for option, flag in SELECTIVE_AC_FLAG_OPTIONS.items() if flag in flags]
    named = ", ".join(f"engine_kwargs.tpu.{option}" for option in options)
    if ac_mode != "selective":
        raise ValueError(
            f"{named}: only for activation_checkpoint=selective; this engine runs with "
            f"activation_checkpoint={ac_mode!r}"
        )
    names = {f.name for f in dataclasses.fields(ac_config)} if dataclasses.is_dataclass(ac_config) else set()
    missing = [flag for flag in flags if flag not in names]
    if missing:
        raise ValueError(
            f"{named}: the selective-AC config {type(ac_config).__qualname__} has no field {missing}. These options "
            "need a torchtitan whose TPUSelectiveAC has the flags (recompute_bmm and save_pallas_attention: "
            "google-pytorch/torchtitan#168; save_kda_residuals: google-pytorch/torchtitan#170)."
        )
    config = dataclasses.replace(ac_config, **flags)
    owner = getattr(type(config), "_owner", None)
    logger.warning(
        "selective AC policy: %s(%s)",
        getattr(owner, "__name__", type(config).__qualname__),
        ", ".join(f"{flag}={getattr(config, flag)}" for flag in SELECTIVE_AC_FLAG_OPTIONS.values() if flag in names),
    )
    return config
