# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Cnprec profiler integration for verl.

This module monkey-patches verl's ``DistProfiler`` dispatcher so that
``global_profiler.tool=cnprec`` routes to cnprec's ``VerlTrainingAnalyzer``
without modifying upstream verl. It reuses ``DistProfiler``'s existing
rank/step gating and only wraps the same ``@DistProfiler.annotate`` stage
ranges already present in verl workers (``actor_compute_log_prob``,
``actor_update``, ``ref_compute_log_prob``, ``critic_update``,
``compute_values``, ``compute_rm_score``).

The dispatch relies on verl/cnprec private APIs (``DistProfiler``'s
``_tool``/``_impl`` layout, ``VerlTrainingAnalyzer``'s ``dumper``/``save_data``
interface). Mismatches are detected and logged loudly instead of silently
producing no profiling output.

Unlike cntrainkit (a decorator-style hook), cnprec attaches forward/backward
hooks to the model plus an optimizer step pre-hook. Therefore the backend
lazily constructs one ``VerlTrainingAnalyzer`` per (role, model, optimizer)
on the first profiled step and drives ``dumper.start(block)`` /
``dumper.step()`` around the annotated stage.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)

_original_dist_profiler_init = None

# verl annotate stage -> cnprec verl block name.
# cnprec verl only defines two dump blocks: compute_log_prob / update_policy.
_STAGE_TO_BLOCK = {
    "actor_compute_log_prob": "compute_log_prob",
    "ref_compute_log_prob": "compute_log_prob",
    "compute_values": "compute_log_prob",  # critic forward; reuse log_prob block
    "compute_rm_score": "compute_log_prob",  # reward model forward; reuse log_prob block
    "actor_update": "update_policy",
    "critic_update": "update_policy",
}

# Attribute paths (most specific first) used to resolve the nn.Module and
# optimizer from the annotated worker instance. Mirrors precision_debugger's
# _MODEL_ATTRS_BY_ROLE, with optimizer paths added.
_ROLE_MODEL_ATTRS = {
    "actor": (
        "actor.engine.module",
        "actor.actor_module",
        "actor.actor_module_fsdp",
        "actor_module_fsdp",
        "actor_module",
    ),
    "ref": (
        "ref.engine.module",
        "ref.actor_module",
        "ref_policy.actor_module",
        "ref_module_fsdp",
        "ref_module",
        "ref_policy.ref_module",
    ),
    "critic": (
        "critic.engine.module",
        "critic.critic_module",
        "critic_module_fsdp",
        "critic_module",
    ),
    "reward_model": (
        "reward_model.engine.module",
        "reward_model_module_fsdp",
        "reward_model_module",
        "rm.reward_model_module",
    ),
}

_STAGE_TO_ROLE = {
    "actor_update": "actor",
    "actor_compute_log_prob": "actor",
    "ref_compute_log_prob": "ref",
    "compute_values": "critic",
    "critic_update": "critic",
    "compute_rm_score": "reward_model",
}


@dataclass
class CnprecToolConfig(BaseConfig):
    """Cnprec profiler config.

    config_path: path to ``analyze_verl_config.json`` (cnprec verl config).
    stages: optional subset of annotate stages to collect; None = all known.
    strict: if True, raise when cnprec is unavailable; otherwise run without.
    """

    name: str = "cnprec"
    config_path: Optional[str] = None
    stages: Optional[list] = None
    strict: bool = False

    def __post_init__(self) -> None:
        if self.config_path is not None and not isinstance(self.config_path, str):
            raise AssertionError(f"config_path must be str or None, got {type(self.config_path)}")
        if self.stages is not None and not isinstance(self.stages, list):
            raise AssertionError(f"stages must be list or None, got {type(self.stages)}")
        if not isinstance(self.strict, bool):
            raise AssertionError(f"strict must be bool, got {type(self.strict)}")


def _import_cnprec():
    """Import cnprec's VerlTrainingAnalyzer lazily.

    Returns the class or raises ImportError when cnprec is unavailable.
    """
    from cnprec.analyzer.rl_analyzer import VerlTrainingAnalyzer

    return VerlTrainingAnalyzer


def _resolve_attr(obj: Any, attr_path: str) -> Any:
    current = obj
    for part in attr_path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _is_valid_model(model: Any) -> bool:
    return model is not None and callable(getattr(model, "forward", None))


def _resolve_model(self_instance: Any, role: str) -> Any:
    for attr in _ROLE_MODEL_ATTRS.get(role, ()):
        value = _resolve_attr(self_instance, attr)
        if _is_valid_model(value):
            return value
        if isinstance(value, (list, tuple)):
            models = [m for m in value if _is_valid_model(m)]
            if models:
                if len(models) > 1:
                    logger.warning(
                        "cnprec binds the first of %d model chunks for role '%s'",
                        len(models),
                        role,
                    )
                return models[0]
    fallback = getattr(self_instance, "module", None)
    return fallback if _is_valid_model(fallback) else None


def _resolve_optimizer(self_instance: Any, role: str) -> Any:
    """Best-effort optimizer resolution. Only actor/critic typically have one.

    verl 的优化器封装在 engine 里（FSDP/Megatron/...），如
    ``WorkerActor.engine`` 的 ``FSDPEngine.optimizer``，故需先沿
    ``actor.engine.optimizer`` / ``engine.optimizer`` 取，再回退到
    actor 直接属性。
    """
    candidates = {
        "actor": (
            "actor.engine.optimizer",  # FSDP/Megatron engine 优化器（首选）
            "engine.optimizer",
            "actor.actor_optimizer",
            "actor.optimizer",
            "actor_optimizer",
            "optimizer",
        ),
        "critic": (
            "critic.engine.optimizer",
            "engine.optimizer",
            "critic.critic_optimizer",
            "critic.optimizer",
            "critic_optimizer",
        ),
    }.get(role, ())
    for attr in candidates:
        value = _resolve_attr(self_instance, attr)
        if value is not None:
            return value
    return None


class CnprecProfiler:
    """Adapter that drives cnprec's ``VerlTrainingAnalyzer`` from verl stages.

    One instance is created per ``DistProfiler`` (i.e. per worker role). The
    underlying ``VerlTrainingAnalyzer`` is constructed lazily on the first
    profiled stage, so non-profiled steps pay no hook overhead.
    """

    def __init__(self, rank: int, config: Any, tool_config: Any = None, **kwargs):
        self.rank = rank
        self.config = config
        self.tool_config = _normalize_tool_config(_extract_cnprec_config(config, tool_config))
        self.profile_step = None
        self._force_profile = False
        # Lazily-built analyzers keyed by (role, block) so ref/actor don't share.
        self._analyzers: dict = {}
        self._stages = self._normalize_stages(self.tool_config.stages)

    @staticmethod
    def _normalize_stages(stages: Optional[list]) -> Optional[set]:
        if stages is None:
            return None
        return set(stages)

    def _should_collect(self, stage: str) -> bool:
        if stage not in _STAGE_TO_BLOCK:
            return False
        if self._stages is not None and stage not in self._stages:
            return False
        return True

    def start(self, **kwargs) -> None:
        """Called by DistProfiler at the beginning of a profiled global step."""
        self.profile_step = kwargs.get("profile_step", kwargs.get("global_step"))
        self._force_profile = True

    def stop(self) -> None:
        """Called by DistProfiler at the end of a profiled global step.

        Advances every active analyzer one step (writes per-step artifacts).
        """
        for analyzer in self._analyzers.values():
            try:
                analyzer.cur_step = self.profile_step if self.profile_step is not None else analyzer.cur_step
                analyzer.step()
            except Exception as e:
                if self.tool_config.strict:
                    raise
                logger.warning("cnprec analyzer.step() failed: %s", e)
        self.profile_step = None
        self._force_profile = False

    def step(self) -> None:
        # DistProfiler.step() is a no-op for non-torch backends; kept for API parity.
        return

    def _get_or_build_analyzer(self, self_instance: Any, role: str, block: str):
        key = (role, block)
        if key in self._analyzers:
            return self._analyzers[key]
        model = _resolve_model(self_instance, role)
        optimizer = _resolve_optimizer(self_instance, role) if block == "update_policy" else None
        if model is None:
            return None
        try:
            VerlTrainingAnalyzer = _import_cnprec()
        except Exception as exc:
            if self.tool_config.strict:
                raise ImportError("cnprec is not available but cnprec.strict is True") from exc
            logger.warning("cnprec unavailable, profiler disabled: %s", exc)
            return None
        # The environment gate CNPREC_VERL_ANALYZE must be true for the analyzer
        # to activate; set it from our own enablement if not already set.
        os.environ.setdefault("CNPREC_VERL_ANALYZE", "true")
        config_path = self.tool_config.config_path or os.environ.get("CNPREC_VERL_CONFIG_PATH")
        if not config_path:
            if self.tool_config.strict:
                raise ValueError(
                    "cnprec.config_path not set (provide <role>.profiler.tool_config.cnprec.config_path, "
                    "e.g. +actor_rollout_ref.actor.profiler.tool_config.cnprec.config_path=/path/to/config.json, "
                    "or env CNPREC_VERL_CONFIG_PATH)"
                )
            logger.warning("cnprec.config_path not set, profiler disabled")
            return None
        try:
            analyzer = VerlTrainingAnalyzer(
                actor_module=model,
                actor_optimizer=optimizer,
                verl_config_path=config_path,
            )
        except Exception as e:
            # cnprec's constructor signature is an internal API; a version
            # mismatch must degrade to "no profiling", not crash the stage.
            if self.tool_config.strict:
                raise
            logger.warning("cnprec VerlTrainingAnalyzer construction failed: %s", e)
            return None
        analyzer.cur_step = self.profile_step if self.profile_step is not None else 0
        analyzer.cur_rank = self.rank
        self._analyzers[key] = analyzer
        return analyzer

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        stage = kwargs_outer.get("role") or message

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                if not self._force_profile or not self._should_collect(stage):
                    return func(self_instance, *args, **kwargs_inner)
                role = _STAGE_TO_ROLE.get(stage)
                block = _STAGE_TO_BLOCK.get(stage)
                if role is None or block is None:
                    return func(self_instance, *args, **kwargs_inner)
                analyzer = self._get_or_build_analyzer(self_instance, role, block)
                if analyzer is None:
                    return func(self_instance, *args, **kwargs_inner)
                try:
                    analyzer.cur_step = self.profile_step if self.profile_step is not None else analyzer.cur_step
                    analyzer.dumper.cur_step = analyzer.cur_step
                    analyzer.dumper.start(block)
                except Exception as e:
                    if self.tool_config.strict:
                        raise
                    logger.warning("cnprec dumper.start(%s) failed: %s", block, e)
                    return func(self_instance, *args, **kwargs_inner)
                try:
                    result = func(self_instance, *args, **kwargs_inner)
                    # cnprec: dump per-step data (log_probs/prompts/responses/...).
                    # save_data expects a dict-like batch; the
                    # verl worker methods receive a TensorDict/DataProto as args[0].
                    try:
                        from cnprec.common.utils import is_rank_to_dump, is_step_to_dump

                        should = is_step_to_dump(
                            analyzer.cur_step, analyzer.verl_config.get("steps", [])
                        ) and is_rank_to_dump(analyzer.cur_rank, analyzer.verl_config.get("ranks", []))
                        if should:
                            batch = args[0] if args else None
                            if batch is not None and hasattr(batch, "to_dict"):
                                batch = batch.to_dict()
                            if batch is not None and hasattr(batch, "get"):
                                analyzer.save_data(batch)
                    except Exception as e:
                        if self.tool_config.strict:
                            raise
                        logger.warning("cnprec save_data/snapshot failed: %s", e)
                    return result
                finally:
                    try:
                        analyzer.dumper.end(block)
                    except Exception as e:
                        if self.tool_config.strict:
                            raise
                        logger.warning("cnprec dumper.end(%s) failed: %s", block, e)

            return wrapper

        return decorator


def _get_config_value(config: Any, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except TypeError:
            pass
    return getattr(config, name, default)


def _extract_cnprec_config(config: Any, tool_config: Any):
    """Locate the cnprec tool config among the profiler config objects.

    Accepts (in order) the tool config itself, a ``cnprec`` key nested inside
    it, or the same two shapes on ``config.tool_config``. A config that
    belongs to another tool (named, or carrying only foreign keys) is never
    mistaken for ours: the fallback returns ``None`` (defaults) instead of
    swallowing it, so a missing cnprec config surfaces as the explicit
    "config_path not set" warning rather than silently profiling nothing.
    """

    def _is_cnprec(source: Any) -> bool:
        name = _get_config_value(source, "name")
        if name == "cnprec":
            return True
        if name is not None:
            return False  # explicitly belongs to another tool
        return any(_get_config_value(source, key) is not None for key in ("config_path", "stages", "strict"))

    if _is_cnprec(tool_config):
        return tool_config
    nested = _get_config_value(tool_config, "cnprec")
    if nested is not None:
        return nested
    config_tool_config = getattr(config, "tool_config", None)
    if _is_cnprec(config_tool_config):
        return config_tool_config
    nested = _get_config_value(config_tool_config, "cnprec")
    if nested is not None:
        return nested
    return None


def _normalize_tool_config(tool_config: Any) -> CnprecToolConfig:
    if isinstance(tool_config, CnprecToolConfig):
        return tool_config
    if tool_config is None:
        return CnprecToolConfig()
    return CnprecToolConfig(
        config_path=_get_config_value(tool_config, "config_path", None),
        stages=_get_config_value(tool_config, "stages", None),
        strict=_get_config_value(tool_config, "strict", False),
    )


def _patch_cnprec_config() -> None:
    """Expose the config class from verl.utils.profiler.config for Hydra targets."""
    import verl.utils.profiler.config as profiler_config

    if not hasattr(profiler_config, "CnprecToolConfig"):
        profiler_config.CnprecToolConfig = CnprecToolConfig


def _patch_dist_profiler() -> None:
    """Patch DistProfiler to dispatch tool='cnprec' to CnprecProfiler."""
    from verl.utils.profiler.profile import DistProfiler

    global _original_dist_profiler_init
    if _original_dist_profiler_init is not None:
        return

    if not callable(getattr(DistProfiler, "annotate", None)):
        logger.warning(
            "[verl_hardware_plugin] verl DistProfiler no longer exposes annotate(); cnprec stage wrapping "
            "is unavailable. verl's profiler private API changed -- the plugin needs adapting; cnprec "
            "profiling will not run."
        )
        return

    _original_dist_profiler_init = DistProfiler.__init__

    @functools.wraps(_original_dist_profiler_init)
    def _patched_init(self, rank: int, config=None, tool_config=None, **kwargs):
        _original_dist_profiler_init(self, rank=rank, config=config, tool_config=tool_config, **kwargs)
        if getattr(self, "_tool", None) == "cnprec":
            # Rank selection stays verl's: the original init above already
            # computed self._this_rank from config.all_ranks / config.ranks,
            # so global_profiler.ranks is honored. Users who need every rank
            # in the profiled range set all_ranks=True (see profiling.md);
            # cnprec's own analyze_verl_config.json "ranks" further filters
            # which ranks actually dump data.
            # Resolve the class at call time (not via this module's globals):
            # test harnesses may reload plugin modules, and closing over the
            # class at patch time would then fail isinstance checks against
            # the reloaded module's class.
            from verl_hardware_plugin.accelerators.mlu.profilers.cnprec_profile import CnprecProfiler

            self._impl = CnprecProfiler(rank=rank, config=self.config, tool_config=self.tool_config, **kwargs)
            self._discrete = True
        elif getattr(config, "tool", None) == "cnprec" and getattr(self, "_enable", False):
            logger.warning(
                "[verl_hardware_plugin] profiler config requests tool='cnprec' but DistProfiler no longer "
                "exposes _tool; cnprec cannot be dispatched. verl's profiler private API changed -- the "
                "plugin needs adapting; profiling will not run."
            )

    DistProfiler.__init__ = _patched_init
    logger.info("[verl_hardware_plugin] Patched DistProfiler: +cnprec")


def apply_cnprec_profiler_patch() -> None:
    """Register cnprec as a verl profiler backend. Idempotent."""
    _patch_cnprec_config()
    _patch_dist_profiler()
