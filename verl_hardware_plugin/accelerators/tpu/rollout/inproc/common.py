# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Logging, switches and small collective helpers shared by the in-process rollout modules."""

from __future__ import annotations

import logging
import os
import re
import traceback
from collections.abc import Mapping
from typing import Any, Optional

# One logger for the whole package; every line starts with ``inproc``. Quiet by default (the plugin's convention).
logger = logging.getLogger("verl_hardware_plugin.accelerators.tpu.rollout.inproc")
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PREFIX = "inproc"
INTEGRITY_TAG = "INTEGRITY FAILED"

# The feature switch (opt-in, default off) and the variable the platform sets on the trainer worker groups.
INPROC_ENV = "VERL_TPU_ROLLOUT_INPROC"
ROLE_ENV = "VERL_TPU_INPROC_ROLE"
# The run-level switches. Every other safety decision is fixed in code.
PARITY_EVERY_ENV = (
    "VERL_TPU_INPROC_PARITY_EVERY"  # post-install parity every N steps (default 5; 0 = only the fixed syncs)
)
LAST_STEP_ENV = "VERL_TPU_INPROC_LAST_STEP"  # the run's last training step: verified too
GC_RSS_GIB_ENV = "VERL_TPU_INPROC_GC_RSS_GIB"  # RSS growth (GiB) that triggers a collect + freeze (default 8; 0 = off)
INTEGRITY_ENV = (
    "VERL_TPU_INPROC_INTEGRITY"  # "fatal" (default) | "report": the staleness checks and the KV capacity check
)
SWITCHES = (ROLE_ENV, PARITY_EVERY_ENV, LAST_STEP_ENV, GC_RSS_GIB_ENV, INTEGRITY_ENV)
SWITCH_PREFIX = "VERL_TPU_INPROC_"

PARITY_EVERY = 5
GC_RSS_GIB = 8.0


def log(msg: str) -> None:
    logger.info(f"{PREFIX} {msg}")


def warn(msg: str) -> None:
    logger.warning(f"{PREFIX} {msg}")


def inproc_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(INPROC_ENV, "").strip() == "1"


def _choice(env: Mapping[str, str], name: str, allowed: tuple[str, ...]) -> str:
    raw = (env.get(name) or "").strip()
    value = raw or allowed[0]
    if value not in allowed:
        raise RuntimeError(f"{name}={raw!r}: expected one of {list(allowed)}")
    return value


def integrity_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """``fatal`` (default) or ``report``: covers only the staleness checks and the KV capacity check."""
    return _choice(os.environ if environ is None else environ, INTEGRITY_ENV, ("fatal", "report"))


def parity_every(environ: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if environ is None else environ
    raw = (env.get(PARITY_EVERY_ENV) or "").strip()
    if not raw:
        return PARITY_EVERY
    if not raw.isdigit():
        raise RuntimeError(f"{PARITY_EVERY_ENV}={raw!r}: expected a non-negative integer")
    return int(raw)


def last_step(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    env = os.environ if environ is None else environ
    raw = (env.get(LAST_STEP_ENV) or "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise RuntimeError(f"{LAST_STEP_ENV}={raw!r}: expected the run's last training step")
    return int(raw)


def gc_rss_limit_bytes(environ: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if environ is None else environ
    raw = (env.get(GC_RSS_GIB_ENV) or "").strip()
    try:
        gib = float(raw) if raw else GC_RSS_GIB
    except ValueError:
        raise RuntimeError(f"{GC_RSS_GIB_ENV}={raw!r}: expected a number of GiB") from None
    if gib < 0:
        raise RuntimeError(f"{GC_RSS_GIB_ENV}={raw!r}: expected a number of GiB >= 0")
    return int(gib * 2**30)


def known_switches() -> tuple[str, ...]:
    """The ``VERL_TPU_INPROC_*`` variables this package reads (the later parts of the package register theirs)."""
    return SWITCHES + tuple(_EXTRA_SWITCHES)


_EXTRA_SWITCHES: list[str] = []


def register_switch(name: str) -> str:
    if name not in _EXTRA_SWITCHES:
        _EXTRA_SWITCHES.append(name)
    return name


def check_switches(environ: Optional[Mapping[str, str]] = None) -> list[str]:
    """Validate the switches (a malformed value refuses) and return the ``VERL_TPU_INPROC_*`` variables that are set
    but not read by this package: none of them changes what the in-process rollout does."""
    env = os.environ if environ is None else environ
    if not inproc_enabled(env):
        raise RuntimeError(f"{INPROC_ENV}={env.get(INPROC_ENV)!r}: the in-process rollout is not switched on")
    integrity_mode(env)
    parity_every(env)
    last_step(env)
    gc_rss_limit_bytes(env)
    for check in _EXTRA_CHECKS:
        check(env)
    known = set(known_switches())
    return sorted(k for k in env if k.startswith(SWITCH_PREFIX) and k not in known)


_EXTRA_CHECKS: list = []


def register_switch_check(fn) -> None:
    if fn not in _EXTRA_CHECKS:
        _EXTRA_CHECKS.append(fn)


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """A field of a verl config dataclass, dict or OmegaConf node (None-safe, never raises)."""
    if config is None:
        return default
    try:
        if hasattr(config, "get"):
            return config.get(key, default)
    except Exception:  # noqa: BLE001
        pass
    try:
        return getattr(config, key, default)
    except Exception:  # noqa: BLE001
        return default


def dist_rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0") or 0), int(os.environ.get("WORLD_SIZE", "1") or 1)


def all_ranks_ok(flag: bool) -> bool:
    """MIN of ``flag`` over the trainer's world (a CPU tensor on the gloo side of the default group).

    Every trainer rank reaches each call site in lockstep (they run inside one worker-group method), so a rank that
    failed a local step makes every rank skip the next collective step instead of leaving the others waiting in it.
    """
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(int(t.item()))


def say_build_failure(stage: str, rank: int, exc: BaseException) -> None:
    """The failing rank's own exception and traceback tail, logged before the cross-rank agreement raises."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tail = " / ".join(x.strip().replace("\n", " ")[:300] for x in tb[-4:])
    warn(f"engine {stage} FAILED rank {rank}: {type(exc).__name__}: {exc} | {tail}")


def matches_any(patterns: tuple[str, ...] | list[str], name: str) -> bool:
    return any(re.search(p, name) for p in patterns)
