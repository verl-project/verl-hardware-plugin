# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""SDK-free declarations shared by accelerator and integration registrations."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendRegistration:
    """Describe one backend without importing its implementation or optional SDKs.

    Module names are relative to ``package``. Each engine tuple is an import
    group: a failure skips the remainder of that group, not subsequent groups.
    Hooks must defer optional imports until called by their registration stage.
    """

    package: str
    platform: str | None = None
    engines: tuple[tuple[str, ...], ...] = ()
    profiler: Callable[[], None] | None = None
    rollout: Callable[[], None] | None = None
    engine_order: int = 0
