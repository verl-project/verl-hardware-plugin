# Copyright 2025 Intel Corporation
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Register ``VtuneProfiler`` for ``profiler.tool: vtune`` by patching ``DistProfiler``.

``DistProfiler.__init__`` resolves ``profiler.tool`` against its built-in tool
names and leaves a ``_NoOpProfiler`` for anything else, so the plugin claims the
``vtune`` name by wrapping that constructor.

:func:`apply_vtune_profiler_patch` is applied by ``PlatformXPU`` once it has
confirmed a live XPU device, never at import time.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _xpu_available() -> bool:
    try:
        import torch

        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:  # pragma: no cover - a probe must never raise
        return False


def apply_vtune_profiler_patch() -> None:
    """Make ``DistProfiler(config=...tool="vtune")`` build a ``VtuneProfiler``.

    Idempotent, and a no-op without a usable XPU device or when verl core does
    not expose the profiler internals this relies on -- a failed profiler
    registration must never break training.
    """
    global _PATCHED
    if _PATCHED:
        return
    if not _xpu_available():
        return

    try:
        from verl.utils.profiler.profile import DistProfiler
    except ImportError:  # pragma: no cover - verl is a hard dependency in practice
        logger.debug("verl.utils.profiler.profile unavailable; skipping vtune registration")
        return

    original_init = DistProfiler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Core has already resolved the tool name; only "vtune" is ours to take over.
        if getattr(self, "_tool", None) != "vtune":
            return
        from verl_hardware_plugin.accelerators.xpu.profilers.itt_profile_xpu import VtuneProfiler

        self._impl = VtuneProfiler(
            rank=self.rank,
            config=self.config,
            tool_config=self.tool_config,
            save_file_prefix=self.save_file_prefix,
        )

    DistProfiler.__init__ = patched_init
    _PATCHED = True
    logger.debug("registered VtuneProfiler for profiler.tool='vtune'")
