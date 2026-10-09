# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Intel XPU profiler extensions.

Nothing is registered from here. ``itt_profile_xpu`` is reached through
``PlatformXPU.profiler_markers()``, and ``VtuneProfiler`` is claimed for
``profiler.tool: vtune`` by ``register_vtune``, which ``PlatformXPU.__init__``
applies once it has confirmed a live XPU device -- not at import time.
"""
