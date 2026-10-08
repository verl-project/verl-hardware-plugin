# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plugin-side monkeypatches, one module per capability, applied only from
the platform class that needs them -- not from this package's `__init__.py`
and not from this plugin's shared top-level `verl_hardware_plugin/__init__.py`.

Each `PlatformXXX()` is only constructed once `verl.plugin.platform.
platform_manager._create_platform()` has actually selected that platform for
the process, so a platform's own `__init__` is the right place to apply its
patches. Applying a patch from a shared location instead would fire on
any host where the corresponding hardware/SDK merely happens to be
importable, regardless of which platform verl actually selects for the run.
"""
