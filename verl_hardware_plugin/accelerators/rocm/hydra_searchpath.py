# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Add the ROCm Hydra config groups (``model_engine=primus``, ``model_engine=maxtext``) to verl's search path.

verl loads this plugin when ``verl`` is imported, before ``main_ppo`` composes its config, so registering the
search-path plugin here is enough. The groups only add new option names and are appended after verl's own
config directory, so verl's options are unaffected.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

_CONFIG_DIR = Path(__file__).resolve().parent / "config"


class VerlRocmSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="verl-hardware-plugin-rocm", path=f"file://{_CONFIG_DIR}")


def register_search_path() -> None:
    Plugins.instance().register(VerlRocmSearchPathPlugin)
