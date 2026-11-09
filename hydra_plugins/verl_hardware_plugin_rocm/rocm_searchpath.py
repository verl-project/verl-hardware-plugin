# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Add the ROCm plugin's Hydra config groups to every Hydra app (verl's ``main_ppo``).

Hydra discovers this module through the ``hydra_plugins`` namespace package. The groups only
add new option names (``model_engine=primus``, ``model_engine=maxtext``), so verl's own options
are unaffected.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "verl_hardware_plugin" / "accelerators" / "rocm" / "config"


class VerlRocmSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if _CONFIG_DIR.is_dir():
            search_path.append(provider="verl-hardware-plugin-rocm", path=f"file://{_CONFIG_DIR}")
