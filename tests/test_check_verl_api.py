# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the static verl API compatibility checker."""

from pathlib import Path

from scripts.check_verl_api import check_contract


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_contract_accepts_definitions_and_reexports(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(
        plugin_root / "adapter.py",
        "from verl.api import PublicClass, public_function\nimport verl.api.impl\n",
    )
    _write(
        verl_root / "api" / "__init__.py",
        "from .impl import PublicClass\n\ndef public_function():\n    pass\n",
    )
    _write(verl_root / "api" / "impl.py", "class PublicClass:\n    pass\n")

    requirements, missing = check_contract(plugin_root, verl_root)

    assert len(requirements) == 3
    assert missing == []


def test_contract_reports_missing_module_and_symbol(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(
        plugin_root / "adapter.py",
        "from verl.api import RemovedClass\nfrom verl.gone import Missing\n",
    )
    _write(verl_root / "api" / "__init__.py", "class CurrentClass:\n    pass\n")

    _requirements, missing = check_contract(plugin_root, verl_root)

    assert [(item.requirement.qualified_name, item.reason) for item in missing] == [
        ("verl.api.RemovedClass", "symbol not found"),
        ("verl.gone.Missing", "module not found"),
    ]


def test_contract_checks_attributes_used_through_module_aliases(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(
        plugin_root / "adapter.py",
        "import verl.profiler.tools as tools\n\ndef patch():\n    return tools.removed_hook\n",
    )
    _write(verl_root / "profiler" / "tools.py", "def current_hook():\n    pass\n")

    _requirements, missing = check_contract(plugin_root, verl_root)

    assert [(item.requirement.qualified_name, item.reason) for item in missing] == [
        ("verl.profiler.tools.removed_hook", "symbol not found")
    ]


def test_contract_accepts_imported_submodules(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(plugin_root / "adapter.py", "from verl.api import helpers\n")
    _write(verl_root / "api" / "__init__.py", "")
    _write(verl_root / "api" / "helpers.py", "")

    _requirements, missing = check_contract(plugin_root, verl_root)

    assert missing == []


def test_contract_follows_relative_star_reexports(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(plugin_root / "adapter.py", "from verl.config import PublicConfig\n")
    _write(verl_root / "config" / "__init__.py", "from .model import *\n")
    _write(
        verl_root / "config" / "model.py",
        '__all__ = ["PublicConfig"]\n\nclass PublicConfig:\n    pass\n',
    )

    _requirements, missing = check_contract(plugin_root, verl_root)

    assert missing == []


def test_contract_honors_all_when_following_star_reexports(tmp_path):
    plugin_root = tmp_path / "plugin"
    verl_root = tmp_path / "upstream" / "verl"
    _write(plugin_root / "adapter.py", "from verl.config import InternalConfig\n")
    _write(verl_root / "config" / "__init__.py", "from .model import *\n")
    _write(
        verl_root / "config" / "model.py",
        '__all__ = ["PublicConfig"]\n\nclass PublicConfig:\n    pass\n\nclass InternalConfig:\n    pass\n',
    )

    _requirements, missing = check_contract(plugin_root, verl_root)

    assert [(item.requirement.qualified_name, item.reason) for item in missing] == [
        ("verl.config.InternalConfig", "symbol not found")
    ]
