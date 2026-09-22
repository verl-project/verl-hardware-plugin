# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for automated compatibility patch validation."""

import subprocess
from pathlib import Path

import pytest

from scripts.validate_verl_compat_changes import validate_staged_patch


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    (repo / "compat").mkdir(parents=True)
    (repo / "compat" / "verl-release.json").write_text('{"tag": "v0.9.0", "commit": "old"}\n', encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "Compatibility Test")
    _git(repo, "config", "user.email", "compat-test@example.com")
    _git(repo, "add", "compat/verl-release.json")
    _git(repo, "commit", "-m", "initial")


def test_validator_accepts_release_marker_update(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    marker = tmp_path / "compat" / "verl-release.json"
    marker.write_text('{"tag": "v0.9.1", "commit": "new"}\n', encoding="utf-8")
    _git(tmp_path, "add", "compat/verl-release.json")
    monkeypatch.chdir(tmp_path)

    assert validate_staged_patch() == (1, 2)


def test_validator_rejects_workflow_changes(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "untrusted.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: untrusted\n", encoding="utf-8")
    _git(tmp_path, "add", ".github/workflows/untrusted.yml")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="outside the compatibility repair allowlist"):
        validate_staged_patch()


def test_validator_rejects_non_python_plugin_files(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    generated = tmp_path / "verl_hardware_plugin" / "generated.json"
    generated.parent.mkdir(parents=True)
    generated.write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "add", "verl_hardware_plugin/generated.json")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="file type is not allowed"):
        validate_staged_patch()
