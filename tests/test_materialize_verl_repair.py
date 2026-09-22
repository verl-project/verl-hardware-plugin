# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for structured Codex compatibility proposals."""

import json

import pytest

from scripts.materialize_verl_repair import materialize_proposal


def test_materialize_proposal_writes_a_unified_diff(tmp_path):
    output = tmp_path / "repair.patch"
    patch = (
        "diff --git a/verl_hardware_plugin/a.py b/verl_hardware_plugin/a.py\n"
        "--- a/verl_hardware_plugin/a.py\n"
        "+++ b/verl_hardware_plugin/a.py\n"
    )

    size = materialize_proposal(json.dumps({"patch": patch, "summary": "Update an import."}), output)

    assert size == len(patch.encode())
    assert output.read_text(encoding="utf-8") == patch


@pytest.mark.parametrize(
    "proposal",
    [
        {"patch": "", "summary": "No safe repair."},
        {"patch": "not a diff", "summary": "Invalid output."},
        {"patch": "diff --git a/a b/a\n", "summary": "Unexpected.", "extra": True},
        {"patch": "diff --git a/.git/config b/.git/config\n", "summary": "Unsafe path."},
        {
            "patch": "diff --git a/verl_hardware_plugin/a.py b/verl_hardware_plugin/a.py\n+++ /dev/null\n",
            "summary": "Deletion.",
        },
    ],
)
def test_materialize_proposal_rejects_invalid_payloads(tmp_path, proposal):
    with pytest.raises(ValueError):
        materialize_proposal(json.dumps(proposal), tmp_path / "repair.patch")
