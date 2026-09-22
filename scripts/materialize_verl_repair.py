#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Materialize a bounded unified diff from a structured Codex proposal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath


def _validate_diff_paths(patch: str) -> None:
    """Reject paths and patch forms outside the plugin Python allowlist."""

    lines = patch.splitlines()
    header_indexes = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not header_indexes:
        raise ValueError("proposal patch does not contain a Git diff header")
    for position, header_index in enumerate(header_indexes):
        header = lines[header_index]
        parts = header.split(" ")
        if len(parts) != 4 or not parts[2].startswith("a/") or not parts[3].startswith("b/"):
            raise ValueError("quoted or malformed Git diff paths are not allowed")
        old_path = parts[2][2:]
        new_path = parts[3][2:]
        normalized = PurePosixPath(new_path)
        if old_path != new_path:
            raise ValueError("automated compatibility patches may not rename files")
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != new_path
            or not new_path.startswith("verl_hardware_plugin/")
            or normalized.suffix != ".py"
        ):
            raise ValueError(f"patch path is outside the Python allowlist: {new_path}")

        next_header = header_indexes[position + 1] if position + 1 < len(header_indexes) else len(lines)
        metadata = lines[header_index + 1 : next_header]
        hunk_start = next((index for index, line in enumerate(metadata) if line.startswith("@@")), len(metadata))
        metadata = metadata[:hunk_start]
        old_markers = [line for line in metadata if line.startswith("--- ")]
        new_markers = [line for line in metadata if line.startswith("+++ ")]
        if len(old_markers) != 1 or old_markers[0] not in {f"--- a/{old_path}", "--- /dev/null"}:
            raise ValueError("patch has a missing or inconsistent old-file marker")
        if new_markers != [f"+++ b/{new_path}"]:
            raise ValueError("patch has a missing or inconsistent new-file marker")

    if "\nGIT binary patch\n" in patch or "\nBinary files " in patch:
        raise ValueError("binary patches are not allowed")


def materialize_proposal(proposal: str, output: Path, max_bytes: int = 60_000) -> int:
    """Validate a repair proposal and write its patch to ``output``."""

    value = json.loads(proposal)
    if not isinstance(value, dict) or set(value) != {"patch", "summary"}:
        raise ValueError("proposal must contain exactly 'patch' and 'summary'")
    patch = value["patch"]
    summary = value["summary"]
    if not isinstance(patch, str) or not isinstance(summary, str):
        raise ValueError("proposal patch and summary must be strings")
    if not patch.startswith("diff --git a/"):
        raise ValueError("proposal patch must be a non-empty Git unified diff")
    if "\0" in patch:
        raise ValueError("proposal patch may not contain NUL bytes")
    _validate_diff_paths(patch)
    encoded = patch.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"proposal patch is {len(encoded)} bytes; limit is {max_bytes}")
    if not patch.endswith("\n"):
        patch += "\n"
        encoded = patch.encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return len(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=60_000)
    args = parser.parse_args()
    size = materialize_proposal(sys.stdin.read(), args.output, args.max_bytes)
    print(f"Materialized compatibility proposal: {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
