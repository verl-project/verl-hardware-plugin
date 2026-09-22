#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Validate a staged, automatically generated verl compatibility patch."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import PurePosixPath

ALLOWED_EXACT = {"compat/verl-release.json"}
ALLOWED_PREFIXES = ("verl_hardware_plugin/",)


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _staged_paths(base: str) -> list[str]:
    output = _git_bytes("diff", "--cached", "--name-only", "--no-renames", "-z", base)
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def _validate_path(path: str) -> None:
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts or normalized.as_posix() != path:
        raise ValueError(f"unsafe path: {path!r}")
    if path in ALLOWED_EXACT:
        if normalized.suffix != ".json":
            raise ValueError(f"release marker must be JSON: {path}")
        return
    if not path.startswith(ALLOWED_PREFIXES):
        raise ValueError(f"path is outside the compatibility repair allowlist: {path}")
    if normalized.suffix != ".py":
        raise ValueError(f"file type is not allowed in an automated compatibility patch: {path}")


def _changed_line_count(base: str) -> int:
    output = _git_bytes("diff", "--cached", "--numstat", "--no-renames", "-z", base)
    total = 0
    for record in output.split(b"\0"):
        if not record:
            continue
        added, deleted, _path = record.split(b"\t", 2)
        if added == b"-" or deleted == b"-":
            raise ValueError("binary files are not allowed in an automated compatibility patch")
        total += int(added) + int(deleted)
    return total


def _validate_index_modes(paths: list[str]) -> None:
    for path in paths:
        output = _git_bytes("ls-files", "--stage", "-z", "--", path)
        if not output:
            continue  # A deleted file has no index entry.
        mode = output.split(b" ", 1)[0].decode("ascii")
        if mode != "100644":
            raise ValueError(f"only regular non-executable files are allowed: {path} has mode {mode}")


def validate_staged_patch(
    base: str = "HEAD", max_files: int = 25, max_lines: int = 1500, max_bytes: int = 1_000_000
) -> tuple[int, int]:
    """Validate allowed paths, file modes, and patch size."""

    paths = _staged_paths(base)
    if not paths:
        raise ValueError("the staged compatibility patch is empty")
    if len(paths) > max_files:
        raise ValueError(f"patch changes {len(paths)} files; limit is {max_files}")
    for path in paths:
        _validate_path(path)
    deleted = _git_bytes("diff", "--cached", "--name-only", "--diff-filter=D", "-z", base)
    if deleted:
        raise ValueError("automated compatibility patches may not delete files")
    _validate_index_modes(paths)
    changed_lines = _changed_line_count(base)
    if changed_lines > max_lines:
        raise ValueError(f"patch changes {changed_lines} lines; limit is {max_lines}")
    patch_size = len(_git_bytes("diff", "--cached", "--binary", "--no-renames", base))
    if patch_size > max_bytes:
        raise ValueError(f"patch is {patch_size} bytes; limit is {max_bytes}")
    for path in paths:
        blob_size = len(_git_bytes("show", f":{path}"))
        if blob_size > max_bytes:
            raise ValueError(f"staged file is {blob_size} bytes; limit is {max_bytes}: {path}")
    return len(paths), changed_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--max-files", type=int, default=25)
    parser.add_argument("--max-lines", type=int, default=1500)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    file_count, line_count = validate_staged_patch(args.base, args.max_files, args.max_lines, args.max_bytes)
    print(f"Validated compatibility patch: {file_count} files, {line_count} changed lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
