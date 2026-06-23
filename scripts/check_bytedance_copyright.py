#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Check that files have a recognized copyright header.

This plugin repo is jointly maintained by Bytedance and BAAI. Acceptable
copyright combinations:
- BAAI only
- Bytedance only
- Both BAAI and Bytedance
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

BYTEDANCE_MARKERS = [
    "Copyright 2024 Bytedance Ltd. and/or its affiliates",
    "Copyright 2025 Bytedance Ltd. and/or its affiliates",
    "Copyright 2026 Bytedance Ltd. and/or its affiliates",
]

BAAI_MARKER = "Copyright (c) 2026 BAAI. All rights reserved."

# A file is valid if it has at least one recognized copyright header.
# Both Bytedance and BAAI copyrights are acceptable independently or together.


def _git_tracked_py_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py", "**/*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def get_py_files(path_arg: Path, tracked: set[Path]) -> Iterable[Path]:
    if path_arg.is_dir():
        return (p for p in tracked if p == path_arg or path_arg in p.parents)
    elif path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


def main():
    parser = ArgumentParser()
    parser.add_argument("--directories", "-d", required=True, type=Path, nargs="+")
    args = parser.parse_args()

    tracked = _git_tracked_py_files()
    pathlist = sorted(set(path for path_arg in args.directories for path in get_py_files(path_arg, tracked)))

    failures = []
    for path in pathlist:
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not content.strip():
            continue

        has_bytedance = any(marker in content for marker in BYTEDANCE_MARKERS)
        has_baai = BAAI_MARKER in content

        # File must have at least one recognized copyright header
        if not has_bytedance and not has_baai:
            failures.append(str(path))

    if failures:
        print(f"Found {len(failures)} file(s) missing a recognized copyright header:")
        for f in failures:
            print(f"  {f}")
        print()
        print("Please add one of:")
        print(f"  # {BAAI_MARKER}")
        print("  # Copyright 2024 Bytedance Ltd. and/or its affiliates")
        sys.exit(1)


if __name__ == "__main__":
    main()
