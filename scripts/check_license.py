#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Check that all Python files have a valid license header.

Aligned with verl-FL's check_license.py — supports multiple recognized
copyright holders (Bytedance, BAAI, etc.).
"""

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

# Recognized license headers (same as verl-FL)
license_headers = [
    "Copyright 2024 Bytedance Ltd. and/or its affiliates",
    "Copyright 2025 Bytedance Ltd. and/or its affiliates",
    "Copyright 2026 Bytedance Ltd. and/or its affiliates",
    "Copyright 2024 PRIME team and/or its affiliates",
    "Copyright 2025 Individual Contributor:",
    "Copyright 2023-2024 SGLang Team",
    "Copyright 2025 ModelBest Inc. and/or its affiliates",
    "Copyright 2025 Amazon.com Inc and/or its affiliates",
    "Copyright 2026 Amazon.com Inc and/or its affiliates",
    "Copyright (c) 2016-     Facebook, Inc",
    "Copyright 2025 Meituan Ltd. and/or its affiliates",
    "Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.",
    "Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.",
    "Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.",
    "Copyright (c) 2026 BAAI. All rights reserved.",
    "Copyright (c) 2026 Google LLC. All rights reserved.",
    "Copyright 2026 Advanced Micro Devices, Inc.",
    "Copyright (c) Meta Platforms, Inc. and affiliates.",
]


def _git_tracked_py_files() -> set[Path]:
    """Return the set of .py files tracked by git (respects .gitignore)."""
    result = subprocess.run(
        ["git", "ls-files", "*.py", "**/*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def get_py_files(path_arg: Path, tracked: set[Path]) -> Iterable[Path]:
    """Get py files under a directory that are git-tracked."""
    if path_arg.is_dir():
        return (p for p in tracked if p == path_arg or path_arg in p.parents)
    elif path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


def main():
    parser = ArgumentParser(description="Check license headers in Python files")
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="List of directories to check for license headers",
    )
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

        has_license = any(lh in content for lh in license_headers)
        if not has_license:
            failures.append(str(path))

    if failures:
        print(f"Missing license header in {len(failures)} file(s):")
        for f in failures:
            print(f"  {f}")
        print()
        print("Accepted headers:")
        for lh in license_headers:
            print(f"  # {lh}")
        sys.exit(1)


if __name__ == "__main__":
    main()
