# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Each accelerator backend owns its pip extra: `pip install "verl-hardware-plugin[<backend>]"`.

The extras are declared in pyproject.toml as `[tool.setuptools.dynamic.optional-dependencies]`, one requirements
file per backend under verl_hardware_plugin/accelerators/<backend>/. These checks run on CPU and need no network
and no build.
"""

from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    from pip._vendor import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
ACCELERATORS = ROOT / "verl_hardware_plugin" / "accelerators"
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
EXTRAS = PYPROJECT["tool"]["setuptools"]["dynamic"]["optional-dependencies"]


def _requirement_lines(path: Path):
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line:
            yield number, line


def test_extras_are_dynamic():
    assert "optional-dependencies" in PYPROJECT["project"]["dynamic"]
    assert "optional-dependencies" not in PYPROJECT["project"], "declare extras only in the dynamic table"


@pytest.mark.parametrize("extra", sorted(EXTRAS))
def test_extra_files_exist_and_parse(extra):
    for rel in EXTRAS[extra]["file"]:
        path = ROOT / rel
        assert path.is_file(), f"extra [{extra}]: {rel} is missing"
        for number, line in _requirement_lines(path):
            assert not line.startswith("-"), (
                f"{rel}:{number}: option lines such as {line.split()[0]} are not allowed in an extra; "
                "put index flags in the install guide"
            )
            try:
                requirement = Requirement(line)
            except InvalidRequirement as e:
                pytest.fail(f"{rel}:{number}: not a valid requirement: {e}")
            assert requirement.url is None, f"{rel}:{number}: PyPI rejects direct URL dependencies in package metadata"


def test_every_backend_has_an_extra():
    backends = {p.name for p in ACCELERATORS.iterdir() if (p / "__init__.py").is_file()}
    missing = sorted(b for b in backends if b not in EXTRAS)
    assert not missing, f"backends without a [{'|'.join(missing)}] extra: add a requirements.txt and a pyproject entry"
    for backend in backends:
        assert (ACCELERATORS / backend / "requirements.txt").is_file()


def _pins(relative_path):
    return {Requirement(line).name: Requirement(line) for _, line in _requirement_lines(ACCELERATORS / relative_path)}


def test_xpu_pins_are_consistent():
    xpu = _pins("xpu/requirements.txt")
    vllm = _pins("xpu/requirements-vllm.txt")
    assert str(xpu["torch"].specifier).startswith("==") and "+xpu" in str(xpu["torch"].specifier)
    assert "+xpu" in str(vllm["vllm"].specifier)
