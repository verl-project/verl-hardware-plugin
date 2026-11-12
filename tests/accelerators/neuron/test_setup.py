# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Exercise installer orchestration without downloading packages or loading Neuron."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/accelerators/neuron/setup_neuron_public.sh"
STUB = """
import json
import os
import pathlib
import sys

command, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
record = {"command": command, "args": args}
if "--excludes" in args:
    record["excluded"] = pathlib.Path(args[args.index("--excludes") + 1]).read_text().splitlines()
with open(os.environ["SETUP_TEST_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
if command == "git" and "rev-parse" in args:
    print((pathlib.Path(args[1]) / ".git" / "revision").read_text().strip())
elif command == "uv":
    if args[:2] == ["python", "find"]:
        print(sys.executable)
    elif args[0] == "venv":
        target = pathlib.Path(args[-1]) / "bin" / "python"
        target.parent.mkdir(parents=True)
        target.write_text(pathlib.Path(sys.argv[0]).read_text())
        target.chmod(0o755)
    elif args[:2] == ["pip", "install"]:
        role = pathlib.Path(args[args.index("--python") + 1]).parents[1].name
        if role == os.environ.get("SETUP_TEST_FAIL"):
            sys.exit(19)
"""


@pytest.fixture
def installer(tmp_path):
    root = tmp_path / "public environments"
    for name, revision in (
        ("torchtitan-neuron", "b7582e986586051d7be85d03956afe169624aeda"),
        ("vllm-neuron", "eb6cb360f7cff6082e22a9437d8ce0ba3891304a"),
    ):
        directory = root / "sources" / name / ".git"
        directory.mkdir(parents=True)
        (directory / "revision").write_text(revision)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("uv", "git"):
        executable = tools / name
        executable.write_text(f"#!{sys.executable}\n{STUB}")
        executable.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    environment = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ.get('PATH', '')}",
        "UV_BIN": str(tools / "uv"),
        "SETUP_TEST_LOG": str(log),
    }
    for name in ("NEURON_TRAINING_REQUIREMENTS", "NEURON_SERVING_REQUIREMENTS", "SETUP_TEST_FAIL"):
        environment.pop(name, None)

    def run(fail=None):
        result = subprocess.run(
            ["bash", str(SCRIPT), str(root)],
            env={**environment, "SETUP_TEST_FAIL": fail or ""},
            capture_output=True,
            text=True,
            check=False,
        )
        return result, [json.loads(line) for line in log.read_text().splitlines()]

    return run


def test_setup_resolves_both_environments_without_external_requirements(installer):
    result, commands = installer()
    assert result.returncode == 0, result.stderr
    installs = [command for command in commands if command["args"][:2] == ["pip", "install"]]
    assert len(installs) == 2
    for command, torch_pin, runtime in zip(
        installs,
        ("torch==2.14.0+cpu", "torch==2.13.0+cpu"),
        ("torch-neuronx==2.14.3.0.281+dc70ffa", "libtorch-neuronx-lite==2.13.0.1.0.3112+653bc48f"),
        strict=True,
    ):
        args = command["args"]
        assert torch_pin in args and runtime in args
        assert args[args.index("--torch-backend") + 1] == "cpu"
        assert "--no-deps" not in args and "--requirements" not in args and "-r" not in args
        assert "verl[verl-core]==0.9.1" in args
        assert "nixl-cu12==1.3.2" in args
        assert {"nixl-cu13", "torchaudio", "flashinfer-python"} <= set(command["excluded"])
        assert not {"torch", "apache-tvm-ffi", "xgrammar", "nixl-cu12"} & set(command["excluded"])
        assert args.count("-e") == 2
    assert "torch-neuronx==2.14.3.0.281+dc70ffa" not in installs[1]["args"]
    assert commands[-1]["args"] == [str(SCRIPT.with_name("check_neuron_environment.py"))]
    assert "export VERL_NEURON_ROLLOUT_PYTHON=" in result.stdout


@pytest.mark.parametrize("role,expected_installs", [("training", 1), ("serving", 2)])
def test_failed_resolution_stops_before_environment_check(installer, role, expected_installs):
    result, commands = installer(fail=role)
    assert result.returncode == 19
    installs = [command for command in commands if command["args"][:2] == ["pip", "install"]]
    assert len(installs) == expected_installs
    assert commands[-1] == installs[-1]
    assert "export VERL_NEURON_" not in result.stdout
