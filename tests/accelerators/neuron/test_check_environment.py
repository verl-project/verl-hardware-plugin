# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Validate the shared runtime requirements without installing the Neuron SDK."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def checker(monkeypatch):
    script = Path(__file__).resolve().parents[3] / "scripts/accelerators/neuron/check_neuron_environment.py"
    spec = importlib.util.spec_from_file_location("check_neuron_environment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {
        "python": [3, 12],
        "executable": sys.executable,
        "libc": ["glibc", "2.43"],
        "forbidden": False,
        "packages": {
            "verl": "0.9.1",
            "torch": "2.13.0+cpu",
            "ray": "2.55.0",
            "TransferQueue": "0.1.11",
            "neuronx-cc": "2.28.7631.0+8f7b0cf5",
            "nki": "0.7.0+37713544100.gfede04bb",
            "nixl-cu12": "1.3.2",
            "torch-neuronx": "2.13.3.0.283+dc70ffa",
            "torchtitan-neuron": "0.2.2+neuron.1",
            "vllm": "0.29.0",
            "vllm-neuron": "0.29.0.1.2.0",
            "libtorch-neuronx-lite": "2.13.0.1.0.3112+653bc48f",
        },
    }
    probe = Mock(side_effect=lambda *args, **kwargs: json.dumps(report))
    monkeypatch.setattr(module.subprocess, "check_output", probe)
    monkeypatch.setattr(module, "check_system", lambda: {})
    monkeypatch.setenv("VERL_NEURON_PYTHON", sys.executable)
    return module, report, probe


def test_check_accepts_both_runtimes_in_one_environment(checker, capsys):
    module, report, probe = checker
    module.check()
    assert json.loads(capsys.readouterr().out) == {"environment": report, "system": {}}
    probe.assert_called_once()
    assert probe.call_args.args[0] == [sys.executable, "-c", module.PROBE]
    assert probe.call_args.kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


@pytest.mark.parametrize("package", ["torch-neuronx", "torchtitan-neuron", "libtorch-neuronx-lite", "vllm-neuron"])
def test_check_requires_training_and_rollout_packages(checker, package):
    module, report, _ = checker
    del report["packages"][package]
    with pytest.raises(SystemExit, match=f"Expected {package}=="):
        module.check()


def test_check_rejects_pytorch_214(checker):
    module, report, _ = checker
    report["packages"]["torch"] = "2.14.0+cpu"
    with pytest.raises(SystemExit, match=r"Expected torch==2\.13\.0\+cpu"):
        module.check()
