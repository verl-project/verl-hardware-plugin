#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Check the shared public environment without loading a device runtime."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

PROBE = """
import importlib.metadata as m, importlib.util, json, platform, sys
packages = ['torch', 'ray', 'verl', 'TransferQueue', 'neuronx-cc', 'nki',
            'torch-neuronx', 'torchtitan-neuron', 'libtorch-neuronx-lite',
            'vllm-neuron', 'vllm', 'nixl-cu12', 'nixl-cu13']
versions = {}
for name in packages:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        pass
print(json.dumps(dict(python=list(sys.version_info[:2]), executable=sys.executable,
                     libc=platform.libc_ver(), packages=versions,
                     forbidden=importlib.util.find_spec('neuron_rl_core') is not None)))
"""


def check_system():
    driver = Path("/sys/module/neuron/version")
    report = {"neuron_driver": driver.read_text().strip() if driver.exists() else None}
    fabric = Path("/opt/amazon/efa/bin/fi_info")
    if fabric.exists():
        output = subprocess.check_output([str(fabric), "--version"], text=True)
        match = re.search(r"^libfabric:\s*(\S+)", output, re.MULTILINE)
        report["libfabric"] = match[1] if match else None
    else:
        report["libfabric"] = None
    for package in ("aws-neuronx-runtime-lib", "aws-neuronx-collectives"):
        result = subprocess.run(["dpkg-query", "-W", "-f=${Version}", package], capture_output=True, text=True)
        report[package] = result.stdout.strip() if result.returncode == 0 else None
    return report


def check(require_hardware=False):
    executable = os.environ["VERL_NEURON_PYTHON"]
    report = json.loads(
        subprocess.check_output([executable, "-c", PROBE], env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    )
    packages = report["packages"]
    expected = {
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
    }
    if report["python"] != [3, 12]:
        raise SystemExit("The public recipe requires Python 3.12")
    if report["libc"][0] != "glibc" or tuple(map(int, report["libc"][1].split("."))) < (2, 43):
        raise SystemExit("Public TorchNeuron requires GLIBC >= 2.43; use a compatible system environment")
    if report["forbidden"]:
        raise SystemExit("Remove neuron-rl-core from the public environment")
    if "nixl-cu13" in packages:
        raise SystemExit("Remove nixl-cu13 so NIXL uses the pinned CUDA 12 backend")
    for name, version in expected.items():
        if packages.get(name) != version:
            raise SystemExit(f"Expected {name}=={version}, found {packages.get(name)}")
    reports = {"environment": report}
    reports["system"] = check_system()
    if require_hardware:
        expected = {
            "neuron_driver": "2.31.7.0",
            "libfabric": "2.7.0amzn2.0",
            "aws-neuronx-runtime-lib": "2.35.30.0-d2725ccb8",
            "aws-neuronx-collectives": "2.35.43.0-354ad3064",
        }
        for name, version in expected.items():
            if reports["system"][name] != version:
                raise SystemExit(
                    f"Public RL requires {name}={version}, found {reports['system'][name]}. "
                    "Use the documented runtime versions and matching host driver."
                )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="also enforce the tested driver/EFA/runtime versions")
    check(require_hardware=parser.parse_args().hardware)
