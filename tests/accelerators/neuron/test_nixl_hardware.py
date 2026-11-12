# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Opt-in native-to-lite HBM prefix writes, repeated payload sizes, and explicit cores."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("RUN_NEURON_NIXL_TESTS") != "1", reason="Requires two idle Trainium cores and EFA")
def test_native_to_lite_nixl_without_engine_dependency(tmp_path):
    cores = [os.environ.get("TRAINER_CORE", "4"), os.environ.get("ROLLOUT_CORE", "8")]
    assert cores[0] != cores[1], "Trainer and receiver require separate cores"
    devices = json.loads(subprocess.check_output(["/opt/aws/neuron/bin/neuron-ls", "--json-output"]))
    assert not any(device.get("neuron_processes") for device in devices), "Run this hardware gate on an idle host"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    children = []
    try:
        for role, core in zip(("sender", "receiver"), cores, strict=True):
            environment = {
                **os.environ,
                "VERL_PLATFORM": "aws",
                "VERL_NEURON_SERVING": "1" if role == "receiver" else "0",
                "VERL_NEURON_PROCESS_ROLE": "rollout" if role == "receiver" else "trainer",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "VLLM_PLUGINS": "",
                "VLLM_NEURON_BACKEND": "neuron_native",
                "NEURON_RT_VISIBLE_CORES": core,
                "NEURON_LOGICAL_NC_CONFIG": "2",
                "NEURON_RT_MAP_HBM": "1",
                "NEURON_RT_ENABLE_P2P": "0",
                "FI_EFA_ENABLE_SHM_TRANSFER": "0",
            }
            environment.pop("FI_EFA_IFACE", None)
            # Child scripts must exercise this checkout, even with an editable install elsewhere.
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(Path(__file__).resolve().parents[3]), environment.get("PYTHONPATH")))
            )
            log_path = tmp_path / f"{role}.log"
            with log_path.open("w") as output:
                executable_variable = "VERL_NEURON_TRAIN_PYTHON" if role == "sender" else "VERL_NEURON_ROLLOUT_PYTHON"
                executable = os.environ.get(executable_variable, sys.executable)
                environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                    filter(None, (str(Path(executable).parent.parent / "lib"), environment.get("LD_LIBRARY_PATH")))
                )
                child = subprocess.Popen(
                    [
                        executable,
                        str(Path(__file__).resolve()),
                        role,
                        str(port),
                        core,
                    ],
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            children.append((child, log_path))
        for child, log_path in children:
            assert child.wait(timeout=240) == 0, log_path.read_text()
            print(log_path.read_text())
    finally:
        for child, log_path in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def run_probe(role, port, core):
    from importlib import import_module
    from importlib.util import find_spec

    assert find_spec("neuron_rl_core") is None, "Run acceptance with the forbidden dependency uninstalled"
    import_module("libtorch_neuronx_lite" if role == "receiver" else "torch_neuronx")
    import torch

    from verl_hardware_plugin.accelerators.neuron.weight_sync.nixl_transport import (
        NixlEndpoint,
        NixlTopology,
        control_store,
        synchronize,
    )

    sender = role == "sender"
    spec = NixlTopology("127.0.0.1", int(port), 1, 1, 120)
    store = control_store(spec, master=sender)
    endpoint = NixlEndpoint(store, "probe", "neuron:0", int(core), sender=sender, lite=not sender, timeout_s=120)
    try:
        endpoint.register(4 * 1024 * 1024)
        address = endpoint.address
        capacity = endpoint.buffer.numel()
        lengths = (capacity, 0, 17, capacity // 2 + 3, 17, 0, capacity)
        expected = torch.zeros(capacity, dtype=torch.bfloat16)
        if sender:
            from torch_neuronx import metrics

            metrics.set_enabled(True)

        def compilation_counts():
            if not sender:
                return {}
            return {
                name: value["total_samples"] if isinstance(value, dict) else value
                for name, value in metrics.get_metrics_dict().items()
                if "compil" in name.lower()
            }

        baseline_compilations = compilation_counts()
        for update in range(2):
            for index, numel in enumerate(lengths):
                if not numel:
                    continue
                pattern = ((torch.arange(capacity) % 127) - 63 + index + update * len(lengths)).to(torch.bfloat16)
                if sender:
                    # Fixed device copy shape; only the CPU-side NIXL byte count changes.
                    endpoint.buffer.copy_(pattern)
                    endpoint.push(numel)
                else:
                    expected[:numel].copy_(pattern[:numel])
                    endpoint.receive(lambda slot: torch.testing.assert_close(slot.cpu(), expected, rtol=0, atol=0))
                assert endpoint.address == address
        synchronize(lite=not sender)
        assert endpoint.sequence == 2 * sum(numel > 0 for numel in lengths)
        assert compilation_counts() == baseline_compilations, "Transfer lengths triggered native compilation activity"
        if sender:
            assert set(endpoint.transfers) == {numel * 2 for numel in lengths if numel}
            print(f"sender: native compilation counters unchanged: {compilation_counts()}")
        print(
            f"{role}: {endpoint.sequence} NIXL/LIBFABRIC writes passed; "
            f"{2 * sum(lengths) * 2} payload bytes; untouched suffixes verified"
        )
    finally:
        endpoint.close()


if __name__ == "__main__":
    run_probe(*sys.argv[1:])
