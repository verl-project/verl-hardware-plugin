# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Interpreter dispatch happens before imports and preserves Ray assignments."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verl_hardware_plugin.accelerators.neuron.runtime import public_runtime_env


@pytest.mark.parametrize(
    "worker_environment,role",
    [
        ({}, "controller"),
        ({"WG_BACKEND": "ray", "RANK": "3", "NEURON_RT_VISIBLE_CORES": "7"}, "trainer"),
        ({"WG_BACKEND": "ray", "VERL_NEURON_SERVING": "1"}, "rollout"),
    ],
)
def test_launcher_selects_interpreter_before_imports(tmp_path, monkeypatch, worker_environment, role):
    scripts = {}
    for kind in ("TRAIN", "ROLLOUT"):
        path = tmp_path / f"{kind.lower()} python"
        path.write_text(f'#!/bin/sh\nexport SELECTED_PYTHON={kind}\nexec "{sys.executable}" "$@"\n')
        path.chmod(0o755)
        scripts[kind] = str(path)
        monkeypatch.setenv(f"VERL_NEURON_{kind}_PYTHON", str(path))
    environment = os.environ.copy()
    for key in ("WG_BACKEND", "VERL_NEURON_SERVING", "RANK", "NEURON_RT_VISIBLE_CORES"):
        environment.pop(key, None)
    environment.update(worker_environment)
    launcher = Path(__file__).parents[3] / "verl_hardware_plugin/accelerators/neuron/worker_launcher.sh"
    code = (
        "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
        "('SELECTED_PYTHON', 'VERL_NEURON_PROCESS_ROLE', 'RANK', 'NEURON_RT_VISIBLE_CORES')}))"
    )
    result = json.loads(subprocess.check_output(["bash", str(launcher), "-c", code], env=environment))
    assert result["VERL_NEURON_PROCESS_ROLE"] == role
    assert result["SELECTED_PYTHON"] == ("TRAIN" if role == "trainer" else "ROLLOUT")
    for key in ("RANK", "NEURON_RT_VISIBLE_CORES"):
        assert result[key] == worker_environment.get(key)
    assert public_runtime_env()["env_vars"]["VERL_NEURON_TRAIN_PYTHON"] == scripts["TRAIN"]


def test_public_runtime_requires_both_executables(monkeypatch):
    monkeypatch.setenv("VERL_NEURON_TRAIN_PYTHON", sys.executable)
    monkeypatch.delenv("VERL_NEURON_ROLLOUT_PYTHON", raising=False)
    with pytest.raises(ValueError, match="Set both"):
        public_runtime_env()


def test_launcher_emits_only_valid_hydra_overrides(monkeypatch):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    script = (root / "scripts/accelerators/neuron/run_neuron.sh").read_text()
    # Exercise the launcher's actual generator: importing installed vLLM can
    # print plugin discovery messages, which must not become Hydra arguments.
    code = script.split("runtime_overrides=$(", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    for name in ("VERL_NEURON_TRAIN_PYTHON", "VERL_NEURON_ROLLOUT_PYTHON"):
        monkeypatch.setenv(name, sys.executable)
    monkeypatch.setenv("VERL_PLATFORM", "aws")
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "controller")
    monkeypatch.setenv("VERL_NEURON_SERVING", "0")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    overrides = result.stdout.splitlines()
    assert all(line.startswith(("ray_kwargs.", "++ray_kwargs.")) for line in overrides), result.stdout
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    runtime = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env, resolve=True)
    expected = public_runtime_env()
    assert runtime["py_executable"] == expected["py_executable"]
    assert runtime["env_vars"] == expected["env_vars"]


def test_controller_cannot_import_training_runtime(monkeypatch):
    from verl_hardware_plugin.accelerators.neuron.runtime import ensure_neuron

    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "controller")
    monkeypatch.delenv("VERL_NEURON_SERVING", raising=False)
    ensure_neuron.cache_clear()
    with pytest.raises(RuntimeError, match="Only Neuron training"):
        ensure_neuron()


def test_controller_deserializes_reward_data_without_training_runtime():
    code = """
import pickle
import sys
import torch
from verl import DataProto
from verl.utils.device import get_torch_device

assert not get_torch_device().is_available()
sent = DataProto.from_dict(tensors={"responses": torch.arange(6).reshape(2, 3)})
received = pickle.loads(pickle.dumps(sent))
torch.testing.assert_close(received.batch["responses"], sent.batch["responses"])
assert "torch_neuronx" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "VERL_PLATFORM": "aws",
            "VERL_USE_EXTERNAL_PLUGINS": "hardware",
            "VERL_NEURON_PROCESS_ROLE": "controller",
            "VERL_NEURON_SERVING": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get("RUN_NEURON_PUBLIC_RAY_TESTS") != "1", reason="requires both public environments")
def test_ray_exchanges_training_batches_between_public_interpreters():
    import ray
    import torch
    from tensordict import TensorDict

    assert not ray.is_initialized(), "Run this gate in its own Ray session"

    @ray.remote
    class TrainingReceiver:
        def exchange(self, batch):
            import os
            import sys

            import torch

            batch["scores"] = batch["tokens"].sum(-1)
            return batch, torch.__version__, sys.executable, os.environ["VERL_NEURON_PROCESS_ROLE"]

    ray.init(num_cpus=2, runtime_env=public_runtime_env(), include_dashboard=False)
    try:
        worker = TrainingReceiver.options(
            runtime_env={"env_vars": {"WG_BACKEND": "ray", "NEURON_RT_VISIBLE_CORES": ""}}
        ).remote()
        sent = TensorDict({"tokens": torch.arange(12).reshape(3, 4)}, batch_size=[3])
        received, version, executable, role = ray.get(worker.exchange.remote(sent))
        assert role == "trainer"
        assert version.split("+")[0] == "2.14.0"
        assert executable == os.environ["VERL_NEURON_TRAIN_PYTHON"]
        torch.testing.assert_close(received["scores"], sent["tokens"].sum(-1))
    finally:
        ray.shutdown()
