# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Workers share an interpreter, select roles before imports, and preserve Ray assignments."""

import importlib
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
        ({"VERL_NEURON_PROCESS_ROLE": "controller"}, "controller"),
        ({"WG_BACKEND": "ray", "RANK": "3", "NEURON_RT_VISIBLE_CORES": "7"}, "trainer"),
        ({"WG_BACKEND": "ray", "VERL_NEURON_PROCESS_ROLE": "controller"}, "trainer"),
        ({"VERL_NEURON_PROCESS_ROLE": "rollout"}, "rollout"),
        ({"WG_BACKEND": "ray", "VERL_NEURON_PROCESS_ROLE": "rollout"}, "rollout"),
    ],
)
def test_launcher_uses_shared_interpreter_for_every_role(tmp_path, monkeypatch, worker_environment, role):
    path = tmp_path / "shared python"
    path.write_text(f'#!/bin/sh\nexport SELECTED_PYTHON=shared\nexec "{sys.executable}" "$@"\n')
    path.chmod(0o755)
    monkeypatch.setenv("VERL_NEURON_PYTHON", str(path))
    environment = os.environ.copy()
    for key in ("WG_BACKEND", "VERL_NEURON_PROCESS_ROLE", "RANK", "NEURON_RT_VISIBLE_CORES"):
        environment.pop(key, None)
    environment.update(worker_environment)
    launcher = Path(__file__).parents[3] / "verl_hardware_plugin/accelerators/neuron/worker_launcher.sh"
    code = (
        "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
        "('SELECTED_PYTHON', 'VERL_NEURON_PROCESS_ROLE', 'RANK', 'NEURON_RT_VISIBLE_CORES')}))"
    )
    result = json.loads(subprocess.check_output(["bash", str(launcher), "-c", code], env=environment))
    assert result["VERL_NEURON_PROCESS_ROLE"] == role
    assert result["SELECTED_PYTHON"] == "shared"
    for key in ("RANK", "NEURON_RT_VISIBLE_CORES"):
        assert result[key] == worker_environment.get(key)
    assert public_runtime_env()["env_vars"]["VERL_NEURON_PYTHON"] == str(path)


def test_public_runtime_without_an_explicit_environment(monkeypatch):
    monkeypatch.delenv("VERL_NEURON_PYTHON", raising=False)
    assert public_runtime_env() == {}


@pytest.mark.parametrize("executable", ["relative/bin/python", "/nonexistent/neuron/bin/python"])
def test_public_runtime_rejects_invalid_executable(monkeypatch, executable):
    monkeypatch.setenv("VERL_NEURON_PYTHON", executable)
    with pytest.raises(ValueError, match="absolute executable path"):
        public_runtime_env()


def test_launcher_emits_only_valid_hydra_overrides(monkeypatch):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[3]
    script = (root / "scripts/accelerators/neuron/run_neuron.sh").read_text()
    # Exercise the launcher's actual generator: importing installed vLLM can
    # print plugin discovery messages, which must not become Hydra arguments.
    code = script.split("runtime_overrides=$(", 1)[1].split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
    monkeypatch.setenv("VERL_NEURON_PYTHON", sys.executable)
    monkeypatch.setenv("VERL_PLATFORM", "aws")
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "controller")
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
    ensure_neuron.cache_clear()
    with pytest.raises(RuntimeError, match="Only Neuron training"):
        ensure_neuron()


@pytest.mark.parametrize("role", ["controller", "rollout"])
@pytest.mark.parametrize("module", ["torchtitan", "torchtitan_neuron", "torch_neuronx"])
def test_non_training_workers_block_installed_training_packages(tmp_path, monkeypatch, role, module):
    from verl_hardware_plugin.accelerators.neuron.runtime import isolate_training_imports

    (tmp_path / f"{module}.py").write_text('raise AssertionError("Training package executed in a non-training worker")')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", role)
    isolate_training_imports()
    isolate_training_imports()
    with pytest.raises(ModuleNotFoundError, match="only available in Neuron training workers"):
        importlib.import_module(module)


def test_training_import_filter_preserves_trainer_and_lite_imports(monkeypatch):
    from verl_hardware_plugin.accelerators.neuron.runtime import _TrainingImportFinder

    finder = _TrainingImportFinder()
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "trainer")
    for module in ("torchtitan", "torchtitan_neuron", "torch_neuronx", "torch_neuronx._C"):
        assert finder.find_spec(module) is None
    monkeypatch.setenv("VERL_NEURON_PROCESS_ROLE", "rollout")
    for module in ("libtorch_neuronx_lite", "libtorch_neuronx_lite._compiler", "vllm_neuron", "nki"):
        assert finder.find_spec(module) is None


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
assert "torchtitan" not in sys.modules
assert not getattr(torch.compile, "_neuron_wrapped", False)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "VERL_PLATFORM": "aws",
            "VERL_USE_EXTERNAL_PLUGINS": "hardware",
            "VERL_NEURON_PROCESS_ROLE": "controller",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("RUN_NEURON_PUBLIC_RAY_TESTS") != "1", reason="requires the public Neuron environment"
)
def test_ray_exchanges_batches_in_shared_environment():
    import ray
    import torch
    from tensordict import TensorDict

    assert not ray.is_initialized(), "Run this gate in its own Ray session"

    @ray.remote
    class BatchReceiver:
        def exchange(self, batch):
            import os
            import sys

            import torch

            batch["scores"] = batch["tokens"].sum(-1)
            return batch, torch.__version__, sys.executable, os.environ["VERL_NEURON_PROCESS_ROLE"]

    ray.init(num_cpus=2, runtime_env=public_runtime_env(), include_dashboard=False)
    try:
        sent = TensorDict({"tokens": torch.arange(12).reshape(3, 4)}, batch_size=[3])
        for expected_role, environment in (
            ("trainer", {"WG_BACKEND": "ray"}),
            ("rollout", {"VERL_NEURON_PROCESS_ROLE": "rollout"}),
            ("controller", {}),
        ):
            worker = BatchReceiver.options(
                runtime_env={"env_vars": {**environment, "NEURON_RT_VISIBLE_CORES": ""}}
            ).remote()
            received, version, executable, role = ray.get(worker.exchange.remote(sent))
            assert role == expected_role
            assert version == "2.13.0+cpu"
            assert executable == os.environ["VERL_NEURON_PYTHON"]
            torch.testing.assert_close(received["scores"], sent["tokens"].sum(-1))
            ray.kill(worker)
    finally:
        ray.shutdown()
