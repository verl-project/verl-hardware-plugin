# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Process setup shared by Ray workers and torchrun entry points."""

import functools
import importlib
import importlib.abc
import importlib.util
import os
import shlex
import sys
from pathlib import Path


def process_role():
    """Return the role selected before importing native Neuron runtimes."""
    role = os.environ.get("VERL_NEURON_PROCESS_ROLE", "trainer")
    if role not in ("controller", "trainer", "rollout"):
        raise ValueError(f"Unknown Neuron process role: {role}")
    # Ray training workers inherit the controller's environment. Verl sets
    # WG_BACKEND before importing their modules; an explicit rollout role wins.
    if role != "rollout" and os.environ.get("WG_BACKEND") == "ray":
        return "trainer"
    return role


def public_runtime_env():
    """Launch every Ray worker in the shared Neuron Python environment.

    Ray applies these environment variables before starting Python. Verl adds
    WG_BACKEND/rank variables for training workers, which process_role() uses to
    select the native runtime before importing it.
    """
    executable = os.environ.get("VERL_NEURON_PYTHON")
    if not executable:
        return {}
    if not Path(executable).is_absolute() or not os.access(executable, os.X_OK):
        raise ValueError(f"Neuron Python must be an absolute executable path: {executable}")
    executable_dir = Path(executable).parent
    environment_paths = {
        "PATH": str(executable_dir),
        "LD_LIBRARY_PATH": str(executable_dir.parent / "lib"),
    }
    for name, directory in environment_paths.items():
        if inherited := os.environ.get(name):
            environment_paths[name] = os.pathsep.join(dict.fromkeys([directory, *inherited.split(os.pathsep)]))
    return {
        "py_executable": shlex.quote(executable),
        "env_vars": {
            **environment_paths,
            "VERL_NEURON_PYTHON": executable,
            "VERL_NEURON_PROCESS_ROLE": "controller",
            "VERL_PLATFORM": "aws",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "NEURON_RT_MAP_HBM": "1",
            "NEURON_RT_ENABLE_P2P": "0",
            "FI_EFA_ENABLE_SHM_TRANSFER": "0",
            "NEURON_LOGICAL_NC_CONFIG": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
            "VLLM_DISABLE_COMPILE_CACHE": os.environ.get("VLLM_DISABLE_COMPILE_CACHE", "0"),
        },
    }


class CoreLease:
    """Reserve Ray resources without importing a serving or training runtime."""

    def allocation(self):
        import ray

        context = ray.get_runtime_context()
        return context.get_node_id(), context.get_accelerator_ids()["neuron_cores"]


class _TrainingImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if process_role() != "trainer" and fullname.split(".")[0] in (
            "torchtitan",
            "torchtitan_neuron",
            "torch_neuronx",
        ):
            raise ModuleNotFoundError(f"{fullname} is only available in Neuron training workers", name=fullname)
        return None


def isolate_training_imports():
    """Keep eager training imports out of controllers and lite rollout workers.

    Verl discovers TorchTitan even in rollout processes. TorchTitan redirects
    torch.compile to the training backend during import; loading that extension
    beside lite leaves conflicting native symbols even if the import fails.
    """
    if not any(isinstance(finder, _TrainingImportFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TrainingImportFinder())


class _CUDAOnlyCheckpointFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verl.checkpoint_engine.nixl_checkpoint_engine":
            raise ImportError("The CUDA NIXL checkpoint engine is unavailable on Trainium; use neuron_nixl")
        return None


def isolate_checkpoint_imports():
    """Keep verl's eager CUDA NIXL import out of the native Neuron process.

    NIXL must be loaded through our native transport after Neuron initializes. Loading
    it earlier either interposes incompatible protobuf symbols or exports
    Neuron's MLIR symbols before the NKI compiler loads.
    """
    if not any(isinstance(finder, _CUDAOnlyCheckpointFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _CUDAOnlyCheckpointFinder())


def configure_environment():
    """Set runtime options before importing the native Neuron extension."""
    defaults = {
        "NEURON_RT_MAP_HBM": "1",
        "NEURON_RT_ENABLE_P2P": "0",
        "FI_EFA_ENABLE_SHM_TRANSFER": "0",
        "NEURON_LOGICAL_NC_CONFIG": "2",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    executable_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys([executable_dir, *os.environ.get("PATH", "").split(os.pathsep)]))


def physical_core(visible, local_rank):
    """Resolve the physical core for either a Ray allocation or a torchrun rank."""
    cores = []
    for part in visible.split(","):
        bounds = [int(value) for value in part.strip().split("-")]
        if len(bounds) == 1:
            cores.extend(bounds)
        elif len(bounds) == 2 and bounds[1] >= bounds[0]:
            cores.extend(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"Invalid NeuronCore allocation {visible!r}")
    if len(cores) == 1:
        return cores[0]
    if not 0 <= local_rank < len(cores):
        raise ValueError("LOCAL_RANK is outside the visible NeuronCore allocation")
    return cores[local_rank]


# Opt-in actor rank placement for TP=8 x FSDP=8 on a 64-core trn2 node. The default (rank r on core r)
# puts each FSDP group on every other device, which the runtime cannot build collectives for.
ACTOR_CORE_ORDER = "VERL_NEURON_ACTOR_CORE_ORDER"
CORES_PER_NODE = 64
_TP8_FSDP8_FIRST_DEVICES = (0, 3, 4, 7, 8, 11, 12, 15)


def actor_core(order, node_rank):
    if order != "tp8_fsdp8":
        raise ValueError(f"Unknown {ACTOR_CORE_ORDER}: {order!r} (supported: tp8_fsdp8)")
    if not 0 <= node_rank < CORES_PER_NODE:
        raise ValueError(f"Node rank {node_rank} is outside a {CORES_PER_NODE}-core node")
    group, tp_rank = divmod(node_rank, 8)
    first = _TP8_FSDP8_FIRST_DEVICES[group]
    device = first if tp_rank < 4 else first ^ 1
    return device * 4 + tp_rank % 4


def prepare_worker_environment():
    """Preserve Ray's physical allocation while selecting local device zero."""
    configure_environment()
    if "RAY_LOCAL_WORLD_SIZE" in os.environ:
        visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
        if not visible:
            raise RuntimeError("Ray did not assign NeuronCores to this training worker")
        order = os.environ.get(ACTOR_CORE_ORDER)
        if order:
            if int(os.environ["RAY_LOCAL_WORLD_SIZE"]) != CORES_PER_NODE:
                raise ValueError(f"{ACTOR_CORE_ORDER} requires actor workers on all {CORES_PER_NODE} cores of a node")
            node_rank = int(os.environ["RANK"]) % CORES_PER_NODE
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(actor_core(order, node_rank))
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = os.environ["RAY_LOCAL_WORLD_SIZE"]


@functools.lru_cache(maxsize=1)
def ensure_neuron():
    """Import the native extension without allocating an accelerator tensor."""
    if process_role() != "trainer":
        raise RuntimeError("Only Neuron training workers may import torch_neuronx")
    configure_environment()
    if importlib.util.find_spec("torch_neuronx") is None:
        raise RuntimeError("AWS Trainium requires the native torch_neuronx package")
    importlib.import_module("torch_neuronx")


def install_local_device_count():
    """Expose Ray's one local device after distributed setup, not the host count.

    DTensor derives its RNG device from rank modulo device_count. The pinned SDK
    otherwise reports every host core despite restricting each process to one.
    """
    import torch

    torch.neuron.device_count = lambda: 1


def install_worker_setup():
    """Install process-group preparation before verl caches the initializer."""
    from verl.utils import distributed

    for name in ("initialize_global_process_group", "initialize_global_process_group_ray"):
        original = getattr(distributed, name)
        if getattr(original, "_neuron_setup", False):
            continue

        def wrap(function):
            @functools.wraps(function)
            def initialize(*args, **kwargs):
                from verl.plugin.platform import get_platform

                native = get_platform().vendor_name == "aws" and kwargs.get("backend") != "cpu:gloo"
                if native:
                    prepare_worker_environment()
                    ensure_neuron()
                result = function(*args, **kwargs)
                if native and "RAY_LOCAL_WORLD_SIZE" in os.environ:
                    install_local_device_count()
                return result

            initialize._neuron_setup = True
            return initialize

        setattr(distributed, name, wrap(original))
