# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Native PyTorch Trainium platform; XLA execution is not supported."""

import importlib.util
import os
from contextlib import nullcontext
from types import ModuleType

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

from .runtime import ensure_neuron, process_role, public_runtime_env


class NeuronDeviceModule(ModuleType):
    def __init__(self):
        super().__init__(__name__ + ".device")

    def is_available(self):
        # DataProto deserialization queries the device module directly, without
        # going through PlatformNeuron.is_available().
        if process_role() == "controller" or os.environ.get("NEURON_RT_VISIBLE_CORES") == "":
            return False
        return self.__getattr__("is_available")()

    def __getattr__(self, name):
        import torch

        if process_role() == "trainer":
            ensure_neuron()
        elif not hasattr(torch, "neuron"):
            raise RuntimeError("The serving controller must not initialize the training Neuron runtime")
        return getattr(torch.neuron, name)

    def mem_get_info(self, device=None):
        total = self.get_device_properties(device).total_memory
        return max(0, total - self.memory_reserved()), total

    def ipc_collect(self):
        return None

    def reset_max_memory_allocated(self, device=None):
        self.reset_peak_memory_stats()


@PlatformRegistry.register(platform="aws")
class PlatformNeuron(PlatformBase):
    def __init__(self):
        self._device_module = NeuronDeviceModule()

    @property
    def device_name(self):
        return "neuron"

    @property
    def vendor_name(self):
        return "aws"

    @property
    def device_module(self):
        return self._device_module

    def is_platform_available(self, use_smi_check=False):
        package = "torch_neuronx" if process_role() == "trainer" else "libtorch_neuronx_lite"
        return importlib.util.find_spec(package) is not None and os.path.exists("/dev/neuron0")

    def is_available(self):
        if process_role() == "controller" or os.environ.get("NEURON_RT_VISIBLE_CORES") == "":
            return False
        if process_role() == "rollout":
            return self.is_platform_available()
        return self.is_platform_available() and self.device_module.is_available()

    def current_device(self):
        return self.device_module.current_device()

    def device_count(self):
        return self.device_module.device_count()

    def set_device(self, device_index):
        self.device_module.set_device(device_index)

    def synchronize(self, device_index=None):
        self.device_module.synchronize()

    def manual_seed(self, seed):
        self.device_module.manual_seed(seed)

    def manual_seed_all(self, seed):
        self.device_module.manual_seed_all(seed)

    def empty_cache(self):
        self.device_module.empty_cache()

    def set_allocator_settings(self, settings):
        return None

    def get_device_capability(self, device_index=0):
        return None, None

    def communication_backend_name(self):
        return "neuron"

    def visible_devices_envvar(self):
        return "NEURON_RT_VISIBLE_CORES"

    def ray_resource_name(self):
        return "neuron_cores"

    def ray_resource_options(self, num_gpus):
        if num_gpus < 0 or int(num_gpus) != num_gpus:
            raise ValueError("Trainium workers require a nonnegative integral NeuronCore allocation")
        return {"resources": {"neuron_cores": int(num_gpus)}} if num_gpus else {}

    def supports_colocated_worker_groups(self):
        return False

    def is_ipc_supported(self):
        return False

    def ray_noset_envvars(self):
        return []

    def get_ray_init_kwargs(self):
        runtime_env = public_runtime_env()
        runtime_env.setdefault("env_vars", {}).update(
            {
                "VERL_PLATFORM": "aws",
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                "NEURON_RT_MAP_HBM": "1",
                "NEURON_RT_ENABLE_P2P": "0",
                "FI_EFA_ENABLE_SHM_TRANSFER": "0",
                "NEURON_LOGICAL_NC_CONFIG": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
            }
        )
        return {"runtime_env": runtime_env}

    def cudart(self):
        return None

    def nvtx_range(self, msg):
        return nullcontext()

    def profiler_start(self):
        return None

    def profiler_stop(self):
        return None
