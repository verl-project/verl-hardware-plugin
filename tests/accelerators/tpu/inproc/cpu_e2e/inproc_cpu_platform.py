# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""A CPU-only verl platform for the in-process rollout CPU tests (``VERL_PLATFORM=cpu``). Test code.

verl has no CPU platform (it falls back to "nvidia"). This registers ``cpu``: device ``cpu``, the gloo backend, no-op
memory management, and the hooks verl calls on a platform. Imported through ``VERL_USE_EXTERNAL_MODULES`` before verl's
plugin auto-discovery; it also forces vLLM's CPU platform (a vLLM source tree without compiled kernels is not detected
as CPU by vLLM itself).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

RESOURCE = "CPUDEV"


def force_vllm_cpu_platform() -> None:
    try:
        import vllm.platforms as vplat
        from vllm.platforms.cpu import CpuPlatform

        if not isinstance(vplat._current_platform, CpuPlatform):
            vplat._current_platform = CpuPlatform()
    except Exception:  # noqa: BLE001 - vLLM absent in a process that never needs it
        pass


class _CpuDeviceModule:
    """The CUDA-style device module verl calls (``get_torch_device()``), for CPU."""

    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def set_device(self, *_):
        return None

    def synchronize(self, *_):
        return None

    def empty_cache(self):
        return None

    def manual_seed(self, seed):
        torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        torch.manual_seed(seed)

    def memory_allocated(self, *_):
        return 0

    def max_memory_allocated(self, *_):
        return 0

    def memory_reserved(self, *_):
        return 0

    def max_memory_reserved(self, *_):
        return 0

    def mem_get_info(self, *_):
        return (0, 0)

    def memory_stats(self, *_):
        return {}

    def reset_peak_memory_stats(self, *_):
        return None

    def get_rng_state(self, *_):
        return torch.get_rng_state()

    def set_rng_state(self, state, *_):
        torch.set_rng_state(state)

    def is_initialized(self):
        return True

    def get_device_name(self, *_):
        return "cpu"

    def get_device_properties(self, *_):
        return type("P", (), {"total_memory": 0, "name": "cpu"})()

    def __getattr__(self, name):
        return getattr(torch.cpu, name)


@PlatformRegistry.register(platform="cpu")
class PlatformCPUTest(PlatformBase):
    def __init__(self):
        super().__init__()
        self._dm = _CpuDeviceModule()

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def vendor_name(self) -> str:
        return "cpu-test"

    @property
    def device_module(self):
        return self._dm

    def is_available(self) -> bool:
        return True

    def is_platform_available(self, use_smi_check=False) -> bool:
        return os.environ.get("VERL_PLATFORM") == "cpu"

    def current_device(self):
        # verl moves tensors with .to(get_device_id()) / device=get_device_id(); an int would mean CUDA ordinal 0 to
        # torch
        return torch.device("cpu")

    def device_count(self) -> int:
        return 1

    def set_device(self, device_index: int) -> None:
        return None

    def synchronize(self, device_index: Optional[int] = None) -> None:
        return None

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_allocator_settings(self, settings: str) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def get_device_capability(self, device_index: int = 0):
        return (None, None)

    def communication_backend_name(self) -> str:
        return "gloo"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        return None

    def profiler_stop(self) -> None:
        return None

    def ray_resource_name(self) -> str:
        return RESOURCE

    def ray_noset_envvars(self) -> list[str]:
        return []

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        n = int(num_gpus)
        return {"resources": {RESOURCE: n}} if n >= 1 else {}

    def supports_colocated_worker_groups(self) -> bool:
        return False

    def is_ipc_supported(self) -> bool:
        return False

    def cudart(self) -> Any:
        return None

    def get_ray_init_kwargs(self) -> dict[str, Any]:
        keep = ("VERL_TPU_",)
        env = {k: v for k, v in os.environ.items() if v and k.startswith(keep)}
        env.update({"VERL_PLATFORM": "cpu", "CUDA_VISIBLE_DEVICES": ""})
        return {"runtime_env": {"env_vars": env}}

    def rollout_env_vars(self) -> dict[str, str]:
        return {}


def _patch_verl_dist_for_cpu() -> None:
    """verl builds its process-group backend as "cpu:gloo,<device>:<backend>", which is "cpu:gloo,cpu:gloo" here (torch
    refuses a
    duplicate device type): the CPU test platform's workers use plain gloo. Test only."""
    try:
        import verl.utils.distributed as vd
    except Exception:  # noqa: BLE001
        return
    if getattr(vd.initialize_global_process_group_ray, "_inproc_cpu", False):
        return
    orig = vd.initialize_global_process_group_ray

    def initialize_global_process_group_ray(timeout_second=None, backend=None):
        return orig(timeout_second=timeout_second, backend=backend or "gloo")

    initialize_global_process_group_ray._inproc_cpu = True
    vd.initialize_global_process_group_ray = initialize_global_process_group_ray


force_vllm_cpu_platform()
if os.environ.get("VERL_PLATFORM") == "cpu":
    _patch_verl_dist_for_cpu()
