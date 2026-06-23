# Development Guide: Adding a New Hardware Platform

This document explains how to add support for a new hardware accelerator to verl via the plugin system, with detailed examples and inline commentary.

> [!NOTE]
> 目前 verl-core 中的 platform/engine 插件机制仍在持续验证中。如果您在适配过程中发现仅通过在本仓库中增加 platform 和 engine 无法满足需求（例如 `PlatformBase` 缺少必要接口、Engine 基类行为与硬件不兼容等），欢迎在 [verl](https://github.com/verl-project/verl) 仓库提出 Issue，并给出需要完善的建议，我们会快速响应。

> **verl core PR**: The platform and engine registry mechanism is implemented in
> [verl#6086](https://github.com/verl-project/verl/pull/6086). Refer to that PR
> for the base class interfaces (`PlatformBase`, `EngineRegistry`) and the
> registration/lookup logic.

## Architecture Overview

```
verl (main framework)
    │
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── platforms/  → @PlatformRegistry.register(platform="vendor_name")
            └── engines/    → @EngineRegistry.register(device=..., vendor=...)
```

The plugin integrates with verl through two registries:
1. **PlatformRegistry** — registers hardware platform abstractions (device management, communication, memory, etc.)
2. **EngineRegistry** — registers training engines (hardware-specific variants of FSDP/Megatron)

### How Discovery Works

verl discovers plugins through Python's `entry_points` mechanism. When verl starts, it imports all packages registered under the `verl.plugins` group. This triggers the `__init__.py` of the plugin package, which calls `register_all_platforms()` and `register_all_engines()` to fire all `@register` decorators.

```toml
# pyproject.toml of this plugin
[project.entry-points."verl.plugins"]
verl_hardware_plugin = "verl_hardware_plugin"
```

No manual configuration in verl is needed — just `pip install` the plugin package.

---

## Steps to Add a New Platform

> **Reference in verl core**: See [verl/plugin/platform](https://github.com/verl-project/verl/tree/main/verl/plugin/platform) for built-in platform implementations (CUDA, NPU).

### Step 1: Create the Platform Class

Create a new file under `verl_hardware_plugin/platforms/`, e.g. `platform_my_vendor.py`.

Below is a **fully annotated template** — every method includes comments explaining what it does and why:

```python
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MyVendor platform implementation.

This module registers a platform for MyVendor hardware with verl's
PlatformRegistry. It will be auto-discovered when verl starts.

Key decisions documented here:
- device_name: The torch device type string (e.g. "cuda", "xpu", "mlu", "npu")
  This must match what PyTorch uses: torch.device("xpu"), torch.xpu.*, etc.
- vendor_name: A human-readable vendor identifier used for engine lookup.
  Multiple vendors can share the same device_name (e.g. MetaX and NVIDIA both
  use "cuda"), but vendor_name must be unique.
"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Optional: ensure vendor-specific torch extension is importable.
# Some hardware (NPU, XPU) requires importing an extension before
# torch.<device> becomes available.
def _ensure_torch_my_device() -> bool:
    """Try to import the vendor's torch extension so that torch.my_device becomes available.

    Returns True if torch.my_device is usable after the attempt.
    This pattern is used by NPU (torch_npu) and XPU (intel_extension_for_pytorch).
    If your hardware uses standard torch.cuda, you don't need this.
    """
    if hasattr(torch, "my_device"):
        return True
    try:
        import my_vendor_torch_extension  # noqa: F401
        return hasattr(torch, "my_device")
    except ImportError:
        return False


@PlatformRegistry.register(platform="my_vendor")
class PlatformMyDevice(PlatformBase):
    """Platform backend for MyVendor hardware.

    Registration key: "my_vendor"
    - Auto-detection: set VERL_PLATFORM=my_vendor, or implement is_platform_available()
    - Engine lookup: engines registered with device="my_device", vendor="my_vendor"
      will be selected automatically
    """

    # ==================================================================
    # Core device management
    # ==================================================================

    @property
    def device_name(self) -> str:
        """torch device type string.

        This is used throughout verl for:
        - torch.device(device_name)
        - torch.<device_name>.is_available()
        - Tensor placement: tensor.to(device_name)

        Examples: "cuda", "xpu", "mlu", "npu"
        """
        return "my_device"

    @property
    def vendor_name(self) -> str:
        """Unique vendor identifier.

        Used for:
        - Engine registry lookup: (device_name, vendor_name) → specific engine class
        - Logging and display purposes
        - Distinguishing multiple vendors sharing the same device_name

        Examples: "nvidia", "metax", "intel", "cambricon", "huawei"
        """
        return "my_vendor"

    @property
    def device_module(self) -> ModuleType:
        """Return the torch.<device> namespace module.

        verl uses this for operations like:
        - platform.device_module.current_device()
        - platform.device_module.memory_allocated()

        Must return the actual module object (e.g. torch.cuda, torch.xpu).
        """
        if not _ensure_torch_my_device():
            raise RuntimeError(
                "torch.my_device is not available. "
                "Install my-vendor-torch-extension."
            )
        return torch.my_device

    def is_available(self) -> bool:
        """Check if devices are visible and usable in the current process.

        Called at runtime to verify device accessibility.
        This should be a fast check — typically just torch.<device>.is_available().

        Note: In Ray CPU actors, this may return False even if the cluster
        has accelerators. Use is_platform_available() for broader detection.
        """
        if not _ensure_torch_my_device():
            return False
        return torch.my_device.is_available()

    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        """Determine if this platform is the target hardware during auto-detection.

        Called ONCE during startup to determine which platform to use.
        Unlike is_available(), this can use broader checks that work even
        in processes without device visibility (e.g. CPU-only Ray actors).

        Args:
            use_smi_check: If True, use relaxed detection (SMI commands,
                          package importability) that works without GPU visibility.

        For CUDA-compatible hardware (like MetaX), you MUST implement SMI-based
        detection to distinguish from NVIDIA:
            if use_smi_check:
                return self.check_smi_command("my-smi")

        For non-CUDA hardware (XPU, MLU, NPU), checking torch extension
        importability is usually sufficient.
        """
        if not _ensure_torch_my_device():
            return False
        if use_smi_check:
            # Option A: SMI command check (for CUDA-compatible hardware)
            return self.check_smi_command("my-smi")
            # Option B: Extension importability (for non-CUDA hardware)
            # return True  # torch extension imported successfully
        return torch.my_device.is_available()

    def current_device(self) -> int:
        """Return the index of the currently active device."""
        return torch.my_device.current_device()

    def device_count(self) -> int:
        """Return the total number of visible devices."""
        return torch.my_device.device_count()

    def set_device(self, device_index: int) -> None:
        """Select the device at the given index.

        Called by verl before performing operations on a specific device,
        particularly in multi-GPU training with Ray actors.
        """
        torch.my_device.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        """Block until all pending operations on the device complete.

        Used for accurate timing and ensuring computation is complete
        before reading results. Critical for profiling and checkpointing.
        """
        if device_index is not None:
            torch.my_device.synchronize(device_index)
        else:
            torch.my_device.synchronize()

    # ==================================================================
    # Random number generator
    # ==================================================================

    def manual_seed(self, seed: int) -> None:
        """Seed the RNG on the current device.

        Called during model initialization for reproducibility.
        """
        torch.my_device.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        """Seed the RNG on ALL devices.

        Used when all workers need identical random state (e.g. model init).
        """
        torch.my_device.manual_seed_all(seed)

    # ==================================================================
    # Memory management
    # ==================================================================

    def set_allocator_settings(self, settings: str) -> None:
        """Configure the device memory allocator.

        verl typically calls this with "expandable_segments:True" to
        reduce memory fragmentation during training.

        If your hardware doesn't support allocator configuration,
        implement as a no-op (pass) or log a warning.
        """
        try:
            torch.my_device.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning(
                "MyVendor: allocator settings not supported, ignoring: %s",
                settings,
            )

    def empty_cache(self) -> None:
        """Release all unused cached memory back to the system.

        Called periodically during training to prevent OOM, especially
        between generation (rollout) and training phases.
        """
        torch.my_device.empty_cache()

    # ==================================================================
    # Device properties
    # ==================================================================

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        """Return (major, minor) compute capability.

        Used by verl/vLLM to select optimal kernel implementations.
        Return (None, None) if your hardware doesn't have this concept.
        """
        # If your hardware has compute capability:
        # return torch.my_device.get_device_capability(device_index)
        return (None, None)

    # ==================================================================
    # Distributed communication
    # ==================================================================

    def communication_backend_name(self) -> str:
        """Return the name of the distributed communication backend.

        This is passed to torch.distributed.init_process_group(backend=...).
        Common values: "nccl" (NVIDIA), "hccl" (Huawei), "xccl" (Intel),
                       "cncl" (Cambricon), "flagcx" (cross-platform).

        FlagCX support: If your hardware supports FlagCX, check the
        USE_FLAGCX environment variable:
            if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"]:
                return "flagcx"
        """
        return "my_ccl"

    def visible_devices_envvar(self) -> str:
        """Return the environment variable name that controls device visibility.

        Examples:
        - NVIDIA: "CUDA_VISIBLE_DEVICES"
        - Huawei: "ASCEND_RT_VISIBLE_DEVICES"
        - Intel:  "ZE_AFFINITY_MASK"
        - Cambricon: "MLU_VISIBLE_DEVICES"

        verl and Ray use this to isolate devices per worker process.
        """
        return "MY_VISIBLE_DEVICES"

    # ==================================================================
    # Ray integration
    # ==================================================================

    def ray_resource_name(self) -> str:
        """Return the Ray resource name for this accelerator type.

        - "GPU" is a built-in Ray resource (NVIDIA, MetaX, and other CUDA-compatible)
        - Custom resources ("NPU", "MLU", "XPU") require Ray cluster configuration

        If using "GPU", Ray's built-in GPU scheduling works out of the box.
        If using a custom name, you must configure it when starting Ray:
            ray start --resources='{"MLU": 8}'
        """
        return "MY_DEVICE"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        """Return the Ray actor resource request dictionary.

        For built-in "GPU" resource:
            return {"num_gpus": num_gpus}

        For custom resources:
            return {"resources": {"MY_DEVICE": num_gpus}}
        """
        return {"resources": {"MY_DEVICE": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        """Return env vars that Ray should NOT auto-set.

        By default, Ray sets CUDA_VISIBLE_DEVICES based on resource allocation.
        For non-CUDA hardware, this causes conflicts. Return the env var name(s)
        that Ray should leave alone, prefixed with RAY_EXPERIMENTAL_NOSET_:
            ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]  # for CUDA-compatible
            ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"]  # for NPU
        """
        return ["RAY_EXPERIMENTAL_NOSET_MY_VISIBLE_DEVICES"]

    # ==================================================================
    # IPC support
    # ==================================================================

    def is_ipc_supported(self) -> bool:
        """Return True if the platform supports IPC (Inter-Process Communication)
        for sharing tensors between processes without copying.

        IPC is used by verl for efficient tensor sharing between Ray actors
        (e.g. sharing model weights between actor and rollout workers).

        If False, verl falls back to serialization-based transfer.
        """
        return False

    # ==================================================================
    # Profiling helpers
    # ==================================================================

    @contextmanager
    def nvtx_range(self, msg: str):
        """Context manager for profiler annotations.

        Used to mark code regions for profiling tools (NVIDIA Nsight,
        Intel VTune, etc.). If your hardware has no profiling API,
        implement as a simple yield (no-op).
        """
        # If your hardware has a profiling API:
        # torch.my_device.profiler.range_push(msg)
        # try:
        #     yield
        # finally:
        #     torch.my_device.profiler.range_pop()

        # No-op implementation:
        yield

    def profiler_start(self) -> None:
        """Start the hardware profiler. No-op if unsupported."""
        pass

    def profiler_stop(self) -> None:
        """Stop the hardware profiler. No-op if unsupported."""
        pass

    # ==================================================================
    # vLLM integration
    # ==================================================================

    def get_device_uuid(self, device_id: int) -> str:
        """Return a unique identifier string for the given device.

        Used by verl's vLLM integration to map Ray resources to physical
        devices. The default implementation in PlatformBase constructs a
        UUID from ray_resource_name() + visible device index (via the
        visible_devices_envvar()).

        Most platforms do NOT need to override this — the base class logic
        handles it. Override only if your hardware has a different UUID scheme.

        Default behavior (inherited from PlatformBase):
            1. If visible_devices_envvar() is set:
               return ray_resource_name() + visible_devices[device_id]
            2. Otherwise:
               return f"{ray_resource_name()}-{device_id}"
        """
        # Usually no need to override — base class implementation is sufficient.
        # Only override if your device UUID scheme differs from the default.
        return super().get_device_uuid(device_id)

    # ==================================================================
    # Model patches
    # ==================================================================

    def apply_model_patches(self, model_type: str) -> None:
        """Apply platform-specific model monkey patches.

        Called before model construction. Use this to replace unsupported
        operators with device-optimized alternatives. For example:
        - Replace flash attention with a compatible implementation
        - Swap RMSNorm/LayerNorm with hardware-optimized versions
        - Patch RoPE embeddings for device compatibility

        Args:
            model_type: The model architecture name (e.g. "qwen2", "llama")

        Example (NPU replaces RMSNorm):
            if model_type == "qwen2":
                from transformers.models.qwen2 import modeling_qwen2
                modeling_qwen2.Qwen2RMSNorm = MyOptimizedRMSNorm
        """
        pass

    # ==================================================================
    # Rollout engine integration
    # ==================================================================

    def rollout_env_vars(self) -> dict[str, str]:
        """Return env vars to inject when launching vLLM rollout workers.

        Used to configure the rollout engine (vLLM) for specific hardware
        quirks. Common use cases:
        - Disable NCCL CUMEM to prevent deadlocks in disaggregated mode
        - Set vendor-specific vLLM configuration flags

        Example (NVIDIA):
            return {"NCCL_CUMEM_ENABLE": "0"}
        Example (Huawei NPU):
            return {"NCCL_CUMEM_ENABLE": "0", "VLLM_ASCEND_AUTO_DETECT_QUANTIZATION": "0"}
        """
        return {}

    # ==================================================================
    # Collective communication
    # ==================================================================

    def get_collective_module(self) -> Any:
        """Return the low-level collective communication module.

        Used for direct NCCL/HCCL/CNCL operations outside PyTorch's
        distributed framework. Return None if not available.

        Example (NVIDIA):
            from cupy.cuda import nccl
            return nccl
        """
        return None

    # ==================================================================
    # Low-level runtime API
    # ==================================================================

    def cudart(self) -> Any:
        """Return the CUDA runtime API object, or None if not applicable.

        Only meaningful for CUDA-compatible hardware.
        Non-CUDA platforms should return None.
        """
        return None
```

### Step 2: Register the Platform Module

Add your platform to `verl_hardware_plugin/platforms/__init__.py`:

```python
def register_all_platforms():
    """Import all platform modules to trigger their @register decorators."""

    # ... existing platforms ...

    # MyVendor
    try:
        from verl_hardware_plugin.platforms import platform_my_vendor  # noqa: F401
        logger.info("Registered platform: my_vendor")
    except Exception as e:
        logger.debug("MyVendor platform not registered: %s", e)
```

**Why try/except?** — The plugin package may be installed on machines without your hardware SDK. Conditional imports prevent import errors from affecting other platforms.

### Step 3: Create the Engine Class (Optional)

If your hardware needs custom training behavior (e.g. different reduction ops, special initialization), create an engine file under `verl_hardware_plugin/engines/`:

```python
# verl_hardware_plugin/engines/fsdp_my_vendor.py
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for MyVendor devices.

Extends the base FSDP engine with hardware-specific initialization:
- Custom reduction operations for the vendor's communication library
- Memory optimization flags specific to the hardware
- Device-specific workarounds for known issues
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(
    model_type="language_model",          # "language_model" or "value_model"
    backend=["fsdp", "fsdp2"],            # which backends this engine supports
    device="my_device",                   # must match platform.device_name
    vendor="my_vendor",                   # must match platform.vendor_name
)
class FSDPMyVendorEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for MyVendor with custom communication backend.

    Engine lookup priority:
    1. Exact match (device="my_device", vendor="my_vendor") ← this class
    2. Device-only fallback (device="my_device")
    3. NVIDIA fallback (for CUDA-compatible devices only)
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPMyVendorEngineWithLMHead initialized")

    def initialize(self):
        """Called after model is constructed but before training begins.

        This is the main extension point. Common customizations:
        - Force sum reduction if the comm backend doesn't support AVG
        - Enable/disable specific FSDP features
        - Apply vendor-specific optimizations
        """
        super().initialize()

        # Example: Force sum-based reduction for custom communication backend
        # (some backends like xccl don't support ReduceOp.AVG)
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
            logger.info("Enabled force_sum_reduction for MyVendor comm backend")


@EngineRegistry.register(
    model_type="value_model",
    backend=["fsdp", "fsdp2"],
    device="my_device",
    vendor="my_vendor",
)
class FSDPMyVendorEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine for MyVendor value model training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def initialize(self):
        super().initialize()
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
```

Then register in `verl_hardware_plugin/engines/__init__.py`:

```python
    # MyVendor engines
    try:
        from verl_hardware_plugin.engines import fsdp_my_vendor  # noqa: F401
        logger.info("Registered engines: fsdp_my_vendor")
    except Exception as e:
        logger.debug("MyVendor FSDP engines not registered: %s", e)
```

### Step 4: Test Registration

Run the registration test to verify everything loads correctly:

```bash
pip install -e .
pytest tests/test_plugin_registration.py -v
```

You can also verify manually:

```bash
# Check platform detection
VERL_PLATFORM=my_vendor python -c "
from verl.plugin.platform import get_platform
p = get_platform()
print(f'device: {p.device_name}')
print(f'vendor: {p.vendor_name}')
print(f'available: {p.is_available()}')
"
```

### Step 5: Add User Documentation

Each hardware platform must have a corresponding user documentation directory `user_guide_<vendor>` under `docs/` for end-user reference.

Naming convention: `user_guide_<vendor>`, e.g. `user_guide_xpu`, `user_guide_mlu`, `user_guide_metax`, `user_guide_flagos`.

Follow the structure of [verl/docs/ascend_tutorial](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial) when creating the documentation. Recommended contents:

```
docs/user_guide_<vendor>/
├── README.md              # Entry point: introduction, directory layout, platform summary
├── install_guidance.md    # Installation: prerequisites, install steps, environment verification
├── quick_start.md         # Quick start: basic validation scenario (e.g. Qwen2.5-0.5B GRPO)
└── faq.md                 # FAQ: common errors, known limitations, diagnostic commands
```

**README.md should include:**
- Platform introduction (supported hardware models)
- Directory structure overview
- Platform summary table (device type, communication backend, Ray resource name, etc.)
- Required environment variables

**install_guidance.md should include:**
- Hardware requirements and supported models
- Prerequisites (vendor PyTorch extension, drivers, communication libraries)
- Installation steps (including Docker-based deployment)
- Environment verification commands

**quick_start.md should include:**
- Data preparation steps
- Complete runnable training commands (recommend using Qwen2.5-0.5B GRPO as baseline validation)
- Key configuration explanations
- Multi-node setup (if applicable)

**faq.md should include:**
- Common errors and solutions
- Known limitations
- Diagnostic tools/commands

Existing reference implementations:
- `docs/user_guide_xpu/` — Intel XPU
- `docs/user_guide_mlu/` — Cambricon MLU
- `docs/user_guide_metax/` — MetaX
- `docs/user_guide_flagos/` — FlagOS

> **Tip**: Refer to `verl/docs/ascend_tutorial` (Huawei NPU) for documentation quality and coverage expectations. That tutorial covers installation, quick start, advanced features, performance tuning, precision analysis, and FAQ.

---

## Complete Real-World Example: Intel XPU

Here's how the Intel XPU platform is actually implemented in this repository, annotated with design rationale:

```python
# Key design decisions for Intel XPU:
#
# 1. device_name = "xpu" — Intel uses torch.xpu.* API (via intel_extension_for_pytorch)
# 2. vendor_name = "intel" — used for engine lookup: (device="xpu", vendor="intel")
# 3. communication_backend = "xccl" — Intel's oneAPI Collective Communications Library
# 4. ray_resource_name = "GPU" — reuses Ray's built-in GPU resource to simplify deployment
# 5. is_ipc_supported = False — Intel XPU does not yet support CUDA-style IPC
# 6. visible_devices_envvar = "ZE_AFFINITY_MASK" — Intel Level Zero environment variable
# 7. _ensure_torch_xpu() — must import intel_extension_for_pytorch before torch.xpu works

@PlatformRegistry.register(platform="intel")
class PlatformXPU(PlatformBase):
    ...
```

And the corresponding engine:

```python
# Key design decisions for XPU FSDP engine:
#
# 1. Extends FSDPEngineWithLMHead — inherits all standard FSDP logic
# 2. Only overrides initialize() — minimal intrusion pattern
# 3. Forces sum reduction — xccl doesn't support ReduceOp.AVG,
#    so gradient averaging is done as sum/world_size instead

@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="xpu", vendor="intel")
class FSDPXPUEngineWithLMHead(FSDPEngineWithLMHead):
    def initialize(self):
        super().initialize()
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
```

---

## Complete Real-World Example: MetaX (CUDA-Compatible Hardware)

MetaX is a special case — it uses `torch.cuda` but is not NVIDIA hardware:

```python
# Key design decisions for MetaX:
#
# 1. device_name = "cuda" — MetaX is CUDA-compatible, uses torch.cuda API
# 2. vendor_name = "metax" — distinguishes from NVIDIA in engine lookup
# 3. is_platform_available() uses check_smi_command("mx-smi") —
#    Since both NVIDIA and MetaX return True for torch.cuda.is_available(),
#    we use the mx-smi tool to detect MetaX hardware specifically.
# 4. communication_backend = "nccl" — MetaX supports standard NCCL
# 5. ray_resource_name = "GPU" — standard CUDA GPU resource

@PlatformRegistry.register(platform="metax")
class PlatformMetaX(PlatformBase):
    def is_platform_available(self, use_smi_check: bool = False) -> bool:
        if not torch.cuda.is_available():
            return False
        if use_smi_check:
            # mx-smi is MetaX's equivalent of nvidia-smi
            # This is the ONLY way to distinguish MetaX from NVIDIA
            return self.check_smi_command("mx-smi")
        return True
```

---

## API Reference

### PlatformBase Methods

| Category | Method | Abstract? | Description |
|----------|--------|-----------|-------------|
| **Core** | `device_name` | Yes | torch device type string |
| | `vendor_name` | Yes | Hardware vendor identifier |
| | `device_module` | Yes | `torch.<device>` module |
| | `is_available()` | Yes | Runtime device check |
| | `is_platform_available(use_smi_check)` | No (has default) | Auto-detection probe |
| | `current_device()` | Yes | Current device index |
| | `device_count()` | Yes | Number of devices |
| | `set_device(idx)` | Yes | Select active device |
| | `synchronize(idx)` | Yes | Synchronize device |
| **RNG** | `manual_seed(seed)` | Yes | Seed current device |
| | `manual_seed_all(seed)` | Yes | Seed all devices |
| **Memory** | `set_allocator_settings(s)` | Yes | Configure memory allocator |
| | `empty_cache()` | Yes | Release cached memory |
| **Properties** | `get_device_capability(idx)` | Yes | `(major, minor)` or `(None, None)` |
| **Communication** | `communication_backend_name()` | Yes | `'nccl'`, `'hccl'`, `'xccl'`, … |
| | `visible_devices_envvar()` | Yes | Env var controlling device visibility |
| | `get_collective_module()` | No | Collective comm module |
| **Ray** | `ray_resource_name()` | Yes | Ray resource name (`'GPU'`, `'NPU'`, …) |
| | `ray_noset_envvars()` | Yes | `RAY_EXPERIMENTAL_NOSET_*` env var names |
| | `ray_resource_options(num_gpus)` | No (has default) | Ray actor resource dict |
| **IPC** | `is_ipc_supported()` | Yes | Whether IPC tensor sharing is supported |
| **Rollout** | `rollout_env_vars()` | No | Env vars for rollout engine launch |
| **vLLM** | `get_device_uuid(device_id)` | No (has default) | Unique device identifier for vLLM resource mapping |
| **Model Patches** | `apply_model_patches(model_type)` | No | Apply platform-specific model monkey patches |
| **Profiling** | `nvtx_range(msg)` | Yes | Context manager for profiler ranges |
| | `profiler_start()` | Yes | Start device profiler |
| | `profiler_stop()` | Yes | Stop device profiler |
| **Low-level** | `cudart()` | Yes | CUDA runtime API object or None |
| | `check_smi_command(cmd)` | No (static) | Run SMI command and check exit code |

### Engine Lookup Logic

The `EngineRegistry.get_engine_cls(model_type, backend)` lookup follows this priority:

1. **Exact match** `(device, vendor)` — vendor-specific engine (e.g. `("cuda", "metax")`)
2. **Device-only fallback** `device` — base engine for that device type (e.g. `"xpu"`)
3. **NVIDIA fallback** — for unknown CUDA vendors, try `("cuda", "nvidia")` then `"cuda"`

Environment variable overrides:
- `VERL_ENGINE_DEVICE` — override the detected device name
- `VERL_ENGINE_VENDOR` — override the detected vendor name

---

## Key Design Principles

1. **Conditional imports**: Platform modules are imported inside `try/except` blocks — a missing hardware SDK does not affect other platforms.
2. **Last writer wins**: A platform registered later with the same name overrides the earlier one, allowing plugins to override built-in implementations.
3. **Auto-detection**: The first platform whose `is_platform_available(use_smi_check=True)` returns True is selected, or it can be forced via `VERL_PLATFORM`.
4. **Minimal intrusion**: Engine extensions inject logic through inheritance + `initialize()` without modifying base class behavior.
5. **Two-dimensional engine key**: `(device, vendor)` allows multiple vendors sharing the same device type (e.g. MetaX and FlagOS both use `"cuda"`) to have distinct engines.
6. **Section-based organization**: Platform implementations should group methods by category using comment section headers for readability and consistency.

---

## Common Pitfalls and Solutions

### Problem: Both NVIDIA and my CUDA-compatible hardware are detected

**Cause**: `torch.cuda.is_available()` returns True for both.

**Solution**: Implement `is_platform_available(use_smi_check=True)` with `check_smi_command("your-smi")`. The auto-detection loop probes platforms in registration order; your vendor SMI tool distinguishes your hardware.

### Problem: Ray doesn't see my custom accelerator resource

**Cause**: Ray only knows about "GPU" (CUDA) by default.

**Solution**: Either:
- Use `ray_resource_name() = "GPU"` if your hardware is CUDA-compatible
- Configure Ray with custom resources: `ray start --resources='{"MY_DEVICE": 8}'`

### Problem: Distributed training hangs during all-reduce

**Cause**: Communication backend doesn't support `ReduceOp.AVG`.

**Solution**: In your engine's `initialize()`, enable force sum reduction:
```python
self.model.set_force_sum_reduction_for_comms(True)
```

### Problem: Import errors when plugin is installed on machines without my hardware

**Cause**: Top-level imports of vendor SDK at module level.

**Solution**: Use lazy imports inside method bodies, or guard top-level imports in a `_ensure_*()` helper function. The `register_all_platforms()` function already wraps imports in try/except.

---

## Project Structure

```
verl-hardware-plugin/
├── pyproject.toml                         # Package config + entry_points
├── verl_hardware_plugin/
│   ├── __init__.py                        # Entry point: register_all_*()
│   ├── platforms/
│   │   ├── __init__.py                    # register_all_platforms()
│   │   ├── platform_xpu.py               # Intel XPU reference
│   │   ├── platform_mlu.py               # Cambricon MLU reference
│   │   ├── platform_cuda_metax.py        # MetaX reference
│   │   └── platform_<vendor>.py          # Your new platform
│   ├── engines/
│   │   ├── __init__.py                    # register_all_engines()
│   │   ├── fsdp_xpu.py                   # Intel FSDP reference
│   │   ├── fsdp_mlu.py                   # Cambricon FSDP reference
│   │   ├── fsdp_metax.py                 # MetaX FSDP reference
│   │   ├── fsdp_<vendor>.py              # Your new FSDP engine
│   │   └── megatron_<vendor>.py          # Your new Megatron engine (optional)
│   └── utils/
│       ├── __init__.py
│       └── config_manager.py
├── tests/
│   └── test_plugin_registration.py
└── docs/
    ├── development.md                     # This file
    └── user_guide.md                      # End-user documentation
```

---

## Reference Implementations

The following files in this repository serve as examples:

| Vendor | Platform File | Engine Files |
|--------|--------------|--------------|
| Intel XPU | `platforms/platform_xpu.py` | `engines/fsdp_xpu.py`, `engines/megatron_xpu.py` |
| Cambricon MLU | `platforms/platform_mlu.py` | `engines/fsdp_mlu.py`, `engines/megatron_mlu.py` |
| MetaX | `platforms/platform_cuda_metax.py` | `engines/fsdp_metax.py`, `engines/megatron_metax.py` |

---

## Acceptance Baseline for New Hardware Adaptation

When submitting a new hardware platform adaptation, you **must** demonstrate that end-to-end GRPO training produces a reward curve consistent with the NVIDIA GPU reference. This section defines the baseline test and acceptance criteria.

### Reference Results

- **Reference run (NVIDIA 8×GPU)**: [SwanLab — verl_grpo_gsm8k_math](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)
- **Key metric**: `critic/rewards/mean` — the first 100 steps must show a clear upward trend consistent with the reference curve.

### Baseline Test Script

Save the following as `scripts/baseline_grpo_gsm8k.sh` and run it on your target hardware. You must adjust `DATA_DIR` and `MODEL_DIR` to your local paths.

```bash
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

########################### user-adjustable ###########################
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_${INFER_BACKEND}_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################
# Modify these paths to your actual data/model locations
DATA_DIR=/workspace/gsm8k
MODEL_DIR=/workspace/Qwen3-0.6B

n_trainer_devices=$NGPUS_PER_NODE

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$DATA_DIR/train.parquet']"
    data.val_files="['$DATA_DIR/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_DIR"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    actor_rollout_ref.rollout.free_cache_engine=False
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

HYDRA_FULL_ERROR=1
########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@" \
    2>&1 | tee "verl_demo.log"
```

### Acceptance Criteria

When submitting a PR for a new hardware platform, you must provide:

1. **SwanLab (or equivalent) training log link** showing the full training run.
2. **`critic/rewards/mean` curve comparison** — screenshot or overlay of your hardware's curve vs. the NVIDIA reference for the **first 100 steps**.
3. The reward curve must show:
   - A clear upward trend starting within the first 20 steps.
   - No divergence or collapse (reward staying flat or dropping) within the first 100 steps.
   - General shape and magnitude consistent with the reference (small deviations due to hardware numerics are acceptable).
4. **Training completes without errors** — the script must run to completion (all epochs) without crashes or hangs.

> **Tip**: Use `trainer.logger='["console","swanlab"]'` to automatically upload metrics to SwanLab for easy comparison. Set `PROJECT_NAME=verl_grpo_gsm8k_math` so all adaptation runs are grouped together.

### Quick Checklist

- [ ] Ran `scripts/baseline_grpo_gsm8k.sh` on target hardware with 8 devices
- [ ] Training completed all epochs without error
- [ ] `critic/rewards/mean` shows upward trend in first 100 steps
- [ ] Provided SwanLab link or equivalent training log
- [ ] Curve is consistent with [NVIDIA reference](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

---

## Related Resources

- **verl core PR**: [verl#6086 — Platform & Engine Registry](https://github.com/verl-project/verl/pull/6086)
- **Platform base class**: `verl/plugin/platform/platform_base.py`
- **Engine base class**: `verl/workers/engine/base.py`
- **Platform README**: `verl/plugin/platform/README.md`
- **User Guide**: [docs/user_guide.md](user_guide.md)