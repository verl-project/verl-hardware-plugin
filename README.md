# verl-hardware-plugin

Multi-chip hardware platform and engine plugin **reference implementations** for [verl](https://github.com/verl-project/verl).

This package provides platform abstraction and training engine extensions for non-CUDA accelerators. It serves as a **template and example** for hardware vendors to adapt verl to their own devices through the unified plugin interface.

## About

This repository is jointly developed by the ByteDance verl team and the [FlagOS](https://github.com/flagos-ai#flagos-a-unified-open-source-ai-system-software-stack) team at Beijing Academy of Artificial Intelligence (BAAI).

## Purpose

The platforms and engines in this repository are **reference implementations** — they demonstrate how vendors can integrate their hardware with verl's plugin system. Hardware vendors can use these as templates to build their own plugins.

## Supported Hardware (Reference Implementations)

> **Note**: The implementations below are **examples only**. Full production support and maintenance require collaboration with the respective hardware vendors. These serve as templates for vendors to adapt and maintain their own integrations.

| Platform | Device | Communication | Status | Doc |
|----------|--------|---------------|--------|-----|
| FlagOS | NVIDIA GPU (verified) | FlagCX / NCCL | ✅ Supported | [User Guide](docs/user_guide_flagos/README.md) |
| Intel XPU | Data Center GPU Max / Arc | xccl (oneCCL) | ✅ Example (requires vendor support) | TBD |
| Cambricon MLU | MLU370 / MLU590 | CNCL | ✅ Example (requires vendor support) | TBD |
| MetaX | MetaX GPUs (CUDA-compatible) | NCCL | ✅ Example (requires vendor support) | TBD |
| Huawei NPU | Ascend 910B | HCCL | Built-in (verl core) | [Ascend Tutorial](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial) |


## Installation

```bash
pip install --no-build-isolation -e .
```

## Usage

After `pip install`, the plugin is automatically discovered by verl through the
`verl.plugins` entry_points group. No additional configuration needed.

For platform-specific usage and configuration, please refer to each platform's documentation in the [Supported Hardware](#supported-hardware-reference-implementations) table above.

## Architecture

```
verl-FL (main framework)
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── PlatformRegistry.register("intel")    → PlatformXPU
            ├── PlatformRegistry.register("cambricon")→ PlatformMLU
            ├── PlatformRegistry.register("metax")    → PlatformMetaX
            ├── PlatformRegistry.register("flagos")   → PlatformFlagOS
            │
            ├── EngineRegistry.register(device="xpu", vendor="intel")
            ├── EngineRegistry.register(device="mlu", vendor="cambricon")
            ├── EngineRegistry.register(device="cuda", vendor="metax")
            └── EngineRegistry.register(device="cuda", vendor="flagos")
```

The plugin uses verl's decorator-based registration:
- `@PlatformRegistry.register(platform="vendor_name")` for platform classes
- `@EngineRegistry.register(model_type=..., backend=..., device=..., vendor=...)` for engine classes

Registration happens at import time. Engine lookup uses a two-level key `(device, vendor)`:
1. Exact match `(device, vendor)` — vendor-specific engine
2. Fallback to device-only key — base engine for that device type
3. For CUDA-compatible devices, fallback to base CUDA engine

### SMI-based Hardware Detection

For CUDA-compatible hardware (MetaX, NVIDIA), `torch.cuda.is_available()` returns True on both. The `is_platform_available(use_smi_check=True)` method enables SMI command checks to distinguish the actual hardware:

- `PlatformCUDA` checks `nvidia-smi`
- `PlatformMetaX` checks `mx-smi`

This check is only performed during first-time auto-detection. The `is_available()` method (without parameters) directly calls the native `torch.<device>.is_available()` and is used for runtime device availability checks.

## Documentation

### User Guides (by Hardware Platform)

Each hardware platform provides a standalone user guide (following the structure of [verl/docs/ascend_tutorial](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial)):

- **[Intel XPU](docs/user_guide_xpu/README.md)** — Intel Data Center GPU Max / Arc user guide
- **[Cambricon MLU](docs/user_guide_mlu/README.md)** — Cambricon MLU370 / MLU590 user guide
- **[MetaX GPU](docs/user_guide_metax/README.md)** — MetaX GPU user guide
- **[FlagOS](docs/user_guide_flagos/README.md)** — FlagOS unified heterogeneous platform user guide

### Developer Guides

- **[Development Guide](docs/development.md)** — How to add a new hardware platform and engine (start here for adaptation)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache License 2.0
