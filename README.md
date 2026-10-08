# verl-hardware-plugin

Multi-chip hardware platform and engine plugin **reference implementations** for [verl](https://github.com/verl-project/verl).

This package provides platform abstraction and training engine extensions for non-CUDA accelerators. It serves as a **template and example** for hardware vendors to adapt verl to their own devices through the unified plugin interface.

## About
This repository is jointly developed by the ByteDance verl team and the [FlagOS](https://github.com/flagos-ai#flagos-a-unified-open-source-ai-system-software-stack) community. FlagOS community is an organiztion jointly launched by the Beijing Academy of Artificial Intelligence (BAAI), together with a broad coalition of research institutes, chipmakers, system vendors, and algorithm and software providers from both China and abroad.

FlagOS is a fully open-sourced AI system software stack for heterogeneous AI chips, allowing AI models to be developed once and seamlessly ported to a wide range of AI hardwares with minimum effort.

## Community

Join the [Feishu discussion group](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=049p82e5-ec84-4a76-90f6-10ae302a9793) to discuss hardware plugin development and usage.

## Purpose

The platforms and engines in this repository are **reference implementations** — they demonstrate how vendors can integrate their hardware with verl's plugin system. Hardware vendors can use these as templates to build their own plugins.

## Supported Hardware (Reference Implementations)

> **Note**: The implementations below are **examples only**. Full production support and maintenance require collaboration with the respective hardware vendors. These serve as templates for vendors to adapt and maintain their own integrations.

| Platform | Device | Communication | Status | Doc |
|----------|--------|---------------|--------|-----|
| FlagOS | NVIDIA GPU (verified) | FlagCX / NCCL | ✅ Supported | [User Guide](docs/integrations/flagos/nvidia/README.md) |
| Intel XPU | Data Center GPU Max / Arc | xccl (oneCCL) | ✅ Example (requires vendor support) | [User Guide](docs/accelerators/xpu/README.md) |
| Cambricon MLU | MLU | CNCL | ✅ Supported | [User Guide](docs/accelerators/mlu/README.md) |
| MetaX | MetaX GPUs (CUDA-compatible) | NCCL API / MCCL | ✅ Supported | [User Guide](docs/accelerators/metax/README.md) |
| Enflame GCU | GCU | ECCL / FlagCX | ✅ Example (requires vendor support) | [User Guide](docs/accelerators/enflame/README.md) |
| Huawei NPU | Ascend 910B | HCCL | Built-in (verl core) | [Ascend Tutorial](https://github.com/verl-project/verl/tree/main/docs/ascend_tutorial) |
| Iluvatar | BI-V150 (CUDA-compatible) | IXCCL | ✅ Supported | [User Guide](docs/accelerators/iluvatar/README.md) |
| Moore Threads | MUSA | MCCL | ✅ Supported | [User Guide](docs/accelerators/musa/README.md) |
| Google TPU | v6e | tpu_dist | Developing and testing | [User Guide](docs/accelerators/tpu/README.md) |
| Biren | SUPA (CUDA-compatible) | BCCL | ✅ Example (requires vendor support) | [User Guide](docs/accelerators/supa/README.md) |


## Installation

```bash
pip install --no-build-isolation -e .
```

## Usage

After `pip install`, the plugin is automatically discovered by verl through the
`verl.plugins` entry_points group. No additional configuration needed.

For platform-specific usage and configuration, please refer to each platform's documentation in the [Supported Hardware](#supported-hardware-reference-implementations) table above.

## Architecture

### Repository Layout

Accelerator-specific code is grouped in `verl_hardware_plugin/accelerators/<backend>/`,
with each backend owning its platform module and any `engines/`, `rollout/`,
`profilers/`, `patches/`, or `utils/` subdirectories it needs. The layout applies
equally to domestic and international hardware, including Intel XPU and Google TPU.

```text
verl_hardware_plugin/
├── __init__.py      # One backend list and load_backends()
├── accelerators/
│   ├── enflame/
│   ├── iluvatar/
│   ├── metax/
│   ├── mlu/
│   ├── musa/
│   ├── supa/       # Biren SUPA
│   ├── tpu/        # Google TPU
│   ├── trainium/   # Reserved for a future AWS Trainium integration
│   └── xpu/        # Intel XPU
├── integrations/
│   └── flagos/     # Software integration spanning accelerator backends
└── utils/          # Shared helpers
```

User guides mirror this layout under `docs/accelerators/<backend>/` and
`docs/integrations/flagos/`. Accelerator-specific tests and scripts live under
`tests/accelerators/<backend>/` and `scripts/accelerators/<backend>/`; shared tests
and scripts stay at their existing top level. The `trainium` directories reserve
space for future work and do not provide or register AWS Trainium support.

Add each backend package once to `BACKEND_MODULES` in the plugin's root
`__init__.py`. `load_backends()` first imports these packages to register their
platforms, then imports each package's `registration.py` for its engines and
optional profiler/rollout hooks. Platforms must be available before upstream
engine imports cache the selected device. Directly importing a backend's
`registration.py` also initializes its package and registers its platform.
Independent optional imports are guarded locally; there is no shared stage
framework or separate component registry list. Integration package initializers
remain inert because they do not register platforms.

### Plugin Registration

```
verl (main framework)
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── PlatformRegistry.register("intel")    → PlatformXPU
            ├── PlatformRegistry.register("cambricon")→ PlatformMLU
            ├── PlatformRegistry.register("metax")    → PlatformMetaX
            ├── PlatformRegistry.register("enflame")  → PlatformENFLAME
            ├── PlatformRegistry.register("iluvatar") → PlatformIluvatar
            ├── PlatformRegistry.register("musa")     → PlatformMUSA
            ├── PlatformRegistry.register("tpu")      → PlatformTPU
            ├── PlatformRegistry.register("biren")    → PlatformSupa
            │
            ├── EngineRegistry.register(device="cuda", vendor="flagos")
            ├── EngineRegistry.register(device="xpu", vendor="intel")
            ├── EngineRegistry.register(device="mlu", vendor="cambricon")
            ├── EngineRegistry.register(device="cuda", vendor="metax")
            ├── EngineRegistry.register(device="gcu", vendor="enflame")
            ├── EngineRegistry.register(device="cuda", vendor="iluvatar")
            ├── EngineRegistry.register(device="musa", vendor="moore_threads")
            ├── EngineRegistry.register(device="tpu", vendor="google")
            ├── EngineRegistry.register(device="supa", vendor="biren")
            │
            └── RolloutReplicaRegistry.register("vllm")  → TPUvLLMReplica on TPU
```

FlagOS registers engines on the CUDA platform rather than a separate platform. The `vllm` rollout
loader returns the TPU rollout on TPU and verl's own vLLM rollout on every other platform.

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

- **[Intel XPU](docs/accelerators/xpu/README.md)** — Intel Data Center GPU Max / Arc user guide
- **[Cambricon MLU](docs/accelerators/mlu/README.md)** — Cambricon MLU user guide
- **[MetaX GPU](docs/accelerators/metax/README.md)** — MetaX GPU user guide
- **[FlagOS](docs/integrations/flagos/README.md)** — FlagOS unified heterogeneous engine user guide ([NVIDIA](docs/integrations/flagos/nvidia/README.md))
- **[Enflame GCU](docs/accelerators/enflame/README.md)** — Enflame GCU user guide
- **[Iluvatar GPU](docs/accelerators/iluvatar/README.md)** — Iluvatar GPU user guide
- **[Moore Threads GPU](docs/accelerators/musa/README.md)** — Moore Threads GPU user guide
- **[Google TPU](docs/accelerators/tpu/README.md)** — Google TPU platform guide
- **[Biren SUPA](docs/accelerators/supa/README.md)** — Biren SUPA accelerator user guide

Future integration location: [AWS Trainium](docs/accelerators/trainium/README.md)
(placeholder only; no implementation).

### Developer Guides

- **[Development Guide](docs/development.md)** — How to add a new hardware platform and engine (start here for adaptation)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache License 2.0
