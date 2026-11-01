# Intel XPU User Guide

Last updated: 10/09/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Intel XPU
(Arc Pro / Data Center GPU Max). Intel XPU support ships as an external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree.

## Directory Structure

```text
verl_hardware_plugin/accelerators/xpu/
├── platform_xpu.py                  # Platform metadata + profiler hook
├── registration.py                  # Engine registration on import
├── engines/fsdp_xpu.py              # FSDP/FSDP2 actor + critic engines
├── engines/megatron_xpu.py          # Megatron engine registration (Work in Progress — not yet validated end-to-end)
├── patches/reduce_avg_allreduce_patch.py  # xccl ReduceOp.AVG workaround
├── profilers/itt_profile_xpu.py     # ITT range markers + VtuneProfiler
└── profilers/register_vtune.py      # claims `profiler.tool: vtune`
```

```text
docs/accelerators/xpu/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start
└── profiling.md           # Intel VTune (ITT) profiling
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — run a GRPO training example and verify the platform
- [Profiling](./profiling.md) — capture an Intel VTune (ITT) trace

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `xpu` |
| Vendor identifier | `intel` |
| Communication backend | `xccl` (oneCCL) |
| Device visibility env var | `ZE_AFFINITY_MASK` |
| Ray resource name | `GPU` |
| IPC support | No |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `VERL_PLATFORM` | Set to `intel` to force platform selection instead of relying on auto-detection |
| `VERL_USE_EXTERNAL_PLUGINS` | `auto` (default) discovers this plugin via entry_points; `none` disables discovery |
| `ZE_AFFINITY_MASK` | Selects physical device indices visible to this process, e.g. `0,1` |
| `ONEAPI_DEVICE_SELECTOR` | Must be left unset alongside `ZE_AFFINITY_MASK` — see [quick_start.md](./quick_start.md) |

## Related Documentation

- [verl plugin system](../../development.md)
