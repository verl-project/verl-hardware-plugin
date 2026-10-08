# VERL Biren SUPA User Guide

## Introduction

This document describes how to use verl for reinforcement learning training on Biren accelerators through the SUPA runtime.

## Directory Structure

```text
verl_hardware_plugin/accelerators/supa/
├── platform_supa.py                  # SUPA platform settings
└── engines/
    ├── fsdp_supa.py                  # FSDP engine support
    └── megatron_supa.py              # Megatron engine support
```

```text
docs/accelerators/supa/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
└── quick_start.md                    # GSM8K GRPO quick start
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — run a GSM8K GRPO training job

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `supa` |
| Vendor identifier | `biren` |
| Runtime | SUPA / `torch_supa` |
| Communication backend | `bccl` |
| Device visibility env var | `SUPA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |
| Hardware detection | `brsmi` when SMI checking is enabled |

## SUPA Platform Integration

SUPA is CUDA-compatible. The platform exposes its device as `supa` with vendor `biren` so verl selects the Biren-specific FSDP and Megatron engines. The runtime is reached through `torch_supa`, which maps `torch.supa` calls to the SUPA runtime and rewrites NCCL process-group setup to BCCL. verl discovers the plugin automatically through the `verl.plugins` entry points.

The plugin does not install separate SUPA compatibility hooks at import time. Install and initialize the Biren SUPA runtime according to the matching Biren release before starting verl.

```bash
export VERL_PLATFORM=biren
export SUPA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```
