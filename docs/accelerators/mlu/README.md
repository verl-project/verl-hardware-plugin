# Cambricon MLU User Guide

Last updated: 06/16/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Cambricon MLU.

## Directory Structure

Here we list all MLU related files for reference, we will continue to add new features. 

```
verl_hardware_plugin/accelerators/mlu/
├── platform_mlu.py                  # basic platform settings
├── engines/
│   ├── cncl_checkpoint_engine.py    # support checkpoint engine
│   ├── cnixl_checkpoint_engine.py   # CNIXL checkpoint engine
│   ├── fsdp_mlu.py                  # fsdp related model support
│   └── megatron_mlu.py              # megatron related model support
├── profilers/
│   └── torch_profile_mlu.py         # MLU torch profiler
└── utils/
    ├── kernels_mlu.py              # MLU kernels
    └── linear_cross_entropy.py     # MLU cross-entropy helper
```
And docs to start.

```
docs/accelerators/mlu/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start
└── profiling.md           # Profiling guide
```

## Getting Started

- [Installation Guide](./install_guidance.md) — Docker setup, component installation
- [Quick Start](./quick_start.md) — Run your first GRPO training job
- [Profiling Guide](./profiling.md) — Use community torch profile on MLU

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `mlu` |
| Vendor identifier | `cambricon` |
| Communication backend | `cncl` |
| Device visibility env var | `MLU_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

