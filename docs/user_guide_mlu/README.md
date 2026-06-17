# Cambricon MLU User Guide

Last updated: 06/16/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Cambricon MLU (MLU370 / MLU590).

## Directory Structure

Here we list all MLU related files for reference, we will continue to add new features. 

```
verl_hardware_plugin/
├── engines
  ├── cncl_checkpoint_engine.py    # support checkpoint engine
  ├── fsdp_mlu.py                  # fsdp related model support
  ├── megatron_mlu.py              # megatron related model support
└── platforms
  └── platform_mlu.py              # basic platform settings
```
And docs to start.

```
user_guide_flagos/
├── README.md              # This file
├── install_guidance.md    # Installation guide
└── quick_start.md         # Quick start
```

## Getting Started

- [Installation Guide](./install_guidance.md) — Docker setup, component installation
- [Quick Start](./quick_start.md) — Run your first GRPO training job

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `mlu` |
| Vendor identifier | `cambricon` |
| Communication backend | `cncl` |
| Device visibility env var | `MLU_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

