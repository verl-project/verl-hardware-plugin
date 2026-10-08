# VERL MUSA User Guide

## Introduction

This document describes how to use verl for reinforcement learning training on
Moore Threads MUSA accelerators.

## Directory Structure

```text
verl_hardware_plugin/accelerators/musa/
├── platform_musa.py                  # MUSA platform settings
└── engines/
    ├── fsdp_musa.py                  # FSDP engine support
    └── megatron_musa.py              # Megatron engine support
```

```text
docs/accelerators/musa/
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
| Device type | `musa` |
| Vendor identifier | `moore_threads` |
| Communication backend | `mccl` |
| Device visibility env var | `MUSA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

## MUSA Migration Patches

MUSA deployments may use two separate compatibility layers:

- MUSA support for the upstream Megatron/MCore implementation is provided by
  the external `megatron-lm-musa-patch` compatibility layer. The patch is loaded
  at runtime from the directory specified by `MUSA_PATCH_PATH` (usually
  `/home/megatron-lm-musa-patch` in the release image); it adapts the
  unmodified Megatron code for MUSA execution.

- MUSA compatibility for VERL and SGLang runtime components is provided by the
  deployment-specific `verl-musa-patch` compatibility layer. The patch is loaded
  at runtime from the directory specified by `VERL_MUSA_PATCH` (usually
  `/home/verl-musa-patch` in the release image) and made available to Ray workers
  through `PYTHONPATH`; it adapts the VERL and SGLang runtime components for MUSA
  execution.
