# FlagOS User Guide

Last updated: 06/04/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on the FlagOS unified heterogeneous platform.

## Directory Structure

```
user_guide_flagos/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start
└── faq.md                 # FAQ and troubleshooting
```

## Getting Started

- [Installation Guide](./install_guidance.md)
- [Quick Start](./quick_start.md)
- [FAQ](./faq.md)

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `cuda` (CUDA-compatible) |
| Vendor identifier | `flagos` |
| Communication backend | `flagcx` (or fallback to `nccl`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

## Environment Variables

```bash
export VERL_PLATFORM=flagos
export USE_FLAGGEMS=true    # Enable FlagGems operator library
export USE_FLAGCX=1         # Enable FlagCX communication library

# Optional: operator whitelist/blacklist (choose one)
export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax
# export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention
```
