# MetaX GPU User Guide

Last updated: 06/04/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on MetaX GPUs.

## Directory Structure

```
user_guide_metax/
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
| Vendor identifier | `metax` |
| Communication backend | `mccl` |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |
| Auto-detection dependency | `mx-smi` command |

## Environment Variables

```bash
export VERL_PLATFORM=metax
# Or ensure mx-smi is on PATH for auto-detection
```
