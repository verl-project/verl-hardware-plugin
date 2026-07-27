# Iluvatar GPU User Guide

Last updated: 06/16/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Iluvatar GPUs.

## Directory Structure

```
user_guide_iluvatar/
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
| Vendor identifier | `iluvatar` |
| Communication backend | `nccl` |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |
| Auto-detection dependency | `ixsmi` command |

## Environment Variables

```bash
export VERL_PLATFORM=iluvatar
# Or ensure ixsmi is on PATH for auto-detection
```
