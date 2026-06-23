# FlagOS User Guide (NVIDIA)

Last updated: 06/09/2026.

## Introduction

This document describes how to use verl for reinforcement learning training with FlagOS multi-chip engine on **NVIDIA GPUs**. FlagOS is designed as a unified heterogeneous engine that supports multiple hardware backends through a single interface — it does not register as a separate platform but as an engine vendor (`flagos`) on top of existing device platforms (e.g., `cuda`).

## Directory Structure

```
user_guide_flagos/nvidia/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start (NVIDIA example)
├── env_reference.md       # Environment variables reference
└── faq.md                 # FAQ and troubleshooting
```

## Getting Started

- [Installation Guide](./install_guidance.md) — Docker setup, component installation
- [Quick Start](./quick_start.md) — Run your first GRPO training job (NVIDIA example)
- [Environment Variables Reference](./env_reference.md) — Operator dispatch and backend control
- [FAQ](./faq.md) — Troubleshooting common issues

## Platform Summary

| Item | Description |
|------|-------------|
| Type | Engine (not a standalone platform) |
| Vendor identifier | `flagos` |
| Design | Multi-chip unified engine |
| Communication backend | `flagcx` (or fallback to device-native, e.g. `nccl`) |
| Ray resource name | `GPU` |
| IPC support | Yes |

### Tested Hardware

| Device | Status |
|--------|--------|
| NVIDIA GPU | Verified |
| Ascend NPU | In progress |
| MetaX | In progress |
| Iluvatar | In progress |

## Environment Variables

```bash
export VERL_ENGINE_VENDOR='flagos'
export USE_FLAGGEMS=true    # Enable FlagGems operator library
export USE_FLAGCX=1         # Enable FlagCX communication library

# Optional: operator whitelist/blacklist (choose one)
export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax
# export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention
```

See [Environment Variables Reference](./env_reference.md) for the full list.
