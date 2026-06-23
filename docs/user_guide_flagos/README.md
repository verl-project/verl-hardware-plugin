# FlagOS User Guide

Last updated: 06/23/2026.

## Introduction

FlagOS is a unified heterogeneous engine that supports multiple hardware backends through a single interface. It does not register as a separate platform but as an engine vendor (`flagos`) on top of existing device platforms (e.g., `cuda`).

This directory contains platform-specific user guides for running verl with FlagOS.

## Guides by Hardware

| Hardware | Status | Guide |
|----------|--------|-------|
| NVIDIA GPU | ✅ Verified | [NVIDIA User Guide](./nvidia/README.md) |
| Ascend NPU | 🚧 In progress | TBD |
| MetaX | 🚧 In progress | TBD |
| Iluvatar | 🚧 In progress | TBD |

## Directory Structure

```
user_guide_flagos/
├── README.md              # This file (index)
└── nvidia/                # NVIDIA GPU guide
    ├── README.md
    ├── install_guidance.md
    ├── quick_start.md
    ├── env_reference.md
    └── faq.md
```

## Platform Summary

| Item | Description |
|------|-------------|
| Type | Engine (not a standalone platform) |
| Vendor identifier | `flagos` |
| Design | Multi-chip unified engine |
| Communication backend | `flagcx` (or fallback to device-native, e.g. `nccl`) |
| Ray resource name | `GPU` |
| IPC support | Yes |
