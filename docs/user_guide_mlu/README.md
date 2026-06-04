# Cambricon MLU User Guide

Last updated: 06/04/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Cambricon MLU (MLU370 / MLU590).

## Directory Structure

```
user_guide_mlu/
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
| Device type | `mlu` |
| Vendor identifier | `cambricon` |
| Communication backend | `cncl` |
| Device visibility env var | `MLU_VISIBLE_DEVICES` |
| Ray resource name | `MLU` (custom) |
| IPC support | No |

## Environment Variables

```bash
export VERL_PLATFORM=cambricon
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## Ray Cluster Configuration

Note: MLU uses a custom Ray resource. You must declare it when starting Ray workers:

```bash
ray start --head --resources='{"MLU": 8}'
```
