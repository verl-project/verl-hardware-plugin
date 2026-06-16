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

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `mlu` |
| Vendor identifier | `cambricon` |
| Communication backend | `cncl` |
| Device visibility env var | `MLU_VISIBLE_DEVICES` |
| Ray resource name | `GPU` |
| IPC support | Yes |

## Getting Started
- Please use Cambricon release docker images to run verl and make sure you are in pytorch_infer env.
- Install verl & verl_hardware_plugin
- Start ray cluster:
  -   ```bash
      ray start --head --dashboard-host=0.0.0.0
      ```
- Run verl scripts
  - We recommend using Ray to lanuch the task to make sure all env vars are set correctly.
    1. Add necessary environment variables in runtime_env.yaml
    ```bash 
      working_dir: ./
      excludes: ["/.git/"]
      env_vars:
        TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
        RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO: "0"
        VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
      ```
    2. You can run the scripts in [verl examples](https://github.com/verl-project/verl/tree/main/examples).


