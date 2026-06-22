# Enflame GCU User Guide

Last updated: 06/24/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Enflame GCU accelerators. Training uses the `torch_gcu` PyTorch extension (`torch.gcu` API), ECCL (or FlagCX) for distributed communication, and the `verl_hardware_plugin` platform/engine integration.

## Directory Structure

```
user_guide_enflame/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start (GRPO baseline)
└── faq.md                 # FAQ and troubleshooting
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites, software stack, environment verification
- [Quick Start](./quick_start.md) — run the acceptance baseline (`scripts/baseline_grpo_gsm8k_enflame.sh`)
- [FAQ](./faq.md) — common errors, known limitations, diagnostic commands

## Software Stack

| Component | Purpose | Required |
|-----------|---------|----------|
| Enflame driver + runtime | GCU hardware access | Yes |
| `torch_gcu` | PyTorch `torch.gcu` API | Yes |
| ECCL | Homogeneous GCU communication (default) | Yes |
| [verl](https://github.com/verl-project/verl) | RL training framework | Yes |
| [verl-hardware-plugin](../development.md) | Platform + FSDP/Megatron engines | Yes |
| [Migration](https://github.com/enflame/Migration) | Runtime patches for verl/vLLM on GCU | Recommended |
| TransformerEngine-FL | Training kernels (vendor or FlagOS path) | Yes (FSDP) |
| vllm-plugin-FL + vLLM | Rollout inference backend | Yes (vLLM rollout) |
| Megatron-LM-FL | Megatron training backend | Optional |

> **Startup order:** set `ENFLAME_ENABLE_AUTO_MIGRATION=1` (if using Migration) **before** importing verl, so runtime patches apply first. Then load `verl_hardware_plugin` and set `VERL_PLATFORM=enflame`.

## Platform Summary

| Item | Description |
|------|-------------|
| Platform key (`VERL_PLATFORM`) | `enflame` |
| Device type (`device_name`) | `gcu` |
| Vendor identifier (`vendor_name`) | `enflame` |
| PyTorch API | `torch.gcu` (via `torch_gcu`) |
| Communication backend | `eccl` (default) or `flagcx` (when `USE_FLAGCX=1`) |
| Device visibility env var | `TOPS_VISIBLE_DEVICES` |
| Ray resource name | `GPU` (built-in; uses `num_gpus`, not custom resources) |
| IPC support | Device-tensor path only (Python SHM unsupported on `torch_gcu`) |

## Acceptance Baseline (E2E Validation)

End-to-end validation uses the repo-standard GRPO baseline on GSM8K with Qwen3-0.6B. Run on Enflame GCU and compare `critic/rewards/mean` against the NVIDIA reference curve.

**Reference (NVIDIA 8×GPU):** [SwanLab — verl_grpo_gsm8k_math](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

**Run on Enflame GCU:**

```bash
# After install_guidance.md steps (data/model at /workspace/...)
bash scripts/baseline_grpo_gsm8k_enflame.sh
```

This wraps [`scripts/baseline_grpo_gsm8k.sh`](../../scripts/baseline_grpo_gsm8k.sh) with Enflame platform env vars and Ray `runtime_env` overrides. Adjust device count if needed:

```bash
NGPUS_PER_NODE=4 TOPS_VISIBLE_DEVICES=0,1,2,3 bash scripts/baseline_grpo_gsm8k_enflame.sh
```

**Acceptance criteria** (first 100 steps):

1. Training completes all epochs without crash or hang.
2. `critic/rewards/mean` shows a clear upward trend (consistent with the [NVIDIA reference](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)).
3. Upload metrics via SwanLab (`trainer.logger='["console","swanlab"]'` is enabled in the baseline script). Use `PROJECT_NAME=verl_grpo_gsm8k_math` to group runs.

See [Quick Start](./quick_start.md) for step-by-step instructions and [development.md](../development.md#acceptance-baseline-for-new-hardware-adaptation) for the full checklist.

## Environment Variables

Core platform settings:

```bash
export VERL_PLATFORM=enflame
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export TOPS_VISIBLE_DEVICES=0,1,2,3
export RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export USE_FLAGCX=0   # ECCL on homogeneous ENFLAME cluster
```

Migration bootstrap (recommended):

```bash
export ENFLAME_ENABLE_AUTO_MIGRATION=1
```

Ray workers do **not** inherit shell exports. Pass critical variables through Hydra `ray_init.runtime_env` (see [Quick Start](./quick_start.md) for the full list):

```bash
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='enflame'
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='verl_hardware_plugin'
+ray_kwargs.ray_init.runtime_env.env_vars.ENFLAME_ENABLE_AUTO_MIGRATION='1'
+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES='1'
+ray_kwargs.ray_init.runtime_env.env_vars.TOPS_VISIBLE_DEVICES='0,1,2,3'
```

## Quick Verification

Run this after installation and before training:

```bash
export ENFLAME_ENABLE_AUTO_MIGRATION=1   # need using Migration
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_PLATFORM=enflame

python3 -c "
import verl_hardware_plugin
import verl
from verl.plugin.platform import get_platform
p = get_platform()
assert p.device_name == 'gcu', f'unexpected device_name: {p.device_name}'
assert p.vendor_name == 'enflame', f'unexpected vendor_name: {p.vendor_name}'
import torch
assert torch.gcu.is_available(), 'torch.gcu is not available'
print('OK: platform=%s device=%s vendor=%s gcu_count=%d' % (
    'enflame', p.device_name, p.vendor_name, torch.gcu.device_count()))
"
```

Expected output: `OK: platform=enflame device=gcu vendor=enflame gcu_count=N`.

## Ray Cluster Configuration

Enflame maps to Ray's built-in **GPU** resource. verl schedules workers with `num_gpus`; you do **not** need a custom Ray resource.

Single-node (verl starts Ray automatically):

```bash
# No manual ray start required for local training
python3 -m verl.trainer.main_ppo ...
```

Multi-node head node:

```bash
ray start --head --port=6379
```

Worker nodes:

```bash
ray start --address='<head-ip>:6379'
```

> **Do not** use `ray start --resources='{"enflame": N}'`. That causes placement groups to stay pending because verl requests `num_gpus`, not a custom `enflame` resource.

## Related Documentation

- [Installation Guide](./install_guidance.md)
- [Quick Start](./quick_start.md)
- [FAQ](./faq.md)
- [FlagOS User Guide](../user_guide_flagos/README.md) — shared FL components (TE-FL, vllm-plugin-FL)
- [Development Guide](../development.md)
