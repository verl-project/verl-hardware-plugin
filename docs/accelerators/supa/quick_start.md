# Biren SUPA Quick Start

This guide walks you through the GSM8K GRPO baseline on Biren SUPA. Complete the [Installation Guide](./install_guidance.md) first.

**Baseline scenario:** Qwen3-0.6B + GSM8K + FSDP actor + SGLang rollout — see [`scripts/baseline_grpo_gsm8k.sh`](../../../scripts/baseline_grpo_gsm8k.sh).

## 1. Prepare Data and Model

The Biren runtime environment should provide the required PyTorch, SUPA, BCCL, Ray, and rollout dependencies. Set the model and dataset directories to paths available in your environment:

```bash
MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B
DATA_DIR=/ipfs/models/gsm8k
```

## 2. Run the Baseline

From the repository root:

```bash

export VERL_PLATFORM=biren
export SUPA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export INFER_BACKEND=sglang
export DATA_DIR=/ipfs/models/gsm8k
export MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B

exec bash "scripts/baseline_grpo_gsm8k.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='biren'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SUPA_VISIBLE_DEVICES='${SUPA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}'" \
    trainer.device=supa \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=supa \
    "$@"

```

The script passes the platform and device environment to Ray workers through `runtime_env`. Shell exports alone are not sufficient for Ray workers.

## 3. Compare Results

Compare `critic/rewards/mean` with the [NVIDIA reference run](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart).

The baseline should:

1. Complete all epochs without a crash or hang.
2. Show an upward reward trend within the first 20 steps.
3. Avoid a flat or collapsing reward curve during the first 100 steps.

## 4. Quick Verification

```bash
python3 -c 'import torch; import torch_supa; print(torch.supa.is_available(), torch.supa.device_count())'
brsmi
```

The output should show that SUPA is available, report the visible device count, and the logs should contain platform bootstrap messages. `brsmi` should list the available Biren devices.

## Multi-Node Setup

Start Ray on the head node and workers, then set `NNODES` and run the baseline:

```bash
# Head node
ray start --head --port=6379
export RAY_ADDRESS='auto'

# Worker nodes
ray start --address='<head-ip>:6379'

export VERL_PLATFORM=biren
export SUPA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

NNODES=2 bash scripts/baseline_grpo_gsm8k.sh \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='biren'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SUPA_VISIBLE_DEVICES='${SUPA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}'" \
    trainer.device=supa \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=supa \
    "$@"
```

As in the baseline above, pass the platform/device environment to Ray workers through `runtime_env`; shell exports alone are not sufficient.

Biren uses Ray's built-in `GPU` resource. Do not configure a custom `biren` or `supa` resource.

## Next Steps

- See [Installation Guide](./install_guidance.md) for runtime setup.
- See [development.md — Acceptance Baseline](../../development.md#acceptance-baseline-for-new-hardware-adaptation) for the PR checklist.
