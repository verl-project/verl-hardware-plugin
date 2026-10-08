# MUSA Quick Start

This guide walks you through the GSM8K GRPO baseline on Moore Threads MUSA.
Complete the [Installation Guide](./install_guidance.md) first.

**Baseline scenario:** Qwen3-0.6B + GSM8K + FSDP actor + SGLang rollout — see
[`scripts/baseline_grpo_gsm8k.sh`](../../../scripts/baseline_grpo_gsm8k.sh).

## 1. Prepare Data and Model

The MUSA image normally provides the runtime dependencies. Set the model and
dataset directories to paths available in your environment:

```bash
MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B
DATA_DIR=/ipfs/models/gsm8k
```

## 2. Run the Baseline

From the repository root:

```bash

export VERL_PLATFORM=musa
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_MUSA_PATCH=/home/verl-musa-patch
export RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES=1
export MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export MCCL_LIB=/usr/local/musa/lib/libmccl.so
export LD_LIBRARY_PATH="/usr/local/musa/lib:${LD_LIBRARY_PATH:-}"
export VLLM_PATCH_MUSA_CUSTOM_OPS=1
export SGLANG_MUSA_GRAPH_COMPAT=1
export PYTHONPATH="${VERL_MUSA_PATCH}:${PYTHONPATH:-}"

export INFER_BACKEND=sglang
export DATA_DIR=/ipfs/models/gsm8k
export MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B

exec bash "scripts/baseline_grpo_gsm8k.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='musa'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_MUSA_PATCH='${VERL_MUSA_PATCH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${VERL_MUSA_PATCH}:${PYTHONPATH:-}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES='${RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO='${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MUSA_VISIBLE_DEVICES='${MUSA_VISIBLE_DEVICES}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MCCL_LIB='${MCCL_LIB}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PATCH_MUSA_CUSTOM_OPS='${VLLM_PATCH_MUSA_CUSTOM_OPS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_MUSA_GRAPH_COMPAT='${SGLANG_MUSA_GRAPH_COMPAT}'" \
    trainer.device=musa \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.device=musa \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=True \
    "$@"

```

The script passes the platform settings, `verl-musa-patch`, SGLang options, and
device environment to Ray workers through `runtime_env`. Shell exports alone
are not sufficient for Ray workers.



## 3. Compare Results

Compare `critic/rewards/mean` with the [NVIDIA reference run](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart).

The baseline should:

1. Complete all epochs without a crash or hang.
2. Show an upward reward trend within the first 20 steps.
3. Avoid a flat or collapsing reward curve during the first 100 steps.

## 4. Quick Verification

```bash
python3 -c 'import torch; print(torch.musa.is_available(), torch.musa.device_count())'
```

The output should show that MUSA is available and report the visible device
count. The logs should also contain `[VERL_MUSA_SITE]` bootstrap messages.


## Multi-Node Setup

Start Ray on the head node and workers, then set `NNODES` and run the baseline:

```bash
# Head node
ray start --head --port=6379
export RAY_ADDRESS='auto'

# Worker nodes
ray start --address='<head-ip>:6379'

NNODES=2 bash scripts/baseline_grpo_gsm8k.sh
```

MUSA uses Ray's built-in `GPU` resource. Do not configure a custom `musa`
resource.

## Next Steps

- See [Installation Guide](./install_guidance.md) for image and dependency setup.
- See [development.md — Acceptance Baseline](../../development.md#acceptance-baseline-for-new-hardware-adaptation) for the PR checklist.
