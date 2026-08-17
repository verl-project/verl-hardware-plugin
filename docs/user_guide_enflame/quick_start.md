# Enflame GCU Quick Start


This guide walks you through running the **acceptance baseline** GRPO job on Enflame GCU. Complete the [Installation Guide](./install_guidance.md) first.

**Baseline scenario:** Qwen3-0.6B + GSM8K + FSDP actor + vLLM rollout — same as [`scripts/baseline_grpo_gsm8k.sh`](../../scripts/baseline_grpo_gsm8k.sh).

**Reference curve (NVIDIA 8×GPU):** [SwanLab — verl_grpo_gsm8k_math](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

## 1. Prepare Data and Model

```bash
cd /workspace

# Model
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B

# Dataset
mkdir -p gsm8k && cd gsm8k
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"
```

## 2. Run the Acceptance Baseline

From the repository root:

```bash
bash scripts/baseline_grpo_gsm8k_enflame.sh
```

The Enflame wrapper sets platform env vars and Ray `runtime_env` overrides, then delegates to the standard baseline script. Default: 8 GCUs (`TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`).

Adjust for your cluster:

```bash
DATA_DIR=/workspace/gsm8k \
MODEL_DIR=/workspace/Qwen3-0.6B \
NGPUS_PER_NODE=4 \
TOPS_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/baseline_grpo_gsm8k_enflame.sh
```

Logs are written to `verl_demo.log`. Metrics are uploaded to SwanLab when configured (`trainer.logger='["console","swanlab"]'` in the baseline script).

## 3. Compare Results

After training starts, open SwanLab and compare **`critic/rewards/mean`** for the first 100 steps against the [NVIDIA reference run](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart).

Pass criteria:

- Upward trend in `critic/rewards/mean` within the first ~20 steps.
- No flat or collapsing reward in the first 100 steps.
- Full run completes all 15 epochs without error.

## What the Enflame Wrapper Adds

[`scripts/baseline_grpo_gsm8k_enflame.sh`](../../scripts/baseline_grpo_gsm8k_enflame.sh) exports Enflame-specific settings before calling the shared baseline:

| Category | Variables / overrides |
|----------|---------------------|
| Migration | `ENFLAME_ENABLE_AUTO_MIGRATION=1` |
| Platform | `VERL_PLATFORM=enflame`, `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin` |
| Devices | `TOPS_VISIBLE_DEVICES`, `RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=1` |
| Communication | `USE_FLAGCX=0` (ECCL) |
| Rollout stability | `enforce_eager=True`, `TORCHDYNAMO_DISABLE=1`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, etc. |
| Ray workers | All critical vars mirrored in `ray_init.runtime_env.env_vars` |

Shell exports alone are **not** enough — the wrapper passes them through Hydra so Ray workers inherit the same environment.

## Expected Success Indicators

```
Registered platform: enflame (gcu)
verl platform initialised: gcu
Registered engines: fsdp_enflame
verl-hardware-plugin loaded successfully
[validate_config] All configuration checks passed successfully!
```

Step-level `critic/rewards/mean` metrics should appear in console and SwanLab.

## Key Configuration Points

| Parameter | Description | Baseline default |
|-----------|-------------|------------------|
| `DATA_DIR` | GSM8K parquet directory | `/workspace/gsm8k` |
| `MODEL_DIR` | Qwen3-0.6B weights | `/workspace/Qwen3-0.6B` |
| `NGPUS_PER_NODE` | GCUs for training | `8` |
| `TOPS_VISIBLE_DEVICES` | Visible GCU indices | `0,1,2,3,4,5,6,7` |
| `PROJECT_NAME` | SwanLab project | `verl_grpo_gsm8k_math` |
| `trainer.device` | Platform selector | `enflame` (via wrapper) |

Baseline hyperparameters (batch size, learning rate, rollout settings) match [`scripts/baseline_grpo_gsm8k.sh`](../../scripts/baseline_grpo_gsm8k.sh) — do not change them when submitting an adaptation PR unless documenting a deliberate deviation.

## Multi-Node Setup

On the head node:

```bash
ray start --head --port=6379
export RAY_ADDRESS='auto'
NNODES=2 bash scripts/baseline_grpo_gsm8k_enflame.sh
```

On each worker node:

```bash
ray start --address='<head-ip>:6379'
```

## Next Steps

- See [FAQ](./faq.md) for troubleshooting.
- See [development.md — Acceptance Baseline](../development.md#acceptance-baseline-for-new-hardware-adaptation) for the PR submission checklist.
