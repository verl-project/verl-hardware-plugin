# Enflame GCU Quick Start


This guide walks you through running the **acceptance baseline** GRPO job on Enflame GCU. Complete the [Installation Guide](./install_guidance.md) first.

**Baseline scenario:** Qwen3-0.6B + GSM8K + FSDP actor + vLLM rollout — same as [`scripts/baseline_grpo_gsm8k.sh`](../../../scripts/baseline_grpo_gsm8k.sh).

**Reference curve (NVIDIA 8×GPU):** [SwanLab — verl_grpo_gsm8k_math](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

**Supported scope:**

- Trainer: FSDP; Megatron with Megatron-LM-FL on `PYTHONPATH`
- Rollout: vLLM, eager mode or GCUGraph capture
- Weight transfer: colocated (device-side IPC handles) and disaggregated/async
  (ECCL checkpoint engine, or FlagCX when `USE_FLAGCX=1`)
- Acceptance baseline: single-node GRPO, 8 GCUs, Qwen3-0.6B on GSM8K

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
bash scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh
```

The Enflame wrapper sets platform env vars and Ray `runtime_env` overrides, then delegates to the standard baseline script. It pins 8 GCUs (`TOPS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`).

Adjust for your cluster:

```bash
DATA_DIR=/workspace/gsm8k \
MODEL_DIR=/workspace/Qwen3-0.6B \
NGPUS_PER_NODE=4 \
bash scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh
```

`TOPS_VISIBLE_DEVICES` and `PYTHONPATH` are not environment-overridable here: the script sets both explicitly, so edit those lines in the wrapper to run on a different device set or Megatron-LM-FL path.

To use FlagCX, add the variables to the Ray environment yourself — the wrapper does not forward them:

```bash
USE_FLAGCX=1 FLAGCX_PATH=/path/to/flagcx \
bash scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh \
  +ray_kwargs.ray_init.runtime_env.env_vars.USE_FLAGCX="'1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.FLAGCX_PATH="'/path/to/flagcx'"
```

Logs are written to `verl_demo.log`. Metrics are uploaded to SwanLab when configured (`trainer.logger='["console","swanlab"]'` in the baseline script).

### Disaggregated / async layout

The baseline above keeps training and rollout on the same devices. For the
separate actor / rollout layout, where weights move through the ECCL checkpoint
engine, use the async example instead:

```bash
DATA_DIR=/workspace/gsm8k MODEL_DIR=/workspace/Qwen3-0.6B \
bash scripts/accelerators/enflame/example_grpo_gsm8k_enflame_async.sh
```

It splits the visible GCUs evenly between training and rollout (`N_GPUS_TRAINING` /
`N_GPUS_ROLLOUT`, default 4 + 4) and sets
`actor_rollout_ref.rollout.mode=async` with
`actor_rollout_ref.rollout.checkpoint_engine.backend=eccl`. It uses only
verl + torch_gcu + ECCL, with no FL stack variables. Logs go to
`verl_demo_async.log`.

## 3. Compare Results

After training starts, open SwanLab and compare **`critic/rewards/mean`** for the first 100 steps against the [NVIDIA reference run](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart).

Pass criteria:

- Upward trend in `critic/rewards/mean` within the first ~20 steps.
- No flat or collapsing reward in the first 100 steps.
- Full run completes all 15 epochs without error.

## What the Enflame Wrapper Adds

[`scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh`](../../../scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh) exports Enflame-specific settings before calling the shared baseline:

| Category | Variables / overrides |
|----------|---------------------|
| Migration | `ENFLAME_ENABLE_AUTO_MIGRATION=1` |
| Platform | `VERL_PLATFORM=enflame`, `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin` |
| Devices | `TOPS_VISIBLE_DEVICES`, `RAY_EXPERIMENTAL_NOSET_TOPS_VISIBLE_DEVICES=1` |
| Communication | ECCL by default; for FlagCX, pass `USE_FLAGCX=1` and `FLAGCX_PATH` through `ray_init.runtime_env.env_vars` yourself — the wrapper does not forward them |
| Rollout stability | `TORCHDYNAMO_DISABLE=1`, `TORCHGCU_INDUCTOR_ENABLE=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `TORCH_ECCL_AVOID_RECORD_STREAMS=1` |
| Ray workers | Critical vars mirrored in `ray_init.runtime_env.env_vars`, except `USE_FLAGCX`/`FLAGCX_PATH` — add those yourself for FlagCX |

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
| `DATA_DIR` | GSM8K parquet directory | set in the wrapper; override with the env var |
| `MODEL_DIR` | Qwen3-0.6B weights | set in the wrapper; override with the env var |
| `NGPUS_PER_NODE` | GCUs for training | `8` |
| `TOPS_VISIBLE_DEVICES` | Visible GCU indices | pinned in the wrapper; edit the script to change |
| `PROJECT_NAME` | SwanLab project | `verl_grpo_gsm8k_math` |
| `trainer.device` | Platform selector | `enflame` (via wrapper) |

Baseline hyperparameters (batch size, learning rate, rollout settings) match [`scripts/baseline_grpo_gsm8k.sh`](../../../scripts/baseline_grpo_gsm8k.sh) — do not change them when submitting an adaptation PR unless documenting a deliberate deviation.

## Multi-Node Setup

On the head node:

```bash
ray start --head --port=6379
export RAY_ADDRESS='auto'
NNODES=2 bash scripts/accelerators/enflame/baseline_grpo_gsm8k_enflame.sh
```

On each worker node:

```bash
ray start --address='<head-ip>:6379'
```

## Next Steps

- See [FAQ](./faq.md) for troubleshooting.
- See [development.md — Acceptance Baseline](../../development.md#acceptance-baseline-for-new-hardware-adaptation) for the PR submission checklist.
