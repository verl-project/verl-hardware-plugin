# rIluvatar Quick Start

Last updated: 06/25/2026.

This guide walks you through running a GRPO training job with verl on the Iluvatar platform. Make sure you have completed the [Installation Guide](./install_guidance.md) first.

## Running a Training Script

Below is a complete example script for GRPO training with Qwen3-0.6B on 8 Iluvatar GPUs. Save it as `run_qwen3-0.6b_iluv.sh`:

```bash
#!/bin/bash
# GRPO | Qwen3-0.6B | GSM8K | FSDP training | ILUVATAR GPUs

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
DATA_DIR=/mnt/share/user_homes/iluv/flagos-q2/data/gsm8k
MODEL_DIR=/mnt/share/user_homes/iluv/flagos-q2/Qwen/Qwen3-0.6B
#SAVE_FREQ=100
#TEST_FREQ=100
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-2048}

########################### user-adjustable ###########################
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_0.6b_grpo_${INFER_BACKEND}_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################
# Modify these paths to your actual data/model locations
DATA_DIR=${DATA_DIR:-/workspace/gsm8k}
MODEL_DIR=${MODEL_DIR:-/workspace/Qwen3-0.6B}

n_trainer_devices=$NGPUS_PER_NODE

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$DATA_DIR/train.parquet']"
    data.val_files="['$DATA_DIR/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_DIR"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    +actor_rollout_ref.rollout.enable_sleep_mode=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

RAY=(
    +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}
)

HYDRA_FULL_ERROR=1
########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@" \
    2>&1 | tee "verl_demo.log"
```

Launch the training:

```bash
bash run_qwen3-0.6b_iluv.sh
```

Training is running successfully if you see step-level progress output in the logs.

**Training Log**: [verl_grpo_gsm8k_math_iluvatar](https://swanlab.cn/@dannyp/verl_grpo_gsm8k_math/runs/qy00qayu/chart)

## Key Configuration Points

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `DATA_DIR` | Path to parquet dataset files | `/workspace/gsm8k` |
| `MODEL_DIR` | Path to model weights | `/workspace/Qwen3-0.6B` |
| `NGPUS_PER_NODE` | Number of devices per node | `8` |
| `NNODES` | Number of nodes | `1` |
| `INFER_BACKEND` | Inference backend for rollout | `vllm` |
| `rollout_tp` | Tensor parallel size for rollout | `1` |

## Next Steps

- See the [Environment Variables Reference](./env_reference.md) for fine-grained control over operator dispatch and backend selection.

