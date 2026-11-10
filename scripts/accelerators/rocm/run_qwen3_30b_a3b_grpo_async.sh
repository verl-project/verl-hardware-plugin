#!/usr/bin/env bash
# Fully async GRPO on Qwen3-30B-A3B: one 8-GPU trainer node (Primus) and one 8-GPU vLLM rollout node.
#
#   PRIMUS_CONFIG=configs/qwen3_30b_a3b_megatron_grpo.yaml MODEL_PATH=/models/Qwen3-30B-A3B \
#   TRAIN_FILES=dapo/train.parquet VAL_FILES=dapo/val.parquet \
#   bash run_qwen3_30b_a3b_grpo_async.sh [verl overrides ...]
#
# Run on the head node of a 2-node Ray cluster (RAY_ADDRESS=<head>:<port>). The Primus YAML sets
# the trainer's parallelism and optimizer; the module's framework selects Megatron or MaxText.
# Megatron: packed THD sequences with context parallelism (from the YAML) and dynamic batch size,
# each GPU taking up to PPO_MAX_TOKEN_LEN_PER_GPU tokens per micro-batch.
# MaxText: PRIMUS_CONFIG=configs/qwen3_30b_a3b_maxtext_grpo.yaml TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8;
# the mini-batch (prompts * ROLLOUT_N) must fill the YAML's compiled batch of 64 sequences.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_CONFIG="${PRIMUS_CONFIG:-${HERE}/configs/qwen3_30b_a3b_megatron_grpo.yaml}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

EXTRA=()
if [ -n "${RAY_ADDRESS:-}" ]; then
  EXTRA+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi
# Megatron YAMLs name the model in hf_path; MODEL_PATH overrides it and is required for MaxText.
if [ -n "${MODEL_PATH:-}" ]; then
  EXTRA+=(actor_rollout_ref.model.path="${MODEL_PATH}")
fi

python3 -m verl_hardware_plugin.accelerators.rocm.launch "${PRIMUS_CONFIG}" \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP:-1}" \
  trainer.v1.separate_async.num_warmup_batches=1 \
  actor_rollout_ref.hybrid_engine=false \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-${MAX_TOKENS}}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-4}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS}" \
  data.train_files="${TRAIN_FILES:?set TRAIN_FILES}" \
  data.val_files="${VAL_FILES:-${TRAIN_FILES}}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-16}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.truncation=left \
  data.filter_overlong_prompts=false \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.project_name="${PROJECT_NAME:-verl_rocm_primus}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_grpo_async}" \
  trainer.total_epochs=100 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-100}" \
  trainer.test_freq=0 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.logger="${TRAINER_LOGGER:-[console]}" \
  "${EXTRA[@]}" \
  "$@"
