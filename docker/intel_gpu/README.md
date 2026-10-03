# verl-hardware-plugin on Intel GPU

## Supported Hardware

- Intel Arc Pro B-Series (Battlemage)

## What This Image Is

This builds verl core (cloned at a pinned ref) with this plugin
(`verl_hardware_plugin`) installed on top of it, from this repo as the build
context. Unlike the CUDA/ROCm/Ascend images that live in `verl-core`'s own
`docker/` directory, this one is intentionally on the other side of the
plugin boundary: it starts from the plugin and pulls verl in, not the other
way round.

The plugin is auto-discovered by verl via the `verl.plugins` setuptools
entry_points group declared in this repo's `pyproject.toml` — no
`VERL_USE_EXTERNAL_MODULES` env var needed. See `verl/plugin/platform/README.md`
in verl-core for the discovery mechanism.

## Quick Start

### Build the Docker image

```bash
# Standard build
docker build -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

# Behind a corporate proxy
docker build \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t verl-intel-gpu:latest -f docker/intel_gpu/Dockerfile.intel_gpu .

```

### Run with GPU access

```bash
# Find render group GID on host
RENDER_GID=$(getent group render | cut -d: -f3)
docker run --rm --device /dev/dri --group-add ${RENDER_GID} \
  -v /dev/dri/by-path:/dev/dri/by-path:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host --shm-size=16g \
  verl-intel-gpu:latest

```

## Example run case

Setup dataset:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k" >/tmp/gsm8k.log 2>&1
```

### Case 1: KL-only baseline (no reward model)

Small GRPO/FSDP2 smoke test with a separate reference policy and
`algorithm.kl_ctrl`/`use_kl_loss` for the KL penalty — no reward model
involved. `NUM_GPUS`, `MODEL_ID`/`MODEL_PATH`, and `DATA_DIR` are overridable
via env vars (defaults shown below); any extra Hydra overrides can be
appended after the script:

```bash
NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
DATA_DIR=${DATA_DIR:-$HOME/data}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/gsm8k/train.parquet \
    data.val_files=$DATA_DIR/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_fsdp2_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_fsdp2' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} "$@"
```

### Case 2: colocated reward model

Same GRPO setup, but with a real discriminative reward model
(`Skywork/Skywork-Reward-V2-Llama-3.2-1B`) colocated on the same GPUs as
training/rollout (`reward.reward_model.enable_resource_pool=False`), the v1
async trainer (`trainer.use_v1=True`, `trainer.v1.trainer_mode=colocate_async`),
`transfer_queue.enable=True`, async rollout, and dynamic batching. This
exercises the `init_colocated()`/`init_standalone()` device_name path on XPU
that case 1 never reaches. `NUM_GPUS`, `MODEL_ID`/`MODEL_PATH`, and
`RM_MODEL_ID`/`RM_MODEL_PATH` are overridable via env vars:

```bash
NUM_GPUS=${NUM_GPUS:-2}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
RM_MODEL_ID=${RM_MODEL_ID:-Skywork/Skywork-Reward-V2-Llama-3.2-1B}
RM_MODEL_PATH=${RM_MODEL_PATH:-${RM_MODEL_ID}}

adv_estimator=grpo
n_resp_per_prompt=4
num_reward_workers=${NUM_REWARD_WORKERS:-4}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${train_prompt_bsz}}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-128}
# Reward-model rollout must fit the full chat (prompt + response + RM template overhead).
rm_prompt_length=$(( max_prompt_length + max_response_length + 512 ))

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=1 \
    transfer_queue.enable=True \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward.num_workers=${num_reward_workers} \
    reward.reward_manager.name=dapo \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.model_path="${RM_MODEL_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.reward_model.rollout.prompt_length=${rm_prompt_length} \
    reward.reward_model.rollout.response_length=${max_response_length} \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_colocate_rm_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_colocate_rm' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} "$@"
```

### Case 3: HSDP (hybrid sharding)

Same KL-only setup as case 1, but shards across 4 GPUs in HSDP groups of 2
(`actor_rollout_ref.actor.fsdp_config.fsdp_size=2` and
`actor_rollout_ref.ref.fsdp_config.fsdp_size=2`) instead of full FSDP —
exercises the hybrid-sharding device-mesh path on XPU with `NUM_GPUS=4`:

```bash
NUM_GPUS=${NUM_GPUS:-4}
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${MODEL_ID}}
DATA_DIR=${DATA_DIR:-$HOME/data}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/gsm8k/train.parquet \
    data.val_files=$DATA_DIR/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_intel_gpu_grpo_hsdp_e2e' \
    trainer.experiment_name='qwen2_5_05b_intel_gpu_grpo_hsdp' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    +ray_kwargs.ray_init.num_gpus=${NUM_GPUS} "$@"
```

## Dependencies

| Package | Version | Source |
|---------|---------|--------|
| PyTorch XPU | from vLLM xpu requirements | `https://download.pytorch.org/whl/xpu` |
| oneCCL runtime | installed in image via oneAPI / oneCCL bundle | bundled in image |
| vLLM | from Dockerfile `VLLM_VERSION` | prebuilt XPU wheel from `wheels.vllm.ai` |
| verl core | from Dockerfile `VERL_GIT_REPO`/`VERL_REF` | git-cloned, `pip install --no-deps -e .` |
| verl deps | from `requirements-intel-gpu.txt` | PyPI and extra indexes |
| verl-hardware-plugin (this repo) | local build context | `pip install --no-deps -e .` |

Runtime sanity checks validated on this image:

- `torch.xpu.is_available() == True`
- `from vllm.platforms import current_platform` reports `xpu`
- oneCCL runtime is available via `CCL_ROOT` and `libccl.so.1`
- `python3 -c "from verl.plugin.platform import get_platform; print(get_platform().device_name)"` reports `xpu`

## Backend Policy

- Default rollout backend on Intel GPU is vLLM.
- sglang is not the default path for Intel GPU in this image.
- Separate image/profile will be released when SGLang is validated on Intel GPU.
