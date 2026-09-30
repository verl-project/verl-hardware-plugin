# Quick Start

Training example for running verl on Intel GPU (Arc / Arc Pro) with this
plugin installed (see [`install_guidance.md`](install_guidance.md) first).

Current software and hardware scope:

- Runtime mode: **Colocate** (FSDP actor + vLLM rollout on the same device).
- Inference engine: **vLLM** validated. SGLang weight sync — Work in
  Progress on Intel GPU.
- Trainer backend: **FSDP**, **FSDP2**. Megatron — Work in Progress.
- Algorithms: GRPO, PPO, SFT — validated on GSM8K / Qwen2.5.
- Hardware targets:
  - Intel Arc Pro B60 (Battlemage, 24 GB) — validated, 1-GPU and 2-GPU
  - Intel Arc Pro B70 (2x GPU, 32 GB each)
  - Multi-node — Work in Progress

## 1. Verify Platform Resolution

```bash
python3 -c '
from verl.plugin.platform import get_platform
p = get_platform()
print("device:  ", p.device_name)
print("vendor:  ", p.vendor_name)
print("backend: ", p.communication_backend_name())
print("ray res: ", p.ray_resource_name())
print("ipc:     ", p.is_ipc_supported())
'
```

Expected output:

```text
device:   xpu
vendor:   intel
backend:  xccl
ray res:  GPU
ipc:      False
```

## 2. Prepare Data

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

## 3. Launch Training

```bash
export ZE_AFFINITY_MASK=0,1            # select physical device indices
unset ONEAPI_DEVICE_SELECTOR           # must NOT be set

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    +ray_kwargs.ray_init.num_gpus=1
```

Expected (Qwen2.5-0.5B-Instruct, GSM8K, batch 16, 1 step, 1x Arc Pro B60):

```text
timing_s/step: ~51   timing_per_token_ms/gen: ~0.35
perf/throughput: ~148 tok/s
```

On 2 GPUs with the same batch size (throughput scales with larger batch —
this keeps the comparison apples-to-apples rather than showing peak
throughput):

```text
timing_s/step: ~93   perf/throughput: ~41 tok/s
```

## 4. Run the Plugin Test Suite

```bash
pytest tests/test_plugin_registration.py -k xpu -v
```

Expected: all XPU cases pass. These run on any host, with or without an
Intel GPU attached.

## Feature Support Matrix

| Category | Status | Notes |
|---|---|---|
| Runtime mode | Colocate | FSDP/FSDP2 actor + vLLM rollout on same GPU(s) |
| Inference engine | vLLM validated | SGLang: `update_weights` — Work in Progress |
| Trainer backend | FSDP, FSDP2 | Megatron — Work in Progress |
| Algorithms | GRPO, PPO, SFT | Validated on GSM8K / Qwen2.5 |
| Hardware (1-GPU) | Validated | Arc Pro B60: 51.2 s/step, 148.2 tok/s (batch 16, Qwen2.5-0.5B) |
| Hardware (2-GPU) | Validated | Arc Pro B60 x2: 93.0 s/step at same batch size |
| Multi-node | Work in Progress | — |
| Profiler | Intel VTune (ITT) — Work in Progress | Pending upstream verl-core merge; see [profiling.md](./profiling.md) |

## Next Steps

See [`docker/intel_gpu/README.md`](../../docker/intel_gpu/README.md) for a
containerized environment with all of the above preconfigured.
