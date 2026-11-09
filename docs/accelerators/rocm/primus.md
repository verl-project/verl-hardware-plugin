# Primus training backends on AMD ROCm

## Introduction

This guide covers the two training backends `verl-hardware-plugin` builds on AMD GPUs
with [Primus](https://github.com/AMD-AGI/Primus), each configured by a Primus experiment YAML given on
the command line.

| Backend | verl strategy | Engine | Selected by |
|---------|---------------|--------|-------------|
| Primus Megatron | `megatron` (vendor `amd`) | `PrimusMegatronEngineWithLMHead` | `framework: megatron` module, `model_engine=primus` |
| Primus MaxText (JAX) | `maxtext` | `MaxTextEngineWithLMHead` | `framework: maxtext` module, `model_engine=maxtext` |

verl's built-in `PlatformROCm` already handles AMD GPUs, so the plugin registers no platform. Both
backends use verl's vLLM rollout and NCCL checkpoint engine unchanged. No verl source changes are
needed.

## Directory Structure

```text
verl_hardware_plugin/accelerators/rocm/
├── registration.py          # Registers the ${primus:...} resolver and, on ROCm torch, both engines
├── launch.py                # python -m ...rocm.launch <primus.yaml> [verl overrides]
├── hydra_searchpath.py      # Appends config/ to verl's Hydra search path (registered by registration.py)
├── config/                  # Hydra groups: model_engine=primus / maxtext
│   ├── model_engine/        # primus.yaml, maxtext.yaml
│   ├── actor/ ref/ critic/  # primus_*.yaml extend verl's megatron_*; maxtext_*.yaml extend actor/ref/critic
│   └── engine/maxtext.yaml  # MaxTextEngineConfig defaults
└── primus/
    ├── yaml_config.py       # Loads a Primus module (base config + model + overrides) as a flat dict
    ├── resolver.py          # ${primus:<key>,<default>[,<engine node>]}: Primus YAML values inside verl configs
    ├── megatron/            # verl's Megatron engine plus Primus patches and Primus YAML parallelism
    │   ├── config.py        # PrimusMcoreEngineConfig, YAML -> engine config, capability checks
    │   ├── engine.py        # Registered as backend=megatron, vendor=amd; plain Megatron without a YAML
    │   ├── patches.py       # Runs Primus' patch phases (kernels by default) around model build
    │   └── megatron_compat.py
    └── maxtext/             # MaxText train state driven step by step, one JAX process per GPU
        ├── config.py        # MaxTextEngineConfig / MaxTextActorConfig
        ├── params.py        # Primus YAML -> MaxText keys, JAX environment, jax.distributed start-up
        ├── runtime.py       # MaxTextSession: train step with the clipped policy loss, log-prob step
        ├── engine.py        # verl BaseEngine; full batch on every rank, rank 0 returns outputs
        ├── batch.py loss.py # Packing verl batches; clipped policy loss (verl's vanilla PPO loss)
        ├── hf_import.py     # HF Qwen3-MoE -> scanned MaxText layout, each process reads its own shards
        ├── export.py        # MaxText -> HF tensors for the rollout (on-GPU, DLPack) and checkpoints
        └── checkpoint.py    # HF safetensors checkpoints
```

## Installation

Two images, one per trainer. Both build vLLM from source against a ROCm torch.

```bash
cd scripts/accelerators/rocm/docker
docker build -f Dockerfile.primus  -t verl-rocm-primus:v26.9  .   # FROM rocm/primus:v26.9
docker build -f Dockerfile.maxtext -t verl-rocm-maxtext:v26.7 .   # FROM rocm/jax-training:maxtext-v26.7
```

The MaxText image installs the ROCm 10 torch wheels with gfx942/gfx950 kernels next to JAX; torch
in the same process carries the weight sync to vLLM.

In the container, install verl and the plugin editable (the plugin loads through verl's
`verl.plugins` entry point in every Ray worker) and point `PRIMUS_PATH` at a Primus checkout:

```bash
pip install --no-deps -e /path/to/verl -e /path/to/verl-hardware-plugin
export PRIMUS_PATH=/path/to/Primus
# Primus Megatron image only: Primus' Megatron-LM is used from the checkout.
export PYTHONPATH=$PRIMUS_PATH:$PRIMUS_PATH/third_party/Megatron-LM:$PYTHONPATH
```

## Usage

The launcher reads the module's `framework` and turns the YAML into verl overrides
(`model_engine`, the engine's `primus_config_path`, and `actor_rollout_ref.model.path` from `hf_path`);
other arguments go to `verl.trainer.main_ppo` unchanged:

```bash
python -m verl_hardware_plugin.accelerators.rocm.launch my_experiment.yaml [verl overrides ...]
```

The same works without the launcher:

```bash
python -m verl.trainer.main_ppo model_engine=primus \
  actor_rollout_ref.actor.megatron.primus_config_path=my_experiment.yaml ...
```

The YAML is the single source of truth for trainer parallelism and the optimizer. The Hydra groups
fill verl's fields (TP/PP/CP/EP, dtype, lr, ...) from it through `${primus:...}`, so verl's
batch checks and token budgets see the real values, and the Megatron engine rejects a verl override
that contradicts the YAML. Change values in the YAML or with
`actor_rollout_ref.actor.<megatron|maxtext>.primus_overrides='[key=value,...]'`.

### Fully async GRPO on Qwen3-30B-A3B

`scripts/accelerators/rocm/run_qwen3_30b_a3b_grpo_async.sh` runs separate-async GRPO with one
8-GPU trainer node and one 8-GPU vLLM node (run it on the head of a 2-node Ray cluster):

```bash
# Primus Megatron: CP=2, EP=4, packed THD sequences, dynamic batch size.
PRIMUS_CONFIG=scripts/accelerators/rocm/configs/qwen3_30b_a3b_megatron_grpo.yaml \
MODEL_PATH=/models/Qwen3-30B-A3B TRAIN_FILES=dapo/train.parquet \
bash scripts/accelerators/rocm/run_qwen3_30b_a3b_grpo_async.sh

# Primus MaxText: EP=8 over the node; 8 prompts x 8 samples fill the compiled 64-sequence batch.
PRIMUS_CONFIG=scripts/accelerators/rocm/configs/qwen3_30b_a3b_maxtext_grpo.yaml \
MODEL_PATH=/models/Qwen3-30B-A3B TRAIN_FILES=dapo/train.parquet \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 \
bash scripts/accelerators/rocm/run_qwen3_30b_a3b_grpo_async.sh
```

## Backend notes

### Primus Megatron

- verl's Megatron engine runs with Primus' patches (`primus_patches: kernels | none | all | <ids>`)
  applied in Primus' phases around model construction. Without `primus_config_path` the engine is
  plain verl Megatron.
- Context parallelism and dynamic batch size are verl's: set `context_parallel_size` in the YAML and
  `actor.use_dynamic_bsz=true` with `ppo_max_token_len_per_gpu`. `use_packed_attention` in the YAML
  turns on THD packing (`use_remove_padding`).
- Primus-Turbo attention is rejected (it does not take packed THD sequences); Turbo GEMM flags are
  accepted only at TP=1.

### Primus MaxText (JAX)

- verl starts one worker per GPU; the workers form one multi-process JAX job (`hardware:
  gpu_multiprocess`). The coordinator address is shared through verl's torch.distributed group.
- Every worker receives the whole batch and runs the same step; data, FSDP and expert parallelism
  live in the MaxText mesh (`ici_*`/`dcn_*` in the YAML). The optimizer and schedule are MaxText's.
- Each step trains on a fixed `(num_devices * per_device_batch_size * gradient_accumulation_steps,
  max_target_length)` batch; the actor mini-batch (`ppo_mini_batch_size * rollout.n`) must fit it.
- Weights reach the rollout as bf16 HF tensors produced on each GPU (DLPack, no host copy);
  checkpoints are fp32 HF safetensors written by rank 0 (optimizer state is not saved).
- Use `trainer.v1.separate_async.parameter_sync_step=1`: the engine keeps one policy copy.
- Qwen3-MoE in MaxText's scanned layout is the supported model; there is no value head (use GRPO).

## Tests

```bash
pytest tests/accelerators/rocm
```

CPU tests cover the YAML resolver and launcher, Hydra composition with verl's configs, the Megatron
config checks, the MaxText loss against verl's vanilla PPO loss, log-prob alignment, and the HF <->
MaxText layout conversion.
