# VERL Google TPU User Guide

## Introduction

This document describes Google TPU support in `verl-hardware-plugin`.

The plugin registers the TPU platform (`PlatformTPU`: device metadata, Ray resource configuration,
and the PJRT worker environment), the TorchTitan TPU training engine (`TorchTitanTPUEngineWithLMHead`
under `verl_hardware_plugin/accelerators/tpu/engines/torchtitan_tpu.py`), a vLLM rollout for TPU, and two checkpoint
engines that send the trainer's weights to it (`ray-plasma-store` and `tpu-sync`).

## Directory Structure

Each subpackage extends one part of verl core:

| Plugin subpackage | verl core counterpart | What it holds |
|-------------------|-----------------------|---------------|
| `engines/` | `verl/workers/engine/` (`BaseEngine`, `EngineRegistry`, `model_engine=torchtitan`) | The training engine: model build, forward/backward, optimizer step, weight export. Algorithm-agnostic; the GRPO/PPO loop is verl's `verl/trainer/`. |
| `losses/` | `verl/workers/utils/losses.py` (`sft_loss`, `ppo_loss`) | Glue that hands the engine's TPU outputs to verl's unmodified losses; the only TPU code that knows the SFT/PPO loss inputs |
| `rollout/` | `verl/workers/rollout/` (`RolloutReplicaRegistry`) | The sampler: the vLLM replica and server on TPU |
| `weight_sync/` | `verl/checkpoint_engine/` (`CheckpointEngineRegistry`, `rollout.checkpoint_engine.backend`) | The checkpoint engines that send the trainer's weights to the rollout |
| `profilers/` | `verl/utils/profiler/` | Opt-in `VERL_TPU_TIMELINE` spans and 1 Hz TPU HBM memory sampling for the RL timeline report |
| `patches/` | — | Runtime patches of torch / verl that TPU needs |

```text
verl_hardware_plugin/accelerators/tpu/
├── __init__.py                       # Registers the TPU platform
├── registration.py                   # Registers the engine, the weight sync engines and the rollout hook
├── platform_tpu.py                   # PlatformTPU: device queries, Ray resources, PJRT mesh environment
├── device_api.py                     # torch.accelerator / torch.tpu device functions
├── options.py                        # Every TPU-only option: engine, tpu-sync, platform; see Configuration
├── engines/                          # verl core: verl/workers/engine/
│   ├── torchtitan_tpu.py             # TorchTitan TPU training engine (FSDP2 / SimpleFSDP, SPMD)
│   ├── tpu_utils.py                  # Attention, SimpleFSDP, compile and sequence-bucketing helpers; option resolvers
│   └── weight_export.py              # Exports the engine's weights as local FSDP shards or full tensors
├── losses/                           # verl core: verl/workers/utils/losses.py
│   └── loss_inputs.py                # Static-shape inputs for verl's unmodified sft_loss / ppo_loss
├── rollout/                          # verl core: verl/workers/rollout/
│   ├── tpu_vllm.py                   # vLLM rollout replica and server for TPU
│   ├── tpu_vllm_patches.py           # vLLM / vllm-torchtpu patches for multi-host TPU
│   └── tpu_sync_worker_extension.py  # vLLM worker extension that receives `tpu-sync` weight syncs
├── weight_sync/                      # verl core: verl/checkpoint_engine/
│   ├── weight_sync_backends.py       # Weight sync backend names, shared helpers and manager dispatch hook
│   ├── ray_checkpoint_engine.py      # `ray-plasma-store` weight sync: through the Ray object store
│   ├── tpu_sync_checkpoint_engine.py # `tpu-sync` weight sync: peer to peer to the vLLM rollout
│   ├── tpu_sync_rendezvous.py        # Per-job actor: controller address, tensor shapes, parity stats
│   └── tpu_sync_parity.py            # Optional `tpu-sync` parity checks (norms every sync, bit-exact at step 0)
├── profilers/                        # verl core: verl/utils/profiler/
│   ├── timeline.py                   # Opt-in `VERL_TPU_TIMELINE` span prints, 1 Hz HBM sampling, and CSV sidecar CLI
│   ├── tpu_vllm_timeline.py          # vLLM rollout server timeline mixin (requests, drain, pause, RPCs)
│   ├── make_report.py                # Builds a self-contained HTML report (Timeline + HBM Memory tabs) from a job log
│   └── index.html                    # Self-contained report viewer (Timeline + HBM Memory tabs; also opens raw logs)
└── patches/
    └── reduce_avg_allreduce_patch.py # all_reduce workaround for TPU tensors on Gloo-only groups
```

```text
tests/accelerators/tpu/
├── test_tpu_device_api.py            # device_api.py
├── test_tpu_allreduce_patch.py       # patches/reduce_avg_allreduce_patch.py
├── test_tpu_engine.py                # platform_tpu.py and engines/ (torchtitan_tpu, tpu_utils, weight_export)
├── test_tpu_options.py               # options.py (engine_kwargs.tpu, engine_kwargs.tpu-sync, deprecated env vars)
├── test_tpu_loss_inputs.py           # losses/loss_inputs.py
├── test_tpu_vllm_rollout.py          # rollout/tpu_vllm.py
├── test_tpu_vllm_patches.py          # rollout/tpu_vllm_patches.py
├── test_tpu_timeline.py              # profilers/ (timeline, tpu_vllm_timeline, make_report, index.html)
├── test_weight_sync_backends.py      # weight_sync/weight_sync_backends.py
├── test_ray_checkpoint_engine.py     # weight_sync/ray_checkpoint_engine.py
└── test_tpu_sync_checkpoint_engine.py # weight_sync/tpu_sync_*.py and rollout/tpu_sync_worker_extension.py
```

```text
docs/accelerators/tpu/
├── README.md                         # This file
├── install_guidance.md               # Installation and environment setup
├── quick_start.md                    # Selecting and verifying the platform
├── rollout.md                        # Running the vLLM rollout on TPU; TorchTitan engine options (engine_kwargs.tpu)
├── weight_sync.md                    # Choosing and configuring the weight sync (ray-plasma-store / tpu-sync)
└── timeline.md                       # Recording (`VERL_TPU_TIMELINE=1`) and viewing the unified Timeline + HBM report
```

## Getting Started

- [Installation Guide](./install_guidance.md) — prerequisites and environment setup
- [Quick Start](./quick_start.md) — select the TPU platform and verify it resolves
- [vLLM Rollout](./rollout.md) — run GRPO with the vLLM rollout on TPU, and set the TorchTitan engine's TPU options (`engine_kwargs.tpu`)
- [Weight Sync](./weight_sync.md) — send the trainer's weights to the rollout peer to peer with `tpu-sync`
- [RL Timeline & HBM Profiler](./timeline.md) — see what the trainer and rollout chips do and their 1 Hz HBM usage in every step (`VERL_TPU_TIMELINE=1`)
- [GRPO example](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo) — end-to-end GRPO recipes in verl (`Dockerfile`, KubeRay `RayCluster`, Qwen3-0.6B / Qwen3-4B scripts)

## Configuration

Every TPU option goes in the verl config, in the per-backend `engine_kwargs` dict of the component it
configures, the same channel verl uses for its own backends. All TPU-only options (the ones verl's GPU config has no
field for) are defined and parsed in one module, `verl_hardware_plugin/accelerators/tpu/options.py`: unknown keys and
values of the wrong type are an error, and each deprecated channel logs a one-time warning that names its
replacement. verl's `TorchtitanEngineConfig` has no `engine_kwargs` field yet; the plugin adds it.

| Component | Where the options go | Read by | Options |
|-----------|----------------------|---------|---------|
| Engine (TorchTitan on TPU) | `actor_rollout_ref.{actor,ref}.torchtitan.engine_kwargs.tpu.*` | `options.py`: `TPUEngineOptions` | `use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`, `tpu_eager_mode`, `optim_impl`, `seq_bucket_size`, `context_parallel_strategy`; see [vLLM Rollout](./rollout.md) |
| Rollout (vLLM on TPU) | `actor_rollout_ref.rollout.engine_kwargs.vllm.*` | verl core, passed to vLLM as is | vLLM engine arguments. The TPU replica forces `enable_sleep_mode=False` and keeps upstream's `distributed_executor_backend=mp`; there are no TPU-only rollout options. |
| Weight sync | `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.<backend>.*` | verl core passes them to the checkpoint engine's constructor; `options.py`: `TPUSyncOptions` | `tpu-sync`: `parallelism`, `verify_parity`, `release_buffers_after_sync`, `tie_word_embeddings`; `ray-plasma-store`: none. See [Weight Sync](./weight_sync.md#configuration) |

Process-wide settings that are not part of any verl config stay environment variables, set in the shell that
runs the recipe; the platform forwards them to every worker: libtpu / XLA flags (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`),
`VERL_TPU_CHIPS_PER_HOST` (`options.tpu_chips_per_host_override`; the platform resolves the topology before any
verl config exists), and `VERL_TPU_TIMELINE` ([RL Timeline & HBM Profiler](./timeline.md)).

Deprecated channels, still read for compatibility (each logs a one-time warning; an environment variable still
overrides the config, as before):

| Deprecated | Replacement |
|------------|-------------|
| `torchtitan.<option>` (`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`, `tpu_eager_mode`) | `torchtitan.engine_kwargs.tpu.<option>` |
| `VERL_TPU_USE_SPLASH_ATTENTION`, `VERL_TPU_USE_SIMPLE_FSDP`, `VERL_TPU_USE_FUSED_LINEAR_CROSS_ENTROPY`, `VERL_TPU_EAGER_MODE`, `VERL_TPU_OPTIM_IMPL`, `VERL_TPU_SEQ_BUCKET_SIZE` | `torchtitan.engine_kwargs.tpu.<option>` |
| `VERL_TPU_SYNC_RELEASE_BUFFERS` | `checkpoint_engine.engine_kwargs.tpu-sync.release_buffers_after_sync` |
| `checkpoint_engine.backend=tpu` | `ray-plasma-store` |

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `tpu` |
| Vendor identifier | `google` |
| Communication backend | `tpu_dist` (registered by `torch_tpu`) |
| Device visibility env var | `CUDA_VISIBLE_DEVICES` (see note below) |
| Ray resource name | `TPU` |
| IPC support | No |
| Colocated worker groups | Not supported — a chip belongs to a single process |

### Why the visibility env var is `CUDA_VISIBLE_DEVICES`

This is intentional. verl assigns `os.environ[<this key>]` when launching vLLM servers. Pointing it
at `TPU_VISIBLE_CHIPS` would overwrite the per-worker chip index that the platform sets through
`get_worker_env_vars()` and reads back through `ray_local_rank_override()`, breaking rank mapping.
The two variables serve different purposes and must not be merged.

## Chip Support

| Generation | HBM per device | Topologies with a built-in mapping |
|------------|----------------|------------------------------------|
| v6e | 32 GB per chip | `v6e-4`, `v6e-8`, `v6e-32`, and by chip count 1 to 256 |
| TPU 7x (`tpu7x`) | 96 GB per TensorCore (192 GB per dual-core chip) | by device count: 8 (one `2x2x1` host), 16, 32, 64, 128 (`4x4x4`) |

On TPU 7x a chip has two TensorCores that `torch_tpu` drives as two devices, one process each: a host is
8 devices, a slice is a 4-D mesh `x,y,z,cores` (`TORCH_TPU_TOPOLOGY`), and libtpu lays the mesh out by process:
when it starts, `torch_tpu` sets `CLOUD_TPU_TASK_ID` to the process's global rank and `TPU_VISIBLE_DEVICES` to
its local rank, i.e. its core. Under `torchrun` and in the rollout, which resolve the mesh inside a TPU pod, a
pool that spans its GKE slice takes the mesh from the slice shape the TPU webhook puts in the pod
(`TPU_HOST_BOUNDS` times `TPU_CHIPS_PER_HOST_BOUNDS`, plus the core axis, as torchtitan's launcher computes it).
A pool that uses part of a slice, and a Ray trainer pool, whose mesh is resolved where its workers are created
(usually on the Ray head, which has no webhook variables), take the whole-host shape for their device count from
`TPU_V7X_TOPOLOGY_MAP` in `platform_tpu.py`.
The generation comes from the `ray.io/tpu-pod-type` node label of the pool's hosts or
from `TPU_ACCELERATOR_TYPE`; set `TORCH_TPU_TOPOLOGY='x,y,z,cores'` to override. Rollout engines on TPU 7x
use one worker per TensorCore, e.g. a TP=8 engine per host; on every generation an engine's mesh is its
own worker count.

### Device memory

`get_torch_device()` (`TPUDeviceModuleProxy`) forwards to `torch.tpu`, except for the calls `torch.tpu`
does not provide:

- The memory statistics (`memory_allocated`, `max_memory_allocated`, `memory_reserved`,
  `max_memory_reserved`, `memory_stats`, `reset_peak_memory_stats`, `mem_get_info`) come from
  `torch.accelerator`. That is where `torch_tpu` reports its PJRT allocator. They read zero until the
  process has started its TPU runtime, and in processes without a TPU. On TPU, `memory_reserved` is
  PJRT's reserved bytes, so unlike on CUDA it is not a superset of `memory_allocated`.
- `empty_cache()` does nothing, because PJRT manages HBM itself. It must not call
  `torch.tpu._clear_cache()`, which evicts every compiled executable.
- HBM capacity: `torch.tpu.get_device_properties().total_memory`.

The plugin never modifies the `torch.tpu` module.

## Related Documentation

- [verl plugin system](../../development.md)
