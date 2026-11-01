# Cambricon MLU Profiling Guide

This guide describes the default profiling support adapted in `verl-hardware-plugin` for Cambricon MLU. The default path reuses verl community `global_profiler.tool=torch` and adds MLU activity support.

## Prerequisites

Install both `verl` and `verl-hardware-plugin` in editable mode, then enable the plugin in Ray runtime:

```yaml
working_dir: ./
excludes: ["/.git/"]
env_vars:
  VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
```

The plugin registers the torch profiler patch when `verl_hardware_plugin` is imported.

## Community Torch Profile

The plugin patches verl's torch profiler config so `contents` accepts `mlu`, and uses `torch.profiler.ProfilerActivity.MLU` when available.

Example Hydra overrides:

```bash
python -m verl.trainer.main_ppo \
  ... \
  global_profiler.tool=torch \
  global_profiler.steps=[2] \
  global_profiler.save_path=outputs/profile \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=False \
  actor_rollout_ref.actor.profiler.ranks=[0] \
  actor_rollout_ref.actor.profiler.tool_config.torch.contents=[mlu,cpu,memory,shapes,stack] \
  actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True
```

Common `contents` values:

| Value | Description |
| --- | --- |
| `mlu` | Collect MLU device activities. |
| `cpu` | Collect CPU activities. |
| `memory` | Enable memory profiling. |
| `shapes` | Record operator input shapes. |
| `stack` | Record Python stack traces. |

Outputs are written as Chrome trace files under `global_profiler.save_path`, for example:

```text
outputs/profile/e2e/prof_rank-0_<pid>_<timestamp>.json.gz
```

Open the trace with Chrome `chrome://tracing` or any compatible trace viewer.

## SFT Profiling

The examples above target PPO, which profiles through `global_profiler.*` and per-role `actor_rollout_ref.*` configs. The SFT trainer (`verl.trainer.sft_trainer_ray`) uses a **different config surface**, so the PPO overrides do not apply:

- SFT reads the top-level `profiler.*` block (default group `profiler@profiler: profiler`), not `global_profiler.*` or `actor_rollout_ref.*`. Overrides such as `global_profiler.tool=...` are silently ignored on the SFT path.
- The profiling step window is selected with `trainer.profile_interval=[start,end]` (inclusive), not `global_profiler.steps`.
- SFT has a single annotated stage, `train_batch` (`@DistProfiler.annotate(role="train_batch")` in `engine_workers.py`). Other stages listed in the PPO examples — `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update` — do not exist in the SFT path, so only `train_batch/` is produced.

Example SFT torch profile overrides:

```bash
python -m verl.trainer.sft_trainer_ray \
  ... \
  trainer.profile_interval=[2,2] \
  profiler.enable=True \
  profiler.all_ranks=False \
  profiler.ranks=[0] \
  profiler.save_path=outputs/profile \
  profiler.tool=torch \
  profiler.tool_config.torch.contents=[mlu,cpu,memory,shapes,stack] \
  profiler.tool_config.torch.discrete=False
```

With `all_ranks=False` and `ranks=[0]`, only rank 0 emits a trace, so a single file is produced (e.g. `outputs/profile/train_batch/language-model_train-batch_rank0-of-8_*.json.gz`). Set `all_ranks=True` if you need per-rank traces for all cards.

## Profiling Steps and Scope

Use `global_profiler.steps` to select training steps. Example: `global_profiler.steps=[2,3]`.

Set `global_profiler.profile_continuous_steps=True` to combine continuous steps into one profiling window. Keep it `False` to generate one output window per selected step.

Enable profiling only on the worker roles you need. For example, use `actor_rollout_ref.actor.profiler.enable=True` for actor paths, and enable `actor_rollout_ref.ref.profiler.enable=True` or `critic.profiler.enable=True` only when those roles are part of the job.

## Troubleshooting

- If `mlu` is rejected in `contents`, confirm `VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"` is set in Ray `runtime_env.yaml`.
- If no files are generated, check that `global_profiler.steps` (PPO) or `trainer.profile_interval` (SFT) matches the actual `global_steps` printed by training.
- On the SFT path, profiling overrides go under `profiler.*` with `trainer.profile_interval`, not `global_profiler.*`/`actor_rollout_ref.*`. PPO-style overrides are silently ignored by the SFT trainer.
- Profiling can add significant overhead; start with one step and rank 0 for torch profile, then expand scope only when needed.