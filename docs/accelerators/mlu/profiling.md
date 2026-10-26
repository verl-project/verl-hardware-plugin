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

The plugin registers its profiler patches (torch and cnprec) when `verl_hardware_plugin` is imported: `accelerators/mlu/registration.py` applies them at import time via `register_all_profiles()`, so no extra call is needed in the training script.

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

## Cnprec Profile

The plugin patches verl `DistProfiler` so `global_profiler.tool=cnprec` dispatches to cnprec's `VerlTrainingAnalyzer` without modifying upstream verl. It reuses existing `@DistProfiler.annotate` stages (`actor_update`, `actor_compute_log_prob`, `ref_compute_log_prob`, `critic_update`, `compute_values`, `compute_rm_score`) and lazily attaches cnprec's module/optimizer hooks on the first profiled step, so non-profiled steps pay no overhead.

This is the **online dump** path for verl train-side precision collection. The dumped artifacts (`module_level/...`, `data/...`, `weight_snapshots/...`) are later consumed by cnprec's offline comparison `analyze_verl_offline`.

### Setup

Install `cnprec` in the Ray runtime environment, and prepare an `analyze_verl_config.json` (see cnprec's bundled template `cnprec.configs.analyze_verl_config.json`). The plugin also honors the `CNPREC_VERL_ANALYZE=true` and `CNPREC_VERL_CONFIG_PATH=...` environment variables.

### PPO example

```bash
python -m verl.trainer.main_ppo \
  ... \
  global_profiler.tool=cnprec \
  global_profiler.steps=[2] \
  global_profiler.save_path=outputs/profile \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=True \
  +actor_rollout_ref.actor.profiler.tool_config.cnprec.config_path=/path/to/analyze_verl_config.json \
  +actor_rollout_ref.actor.profiler.tool_config.cnprec.stages=[actor_update,actor_compute_log_prob] \
  +actor_rollout_ref.actor.profiler.tool_config.cnprec.strict=False
```

The upstream profiler yaml does not define a `cnprec` block, so Hydra overrides must use the `+` prefix to append these keys.

Cnprec options:

| Option | Default | Description |
| --- | --- | --- |
| `config_path` | `null` | Path to cnprec `analyze_verl_config.json`. Required (or set env `CNPREC_VERL_CONFIG_PATH`). |
| `stages` | `null` | Subset of annotate stages to collect (`actor_update`, `actor_compute_log_prob`, `ref_compute_log_prob`, `critic_update`, `compute_values`, `compute_rm_score`); `null` = all known. |
| `strict` | `False` | If `True`, fail when cnprec is unavailable or config_path is missing; otherwise run without profiling. |

### Notes

- Rank selection follows verl's native gating: `global_profiler.ranks` / `all_ranks` decide which ranks enter the profiled range (cnprec no longer forces every rank). cnprec's own `analyze_verl_config.json` `ranks` further filters which ranks actually dump data. Hooks may touch process-global state, so prefer `all_ranks=True` when in doubt.
- One `VerlTrainingAnalyzer` is built lazily per (role, block) and reused across steps; `stop()` advances it one step (writes per-step artifacts under the `output_dir` from `analyze_verl_config.json`, not `global_profiler.save_path`).
- For rollout generation (`rollout.mode=async`), the inference-side collection uses cnprec's `VllmAnalyzer` separately; keep trainer-side stages on cnprec and drive rollout profiling as needed.
- `global_profiler.save_path` is currently informational for cnprec (artifacts go to `analyze_verl_config.json`'s `output_dir`); it is kept for config compatibility.

## Profiling Steps and Scope

Use `global_profiler.steps` to select training steps. Example: `global_profiler.steps=[2,3]`.

Set `global_profiler.profile_continuous_steps=True` to combine continuous steps into one profiling window. Keep it `False` to generate one output window per selected step.

Enable profiling only on the worker roles you need. For example, use `actor_rollout_ref.actor.profiler.enable=True` for actor paths, and enable `actor_rollout_ref.ref.profiler.enable=True` or `critic.profiler.enable=True` only when those roles are part of the job.

## Troubleshooting

- If `mlu` is rejected in `contents`, confirm `VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"` is set in Ray `runtime_env.yaml`.
- If no files are generated, check that `global_profiler.steps` (PPO) or `trainer.profile_interval` (SFT) matches the actual `global_steps` printed by training.
- On the SFT path, profiling overrides go under `profiler.*` with `trainer.profile_interval`, not `global_profiler.*`/`actor_rollout_ref.*`. PPO-style overrides are silently ignored by the SFT trainer.
- Profiling can add significant overhead; start with one step and rank 0 for torch profile, then expand scope only when needed.