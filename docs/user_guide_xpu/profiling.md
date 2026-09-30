# Intel VTune Profiling Guide

Last updated: 09/22/2026.

This guide describes Intel VTune (ITT) profiling support in `verl-hardware-plugin`
for Intel XPU. Unlike Cambricon MLU's profiling support (see
[`user_guide_mlu/profiling.md`](../user_guide_mlu/profiling.md)), which reuses
verl's built-in `global_profiler.tool=torch` path, there is no PyTorch-native
profiler activity for Intel XPU equivalent to `torch.profiler.ProfilerActivity.MLU`.
VTune support is wired in purely through the newer `PlatformBase` plugin hooks —
`PlatformXPU.profiler_markers()` and `PlatformXPU.dist_profiler_cls()` — which
verl core discovers without needing to know about ITT or XPU itself.

## Prerequisites

- These hooks (`attention_utils_module`, `profiler_markers`,
  `dist_profiler_cls`) only exist on a verl-core build that
  includes [verl-project/verl#7917](https://github.com/verl-project/verl/pull/7917)
  ("enable intel XPU to Verl with plugin mechanism with extra General API
  abstraction"), currently **open, not yet merged**. Against stock verl-core
  `main`, `PlatformBase` has no such methods to override, so `profiler.tool=vtune`
  silently falls through to `DistProfiler`'s no-op fallback instead of erroring.
- Install both `verl` (from the `#7917` branch/ref) and `verl-hardware-plugin` in
  editable mode, and enable the plugin in Ray runtime:

  ```yaml
  working_dir: ./
  excludes: ["/.git/"]
  env_vars:
    VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
  ```

## Enabling VTune

Selecting the tool is all that is required:

```bash
python -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.actor.profiler.tool=vtune \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=False \
  actor_rollout_ref.actor.profiler.ranks=[0]
```

There is no dedicated `tool_config.vtune` schema entry in verl-core's generated
config (only `nsys`/`npu`/`torch`/`torch_memory`/`precision_debugger` have one),
and `vtune` does not need one: the only field `VtuneProfiler` reads is
`tool_config.discrete`, which has no effect on XPU because
`PlatformXPU.profiler_start`/`profiler_stop` are no-ops. It therefore defaults to
`False` and no `+tool_config.vtune.*` override is needed.

If you want the config to carry an explicit entry anyway, add it with Hydra's `+`
— `VtuneProfiler` reuses `NsightToolConfig`'s shape:

```bash
  +actor_rollout_ref.actor.profiler.tool_config.vtune._target_=verl.utils.profiler.config.NsightToolConfig \
  +actor_rollout_ref.actor.profiler.tool_config.vtune.discrete=False
```

This changes nothing functionally on XPU.

## How this differs from Nsight/torch/NPU profiling

Nsight, torch, and NPU profiling are process-level: `profiler.start()`/`stop()`
tell the backend when to begin and end recording, and each writes a self-contained
trace file under `global_profiler.save_path`. **VTune does not work this way.**
`PlatformXPU.profiler_start()`/`profiler_stop()` are no-ops by design — VTune
attaches externally as a collector and observes `range_push`/`range_pop` events
rather than being started/stopped by the profiled process itself. The real
signal is the ITT range markers (`mark_start_range`/`mark_end_range`, wrapping
each named stage — `compute_log_prob`, `update_actor`, etc. — via
`profiler_markers()`), which are only visible when the training process runs
*under* an actual VTune collector, e.g.:

```bash
vtune -collect hotspots -result-dir ./vtune_results -- \
  python -m verl.trainer.main_ppo ... actor_rollout_ref.actor.profiler.tool=vtune ...
```

No trace file appears under `global_profiler.save_path` for the `vtune` tool —
open `./vtune_results` in the VTune GUI/CLI instead. Because `profiler_start`/
`profiler_stop` are no-ops, `tool_config.discrete` has little practical effect
for `vtune` specifically (unlike `torch`/`nsys`, where it materially changes
what gets recorded) — the ITT ranges are emitted the same way either way.

## Troubleshooting

- If `profiler.tool=vtune` appears to do nothing, confirm the running verl-core
  build actually includes #7917 (see Prerequisites) — against stock `main` this
  is a silent no-op, not an error.
- If no ITT ranges show up in VTune, confirm the process was launched *under* a
  VTune collector (`vtune -collect ... -- python ...`), not run standalone —
  standalone runs execute the same code but nothing is listening for the events.
