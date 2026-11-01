# Intel VTune Profiling Guide

Last updated: 10/09/2026.

This guide describes Intel VTune (ITT) profiling support in `verl-hardware-plugin`
for Intel XPU. There is no PyTorch-native profiler activity for Intel XPU, so
VTune support is wired in two ways instead: the tracing markers come from the
`PlatformXPU.profiler_markers()` plugin hook, which verl core discovers without
needing to know about ITT or XPU itself, while `VtuneProfiler` is registered for
`profiler.tool=vtune` by a plugin-side monkeypatch applied from
`PlatformXPU.__init__` (verl core has no profiler registry).

## Prerequisites

- `profiler_markers()` was merged into verl-core `main` on 2026-10-08. Against
  an older checkout, `PlatformBase` has no such method to override, so ITT
  ranges are not emitted — it falls back to nvtx or the generic no-op markers
  instead of erroring. That same change also lifted the `tool_config`
  allowlist in `engine_workers.py`; without it a `tool_config.vtune` entry is
  dropped. The `VtuneProfiler` registration itself is pure monkeypatch and
  needs no core change either way.
- **A VTune collector must be installed separately.** ITT is notify-only: the
  markers this plugin emits are handed to whatever collector has attached to the
  process, and are discarded when none has. The plugin side needs nothing beyond
  PyTorch's ITT bindings, which Intel's XPU wheels ship
  (`torch.profiler.itt.is_available()` is `True` on `torch 2.13.0+xpu`); the
  collector is the `intel-vtune` oneAPI component, which a plain oneAPI
  compiler/runtime install (ccl, compiler, mkl, mpi, pti, …) does **not**
  include on its own. Capturing a trace therefore needs a host with VTune
  installed, and enough sampling permission for the collection type you pick
  (`hotspots` reads `perf_event_open`, so a container typically needs
  `kernel.perf_event_paranoid <= 2` or `CAP_PERFMON`).
- Install `verl` (`main` includes `profiler_markers()` as of 2026-10-08) and
  `verl-hardware-plugin` in editable mode, and enable the plugin in Ray
  runtime:

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
  build actually includes `profiler_markers()` (merged 2026-10-08; see
  Prerequisites) — on an older checkout this is a silent no-op, not an error.
- If no ITT ranges show up in VTune, confirm the process was launched *under* a
  VTune collector (`vtune -collect ... -- python ...`), not run standalone —
  standalone runs execute the same code but nothing is listening for the events.
- If `vtune: command not found`, the collector isn't installed (see
  Prerequisites). Check with `ls /opt/intel/oneapi` for a `vtune` entry; a
  standalone run is still harmless, it just records nothing. To tell a missing
  collector apart from broken ITT bindings, check
  `python -c "import torch.profiler.itt as i; print(i.is_available())"` — `True`
  means the plugin's markers are live and only the collector is absent.
