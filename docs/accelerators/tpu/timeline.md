# RL Timeline & HBM Profiler for TPU Runs

When `VERL_TPU_TIMELINE=1` is set, the TPU trainer engine, `tpu-sync` weight sync, and vLLM rollout server print finished spans and 1 Hz TPU HBM usage (`tpu-info`) to stdout. Ray forwards these lines into the job log, and `make_report` turns the log into a **single self-contained, offline-sharable HTML file** (`report.html`) with two tabs at the same URL:

* **Timeline (`report.html#tab=timeline`)**:
  * **Trainer loop** (driver): `step`, `gen` (waiting for rollout), `old_log_prob`, `ref`, `adv`, `update_actor`, and `update_weights` (`sync.*` sub-phases).
  * **Trainer ranks & HBM**: What trainer rank 0 executes on its TPU chips (`update_actor` -> `fwd_bwd` + `optimizer_step`, `compute_ref_log_prob`, `update_weights` -> `gather` / `bind` / `d2h`, `release_tpu_sync`) plus a 1 Hz per-chip HBM lane (`GiB` / `GB`).
  * **Rollout replica N & HBM**: State (`generating`, draining behind closed submission gate, `paused` for weight transfer, `idle`), served weight version (`vK`), in-flight/queued requests, weight-sync RPCs, one bar per request, and a 1 Hz per-chip HBM lane.
* **HBM Memory (`report.html#tab=memory`)**:
  * **Summary Cards**: Trainer peak HBM, idle floor, `compute_ref_log_prob` max, `update_actor` peak, `update_weights` max, and Rollout peak/floor (`GiB` and `GB`).
  * **Per-Chip SVG Charts**: (1) Startup & initial build-up (`t = 0` to Step 3) with RL phase shading, (2) Zoomed / steady-state window synced with the top bar's step selector, (3) Full-run Trainer per-chip HBM, and (4) Full-run Rollout per-chip HBM.
  * **Per-Step HBM Table**: Step-by-step HBM breakdown (`GiB` and `GB`); click any row to zoom to that step.

---

## 1. Record

Set `VERL_TPU_TIMELINE=1` in the driver shell or in the Ray job's `runtime_env` (`PlatformTPU` forwards all `VERL_TPU_*` variables to every worker automatically):

| Environment Variable | Default | Description |
|---|---|---|
| `VERL_TPU_TIMELINE` | `0` | Set to `1` to enable timeline spans and 1 Hz HBM sampling. Zero overhead when unset. |
| `VERL_TPU_TIMELINE_RANKS` | `0` | Trainer ranks that emit chip spans (`0,1` or `all`). |
| `VERL_TPU_TIMELINE_HBM_PERIOD` | `1.0` | Local chip HBM polling interval in seconds via `tpu-info` (`0` to disable HBM polling). |

```bash
ray job submit --working-dir . \
  --runtime-env-json '{"env_vars": {"VERL_TPU_TIMELINE": "1",
                                    "VERL_PLATFORM": "tpu",
                                    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin"}}' \
  -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
```

---

## 2. Build & Share the Report

`make_report` extracts the timeline spans, 1 Hz HBM samples, and `timing_s/*` step metrics from the Ray job log and embeds them directly into `report.html`. The output HTML file has **zero external dependencies** and can be shared, emailed, or opened directly from disk (`file://`) in any browser.

```bash
# Directly from a live Ray cluster:
python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report \
  --ray-address http://localhost:8265 --job <submission-id> \
  --save-log job.log -o report.html --summary

# Or from a saved log (optionally filtering to a steady-state step range):
python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report job.log \
  --steps 10-20 -o report.html --summary
```

### Optional: Pre-Init HBM Sidecar

The built-in 1 Hz HBM sampler starts when `TorchTitanTPUEngine.initialize()` and `TPUvLLMHttpServer.launch_server()` complete. To also capture HBM ramp-up from `0 GiB` during initial checkpoint loading before engine initialization finishes, run `timeline.py` as a sidecar on the TPU worker pods and merge the CSVs with `--hbm-csv`:

```bash
python3 -m verl_hardware_plugin.accelerators.tpu.profilers.timeline /tmp/hbm_monitor.csv 1.0
python3 -m verl_hardware_plugin.accelerators.tpu.profilers.make_report job.log \
  --hbm-csv trainer_workers=trainer_hbm.csv --hbm-csv rollout_replica_0=rollout_hbm.csv \
  -o report.html --summary
```
