# Google TPU Weight Sync

Two checkpoint engines send the trainer's updated weights to the [vLLM rollout](./rollout.md) on TPU.
Pick one with `actor_rollout_ref.rollout.checkpoint_engine.backend`:

| Backend | How the weights travel |
|---------|------------------------|
| `tpu` | Trainer rank 0 gathers the full model and puts it in the Ray object store. Every rollout worker then pulls it and cuts out its own shard. |
| `raiden` | The trainer chips send straight to the rollout chips over the network with [tpu-sync](https://github.com/google/tpu-sync) (Raiden). Each rollout chip receives only its own tensor-parallel shard. |

`raiden` is faster: the weights skip the round trip through the object store, and no rollout worker
receives the whole model.

## How a `raiden` sync works

The driver runs each sync (`update_raiden_weights`, which replaces `CheckpointEngineManager.update_weights`
for this backend):

1. It pauses generation. The initial sync at step 0 skips this.
2. Every trainer rank binds the full weights, which verl gathers on every rank, to a new tpu-sync
   `WeightSynchronizer`, registers them with the Raiden controller and copies them to host memory. The
   driver starts that controller on the first sync. The trainer also publishes the full shape of every
   tensor in the `RayWeightRegistry` actor.
3. On the first sync only, every rollout worker allocates tensor-parallel receive buffers of those shapes
   and registers them.
4. The controller moves the weights from the trainer hosts to the rollout hosts.
5. The trainer frees its synchronizer and send buffers. Meanwhile, each rollout worker copies the weights to
   HBM and fuses q/k/v and gate/up into vLLM's parameters.
6. Generation resumes, tagged with the new weight version.

## Requirements

- `tpu-sync-torch` installed on every trainer and rollout host, built for the installed `torch-tpu`.
  The tpu-sync torch extension is ABI-locked to the `torch-tpu` build it was compiled against. A
  mismatched pair crashes in the first transfer. The backend is tested with `torch-tpu`
  `0.1.1+release.2026.9.22.20260927223741` and `tpu-sync-torch` `0.0.1.dev20261001132139`.
- Network connectivity between the hosts:
  - from all trainer and rollout hosts to the Raiden controller, which runs in the driver process;
  - from the trainer hosts to the rollout hosts. Rollout worker `k` listens on port `12000 + k`.

## Configuration

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=raiden
# Optional, see the table below:
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.parallelism=8
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity=True
```

| Option (`engine_kwargs.raiden.*`) | Default | Description |
|-----------------------------------|---------|-------------|
| `parallelism` | `8` | Parallel transfer streams per worker. |
| `verify_parity` | `False` | After every sync, compare weight norms between trainer and rollout and log the result. Adds a pass over all weights on both sides, so it is meant for bring-up. |

## Metrics

Each sync returns its phase timings. Trainers that record weight-sync metrics, such as the separate-async
trainer, log them with the step metrics:

| Metric | Phase |
|--------|-------|
| `timing_s/tpu-sync/quiesce` | Pausing generation |
| `timing_s/tpu-sync/trainer_init` | Trainer ranks gathering their weights, creating and registering a synchronizer, and copying the weights to host memory |
| `timing_s/tpu-sync/sampler_init` | Rollout workers allocating and registering their receive buffers |
| `timing_s/tpu-sync/barrier` | Waiting until every worker has registered |
| `timing_s/tpu-sync/p2p_transfer` | The transfer |
| `timing_s/tpu-sync/sampler_h2d` | Rollout workers copying the weights to HBM and into vLLM's parameters |
| `timing_s/tpu-sync/sampler_h2d_pure` | The host-to-device copy alone, on the slowest rollout worker |
| `timing_s/tpu-sync/total_sync` | The whole sync after pausing generation |

`sampler_h2d_pure` and the `verify_parity` check depend on the rollout workers' return values, which
verl's vLLM server does not yet pass back from `collective_rpc`. Until it does, the sync still works, but
`sampler_h2d_pure` is not reported and the parity check logs a warning instead of a result.

## Verify

On any host, without tpu-sync, vLLM or a TPU:

```bash
pytest tests/accelerators/tpu/test_raiden_checkpoint_engine.py -v
```

On TPU, set `VERL_LOGGING_LEVEL=INFO` and `SAGEMAKER_CONTAINER_LOG_LEVEL=INFO` in the job's `env_vars`, as
in the [vLLM Rollout](./rollout.md) example. Every sync then logs its phase times. For example, from a
Qwen3-0.6B run with one v6e-8 slice for training and one for rollout:

```text
[RAIDEN TELEMETRY | Orchestrator] Step 1 Completed in 1.9739s:
  * Sampler Quiesce/Pause  : 0.0338s
  * Trainer Raiden Init    : 1.0341s
  * Sampler Raiden Init    : 0.0063s
  * Raiden Barrier Check   : 0.0002s
  * RaidenController P2P   : 0.1317s
  * Sampler H2D DMA        : 0.0318s
  * Total End-to-End Sync  : 1.9739s
```

The first sync (step 0) is slower, because the rollout workers allocate and register their receive
buffers then. With `verify_parity=True`, every sync after the first also logs

```text
[RAIDEN PARITY VERIFIED | Step 1] 100% DISTRIBUTED NORM PARITY CONFIRMED!
```

Both lines come from the driver process. There, vLLM's serving code imports
`model_hosting_container_standards`, which sets Python's root logger to `SAGEMAKER_CONTAINER_LOG_LEVEL`
(default `ERROR`). Without the variable, the driver drops the plugin's `INFO` and `WARNING` messages.
Errors, such as a parity mismatch, are always logged.

## Related Documentation

- [User Guide](./README.md)
- [vLLM Rollout](./rollout.md)
- For developers: `verl_hardware_plugin/accelerators/tpu/engines/raiden_checkpoint_engine.py` (trainer and driver side)
  and `verl_hardware_plugin/accelerators/tpu/rollout/tpu_raiden.py` (rollout side).
