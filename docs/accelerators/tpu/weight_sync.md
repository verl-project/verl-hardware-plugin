# Google TPU Weight Sync

Two checkpoint engines send the trainer's updated weights to the [vLLM rollout](./rollout.md) on TPU. They live
in `verl_hardware_plugin/accelerators/tpu/weight_sync/`, the plugin's counterpart of verl core's
`verl/checkpoint_engine/`, and register with verl's `CheckpointEngineRegistry`.
Pick one with `actor_rollout_ref.rollout.checkpoint_engine.backend`:

| Backend | How the weights travel |
|---------|------------------------|
| `ray-plasma-store` | Trainer rank 0 gathers the full model and puts it in the Ray object store. Every rollout worker then pulls it and cuts out its own shard. |
| `tpu-sync` | The trainer chips send straight to the rollout chips over the network with [tpu-sync](https://github.com/google/tpu-sync). Each rollout chip receives only its own tensor-parallel shard. |

`tpu-sync` is faster: the weights skip the round trip through the object store, and no rollout worker
receives the whole model.

The former name of the `ray-plasma-store` backend, `tpu`, is still accepted and prints a deprecation warning
(`FutureWarning`).

## How a `tpu-sync` sync works

The driver runs each sync (`update_tpu_sync_weights`, which replaces `CheckpointEngineManager.update_weights`
for this backend):

1. It pauses generation. The initial sync at step 0 skips this.
2. Every trainer rank binds its weights to its tpu-sync `WeightSynchronizer`
   (`engines/weight_export.py` decides how each weight is exported):
   - an FSDP shard (`Shard(0)` on a 1-D mesh, dim 0 divisible by the mesh size) is registered as this rank's
     block of the full tensor; tpu-sync reshards the blocks of all ranks for the rollout, so no rank gathers it;
   - a replicated weight is already full on every rank and is registered full, without further communication.
     Note that q/k/v_proj are replicated only because torchtitan's `FusedQKVLinear` `state_dict()` hook already
     all-gathers the FSDP-sharded `wqkv` before splitting it (TODO: split it locally);
   - every other layout (uneven cuts, FSDP x TP or HSDP 2-D meshes, fused expert stacks) is all-gathered and
     registered full. Each such tensor is logged at INFO, with a per-sync summary line.

   On the first sync it creates that synchronizer and registers it with the tpu-sync controller, which the
   driver starts at the same time; later syncs rebind the new tensors to the existing synchronizer, as long as
   the tensor names, shapes, dtypes and shard layout do not change. The trainer also publishes the full shape
   of every tensor in the job's `TpuSyncRendezvous` actor (`weight_sync/tpu_sync_rendezvous.py`). The driver
   creates that actor once per job, named after the Ray job and not detached, so jobs sharing a cluster never
   see each other's.
3. On the first sync only, every rollout worker allocates tensor-parallel receive buffers of those shapes
   and registers them.
4. The controller moves the weights from the trainer chips, through the trainer hosts, to the rollout hosts.
   Its planner decides per tensor, from the source and destination slices together, whether the copy can
   skip the CPU (de)tiling pass.
5. The trainer unbinds its send buffers, returning their HBM to training, and keeps the synchronizer (its
   pinned host buffers and controller registration) for the next sync. Meanwhile, each rollout worker copies
   the weights to HBM and fuses q/k/v and gate/up into vLLM's parameters.
6. Generation resumes, tagged with the new weight version.

## Requirements

- `tpu-sync-torch` installed on every trainer and rollout host, built for the installed `torch-tpu`.
  The tpu-sync torch extension is ABI-locked to the `torch-tpu` build it was compiled against. A
  mismatched pair crashes in the first transfer. The backend is tested with `torch-tpu`
  `0.1.2.dev20261006000518` and `tpu-sync-torch` `0.0.1.dev20261006025601` (the versions in the
  `Dockerfile` of verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo)).
  - `WeightSynchronizer.unbind_weights()` needs `tpu-sync-torch` `0.0.1.dev20261006025601` or newer. With
    an older build (e.g. `0.0.1.dev20261001132139`) the trainer logs a warning once and falls back to
    destroying and re-creating its synchronizer every sync, which adds a few seconds per sync for an 8B
    model.
- Network connectivity between the hosts:
  - from all trainer and rollout hosts to the tpu-sync controller, which runs in the driver process;
  - from the trainer hosts to the rollout hosts. Every trainer and rollout worker listens on a port the
    OS assigns and registers it with the controller, so no fixed port range needs to be open.

## Configuration

```bash
actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync
# Optional, see the table below:
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.parallelism=8
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.verify_parity=all
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.release_buffers_after_sync=True
```

| Option (`engine_kwargs.tpu-sync.*`) | Default | Description |
|-------------------------------------|---------|-------------|
| `parallelism` | `8` | Parallel transfer streams per worker. |
| `verify_parity` | `off` | Weight-sync check: `off`, `norm`, `exact` or `all` (`True` means `norm`). See [Parity checks](#parity-checks). A failed check stops the run. Meant for bring-up and after changing sharding or upgrading tpu-sync. |
| `release_buffers_after_sync` | `True` | Unbind the send buffers (a bf16 copy of the full model on every trainer chip) after every transfer, so training gets that HBM back between syncs. Set to `False` to keep them resident. The deprecated environment variable `VERL_TPU_SYNC_RELEASE_BUFFERS` still overrides the option on the trainer workers and logs a deprecation warning. |

## Parity checks

`verify_parity` selects two complementary checks:

| Value | Check | When it runs | What it catches |
|-------|-------|--------------|-----------------|
| `norm` (`True`) | Trainer ranks post per-tensor L1/L2 norms; the driver compares them with every rollout replica's | Every sync | Lost or scaled weights (L1 within `1e-3`). Blind to misplaced bytes: a mis-sliced or permuted tensor keeps its norms. |
| `exact` | Every rollout worker compares each received tensor bit for bit with the vLLM parameter it overwrites | Step-0 sync only, when vLLM still holds the checkpoint the trainer loaded | Any wrong byte: shard slices, tiling, fusion, transposes |
| `all` | Both | | |

Run `exact` after changing the trainer or rollout sharding, or after upgrading tpu-sync / torch-tpu. It costs one
extra compare per tensor, once. The driver logs one line per replica:

```text
[TPU-SYNC PARITY EXACT | Step 0] 8 rollout ranks, [227] tensors checked per rank, 0 mismatched, 0 unresolved: every received tensor matches the checkpoint bit for bit
```

followed, on a mismatch, by one line per mismatched tensor and rank. It covers every tensor installed directly into
vLLM's parameters. Qwen3.5's stacked experts are not compared. Models installed through vLLM's weight loaders cannot
be checked bit for bit (the loaders re-shard the tensors themselves); the check logs `not run` and fails, so use
`verify_parity=norm` for them.

An enabled check that fails raises `TPUSyncParityError` on the driver, which stops the run. It fails on a mismatch
(including `unresolved` tensors in the `exact` check), and also when it cannot collect what it needs: trainer ranks
that did not post their norms, rollout workers that returned no stats or no `exact` result, or an `exact` check
that could not run (`not run`). The checks run before generation resumes, so the rollout never generates with
weights that failed a check.

## Several rollout replicas

verl creates one rollout replica per `tensor_model_parallel_size` chips of the rollout pool. With
`tensor_model_parallel_size` equal to the number of rollout chips there is one replica (the recipes' default);
with the chips of one host, one replica per rollout host. `tpu-sync` syncs any number of them: every replica
registers under its own job name (`sampler<replica_rank>`), all of them receive in one transfer (tpu-sync
shards it within each replica and broadcasts across the replicas), and the parity check (`verify_parity`)
compares each replica with the trainer on its own, naming the replica in its result line:

```text
[TPU-SYNC PARITY VERIFIED | Step 3 | replica 0] 100% DISTRIBUTED NORM PARITY CONFIRMED!
[TPU-SYNC PARITY MISMATCH | Step 3 | replica 1] Norms do NOT match!
```

Every replica tested so far sits on one host; replicas that span several hosts are not covered.

## Metrics

Each sync returns its phase timings. Trainers that record weight-sync metrics, such as the separate-async
trainer, log them with the step metrics:

| Metric | Phase |
|--------|-------|
| `timing_s/tpu-sync/quiesce` | Pausing generation |
| `timing_s/tpu-sync/trainer_init` | Trainer ranks gathering their weights, and creating and registering a synchronizer (first sync) or rebinding the existing one |
| `timing_s/tpu-sync/sampler_init` | Rollout workers allocating and registering their receive buffers |
| `timing_s/tpu-sync/barrier` | Waiting until every worker has registered |
| `timing_s/tpu-sync/p2p_transfer` | The transfer |
| `timing_s/tpu-sync/sampler_h2d` | Rollout workers copying the weights to HBM and into vLLM's parameters |
| `timing_s/tpu-sync/sampler_h2d_pure` | The host-to-device copy alone, on the slowest rollout worker |
| `timing_s/tpu-sync/total_sync` | The whole sync after pausing generation |

`sampler_h2d_pure` and the `verify_parity` checks read the rollout workers' return values, which verl's vLLM server
passes back from `collective_rpc` (since verl-project/verl#8155). With an older verl, the sync still works, but
`sampler_h2d_pure` is not reported and an enabled parity check fails for lack of results.

## Verify

On any host, without tpu-sync, vLLM or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_sync_checkpoint_engine.py -v
```

On TPU, set `VERL_LOGGING_LEVEL=INFO` and `SAGEMAKER_CONTAINER_LOG_LEVEL=INFO` in the job's `env_vars` (see
[vLLM Rollout](./rollout.md#run)). Every sync then logs its phase times. For example, from a
Qwen3-0.6B run with one v6e-8 slice for training and one for rollout:

```text
[TPU-SYNC TELEMETRY | Orchestrator] Step 1 Completed in 1.9739s:
  * Sampler Quiesce/Pause  : 0.0338s
  * Trainer Init           : 1.0341s
  * Sampler Init           : 0.0063s
  * Barrier Check          : 0.0002s
  * Controller P2P         : 0.1317s
  * Sampler H2D DMA        : 0.0318s
  * Total End-to-End Sync  : 1.9739s
```

The first sync (step 0) is slower, because the rollout workers allocate and register their receive
buffers then. With `verify_parity=norm` (or `True`), every sync after the first also logs

```text
[TPU-SYNC PARITY VERIFIED | Step 1] 100% DISTRIBUTED NORM PARITY CONFIRMED!
```

Both lines come from the driver process. There, vLLM's serving code imports
`model_hosting_container_standards`, which sets Python's root logger to `SAGEMAKER_CONTAINER_LOG_LEVEL`
(default `ERROR`). Without the variable, the driver drops the plugin's `INFO` and `WARNING` messages.
Errors are always logged, and a failed parity check raises `TPUSyncParityError` with its result lines.

## Related Documentation

- [User Guide](./README.md)
- [vLLM Rollout](./rollout.md)
- For developers: `verl_hardware_plugin/accelerators/tpu/weight_sync/tpu_sync_checkpoint_engine.py` (trainer and driver side)
  and `verl_hardware_plugin/accelerators/tpu/rollout/tpu_sync_worker_extension.py` (rollout side).
