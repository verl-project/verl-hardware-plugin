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

1. Every trainer rank binds its weights to its tpu-sync `WeightSynchronizer`
   (`engines/weight_export.py` decides how each weight is exported):
   - an FSDP shard (`Shard(0)` on a 1-D mesh, dim 0 divisible by the mesh size) is registered as this rank's
     block of the full tensor; tpu-sync reshards the blocks of all ranks for the rollout, so no rank gathers it;
   - a replicated weight is already full on every rank and is registered full, without further communication.
     Note that q/k/v_proj are replicated only because torchtitan's `FusedQKVLinear` `state_dict()` hook already
     all-gathers the FSDP-sharded `wqkv` before splitting it (TODO: split it locally);
   - routed experts with `1 < expert_parallel_size < world size` (so `efsdp > 1`) are
     `(_StridedShard(0, split_factor=ep), Shard(0))` on the `(efsdp, ep)` mesh: every rank still holds contiguous
     whole experts, block `i_ep * efsdp + i_efsdp` of `efsdp * ep`, and is registered locally like an FSDP shard;
   - Qwen3 MoE expert stacks (torchtitan's `[E, I, H]` / `[E, H, I]` per projection) are sent as is, one 3-D
     tensor per projection and layer (`model.layers.N.mlp.experts.gate_proj.weight` `[E, I, H]`, likewise
     `up_proj`, and `down_proj` `[E, H, I]`, as Qwen3.5's expert pieces). An FSDP shard of whole experts is a dim-0
     block of that tensor and is registered locally; any other expert layout (e.g. torchtitan's dim-1 expert
     sharding once the FSDP degree exceeds the expert count) is all-gathered first. Other MoE models' expert stacks
     are all-gathered and sent as one tensor per expert;
   - every other layout (uneven cuts, FSDP x TP or HSDP 2-D meshes) is all-gathered and registered full. Each
     such tensor is logged at INFO, with a per-sync summary line.

   On the first sync it creates that synchronizer and registers it with the tpu-sync controller, which the
   driver starts at the same time; later syncs rebind the new tensors to the existing synchronizer, as long as
   the tensor names, shapes, dtypes and shard layout do not change. The trainer also publishes the full shape
   of every tensor in the job's `TpuSyncRendezvous` actor (`weight_sync/tpu_sync_rendezvous.py`). The driver
   creates that actor once per job, named after the Ray job and not detached, so jobs sharing a cluster never
   see each other's.
2. On the first sync only, every rollout worker allocates tensor-parallel receive buffers of those shapes
   and registers them.
3. The controller moves the weights from the trainer chips, through the trainer hosts, to the rollout hosts.
   Its planner decides per tensor, from the source and destination slices together, whether the copy can
   skip the CPU (de)tiling pass.
4. It pauses generation (the initial sync at step 0 skips this). Until then the rollout keeps generating: steps
   1-3 do not write vLLM's parameters, as the transfer only writes the rollout hosts' memory.
5. The trainer unbinds its send buffers, returning their HBM to training, and keeps the synchronizer (its
   pinned host buffers and controller registration) for the next sync. Meanwhile, each rollout worker copies
   the weights to HBM and fuses q/k/v and gate/up into vLLM's parameters (see [Rollout install](#rollout-install)).
6. Generation resumes, tagged with the new weight version. Generations that finished before the pause used the
   previous weights and keep their tag.

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
| `tie_word_embeddings` | unset | Whether `lm_head` is tied to the input embedding, so it need not be sent. Set, the trainer does not send `lm_head.weight`, and every rollout worker copies its embedding into its LM head after the install. Each sync checks that copy: `lm_head` must have been copied from the embedding, or share its tensor, and be bitwise equal to it (with `verify_parity=exact` or `all`, the step-0 sync also requires the checkpoint's `lm_head` to equal the embedding); otherwise the sync fails before generation resumes. The check needs a direct [install mode](#rollout-install): with `vllm loaders`, a sync with the option set fails. With a direct install, the driver logs one `[TPU-SYNC TIED LM_HEAD ...]` line per rollout worker and sync, also with the option unset. Unset sends `lm_head`, which is correct for tied and untied models alike. |
| `return_after_d2h` | `False` | Return to the trainer as soon as its ranks have copied their weights to host memory and freed the device buffers, and finish the sync (transfer, pause, install, resume) in the background while training goes on. The next sync waits for it first and raises its error, if any. Applies from the second sync on (tpu-sync keeps the weights bound across the first transfer to a rollout topology) and needs a `tpu-sync-torch` whose `WeightSynchronizer.d2h()` returns a future; otherwise the sync runs to the end as before. See [Returning after the D2H](#returning-after-the-d2h). |

## Rollout install

Each rollout worker picks an install mode on the first sync, logged as `install mode ...`:

| Mode | Models | How |
|------|--------|-----|
| `direct` | Dense models (Qwen3, Llama) | Each rank receives its TP shard and copies it into vLLM's (fused, possibly transposed) parameters. |
| `qwen3_5 direct` | Qwen3.5 | As `direct`, plus the GatedDeltaNet and expert layouts of `rollout/qwen3_5_layout.py` (experts sliced on their intermediate dimension). |
| `qwen3_moe direct` | Qwen3 MoE (e.g. Qwen3-30B-A3B), with vLLM expert parallelism over the whole TP group (`enable_expert_parallel`, EP size = TP size) or without it | Each rank receives one block of each 3-D expert stack: with EP its own whole experts (dim 0), without EP its slice of every expert's intermediate dimension (as Qwen3.5); it writes them into vLLM's `w13_weight` / `w2_weight`. The router is received whole (`rollout/qwen3_moe_layout.py`). |
| `vllm loaders` | Any other hybrid or MoE model or layout (e.g. quantized experts, EP that does not cover the TP group) | Every rank all-gathers every tensor and runs vLLM's own checkpoint loaders. Correct, but slow; the log says why no direct install was possible. |

The model-specific direct modes are the entries of `rollout/direct_install.py`, tried in order; the first whose
checks pass on the live vLLM model is used. To add one, write `rollout/<model>_layout.py` with a
`DirectInstallLayout` and add it to that table; the worker extension stays unchanged. The expert writers for vLLM's
fused-MoE weights are shared in `rollout/moe_weights.py`.

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
vLLM's parameters, including each written region of the experts' `w13_weight` (`[gate]` / `[up]`) and `w2_weight`
for Qwen3.5 and Qwen3 MoE. Models installed through vLLM's weight loaders cannot be checked bit for bit (the loaders
re-shard the tensors themselves); the check logs `not run` and fails, so use `verify_parity=norm` for them.

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
TPUSyncParityError: [TPU-SYNC PARITY MISMATCH | Step 3 | replica 1] Norms do NOT match!
```

Every replica tested so far sits on one host; replicas that span several hosts are not covered.

## Metrics

Each sync returns its phase timings. Trainers that record weight-sync metrics, such as the separate-async
trainer, log them with the step metrics:

| Metric | Phase |
|--------|-------|
| `timing_s/tpu-sync/quiesce` | Pausing generation, right after the transfer |
| `timing_s/tpu-sync/trainer_init` | Trainer ranks gathering their weights, and creating and registering a synchronizer (first sync) or rebinding the existing one |
| `timing_s/tpu-sync/sampler_init` | Rollout workers allocating and registering their receive buffers |
| `timing_s/tpu-sync/barrier` | Waiting until every worker has registered |
| `timing_s/tpu-sync/p2p_transfer` | The transfer |
| `timing_s/tpu-sync/sampler_h2d` | Rollout workers copying the weights to HBM and into vLLM's parameters |
| `timing_s/tpu-sync/sampler_h2d_pure` | The host-to-device copy alone, on the slowest rollout worker |
| `timing_s/tpu-sync/sampler_resume` | Resuming generation, after the parity checks (if enabled) |
| `timing_s/tpu-sync/rollout_paused` | From the pause (abort) to the resume: the quiesce, the install, the parity checks and the resume |
| `timing_s/tpu-sync/trainer_release` | Waiting for the trainer to free its send buffers; generation has already resumed |
| `timing_s/tpu-sync/total_sync` | The whole sync |
| `timing_s/tpu-sync/trainer_d2h` | With `return_after_d2h`: the trainer ranks' own device-to-host copy, on the slowest rank |
| `timing_s/tpu-sync/wait_previous_sync` | With `return_after_d2h`: waiting for the background part of the previous sync |
| `tpu-sync/returned_after_d2h` | With `return_after_d2h`: 1 when the sync returned after the trainer's D2H |

`sampler_h2d_pure` and the `verify_parity` checks read the rollout workers' return values, which verl's vLLM server
passes back from `collective_rpc` (since verl-project/verl#8155). With an older verl, the sync still works, but
`sampler_h2d_pure` is not reported and an enabled parity check fails for lack of results.

Generation resumes as soon as the rollout holds the new weights and has passed the parity checks; it does not wait
for the trainer to free its send buffers, which only gives trainer HBM back before the next training step.

The rollout keeps generating until the transfer is done, so `rollout_paused` covers only the quiesce, the install, the
parity checks and the resume. The trainer waits longer instead: the quiesce no longer overlaps `trainer_init`, and the
transfer shares the rollout hosts with generation, which makes its time vary more. On 2 × v6e-8 (Qwen3-8B, trainer
FSDP8, rollout TP8, `verify_parity=norm`, steps 2–10), the rollout was paused for 0.73 s per sync instead of the whole
sync (2.1 s), and `update_weights` took 2.71 s instead of 2.11 s. The transfer took 0.94–1.01 s with the rollout
paused and 0.89–1.83 s with it generating; the receiving hosts show TCP receive-queue pruning and drops in both cases.

## Returning after the D2H

With `return_after_d2h`, a sync returns to the trainer after `trainer_init` (which then includes the ranks' own D2H,
reported as `trainer_d2h`), and the transfer reads the trainer's host staging buffers (`skip_d2h`). The other phases run
in the background and are reported by the next sync, together with `wait_previous_sync`: the step metrics of a sync
show the background phases of the sync before it.

The plugin routes verl's `CheckpointEngineManager.wait_for_weight_update` to the pending background sync, so a trainer
that calls it before validation and before its training loop exits sees the new weights installed. With a verl whose
trainer does not call it yet, a validation right after a sync can run on the previous weights, and the background part
of the last sync may not finish before the process exits; keep `return_after_d2h` off there.

## Verify

On any host, without tpu-sync, vLLM or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_sync_checkpoint_engine.py -v
```

On TPU, set `VERL_LOGGING_LEVEL=INFO` and `SAGEMAKER_CONTAINER_LOG_LEVEL=INFO` in the job's `env_vars` (see
[vLLM Rollout](./rollout.md#run)). Every sync then logs its phase times. For example, from a
Qwen3-8B run with one v6e-8 slice for training (FSDP8) and one for rollout (one TP=8 replica, `verify_parity=norm`):

```text
[TPU-SYNC TELEMETRY | Orchestrator] Step 4 Completed in 2.6860s:
  * Trainer Init           : 0.5383s
  * Sampler Init           : 0.0529s
  * Barrier Check          : 0.0002s
  * Controller P2P         : 1.3000s
  * Sampler Quiesce/Pause  : 0.1809s (after the transfer)
  * Sampler H2D DMA        : 0.0935s
  * Parity Check           : 0.5100s
  * Sampler Resume         : 0.0068s
  * Rollout Paused         : 0.7929s
  * Trainer Buffer Release : 0.0004s (rollout already generating)
  * Total End-to-End Sync  : 2.6860s
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
