# Enflame GCU feature support

This page lists the verl features available on Enflame GCU, how to enable them,
and the current limitations. Everything is implemented in
`verl_hardware_plugin`; no changes to verl core are required.

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Platform registration | Supported | `VERL_PLATFORM=enflame`; device `gcu`, vendor `enflame` |
| FSDP training | Supported | Path used by the acceptance baseline |
| Megatron training | Supported with Megatron-LM-FL | Registration is skipped when the package is missing |
| vLLM rollout | Supported | Set the GCU stability env vars (the baseline script does it) |
| Colocated weight update | Supported | Device-side IPC handles; Python SHM is not used |
| Disaggregated / async weight transfer | Supported with the vendor Ray build | ECCL checkpoint engine (`eccl`, alias `nccl`) |
| Profiler | Not supported | Out of scope for this backend |

## Training engines

### FSDP

`fsdp_enflame` registers the FSDP strategy for `(device="gcu", vendor="enflame")`
and is the path used by the acceptance baseline. TransformerEngine-FL must be
installed in every Ray worker; see [install_guidance.md](./install_guidance.md).

### Megatron

`megatron_enflame` is registered when Megatron-LM-FL is installed and on
`PYTHONPATH` for the driver and every Ray worker. On a node without it the
registration is skipped and the FSDP path is unaffected.

## vLLM rollout

`actor_rollout_ref.rollout.name=vllm` is supported, in eager mode and with
execution captured by GCUGraph. The GCU stack needs a small set of stability
variables:

| Setting | Value |
|---------|-------|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` |
| `TORCHDYNAMO_DISABLE` | `1` |
| `TORCHGCU_INDUCTOR_ENABLE` | `0` |
| `TORCH_ECCL_AVOID_RECORD_STREAMS` | `1` |

The baseline script sets these for you; see [quick_start.md](./quick_start.md).

## Weight transfer

Weights move between the trainer and the rollout workers in one of two ways,
depending on the layout.

### Colocated actor and rollout

verl takes its `use_shm=False` path: Python multiprocessing shared memory is
not used. `BucketedWeightSender` synchronizes the weights through device-side
IPC handles (`reduce_tensor` + `rebuild_ipc`). This path needs torch_gcu's IPC
handle layout to match what verl expects.

### Disaggregated or async actor and rollout

The ECCL checkpoint engine sends per-bucket tensor metadata over ZeroMQ and
broadcasts the bucket bytes with `ray.util.collective` on two alternating device
buffers. [`scripts/accelerators/enflame/example_grpo_gsm8k_enflame_async.sh`](../../../scripts/accelerators/enflame/example_grpo_gsm8k_enflame_async.sh)
runs the full separate actor / rollout layout; the key setting is:

```bash
python3 -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.checkpoint_engine.backend=eccl
```

The registry name and the collective backend are separate:

| Selection | Registry key | Collective backend |
|-----------|--------------|--------------------|
| default, `backend=eccl` | `eccl` | `eccl` |
| `backend=nccl` | `nccl` (compatibility alias) | `eccl` |
| `USE_FLAGCX=1` | unchanged | `flagcx` |

`nccl` is an alias for scripts that opt into the async path and set
`checkpoint_engine.backend=nccl` explicitly; Ascend's HCCL engine claims the
same name. `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`
sets the bucket size (default `2048`).

Because Ray workers do not inherit shell exports, `USE_FLAGCX` and `FLAGCX_PATH`
must also be passed through `ray_init.runtime_env.env_vars`. The example scripts
do not forward them, and a worker that does not see `USE_FLAGCX` falls back to
`eccl`.

## Profiler

This backend does not provide a profiler: `nvtx_range`, `profiler_start` and
`profiler_stop` are no-ops, and `tool=topsprof` is not a supported value.

verl's generic `tool=torch` path can be enabled and uses torch_gcu's
PrivateUse1 profiler. The plugin makes no guarantee about that path:

```bash
python3 -m verl.trainer.main_ppo \
  ... \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.tool=torch
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `VERL_PLATFORM` | Set to `enflame` |
| `VERL_USE_EXTERNAL_MODULES` | Set to `verl_hardware_plugin` |
| `TOPS_VISIBLE_DEVICES` | GCUs visible to each process |
| `USE_FLAGCX` | `1` selects the FlagCX collective backend |
| `FLAGCX_PATH` | FlagCX installation path when `USE_FLAGCX=1` |
| `ENFLAME_ENABLE_AUTO_MIGRATION` | `1` applies Migration's runtime patches before verl starts (required for training) |

The same values must be present in `ray_init.runtime_env.env_vars` for Ray
workers. See [quick_start.md](./quick_start.md) for a complete launch example.

## Known limitations

- Python multiprocessing shared-memory weight transfer is not supported.
  Colocated rollout uses device-side IPC handles; disaggregated rollout uses the
  ECCL checkpoint engine.
- The checkpoint engine needs a Ray build that exposes the `eccl` collective
  backend (and `flagcx` when `USE_FLAGCX=1`).
- Profiling is not provided by the plugin.
- Megatron training requires Megatron-LM-FL in every Ray worker.
- The acceptance baseline is single-node GRPO with an FSDP actor and a vLLM
  rollout. Disaggregated, async and multi-node runs use the same platform
  registration and checkpoint engine.
