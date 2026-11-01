# Google TPU vLLM Rollout

The plugin runs verl's vLLM rollout on TPU. With `VERL_PLATFORM=tpu`, the usual
`actor_rollout_ref.rollout.name=vllm` setting selects the TPU rollout; other platforms keep verl's
stock vLLM rollout.

Training and rollout run on separate TPU slices (a slice is a group of TPU hosts joined by a fast
chip interconnect). One vLLM engine spans all chips of the rollout slice and receives the trainer's
updated weights through the `ray-plasma-store` or `tpu-sync` checkpoint engine (see [Weight Sync](./weight_sync.md)).

## Requirements

- The prerequisites from the [Installation Guide](./install_guidance.md), plus vLLM `v0.32.0` and
  [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) `732c4346`, the versions the rollout is tested with (see the `Dockerfile` in verl's
  [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo)).
- A verl checkout with the TPU GRPO examples in `examples/tpu/grpo`.
- At least two TPU slices, one for training and one for rollout. A TPU chip belongs to a single
  process, so training and rollout cannot share chips.
- On KubeRay, a TPU worker group named `tpu-group` with one replica per slice. Ray then labels the
  slices `tpu-group-0`, `tpu-group-1`, ..., which the plugin uses to give training and rollout a
  slice each.

## Run

The reference recipes are in verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo):
GRPO on GSM8K with Qwen3-0.6B (a 5-step smoke test) and Qwen3-4B, on two TPU v6e-4 slices. Its README walks
through the image, the KubeRay cluster and the `ray job submit` command, which loads the plugin in every Ray
worker (`py_modules` and `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin`).

To keep the plugin's `INFO` and `WARNING` logs from the driver process, such as the weight-sync timings (see
[Weight Sync](./weight_sync.md#verify)), add `"VERL_LOGGING_LEVEL": "INFO"` and
`"SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO"` to the job's `env_vars`.

libtpu and XLA flags (`LIBTPU_INIT_ARGS`, `XLA_FLAGS`), `VERL_TPU_CHIPS_PER_HOST`, and the deprecated `VERL_TPU_*` /
`VERL_TPU_SYNC_RELEASE_BUFFERS` variables (see [Configuration](./README.md#configuration)) go in the shell that runs the
recipe, as the recipes do with `export LIBTPU_INIT_ARGS=...`. That shell is the driver process; Ray workers
do not inherit its environment, so the platform names these variables in the job's `runtime_env`
(`get_ray_init_kwargs`), which reaches every actor of the job, the rollout servers and vLLM's engine workers
included. This relies on `RAY_OVERRIDE_JOB_RUNTIME_ENV=1`, which the recipes export, when the job is
submitted with `ray job submit`. A forwarded value replaces one set in a worker pod's environment, so put
flags meant for every process in the recipe. To check that a flag arrived, look for it in the `argv[...]`
lines of the libtpu start-up dump that each TPU worker prints.

TPU options of the TorchTitan engine go in its config, under `engine_kwargs.tpu`, like verl's other
per-backend options (`rollout.engine_kwargs.vllm`), e.g.
`+actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=False`. Supported keys:
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `fuse_qkv`, `tpu_eager_mode`,
`optim_impl`, `seq_bucket_size` and `context_parallel_strategy` (see `TPUEngineOptions` in `options.py`); an unknown key is
an error. The older `torchtitan.<option>` fields and `VERL_TPU_*` environment variables still work but are
deprecated and log a warning.

The rollout settings that matter on TPU (the recipes already set them):

| Setting | Value on TPU |
|---------|--------------|
| `actor_rollout_ref.rollout.name` | `vllm` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | All chips of the rollout slice, e.g. `4` on v6e-4, for one replica; the chips of one host for one replica per host (see [Weight Sync](./weight_sync.md#several-rollout-replicas)) |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` (the default); larger values are rejected |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `ray-plasma-store`, or `tpu-sync` (see [Weight Sync](./weight_sync.md)) |
| `actor_rollout_ref.hybrid_engine` | `False` |

## Verify

On any host, without vLLM or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_vllm_rollout.py -v
```

On TPU, the job log contains

```text
Registered rollout replica loader: vllm (TPU-aware)
Reusing 2 verl placement group(s) for TPU rollout
```

where the number is the number of rollout hosts, and the job ends with status `SUCCEEDED`. The rollout is
tested with Qwen3-0.6B and Qwen3-4B on two v6e-4 slices and on two v6e-8 slices.

## Related Documentation

- [User Guide](./README.md)
- [Installation Guide](./install_guidance.md)
- For developers: the code in `verl_hardware_plugin/accelerators/tpu/rollout/` documents how each part works and which
  upstream fix would make each vLLM patch unnecessary.
