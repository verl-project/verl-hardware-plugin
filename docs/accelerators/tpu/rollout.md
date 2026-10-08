# Google TPU vLLM Rollout

The plugin runs verl's vLLM rollout on TPU. With `VERL_PLATFORM=tpu`, the usual
`actor_rollout_ref.rollout.name=vllm` setting selects the TPU rollout; other platforms keep verl's
stock vLLM rollout.

Training and rollout run on separate TPU slices (a slice is a group of TPU hosts joined by a fast
chip interconnect). One vLLM engine spans all chips of the rollout slice and receives the trainer's
updated weights through the `tpu` or `raiden` checkpoint engine (see [Weight Sync](./weight_sync.md)).

## Requirements

- The prerequisites from the [Installation Guide](./install_guidance.md), plus vLLM `v0.29.0` and
  vllm-torchtpu `9faafb17`, the versions the rollout is tested with (image
  `us-west2-docker.pkg.dev/tpu-pytorch/raycluster/verl-tpu:v20261007-tsync1001132139`).
- A verl checkout with the TPU rollout changes. Until they are merged into verl, use the
  `pr34-grpo-0.6b-core-fixes` branch of
  [jialei777/verl-upstream](https://github.com/jialei777/verl-upstream/tree/pr34-grpo-0.6b-core-fixes).
- At least two TPU slices, one for training and one for rollout. A TPU chip belongs to a single
  process, so training and rollout cannot share chips.
- On KubeRay, a TPU worker group named `tpu-group` with one replica per slice. Ray then labels the
  slices `tpu-group-0`, `tpu-group-1`, ..., which the plugin uses to give training and rollout a
  slice each.

## Run

The reference recipe is `examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh` on the verl branch above:
GRPO on GSM8K with Qwen3-0.6B, on two v6e-8 slices of 2 hosts x 4 chips. Submit it as a Ray job
from the verl checkout, with the plugin loaded in every Ray worker:

```bash
ray job submit --address "${RAY_ADDRESS}" \
  --working-dir . \
  --runtime-env-json '{
    "py_modules": ["/path/to/verl-hardware-plugin/verl_hardware_plugin"],
    "excludes": [".git", "logs", "*.log", "*.pt", "*.bin"],
    "env_vars": {
      "PYTHONPATH": ".",
      "PYTHONUNBUFFERED": "1",
      "VERL_PLATFORM": "tpu",
      "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
      "VERL_LOGGING_LEVEL": "INFO",
      "SAGEMAKER_CONTAINER_LOG_LEVEL": "INFO",
      "RAY_memory_monitor_refresh_ms": "0",
      "RAY_memory_usage_threshold": "0.99",
      "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
      "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1"
    }
  }' \
  -- bash -c 'SMOKE_TEST=1 bash examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh'
```

`SMOKE_TEST=1` runs a 5-step bring-up check; drop it for the full 100-step run. Drop `py_modules`
if the plugin is installed in the image. `SAGEMAKER_CONTAINER_LOG_LEVEL=INFO` keeps the plugin's `INFO`
logs from the driver process, such as the weight-sync timings (see [Weight Sync](./weight_sync.md#verify)).

The rollout settings that matter on TPU (the recipe already sets them):

| Setting | Value on TPU |
|---------|--------------|
| `actor_rollout_ref.rollout.name` | `vllm` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | All chips of the rollout slice, e.g. `8` on v6e-8 |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` (the default); larger values are rejected |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `tpu`, or `raiden` (see [Weight Sync](./weight_sync.md)) |
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

where `2` is the number of rollout hosts, and the job ends with status `SUCCEEDED`. The rollout is
tested with Qwen3-0.6B and Qwen3-4B on two v6e-8 slices.

## Related Documentation

- [User Guide](./README.md)
- [Installation Guide](./install_guidance.md)
- For developers: the code in `verl_hardware_plugin/accelerators/tpu/rollout/` documents how each part works and which
  upstream fix would make each vLLM patch unnecessary.
