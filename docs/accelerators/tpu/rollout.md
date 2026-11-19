# Google TPU Rollout (vLLM & SGLang)

The plugin runs verl's **vLLM** and **SGLang** rollouts on TPU. With `VERL_PLATFORM=tpu`, setting
`actor_rollout_ref.rollout.name=vllm` or `actor_rollout_ref.rollout.name=sglang` selects the corresponding
TPU rollout (`TPUvLLMReplica` or `TPUSGLangReplica`); other platforms keep verl's stock GPU rollouts.

Training and rollout run on separate TPU slices (a slice is a group of TPU hosts joined by a fast
chip interconnect). One vLLM or SGLang engine spans all chips of the rollout slice and receives the trainer's
updated weights through the `ray-plasma-store` or `tpu-sync` checkpoint engine (see [Weight Sync](./weight_sync.md)).

A rollout replica is laid out as upstream verl lays it out on GPU: one server actor per host, the leader
running the HTTP API and engine coordinator, and follower hosts running worker processes headless:

- **vLLM (`actor_rollout_ref.rollout.name=vllm`)**: vLLM's `mp` executor starts one worker per chip
  (per TensorCore on TPU 7x) on each host, and `vllm-torchtpu`'s `mp` multi-host backend
  (`TPU_MULTIHOST_BACKEND=mp`) joins the replica's hosts into one TPU mesh.
- **SGLang (`actor_rollout_ref.rollout.name=sglang`)**: `Engine._launch_subprocesses` spawns one TP
  scheduler process per local chip on each host via [sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu)
  (`SGLANG_PLATFORM=tpu`), joining multi-host slices via `dist_init_addr` and TPU topology normalization,
  and servicing `tpu-sync` / `ray-plasma-store` weight sync RPCs over SGLang's internal ZMQ RPC channel and
  `tp_group.cpu_group`.

## Requirements

- The prerequisites from the [Installation Guide](./install_guidance.md), plus either:
  - **vLLM**: vLLM `v0.32.0` and [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) `732c4346`
    (see the `Dockerfile` in verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo)), or
  - **SGLang**: [sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu) (`SGLANG_BUILD_RUST_EXTS=none` with `python/pyproject_tpu.toml` and `-e .`).
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

For TPU-specific training options of the TorchTitan engine (`engine_kwargs.tpu`), such as
`use_splash_attention`, `use_simple_fsdp`, `use_fused_linear_cross_entropy`, `trainer_mode: graph_trainer`,
and MoE settings, see the [Training Engine Guide](./engine.md).

The rollout settings that matter on TPU (the recipes already set them):

| Setting | Value on TPU |
|---------|--------------|
| `actor_rollout_ref.rollout.name` | `vllm` or `sglang` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | All chips of the rollout slice, e.g. `4` on v6e-4, for one replica; the chips of one host for one replica per host (see [Weight Sync](./weight_sync.md#several-rollout-replicas)) |
| `actor_rollout_ref.rollout.data_parallel_size` | `1` (the default); larger values are rejected |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | `ray-plasma-store`, or `tpu-sync` (see [Weight Sync](./weight_sync.md)) |
| `actor_rollout_ref.hybrid_engine` | `False` |

### Several engines per host (TPU 7x)

With `tensor_model_parallel_size` 2 or 4 on TPU 7x (one or two chips), several replicas share a host, e.g.
16 x TP=2 or 8 x TP=4 on four `2x2x1` rollout hosts. `auto_assign_accelerator_type` spreads the replicas evenly
over the rollout slices, and each engine on a host gets its own slot of the host's devices
(`TPU_LOCAL_RANK_OFFSET`), fixed ports below the ephemeral range (from `VERL_TPU_ROLLOUT_SUBHOST_PORT_BASE`,
default `19000`, 20 ports per device), its own libtpu MegaScale coordinator, and compile caches under
`/tmp/*_tp<N>_r<replica>`. The replicas on a host must all use the same TP size. TP=8 (a whole host) and
multi-host replicas launch as before.

## Verify

On any host, without vLLM, SGLang, or a TPU:

```bash
pytest tests/accelerators/tpu/test_tpu_vllm_rollout.py tests/accelerators/tpu/test_tpu_sglang_rollout.py -v
```

On TPU, the job log contains

```text
Registered rollout replica loader: vllm (TPU-aware)
Registered rollout replica loader: sglang (TPU-aware)
[TPU-SYNC CONTROLLER] All 8 Trainer and 8 Sampler units verified and registered.
```

where the sampler count is the number of rollout chips (with `tpu-sync`), and the job ends with status
`SUCCEEDED`. The rollout is tested with Qwen3-4B on a v6e-8 trainer slice and two 2-host v6e-8 rollout
slices (two replicas, each spanning two hosts).

## Related Documentation

- [User Guide](./README.md)
- [Training Engine Guide](./engine.md)
- [Installation Guide](./install_guidance.md)
- For developers: the code in `verl_hardware_plugin/accelerators/tpu/rollout/` documents how each part works and which
  upstream fix would make each vLLM patch unnecessary.
