# TPU vLLM Rollout

This page covers the vLLM rollout path on Google TPU: how it plugs into verl, which process runs
each patch, which patches are temporary workarounds for upstream repos (`verl`, `vllm-torchtpu`,
`vllm`), and how to test it.

## How it plugs in

```text
verl_hardware_plugin/rollout/
├── __init__.py                 # registers a TPU-aware "vllm" loader in RolloutReplicaRegistry
├── tpu_vllm.py                 # TPUvLLMReplica, TPUvLLMHttpServer, launch_tpu_vllm_servers
└── tpu_vllm_patches.py         # patch_vllm_for_tpu(): vLLM / vllm-torchtpu runtime patches
```

`register_all_rollouts()` wraps the `vllm` entry of `RolloutReplicaRegistry`. The loader returns
`TPUvLLMReplica` when `get_resource_name() == "TPU"` and otherwise calls the loader that was
registered before it, so other platforms are unaffected. No verl core file is modified.

| Hook (verl) | TPU override | Why |
|---|---|---|
| `vLLMReplica.launch_servers` | `launch_tpu_vllm_servers` | One server actor per replica, on the first worker's node, with every TPU env var forwarded. vLLM spans the hosts through its Ray executor instead of one `mp` server per node. `data_parallel_size > 1` is rejected: the TPU runtime compiles one XLA program for the whole mesh of the slice, while a DP group would only span `tensor_model_parallel_size` chips. Use `data_parallel_size=1`; verl then creates one replica per `tensor_model_parallel_size` chips, which gives the same parallelism. |
| `RolloutReplica.rollout_worker_use_gpu` | `False` | Rollout workers must not claim a `GPU` resource. |
| `vLLMHttpServer._preprocess_engine_kwargs` | `distributed_executor_backend=external_launcher`, `enable_sleep_mode=False`, `TPU_MULTIHOST_BACKEND=ray`, `VLLM_DISABLE_COMPILE_CACHE=1` | `patch_vllm_for_tpu` then selects the Ray executor for multi-host engines at config time. Sleep mode is off because rollout runs on its own slice (TPU chips cannot be shared between colocated worker groups), so nothing needs the HBM back. |
| `vLLMHttpServer.collective_rpc` | returns the engine result | verl's version awaits `engine.collective_rpc` and returns `None`. The TPU weight-sync paths need the per-worker return values. |
| `PlatformTPU.auto_assign_accelerator_type` | gives each pool the first `tpu-group-<n>` slice that no earlier pool claimed | Keeps every host of a multi-host pool within one slice and puts the trainer and rollout pools on different slices. |

The server actor's Ray `max_concurrency` comes from `RolloutConfig.ray_actor_max_concurrency` when
verl has it and from `vLLMReplica.max_concurrency` otherwise.

## Where each patch runs and when it can be removed

A multi-slice TPU rollout job involves five kinds of processes:

1. **Driver (`main_ppo`)** — starts the Ray job on the head node and launches `TaskRunner`.
2. **`TaskRunner` Ray actor** — creates `RayResourcePool` placement groups and spawns trainer and rollout worker actors + `TPUvLLMReplica`.
3. **`Worker` / `CheckpointEngineWorker` Ray actors** — verl worker actors placed in the trainer and rollout placement groups.
4. **`TPUvLLMHttpServer` Ray actor & `EngineCoreProc` child process** — the per-replica vLLM HTTP server actor on the first rollout node and the `VLLM::EngineCore` subprocess it spawns via `multiprocessing`.
5. **`RayWorkerWrapper` Ray actors** — the per-chip TPU worker actors spawned by vLLM's `RayDistributedExecutor`.

### 1. verl-core requirements

The TPU rollout needs TPU call sites in verl core that verl main does not have yet. Until they are
upstreamed, they live on the `pr34-grpo-0.6b-core-fixes` branch of
[jialei777/verl-upstream](https://github.com/jialei777/verl-upstream/tree/pr34-grpo-0.6b-core-fixes):

| verl change | Why |
|---|---|
| `Worker._setup_env_cuda_visible_devices` takes `LOCAL_RANK` from `TPU_VISIBLE_CHIPS` on TPU | `get_worker_env_vars` pins each worker to one chip through `TPU_VISIBLE_CHIPS`; Ray cannot map its host-level chip id into that one-chip list, so the generic lookup raises `IndexError`. |
| `RayResourcePool.get_placement_groups` labels TPU bundles with `auto_assign_accelerator_type` | Ray places each per-host placement group independently; without the label a multi-host pool can straddle two slices. |
| `RolloutReplica.init_standalone` sets `use_gpu=supports_colocated_worker_groups()` | The replica's `CheckpointEngineWorker`s must not take the chips that vLLM's own workers need. |

### 2. vLLM / vllm-torchtpu runtime patches (`patch_vllm_for_tpu`)

`patch_vllm_for_tpu()` targets the pinned stack (vLLM `v0.29.0`, vllm-torchtpu `9faafb17`). It is
installed in `TPUvLLMHttpServer`, propagated into the `EngineCoreProc` subprocess (via the
`multiprocessing.process.BaseProcess` and `run_engine_core` wrappers), and invoked in each pooled
vLLM `RayWorkerWrapper` actor (via `RayWorkerWrapper.__init__`).

#### Group A — verl ↔ vLLM Ray integration (permanent plugin glue unless vLLM adds native hooks)

| Patch | Process where it executes | What it does |
|---|---|---|
| `allow_in_graph` on `c10d_functional` ops | `EngineCoreProc` & `RayWorkerWrapper` | Keeps PyTorch functional collectives inside the `torch.compile` / Dynamo graph. |
| `os.environ.__setitem__` guard | `EngineCoreProc` & `RayWorkerWrapper` | Prevents the driver's single-host/default topology env vars from overwriting a TPU pod's own topology; strips `megachip_tccontrol` from `LIBTPU_INIT_ARGS`. |
| `available_resources_per_node` forces `TPU >= 4` | `EngineCoreProc` | Satisfies vLLM's per-node TPU resource check when verl's placement group is already created. |
| `initialize_ray_cluster`: reuse verl placement groups via `VERL_TPU_PG_IDS` | `EngineCoreProc` | Connects to Ray with vLLM's `ray_runtime_env` (preserving `py_modules`) and attaches directly to the replica's `RayResourcePool` placement groups (`VERL_TPU_PG_IDS`) instead of creating a duplicate placement group. |
| `initialize_dummy_weights` no-op, `torch.set_grad_enabled(False)` in `init_worker` | `RayWorkerWrapper` | Skips random weight initialization under `load_format=dummy` because trainer weights are synced right after engine init. |
| `EngineArgs.create_engine_config` | `TPUvLLMHttpServer` & `EngineCoreProc` | Selects the `ray` executor with `async_scheduling=False` for multi-host slices, and the local executor with `async_scheduling=True` for single-host slices; clears stale DP fields when `data_parallel_size <= 1`. |

#### Group B — Upstream bugs / gaps to file against `vllm-torchtpu` and `vllm` (removable once fixed upstream)

These patches exist only because of missing methods or bugs in `vllm-torchtpu` (`9faafb17`) or
`vllm` (`v0.29.0`). They can be filed as an upstream issue list and removed from the plugin as soon
as `vllm-torchtpu` / `vllm` lands the fixes:

| # | Target repo | File / Symbol in upstream | Bug / Gap description | Suggested upstream fix | Plugin workaround today |
|---|---|---|---|---|---|
| 1 | `vllm-torchtpu` | `vllm_torchtpu/worker/tpu_worker.py` (`TPUWorker.reset_encoder_cache`) | `TPUWorker` does not implement `reset_encoder_cache()`. When verl calls `clear_kv_cache` / `reset_prefix_cache` after weight sync, vLLM V1 invokes `worker.reset_encoder_cache()` and crashes with `NotImplementedError` / `AttributeError`. | Implement a no-op (or encoder cache clear) `reset_encoder_cache(self)` method on `TPUWorker`. | Monkey-patches `TPUWorker.reset_encoder_cache` with a no-op stub in `EngineCoreProc` & `RayWorkerWrapper`. |
| 2 | `vllm-torchtpu` | `vllm_torchtpu/executors/ray_distributed_executor.py` (`RayDistributedExecutor`) | When `distributed_executor_backend="ray"` is selected, vLLM resolves to its generic `vllm.v1.executor.ray_executor.RayDistributedExecutor` instead of `vllm_torchtpu`'s `RayDistributedExecutor`, or `vllm_torchtpu`'s `_init_workers_ray` fails under an external Ray placement group because it assumes bundles reserve `TPU` and does not compute per-host `CLOUD_TPU_TASK_ID`, `TPU_VISIBLE_CHIPS`, `TPU_PROCESS_PORT`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`, and `TPU_PROCESS_ADDRESSES` from the placement group's assigned nodes. | Register `vllm_torchtpu`'s `RayDistributedExecutor` as the TPU V1 Ray executor and support placement groups whose bundles do not pre-reserve `TPU` (or compute per-worker TPU mesh env vars in `vllm_torchtpu`'s `_init_workers_ray`). | Replaces `RayDistributedExecutor._init_workers_ray` in `EngineCoreProc` to spawn `RayWorkerWrapper` actors and populate per-worker TPU mesh env vars. |
| 3 | `vllm-torchtpu` / `vllm` | `vllm/v1/executor/ray_executor.py` (`RayDistributedExecutor._execute_dag`) | vLLM V1's `RayDistributedExecutor` uses a Ray compiled DAG (`forward_dag`) by default, which hangs or fails with TPU `RayWorkerWrapper` actors across multiple hosts. | Disable Ray compiled DAG on TPU (`VLLM_USE_RAY_COMPILED_DAG=0` by default on `tpu_platform`) or fall back to standard `ray.get([w.execute_model_ray.remote(...)])` fan-out in `vllm-torchtpu`. | Replaces `RayDistributedExecutor._execute_dag` in `EngineCoreProc` with a direct `ray.get` fan-out. |
| 4 | `vllm` / `vllm-torchtpu` | `vllm` AOT compile cache (`VLLM_DISABLE_COMPILE_CACHE`) | On warm starts, vLLM can reload a degenerate AOT compile artifact (`num_artifacts=0`) that silently skips `torch.compile` and runs the model eagerly on TPU, triggering a fatal `libtpu` `IsFusibleUnalignedDUS` `CHECK` crash (`b/501165531`). | Validate that a reloaded AOT artifact on TPU has `num_artifacts > 0` before skipping compilation, or default `VLLM_DISABLE_COMPILE_CACHE=1` in `vllm-torchtpu` until AOT cache serialization supports PJRT/TPU artifacts. | Sets `VLLM_DISABLE_COMPILE_CACHE=1` by default in `prepare_tpu_server_env` and `patch_vllm_for_tpu`. |

## Testing

CPU (same as the plugin's GitHub Actions):

```bash
ruff check . && ruff format --check .
mypy --ignore-missing-imports verl_hardware_plugin/
python scripts/check_license.py --directories .
python scripts/check_bytedance_copyright.py --directories .
python scripts/check_verl_api.py --plugin-root . --verl-root /path/to/verl/verl
pytest -q tests
```

`tests/test_tpu_vllm_rollout.py` stubs `vllm_async_server` and every Ray handle, so it runs
without vLLM or a TPU.

TPU (GKE, two slices of 2 hosts x 4 v6e chips): run
`examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh` with `SMOKE_TEST=1` against verl main, with the
plugin shipped through Ray `py_modules` and `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin`, then
check the log with `tests/special_tpu/verify_tpu_e2e_log.py grpo <log> 1` and that it contains
`Registered rollout replica loader: vllm (TPU-aware)`, no `Traceback` during training and no
`IsFusibleUnalignedDUS`. Cover `checkpoint_engine.backend=tpu` twice on the same pods (cold, then
warm compile cache).
