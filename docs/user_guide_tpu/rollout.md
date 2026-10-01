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
| `RolloutReplica.init_standalone` | copy with `use_gpu=self.rollout_worker_use_gpu()` | verl hardcodes `use_gpu=True` in standalone mode. Drop the override once verl asks the hook, as `init_colocated` already does. |
| `vLLMHttpServer._preprocess_engine_kwargs` | `distributed_executor_backend=external_launcher`, `enable_sleep_mode=False`, `TPU_MULTIHOST_BACKEND=ray`, `VLLM_DISABLE_COMPILE_CACHE=1` | `patch_vllm_for_tpu` then selects the Ray executor for multi-host engines at config time. Sleep mode is off because rollout runs on its own slice (TPU chips cannot be shared between colocated worker groups), so nothing needs the HBM back. |
| `vLLMHttpServer.collective_rpc` | returns the engine result | verl's version awaits `engine.collective_rpc` and returns `None`. The TPU weight-sync paths need the per-worker return values. |
| `PlatformTPU.auto_assign_accelerator_type` / `configure_placement_group_bundle` | rollout/reward/teacher pools pinned to the second `tpu-group-<n>` slice; their bundles do not reserve `TPU` | Otherwise the rollout placement groups ask for the slice the trainer already holds and wait forever; vLLM's Ray executor claims the chips itself. |

The server actor's Ray `max_concurrency` comes from `RolloutConfig.ray_actor_max_concurrency` when
verl has it and from `vLLMReplica.max_concurrency` otherwise.

## Where each patch runs and when it can be removed

A multi-slice TPU rollout job involves five kinds of processes:

1. **Driver (`main_ppo`)** — starts the Ray job on the head node and launches `TaskRunner`.
2. **`TaskRunner` Ray actor** — creates `RayResourcePool` placement groups and spawns trainer and rollout worker actors + `TPUvLLMReplica`.
3. **`Worker` / `CheckpointEngineWorker` Ray actors** — verl worker actors placed in the trainer and rollout placement groups.
4. **`TPUvLLMHttpServer` Ray actor & `EngineCoreProc` child process** — the per-replica vLLM HTTP server actor on the first rollout node and the `VLLM::EngineCore` subprocess it spawns via `multiprocessing`.
5. **`RayWorkerWrapper` Ray actors** — the per-chip TPU worker actors spawned by vLLM's `RayDistributedExecutor`.

### 1. verl-core bridge patches (`verl_hardware_plugin/patches/tpu/`)

verl main calls only part of the platform interface. Two call sites are bridged by small
patches in `verl_hardware_plugin/patches/tpu/` that route the existing `PlatformTPU` hooks
through; **both are temporary and should be deleted once upstream `verl` calls those hooks directly**:

| Patch | Process where it executes | verl call site | What it routes | Removal condition |
|---|---|---|---|---|
| `ray_resource_pool_patch` | **`TaskRunner` actor** (on the head/coordinator node, where resource pools and placement groups are created) | `RayResourcePool.__init__` / `get_placement_groups` | `auto_assign_accelerator_type` (rollout pools go to the second slice) and `configure_placement_group_bundle` (rollout bundles do not reserve `TPU`). | Upstream `verl` calls `auto_assign_accelerator_type` and `configure_placement_group_bundle` in `RayResourcePool`. |
| `worker_local_rank_patch` | **Every verl `Worker` & `CheckpointEngineWorker` actor** (on the TPU worker nodes) | `Worker._setup_env_cuda_visible_devices` | `ray_local_rank_override` (`LOCAL_RANK` from `TPU_VISIBLE_CHIPS`) and skips eager `set_device` in `CheckpointEngineWorker`. | Upstream `verl` calls `ray_local_rank_override` in `Worker._setup_env_cuda_visible_devices`. |

verl main also does not call `PlatformTPU.get_ray_init_kwargs()`, so the Ray
`worker_process_setup_hook` that installs these patches in every Ray worker process has to be set on
the command line:

```bash
python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=verl_hardware_plugin.platforms.platform_tpu.patch_ray_worker \
    ...
```

If the hook is missing, `PlatformTPU.get_worker_env_vars` logs an error the first time a pool is
used instead of letting the job hang.

### 2. vLLM / vllm-torchtpu runtime patches (`patch_vllm_for_tpu`)

`patch_vllm_for_tpu()` targets the pinned stack (vLLM `v0.29.0`, vllm-torchtpu `9faafb17`). It is
installed in `TPUvLLMHttpServer`, propagated into the `EngineCoreProc` subprocess (via the
`multiprocessing.process.BaseProcess` and `run_engine_core` wrappers), and invoked in each pooled
vLLM `RayWorkerWrapper` actor (via `RayWorkerWrapper.__init__`).

#### Group A — verl ↔ vLLM Ray integration (permanent plugin glue unless vLLM adds native hooks)

| Patch | Process where it executes | What it does |
|---|---|---|
| `allow_in_graph` on `c10d_functional` ops | `EngineCoreProc` & `RayWorkerWrapper` | Keeps PyTorch functional collectives inside the `torch.compile` / Dynamo graph. |
| Strip `worker_process_setup_hook` from `ray.init` | `TPUvLLMHttpServer` & `EngineCoreProc` | Ray forbids passing a job-level `worker_process_setup_hook` when a child process calls `ray.init` to attach to an existing cluster. |
| `os.environ.__setitem__` guard | `EngineCoreProc` & `RayWorkerWrapper` | Prevents the driver's single-host/default topology env vars from overwriting a TPU pod's own topology; strips `megachip_tccontrol` from `LIBTPU_INIT_ARGS`. |
| `available_resources_per_node` forces `TPU >= 4` | `EngineCoreProc` | Satisfies vLLM's per-node TPU resource check when verl's placement group is already created. |
| `initialize_ray_cluster`: PG discovery and swallowed size-validation `ValueError` | `EngineCoreProc` | Connects to Ray with vLLM's `ray_runtime_env` (preserving `py_modules`), attaches to verl's `rollout_pool` placement group, and bypasses vLLM's bundle-resource validation (since rollout bundles intentionally omit `TPU: 1`). |
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
