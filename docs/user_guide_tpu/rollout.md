# TPU vLLM Rollout

This page covers the vLLM rollout path on Google TPU: what the plugin changes relative to
upstream verl, the runtime patches it still needs, and how to test it.

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

| Hook (upstream verl) | TPU override | Why |
|---|---|---|
| `vLLMReplica.launch_servers` | `launch_tpu_vllm_servers` | One server actor per replica, on the first worker's node, with every TPU env var forwarded. vLLM spans the hosts through its Ray executor instead of one `mp` server per node. `data_parallel_size > 1` is rejected: each compiled program has to span the whole TPU mesh. |
| `RolloutReplica.rollout_worker_use_gpu` | `False` | Rollout workers must not claim a `GPU` resource. |
| `RolloutReplica.init_standalone` | copy with `use_gpu=self.rollout_worker_use_gpu()` | Upstream hardcodes `use_gpu=True` in standalone mode. Drop the override once upstream asks the hook, as `init_colocated` already does. |
| `vLLMHttpServer._preprocess_engine_kwargs` | `distributed_executor_backend=external_launcher`, `enable_sleep_mode=False`, `TPU_MULTIHOST_BACKEND=ray`, `VLLM_DISABLE_COMPILE_CACHE=1` | `patch_vllm_for_tpu` then selects the Ray executor for multi-host engines at config time. Sleep mode is off because rollout runs on its own slice (TPU chips cannot be shared between colocated worker groups), so nothing needs the HBM back; this is unchanged from `tpu-main`. |
| `vLLMHttpServer.collective_rpc` | returns the engine result | Upstream drops it. The TPU weight-sync paths read per-worker results. |
| `vLLMHttpServer._get_worker_extension_cls` | Raiden extension when `checkpoint_engine.backend=raiden`, upstream's otherwise | Raiden needs its own worker methods. |
| `PlatformTPU.auto_assign_accelerator_type` / `configure_placement_group_bundle` | rollout/reward/teacher pools pinned to the second `tpu-group-<n>` slice; their bundles do not reserve `TPU` | Otherwise the rollout placement groups ask for the slice the trainer already holds and wait forever; vLLM's Ray executor claims the chips itself. Same rule as `tpu-main`. |

The server actor's Ray `max_concurrency` comes from `RolloutConfig.ray_actor_max_concurrency` when
verl has it and from `vLLMReplica.max_concurrency` otherwise.

## `patch_vllm_for_tpu`

The patches are ported from verl's `tpu-main` branch and audited against the pinned stack
(vLLM `v0.29.0`, vllm-torchtpu `9faafb17`). They run in the driver, in each `TPUvLLMHttpServer`
actor, in the vLLM EngineCore (through the `multiprocessing` and `run_engine_core` wrappers), and
in pooled vLLM Ray workers (through the torchtpu `RayWorkerWrapper.__init__` wrapper).

### Removed from the `tpu-main` version

These were no-ops or dead code on the pinned stack:

| Removed | Why |
|---|---|
| `VLLM_USE_V1=0`, `use_v1=False` overrides, `run_server` override | vLLM `v0.29.0` has no V0 engine; `VLLM_USE_V1` is not read anywhere. |
| `init_cached_hf_modules` stub | The function does not exist in vLLM `v0.29.0`. |
| `RayWorkerWrapper.setup_device_if_necessary` override | vllm-torchtpu's own implementation is already a no-op on TPU. |
| torchtpu topology-map update | vllm-torchtpu has no `TPU_TOPOLOGY_MAP`. |
| Bundle-index `_init_workers_ray` wrapper | Overwritten by the full `_init_workers_ray` replacement on the same class. |
| `TPU_POD_IP_TO_SLICE` / `VERL_ROLLOUT_PG_NAME` branches | Nothing sets these variables. |
| Concat-free RoPE patch and its worker extension | Only needed when the model runs eagerly. The eager fallback comes from vLLM reloading an empty AOT artifact; `VLLM_DISABLE_COMPILE_CACHE=1` avoids that, and Raiden runs never loaded the RoPE patch. |
| Two of three `reset_encoder_cache` stubs | One class-level stub on `TPUWorker` covers it. |

### Still present, to be verified on TPU

Each of these either gets a reason recorded here after a TPU run, or is removed.

| Patch | What it does |
|---|---|
| `allow_in_graph` on `c10d_functional` ops | Keeps functional collectives inside the Dynamo graph. |
| Strip `worker_process_setup_hook` from `ray.init` | Keeps the job-level setup hook out of processes that call `ray.init` later. |
| `os.environ.__setitem__` guard | Stops the driver's topology values from overwriting a pod's own; strips `megachip_tccontrol` from `LIBTPU_INIT_ARGS`. |
| `available_resources_per_node` forces `TPU >= 4` | vLLM's per-node resource check against verl's TPU placement groups. |
| `initialize_ray_cluster`: PG discovery and swallowed size-validation `ValueError` | vLLM does not find verl's rollout placement group on its own. The wrapper first connects to Ray with vLLM's captured `ray_runtime_env`; a Ray call before that would auto-connect with an empty one and the vLLM workers would start without the plugin's `py_modules`. |
| `initialize_dummy_weights` no-op, `torch.set_grad_enabled(False)` in `init_worker` | Skips random init under `load_format=dummy`; weights arrive by sync. |
| `EngineArgs.create_engine_config` | Ray executor + async scheduling off for multi-host, local executor + async scheduling on for single-host; clears stale DP fields. |

### Known gaps in other projects

| Gap | Patch | Owner |
|---|---|---|
| `TPUWorker.reset_encoder_cache` not implemented | class-level no-op stub | vllm-torchtpu |
| Per-worker TPU environment (`TPU_VISIBLE_CHIPS`, `TPU_PROCESS_PORT`, `CLOUD_TPU_TASK_ID`, host/chip bounds, slice-builder addresses) when vLLM's generic Ray executor drives torchtpu workers | full `_init_workers_ray` replacement | vllm-torchtpu |
| Compiled Ray DAG path | `_execute_dag` replaced with a plain `ray.get` fan-out | vllm-torchtpu / vLLM |
| Empty AOT artifact reloads silently as eager execution | `VLLM_DISABLE_COMPILE_CACHE=1` | vLLM |

Most of the torchtpu rows collapse once vllm-torchtpu's own `RayDistributedExecutor` can run under
verl. Today `create_engine_config` forces vLLM's generic executor, so none of vllm-torchtpu's
executor code is exercised.

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
`examples/tpu/grpo/run_qwen3_0_6b_torchtitan.sh` with `SMOKE_TEST=1` from a verl checkout whose
in-tree TPU rollout has been removed, so only the plugin is exercised. Ship the plugin through Ray
`py_modules` with `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin`, then check the log with
`tests/special_tpu/verify_tpu_e2e_log.py grpo <log> 1` and that it contains
`Registered rollout replica loader: vllm (TPU-aware)`, no `Traceback` and no
`IsFusibleUnalignedDUS`. Cover `checkpoint_engine.backend=tpu` twice on the same pods (cold, then
warm compile cache) and the default `backend=raiden`.
