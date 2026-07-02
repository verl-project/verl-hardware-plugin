# Enflame GCU FAQ and Troubleshooting

## Common Questions

### Q: What is the difference between `enflame` and `gcu`?

| Name | Where used | Value |
|------|------------|-------|
| `enflame` | `VERL_PLATFORM`, platform registry key | Platform selector |
| `gcu` | `device_name`, engine registry `device=` | PyTorch device API (`torch.gcu`) |
| `enflame` | `vendor_name`, engine registry `vendor=` | Vendor-specific engine lookup |

Set `VERL_PLATFORM=enflame`. Engines register as `(device="gcu", vendor="enflame")`.

### Q: Should I use ECCL or FlagCX?

For a homogeneous Enflame GCU cluster, use ECCL (default):

```bash
export USE_FLAGCX=0
```

Set `USE_FLAGCX=1` only when FlagCX is installed and configured for your cluster. See the [FlagOS User Guide](../user_guide_flagos/README.md) for FlagCX setup.

### Q: Do I need Migration?

Migration is **recommended** for verl + vLLM on GCU. It applies runtime patches (device normalization, vLLM compat, checkpoint fixes) before verl initializes.

```bash
export ENFLAME_ENABLE_AUTO_MIGRATION=1
pip install migration
```

Without Migration, some verl/vLLM code paths may fail on `torch_gcu`. If you hit import or device errors, install Migration first.

### Q: How do I validate my Enflame setup end-to-end?

Run the acceptance baseline and compare against the NVIDIA reference:

```bash
bash scripts/baseline_grpo_gsm8k_enflame.sh
```

Reference curve: [SwanLab — verl_grpo_gsm8k_math](https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart)

Check that `critic/rewards/mean` trends upward in the first 100 steps. See [Quick Start](./quick_start.md) and [development.md](../development.md#acceptance-baseline-for-new-hardware-adaptation).

### Q: Do I need Megatron-LM-FL?

No, for the FSDP GRPO baseline in [Quick Start](./quick_start.md). Megatron-LM-FL is only required for the Megatron training backend.

If you see `Failed to register Enflame Megatron engines (required): 'torchtitan'` on the **driver** process but Ray workers later log `Registered engines: megatron_enflame`, the FSDP path is unaffected.

### Q: Why must I set env vars in both shell and Hydra `runtime_env`?

Ray workers are separate processes. Shell exports are not propagated unless passed via:

```bash
+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PLATFORM='enflame'
```

Always mirror `ENFLAME_ENABLE_AUTO_MIGRATION`, `VERL_PLATFORM`, `TOPS_VISIBLE_DEVICES`, and vLLM/GCU stability vars in `runtime_env`.

---

## Troubleshooting

### Platform resolves to `cpu` instead of `gcu`

**Symptoms:** `get_platform().device_name` returns `cpu`; training fails early.

**Fix:**

```bash
export VERL_PLATFORM=enflame
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
python3 -c "import verl_hardware_plugin; import verl; from verl.plugin.platform import get_platform; print(get_platform().device_name)"
```

Ensure `verl_hardware_plugin` is installed and `torch.gcu.is_available()` returns `True`.

### Placement group stays pending / workers never start

**Symptoms:** Ray logs show placement group pending indefinitely.

**Cause:** Custom Ray resource used instead of built-in GPU.

**Fix:** Do **not** start Ray with `--resources='{"enflame": N}'`. Enflame maps to Ray `GPU` via `num_gpus`. Use:

```bash
ray start --head   # default GPU detection
trainer.n_gpus_per_node=4
```

### `get_accelerator_ids` or Ray resource mismatch

**Symptoms:** Worker fails to acquire devices; resource request errors.

**Fix:** Align three values:

1. `trainer.n_gpus_per_node` = number of GCUs to use
2. `TOPS_VISIBLE_DEVICES` = comma-separated GCU indices
3. `runtime_env` `TOPS_VISIBLE_DEVICES` matches the shell value

### vLLM rollout crashes (compile, inductor, multiprocessing)

**Symptoms:** Rollout worker fails during vLLM init or first generation.

**Fix:** Add stability env vars (included in [Quick Start](./quick_start.md)):

```bash
export TORCHGCU_INDUCTOR_ENABLE=0
export TORCHDYNAMO_DISABLE=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
```

And set `actor_rollout_ref.rollout.enforce_eager=True`.

### `torch.compile` / dynamo errors during FSDP training

**Fix:**

```bash
export TORCHDYNAMO_DISABLE=1
export TORCHGCU_INDUCTOR_ENABLE=0
```

Or disable compile in actor config: `actor_rollout_ref.actor.fsdp_config.use_torch_compile=False`.

### Ray dashboard failed to start

**Symptoms:** `Failed to start the dashboard` / `dashboard/client/build/static does not exist`.

**Impact:** Non-fatal. Training continues if Ray worker startup succeeds (`Started a local Ray instance`).

**Fix (optional):** Reinstall Ray with dashboard assets, or ignore if training proceeds.

### Weight transfer / IPC / SHM errors

**Symptoms:** Errors mentioning `ipc_collect`, shared memory, or weight sync between actor and rollout.

**Cause:** `torch_gcu` does not support Python multiprocessing SHM for weight transfer.

**Fix:** PlatformENFLAME sets `is_ipc_supported=True` so verl uses the device-tensor path. Ensure you are on a recent `verl_hardware_plugin` version. If errors persist, check Migration patches are active (`ENFLAME_ENABLE_AUTO_MIGRATION=1`).

### ECCL hang or timeout

**Fix:**

```bash
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

Verify all ranks see the same `TOPS_VISIBLE_DEVICES` via `runtime_env`. For multi-node, check network connectivity between nodes.

### Migration patches not applied

**Symptoms:** CUDA-specific code paths run on GCU; import errors in verl/vLLM.

**Fix:**

```bash
export ENFLAME_ENABLE_AUTO_MIGRATION=1
# Must be set BEFORE: python3 -m verl.trainer.main_ppo
```

Also pass `ENFLAME_ENABLE_AUTO_MIGRATION='1'` in `ray_init.runtime_env.env_vars`.

---

## Diagnostic Commands

```bash
# GCU hardware
python3 -c "import torch; print('gcu:', torch.gcu.is_available(), torch.gcu.device_count())"

# Platform
python3 -c "
import verl_hardware_plugin, verl
from verl.plugin.platform import get_platform
p = get_platform()
print('device_name:', p.device_name)
print('vendor_name:', p.vendor_name)
print('comm_backend:', p.communication_backend_name())
print('ray_resource:', p.ray_resource_name())
"

# Engine registration
python3 -c "
import verl_hardware_plugin
from verl.workers.engine.base import EngineRegistry
print(EngineRegistry._engines['language_model']['fsdp'][('gcu', 'enflame')])
"

# Ray resource view (after ray start)
ray status
```

---

## Known Limitations

- Python multiprocessing shared-memory weight transfer is not supported; verl uses device-tensor transfer instead.
- `torch_gcu` patches `torch.cuda` APIs; platform auto-detection must probe `torch.gcu` before CUDA.
- vLLM on GCU requires `enforce_eager=True` and the stability env vars listed above for reliable rollout.
- Megatron backend requires Megatron-LM-FL installed and on `PYTHONPATH` for all Ray workers.
