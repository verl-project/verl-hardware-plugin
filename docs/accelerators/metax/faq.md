# MetaX FAQ and Troubleshooting

## Common Questions

### Q: How do I select the MetaX platform?

Set the environment variable:

```bash
export VERL_PLATFORM=metax
```

Or let auto-detection handle it — ensure `mx-smi` is available, and verl will detect MetaX hardware automatically.

### Q: What is the difference between `metax` and `nvidia` in CUDA-compatible mode?

Both MetaX and NVIDIA use `torch.cuda` underneath (device name: `cuda`). The `vendor_name` distinguishes them:
- `metax` → MetaX GPU hardware (detected via `mx-smi`)
- `nvidia` → NVIDIA GPU hardware (detected via `nvidia-smi`)

The vendor distinction ensures the correct platform-specific engine is selected during training.

### Q: How does verl distinguish MetaX from NVIDIA?

During auto-detection, verl runs the SMI command (`mx-smi` for MetaX, `nvidia-smi` for NVIDIA). Since both are CUDA-compatible and `torch.cuda.is_available()` returns `True` on both, the SMI check is the only reliable way to distinguish them.

You can bypass auto-detection by setting `VERL_PLATFORM=metax` explicitly.

### Q: What communication backend does MetaX use?

MetaX uses  **MCCL** for distributed communication. 

### Q: Does MetaX support FSDP and Megatron training?

Yes. Both FSDP and Megatron engines are registered for MetaX:

- **FSDP**: `FSDPMetaXEngineWithLMHead` / `FSDPMetaXEngineWithValueHead` (backend: `fsdp`, `fsdp2`)
- **Megatron**: `MegatronMetaXEngineWithLMHead` (backend: `megatron`)

---

## Troubleshooting

### CUDA out of memory during training

- Enable parameter offload and optimizer offload in FSDP config:
  ```
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  ```
- Reduce `ppo_max_token_len_per_gpu` or `train_batch_size`.
- Lower `rollout_gpu_mem_util` to reserve more memory for training.


### Ray device detection issues in Docker

If Ray cannot detect devices inside a `--privileged` Docker container:

```bash
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
```

This disables the override that prevents device detection when the visibility environment variable resolves to zero devices.

### Platform not detected during import

If `verl_hardware_plugin` imports but MetaX is not registered, check the logs:

```bash
export VERL_LOGGING_LEVEL=DEBUG
python3 -c "import verl_hardware_plugin"
```

Look for messages like:
- `Registered platform: metax (cuda)` — Platform registered successfully
- `MetaX platform not registered: <error>` — Import failed, check the error message

### Verifying MetaX GPU accessibility

```bash
# Check torch CUDA access
python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check mx-smi
mx-smi -L

# Check from within verl
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print(f'Platform: {p.vendor_name}')
print(f'Device: {p.device_name}')
print(f'Devices: {p.device_count()}')
print(f'Available: {p.is_available()}')
"
```
