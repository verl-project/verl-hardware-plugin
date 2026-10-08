# FlagOS FAQ and Troubleshooting

Last updated: 06/09/2026.

## Common Questions

### Q: How do I disable FlagGems and use native device operators?

For inference (vllm-plugin-FL):

```bash
export USE_FLAGGEMS=0
```

For training:

```bash
export TRAINING_FL_FLAGGEMS_ENABLE=0
```

### Q: How do I disable FlagCX and use the native communication backend?

Unset the FlagCX path:

```bash
unset FLAGCX_PATH
```

This will fall back to the device-native communication backend (e.g., NCCL for NVIDIA, HCCL for Ascend).

### Q: How do I select which vLLM plugin to use?

If multiple vLLM plugins are installed, specify the FlagOS plugin explicitly:

```bash
export VLLM_PLUGINS='fl'
```

### Q: Can I use only specific FlagGems operators?

Yes. Use a whitelist to enable only certain operators:

```bash
# Inference: only use FlagGems for these ops
export VLLM_FL_FLAGOS_WHITELIST="silu_and_mul,rms_norm"

# Training: only use FlagGems for these ops
export TRAINING_FL_FLAGOS_WHITELIST="rmsnorm,layernorm,softmax"
```

Or use a blacklist to exclude certain operators:

```bash
# Inference
export VLLM_FL_FLAGOS_BLACKLIST="mm,to_copy,zeros"

# Training
export TRAINING_FL_FLAGOS_BLACKLIST="flash_attention"
```

Whitelist and blacklist are mutually exclusive — do not set both at the same time.

### Q: What hardware platforms are supported?

| Chip Vendor | Status |
|-------------|--------|
| NVIDIA | Supported |
| Ascend | TBD |
| MetaX | TBD |
| Pingtouge-Zhenwu | TBD |
| Iluvatar | TBD |
| Tsingmicro | TBD |
| Moore Threads | TBD |
| Hygon | TBD |

---

## Troubleshooting

### FlagGems compilation errors

If you encounter errors during FlagGems build:

1. Ensure build dependencies are up to date:
   ```bash
   pip install -U scikit-build-core>=0.11 pybind11 ninja cmake
   ```

2. Make sure the device toolkit is properly installed (e.g., `CUDA_HOME` for NVIDIA):
   ```bash
   export CUDA_HOME=/usr/local/cuda
   ```

### FlagCX import errors

If FlagCX fails to load:

1. Verify the `FLAGCX_PATH` points to the correct installation:
   ```bash
   ls $FLAGCX_PATH/build/lib/libflagcx.so
   ```

2. Ensure submodules were initialized:
   ```bash
   cd $FLAGCX_PATH
   git submodule update --init --recursive
   ```

### Operator dispatch not using expected backend

Enable debug logging to trace dispatch decisions:

```bash
export VLLM_FL_LOG_LEVEL=DEBUG
export VLLM_FL_DISPATCH_DEBUG=1
```

Check the logs for which backend was selected for each operator and why.

### TransformerEngine-FL fallback behavior

By default, TransformerEngine-FL prefers the FlagOS backend (`TE_FL_PREFER=flagos`). If an operator is unavailable, it falls back to vendor → reference. To enforce strict mode (no fallback):

```bash
export TE_FL_STRICT=1
```

### Out of memory during training

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
