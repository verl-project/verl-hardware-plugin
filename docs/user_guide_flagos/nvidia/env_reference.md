# FlagOS Environment Variables Reference

Last updated: 06/09/2026.

> These environment variables apply to all hardware platforms supported by FlagOS (NVIDIA, Ascend, MetaX, etc.) and are not chip-specific.

---

## verl Platform Variables

Environment variables introduced at the verl framework level for platform selection and training behavior:

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `VERL_ENGINE_DEVICE` | Device type | `cuda` | - |
| `VERL_ENGINE_VENDOR` | Vendor identifier | `flagos` | - |
| `TRAINING_FL_FLAGGEMS_ENABLE` | Enable FlagGems for training | `1` / `0` | `0` |
| `TRAINING_FL_FLAGOS_WHITELIST` | Training operator whitelist | `rmsnorm,layernorm,softmax` | (none) |
| `TRAINING_FL_FLAGOS_BLACKLIST` | Training operator blacklist | `flash_attention` | (none) |
| `USE_FLAGCX` | Enable FlagCX communication library | `1` / `0` | `0` |
| `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO` | Ray GPU detection override | `1` / `0` | `1` |

---

## vllm-plugin-FL Inference Dispatch Variables

Controls operator dispatch during inference. For full details, see [dispatch documentation](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/vllm_fl/dispatch/README.md#environment-variables).

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PREFER_ENABLED` | `true` | Global switch; set `false` to disable all dispatch |
| `VLLM_FL_PREFER` | `flagos` | Preferred backend: `flagos` / `vendor` / `reference` |
| `VLLM_FL_STRICT` | `0` | Strict mode: `1` = fail on error, `0` = fallback |
| `VLLM_FL_PER_OP` | (none) | Per-operator backend order, e.g. `rms_norm=vendor\|flagos` |
| `VLLM_FL_ALLOW_VENDORS` | (none) | Vendor whitelist (comma-separated) |
| `VLLM_FL_DENY_VENDORS` | (none) | Vendor blacklist (comma-separated) |
| `USE_FLAGGEMS` | `true` | Enable/disable FlagGems |
| `VLLM_FL_FLAGOS_WHITELIST` | (none) | FlagGems operator whitelist |
| `VLLM_FL_FLAGOS_BLACKLIST` | (none) | FlagGems operator blacklist |
| `VLLM_FL_OOT_ENABLED` | `1` | Enable OOT operator registration |
| `VLLM_FL_OOT_WHITELIST` | (none) | OOT operator whitelist |
| `VLLM_FL_OOT_BLACKLIST` | (none) | OOT operator blacklist |
| `VLLM_FL_CONFIG` | (none) | YAML config file path (complete override) |
| `VLLM_FL_PLATFORM` | (auto) | Force platform: `cuda` / `ascend` |
| `VLLM_FL_LOG_LEVEL` | `INFO` | Log level |
| `VLLM_PLUGINS` | (none) | Specify `fl` when multiple plugins are installed |
| `FLAGCX_PATH` | (none) | FlagCX installation path; enables flagcx communication backend |

---

## TransformerEngine-FL Training Dispatch Variables

Controls backend selection at the TransformerEngine level. For full details, see [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL).

| Variable | Default | Description |
|----------|---------|-------------|
| `TE_FL_PREFER` | `flagos` | Preferred backend: `flagos` / `vendor` / `reference` |
| `TE_FL_PREFER_VENDOR` | `0` | Prefer vendor backend (legacy) |
| `TE_FL_STRICT` | `0` | Strict mode; fail without fallback |
| `TE_FL_ALLOW_VENDORS` | (none) | Vendor whitelist, e.g. `nvidia,amd` |
| `TE_FL_DENY_VENDORS` | (none) | Vendor blacklist |
| `TE_FL_PER_OP` | (none) | Per-operator backend order, e.g. `rmsnorm_fwd=vendor:cuda\|default` |
| `TE_FL_PLUGIN_MODULES` | (none) | Plugin modules to load |
| `TE_FL_SKIP_CUDA` | `0` | Skip CUDA backend during build |
| `TEFL_LOG_LEVEL` | `INFO` | Log level |

---

## Common Configuration Examples

Minimal setup (default FlagGems + FlagCX):

```bash
export VERL_ENGINE_DEVICE='cuda'
export VERL_ENGINE_VENDOR='flagos'
export TRAINING_FL_FLAGGEMS_ENABLE=1
export FLAGCX_PATH=/workspace/FlagCX/
```

Disable FlagGems and use native CUDA operators:

```bash
export USE_FLAGGEMS=0
export TRAINING_FL_FLAGGEMS_ENABLE=0
```

Disable FlagCX and use NCCL:

```bash
export USE_FLAGCX=0
unset FLAGCX_PATH
```

Debug mode:

```bash
export VLLM_FL_LOG_LEVEL=DEBUG
export TEFL_LOG_LEVEL=DEBUG
```
