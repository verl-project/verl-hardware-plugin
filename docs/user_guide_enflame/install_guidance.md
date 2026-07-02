# Enflame GCU Installation Guide

This guide covers prerequisites, software installation, and environment verification for running verl on Enflame GCU.

## Prerequisites

- Enflame GCU hardware with driver and runtime installed
```bash
  # install kmd and runtime
   ./TopsPlatform_*_deb_amd64.run -y
  # install torch_gcu whl
   pip install  x86_64-linux-rel/python_packages/torch_gcu-2.9.1+3.7.1.16-cp312-cp312-linux_x86_64.whl
  # install migration whl
   pip install  x86_64-linux-rel/python_packages/migration-3.7.20260519+gcu-py3-none-any.whl
  # install triton_gcu whl
   pip install  x86_64-linux-rel/python_packages/triton_gcu-3.5.1+1.0.20260522.cc.1.5.1.101903-py3-none-any.whl
  # install vllm_gcu whl
  pip install  x86_64-linux-rel/python_packages/vllm_gcu-0.14.1+3.7.20260521-cp39-abi3-linux_x86_64.whl 
```
- Linux host with network access (for pip/git and dataset downloads)
- Python 3.10+ (3.12 verified)
- Sufficient shared memory (`--shm-size` ≥ 64G recommended in Docker)

Verify GCU visibility on the host:

```bash
# Vendor-specific SMI tool (name may vary by driver version)
efsmi
# or
echo $TOPS_VISIBLE_DEVICES
python3 -c "import torch_gcu, torch; print(torch.gcu.device_count())"
```

## 1. Install Vendor Runtime (torch_gcu + ECCL)

Install `torch_gcu`, ECCL, and the Enflame driver stack according to your vendor documentation or pre-built container image.

After installation, verify:

```bash
python3 -c "
import torch_gcu
import torch
print('gcu available:', torch.gcu.is_available())
print('gcu count:', torch.gcu.device_count())
"
```

> **Note:** `torch_gcu` replaces `torch.cuda.*` with `torch.gcu.*` at import time and sets the default distributed backend to ECCL. This is expected behavior.

## 2. Prepare Workspace

```bash
mkdir -p /workspace && cd /workspace
# Use conda/venv as preferred
python3 -m venv verl-gcu-env
source verl-gcu-env/bin/activate
```

## 3. Install verl and verl-hardware-plugin

```bash
cd /workspace

# Install verl (see https://verl.readthedocs.io/en/latest/start/install.html)
pip install verl

# Install this plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install --no-build-isolation -e .
```

The plugin is auto-discovered via the `verl.plugins` entry point. For development, you can also set:

```bash
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
```

## 4. Install Migration (Recommended)

Migration applies runtime patches for verl, vLLM, and related libraries on Enflame GCU. Install before your first training run:

```bash
pip install migration   # or vendor-provided wheel
export ENFLAME_ENABLE_AUTO_MIGRATION=1
```

Migration activates on import when `ENFLAME_ENABLE_AUTO_MIGRATION` is `1`, `true`, or `yes`. It must be set **before** importing verl.

Optional debug paths:

```bash
export ENFLAME_MIGRATION_CACHE_DIR=./migration_cache
export ENFLAME_MIGRATION_DUMP_DIR=./migration_debug
export ENFLAME_MIGRATION_LOG_LEVEL=INFO
```

## 5. Install Training and Rollout Components

### 5.1 TransformerEngine-FL (FSDP training)

Required for FSDP training with TE-backed kernels. See [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL).

```bash
# PyPI (example version; match your vLLM/verl stack)
pip install transformer_engine --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

# Or from source
git clone https://github.com/flagos-ai/TransformerEngine-FL.git
cd TransformerEngine-FL && pip install --no-build-isolation -v .
```

Vendor-native TE path (without FlagOS):

```bash
export TE_FL_SKIP_CUDA=1
export TE_FL_PREFER=vendor
export TE_FL_STRICT=0
```

### 5.2 vllm-plugin-FL + vLLM (rollout)

Required when `actor_rollout_ref.rollout.name=vllm`. See [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL).

```bash
pip install vllm-plugin-fl --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
export VLLM_FL_OOT_ENABLED=1
```

Install a vLLM version compatible with your `vllm-plugin-fl` build (e.g. v0.13.0).

### 5.3 Megatron-LM-FL (optional, Megatron backend only)

Only needed if you use the Megatron training backend instead of FSDP:

```bash
git clone https://github.com/flagos-ai/Megatron-LM-FL.git
cd Megatron-LM-FL && pip install --no-build-isolation -v .
export PYTHONPATH="/workspace/Megatron-LM-FL:${PYTHONPATH:-}"
```

FSDP-only GRPO does not require Megatron-LM-FL. A log line such as `Failed to register Enflame Megatron engines` on the driver process is harmless if Megatron is not installed yet; Ray workers with Megatron-LM-FL on `PYTHONPATH` will register successfully.

## 6. Prepare Data and Model

```bash
cd /workspace

# Model: Qwen3-0.6B (baseline validation model)
pip install modelscope   # if not installed
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B

# Dataset: GSM8K
mkdir -p gsm8k && cd gsm8k
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"
```

## 7. Environment Verification

Run all checks before launching training:

```bash
export ENFLAME_ENABLE_AUTO_MIGRATION=1
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_PLATFORM=enflame

# 1. Hardware
python3 -c "import torch; assert torch.gcu.is_available(); print('torch.gcu OK, count=', torch.gcu.device_count())"

# 2. Plugin + platform
python3 -c "
import verl_hardware_plugin
import verl
from verl.plugin.platform import get_platform
p = get_platform()
assert p.device_name == 'gcu' and p.vendor_name == 'enflame'
print('platform OK:', p.device_name, p.vendor_name)
"

# 3. Engines
python3 -c "
import verl_hardware_plugin
from verl.workers.engine.base import EngineRegistry
key = ('gcu', 'enflame')
assert key in EngineRegistry._engines['language_model']['fsdp'], 'fsdp_enflame not registered'
print('fsdp_enflame engine OK:', EngineRegistry._engines['language_model']['fsdp'][key].__name__)
"

# 4. Optional components
python3 -c "import transformer_engine; print('TransformerEngine OK')" 2>/dev/null || echo "TransformerEngine: skip or install"
python3 -c "import vllm; print('vLLM OK')" 2>/dev/null || echo "vLLM: skip or install"
```

All required checks should pass without assertion errors.

## 8. Run Acceptance Baseline

Validate end-to-end training with the standard baseline script:

```bash
cd /path/to/verl-hardware-plugin
bash scripts/baseline_grpo_gsm8k_enflame.sh
```

Compare `critic/rewards/mean` against the NVIDIA reference:
https://swanlab.cn/@heavyrain/verl_grpo_gsm8k_math/runs/8h196r8o/chart

See [Quick Start](./quick_start.md) for details.

## 9. Docker Deployment (Optional)

If you use a vendor-provided Enflame GCU container, ensure:

- GCU devices are mounted (`/dev/` or vendor-specific device nodes)
- `TOPS_VISIBLE_DEVICES` is set inside the container
- Shared memory is large enough (`--shm-size 64g` or higher)
- `ENFLAME_ENABLE_AUTO_MIGRATION=1` is exported in the container entrypoint or training script

Adjust mount flags according to your vendor image documentation.

## Next Steps

Proceed to [Quick Start](./quick_start.md) to run the GRPO baseline training job.
