# FlagOS Installation Guide

Last updated: 06/09/2026.

## Prerequisites

- Supported hardware (NVIDIA GPU, Ascend NPU, MetaX, Iluvatar, etc.)
- Docker environment
- Network access to pull images and download models

## 1. Pull the Base Image

```bash
docker pull harbor.baai.ac.cn/flagscale/flagscale-rl:dev-cu128-py3.12-20260402105433
```

Or use the verl official image, see [verl installation docs](https://verl.readthedocs.io/en/latest/start/install.html).

Start a container (NVIDIA example):

```bash
docker_image=harbor.baai.ac.cn/flagscale/flagscale-rl:dev-cu128-py3.12-20260402105433
docker_name=verl_test
sudo docker run -itd \
    --name ${docker_name} \
    --privileged \
    --network=host \
    --ipc=host \
    --device=/dev/infiniband \
    --pid=host \
    --cap-add=ALL \
    --shm-size 512G \
    --ulimit memlock=-1 \
    --gpus all \
    -v /dev/:/dev/ \
    -v /usr/src/:/usr/src/ \
    -v /lib/modules/:/lib/modules/ \
    -w /workspace \
    ${docker_image} \
    /bin/bash

docker exec -it verl_test bash
```

> **Note:** Adjust Docker options according to your hardware platform. For example, Ascend NPU requires different device mounts.

## 2. Prepare Data and Models

```bash
cd /workspace
conda activate flagscale-RL

# Download model (example: Qwen3-0.6B)
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B

# Download dataset (example: GSM8K)
mkdir gsm8k && cd gsm8k
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"
```

## 3. Install FlagOS Components

### 3.1 Install FlagCX (Communication Library)

FlagCX is the unified communication library for FlagOS, supporting multiple hardware backends. For detailed installation options, see the official documentation: [FlagCX Getting Started](https://github.com/flagos-ai/FlagCX/blob/main/docs/getting_started.md#build-and-installation).

```bash
cd /workspace
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git checkout v0.9.0
git submodule update --init --recursive

# Build (set the flag for your platform: USE_NVIDIA, USE_ASCEND, etc.)
make USE_NVIDIA=1

# Set environment
export FLAGCX_PATH="$PWD"

# Install Python bindings (set FLAGCX_ADAPTOR to your platform: nvidia, ascend, etc.)
cd plugin/torch/
FLAGCX_ADAPTOR=nvidia pip install . --no-build-isolation
```

> **Note:** Replace `USE_NVIDIA=1` and `FLAGCX_ADAPTOR=nvidia` with the appropriate platform flag for your hardware.

### 3.2 Install FlagGems (Operator Library)

FlagGems is the unified operator library for FlagOS, providing optimized operator implementations across different hardware. For detailed installation options, see the official documentation: [FlagGems Getting Started](https://github.com/flagos-ai/FlagGems/blob/master/docs/getting-started.md#quick-installation).

```bash
cd /workspace

# Install build dependencies
pip install -U scikit-build-core>=0.11 pybind11 ninja cmake

# Clone and install
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -v .
```

### 3.3 Install vllm-plugin-FL (Inference Plugin)

vllm-plugin-FL extends vLLM with FlagOS multi-chip backend support for inference. For detailed installation options, see the official documentation: [vllm-plugin-FL README](https://github.com/flagos-ai/vllm-plugin-FL#quick-start).

**Method A: Install from PyPI**

```bash
pip install vllm-plugin-fl==0.1.0+vllm0.13.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

**Method B: Install from source**

```bash
cd /workspace
git clone --branch v0.1.0+vllm0.13.0 https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-fl
pip install --no-build-isolation -v .
```

> **Note:** Requires vLLM v0.13.0. Install from [official release](https://github.com/vllm-project/vllm/tree/v0.13.0) or the fork [vllm-FL](https://github.com/flagos-ai/vllm-FL) if not already installed.

### 3.4 Install TransformerEngine-FL and Megatron-LM-FL

TransformerEngine-FL provides the FlagOS-enabled transformer engine for training across multiple hardware platforms. For detailed information, see: [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL).

**TransformerEngine-FL:**

```bash
cd /workspace

# Method A: Install from PyPI
pip install transformer_engine==0.1.0+te2.9.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

# Method B: Install from source
git clone --branch v0.1.0+te2.9.0 https://github.com/flagos-ai/TransformerEngine-FL.git
cd TransformerEngine-FL
pip install --no-build-isolation -v .
```

Megatron-LM-FL provides the FlagOS-enabled Megatron core for distributed training. For detailed information, see: [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL).

**Megatron-LM-FL:**

```bash
cd /workspace

# Method A: Install from PyPI
pip install megatron_core==0.1.0+megatron0.15.0rc7 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple

# Method B: Install from source
git clone --branch v0.1.0+megatron0.15.0rc7 https://github.com/flagos-ai/Megatron-LM-FL.git
cd Megatron-LM-FL
pip install --no-build-isolation -v .
```

## 4. Install verl and verl-hardware-plugin

verl is the RL training framework. For detailed installation options, see: [verl Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html).

verl-hardware-plugin provides the FlagOS hardware platform integration for verl. For detailed information, see: [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin).

```bash
cd /workspace

# Install verl
pip install verl

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install --no-build-isolation -v -e .
```

## Verification

After installation, verify the components are properly installed:

```bash
python3 -c "import flaggems; print('FlagGems OK')"
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import transformer_engine; print('TransformerEngine-FL OK')"
python3 -c "import megatron.core; print('Megatron-LM-FL OK')"
python3 -c "import verl; print('verl OK')"
python3 -c "import verl_hardware_plugin; print('verl-hardware-plugin OK')"
```
