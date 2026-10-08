# MUSA Installation Guide

## Prerequisites

- A MUSA Docker image with the matching driver/runtime, `torch_musa`, MCCL,
  SGLang, and other MUSA dependencies.
- Network access to download models and datasets.
- A VERL checkout and this plugin checkout.

The standard MUSA images already include SGLang and the external Megatron-LM
MUSA patch (usually `/home/megatron-lm-musa-patch`). Other runtime dependencies
such as Ray are also normally pre-installed. Do not add CUDA versions of these
packages, as they may override the MUSA packages.

## 1. Start the MUSA Docker Image

Use the MUSA release image provided for your hardware. The exact image name and
device mounts depend on the driver release; the following is a generic example:

```bash
docker_image="${MUSA_DOCKER_IMAGE:-}"
docker_name="${MUSA_DOCKER_NAME:-verl_musa}"

docker container create \
    --name "${docker_name}" \
    --privileged \
    --net host \
    --pid=host \
    --shm-size 100g \
    --ulimit memlock=-1 \
    -v /home:/home \
    -it \
    "${docker_image}" \
    /bin/bash

docker start -ai "${docker_name}"
```

Inside the container, verify that the pre-installed components are available:

```bash
ls /home
python3 -c 'import torch; import sglang; print(torch.musa.is_available())'
```

A public image is:

`registry.mthreads.com/mcctest/training-suite:v2.1.7.rc3-ut-verify`

## 2. Install verl and verl-hardware-plugin

```bash
# Install verl
git clone https://github.com/verl-project/verl.git
cd verl
pip install -e .

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install -e .
```

## 3. Prepare Data and Models

The baseline scripts use Qwen3-0.6B and GSM8K. Set `MODEL_DIR` and `DATA_DIR`
to the paths available in your environment, for example:

```text
MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B
DATA_DIR=/ipfs/models/gsm8k
```

## 4. Verify the Environment

```bash
python3 -c 'import torch; print(torch.musa.is_available(), torch.musa.device_count())'
```

The output should show that MUSA is available and report the visible device
count. Then follow the [Quick Start](./quick_start.md) to run a VERL script.
