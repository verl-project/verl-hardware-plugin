# Biren SUPA Installation Guide

## Prerequisites

- Biren Docker images from the [community website](https://developers.birentech.com/)
- Biren release environment with matching driver, SUPA runtime, and `torch_supa`
- Network access to download models and datasets
- A VERL checkout and this plugin checkout

Use one compatible release set for driver, SUPA runtime, `torch_supa`, BCCL, PyTorch, and VERL. Do not mix with CUDA/NVIDIA packages.

## 1. Start the Biren Runtime Environment

Use the Biren Docker image or host environment provided for your driver release. The exact image name, device mounts, and library paths depend on that release. Example container start:

```bash
docker_image="${BIREN_DOCKER_IMAGE:-}"
docker_name=verl_biren
docker run -itd \
    --name ${docker_name} \
    --privileged \
    --network=host \
    --pid=host \
    --shm-size 100g \
    --ulimit memlock=-1 \
    --device /dev/biren/card_0 \
    -v /home:/home \
    ${docker_image}

docker exec -it verl_biren bash
```

Inside the environment, verify the runtime:

```bash
python3 -c 'import torch; import torch_supa; print(torch.supa.is_available(), torch.supa.device_count())'
brsmi
```

The first command should report that SUPA is available and show the visible device count. `brsmi` should list the available Biren devices.

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

Install `torch_supa` and the Biren runtime using the package and installation instructions for the matching Biren release. This repository does not bundle the Biren driver, SUPA SDK, or BCCL runtime.

## 3. Prepare Data and Models

The baseline scripts use Qwen3-0.6B and GSM8K. Set `MODEL_DIR` and `DATA_DIR` to the paths available in your environment, for example:

```text
MODEL_DIR=/ipfs/models/Qwen/Qwen3-0.6B
DATA_DIR=/ipfs/models/gsm8k
```

## Verification

After installation, verify the components are properly installed:

```bash
python3 -c 'import torch; import torch_supa; print(torch.supa.is_available(), torch.supa.device_count())'
python3 -c 'import verl; print("verl OK")'
python3 -c "from verl.plugin.platform import get_platform; p = get_platform(); print(f'device: {p.device_name}'); print(f'vendor: {p.vendor_name}'); print(f'available: {p.is_available()}')"
```

The output should show that SUPA is available, report the visible device count, and confirm the platform resolves to `device: supa`, `vendor: biren`. Then follow the [Quick Start](./quick_start.md) to run a VERL script.
