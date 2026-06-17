# Cambricon Installation Guide

Last updated: 06/16/2026.

## Prerequisites

- Docker environment
- Network access to pull images and download models

## 1. Pull the Base Image

Please contact Cambricon engineer to get the cambricon_release docker images.

Start a container (example):

```bash
docker_image=cambricon_release_image
docker_name=verl_test
sudo docker run -itd \
    --name ${docker_name} \
    --privileged \
    --network=host \
    --ipc=host \
    --pid=host \
    --cap-add=ALL \
    --shm-size 512G \
    --ulimit memlock=-1 \
    -v /dev/:/dev/ \
    -v /usr/src/:/usr/src/ \
    -v /lib/modules/:/lib/modules/ \
    ${docker_image} \
    /bin/bash

docker exec -it verl_test bash
```

## 2. Prepare Data and Models

### Download model
```bash
cd /root

# Download model (example: Qwen3-0.6B)
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
```

### Download dataset (example: GSM8K)

See [verl dataset guide](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html) to download and process dataset.

## 3. Install verl and verl-hardware-plugin

verl is the RL training framework. For detailed installation options, see: [verl Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html).

verl-hardware-plugin provides the Cambricon hardware platform integration for verl. For detailed information, see: [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin).

```bash
# Install verl
git clone https://github.com/verl-project/verl
cd verl
pip install -e .

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install -e .
```
Other dependency like vllm, ray are already installed in cambricon iamges Now you can run a verl script.
