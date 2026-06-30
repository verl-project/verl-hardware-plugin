# Iluvatar Installation Guide

Last updated: 06/24/2026.

## Prerequisites

- Docker environment
- Network access to pull images and download models

## 1. Pull the Base Image

```bash
docker pull harbor.baai.ac.cn/flagos21-base/iluvatarcorex-4.4.0-ubuntu24-py312-base:20260601v1 
```

Or use the verl official image, see [verl installation docs](https://verl.readthedocs.io/en/latest/start/install.html).

Start a container (e.g. from iluvatarcorex-4.4.0-ubuntu24-py312-base:20260601v1):

```bash
docker_image=harbor.baai.ac.cn/flagos21-base/iluvatarcorex-4.4.0-ubuntu24-py312-base:20260601v1 
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
    -v /dev/:/dev/ \
    -v /usr/src/:/usr/src/ \
    -v /lib/modules/:/lib/modules/ \
    ${docker_image} \
    /bin/bash

docker exec -it verl_test bash
```

## 2. Prepare Data and Models

```bash
cd /root

# Download model (example: Qwen3-0.6B)
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B

# Download dataset (example: GSM8K)
mkdir gsm8k && cd gsm8k
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"
```

## 3. Install verl and verl-hardware-plugin

verl is the RL training framework. For detailed installation options, see: [verl Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html).

verl-hardware-plugin provides the Iluvatar hardware platform integration for verl. For detailed information, see: [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin).

```bash
cd /root

# Install verl (ver > v0.8.0, #6086)
pip install -v -e "git+https://github.com/verl-project/verl.git@ed89419c23653730e95c43954c00e6c24277e1c8#egg=verl" --no-build-isolation

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install --no-build-isolation -v -e .
```

## Verification

After installation, verify the components are properly installed:

```bash
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import transformer_engine; print('TransformerEngine OK')"
python3 -c "import megatron.core; print('Megatron-LM OK')"
python3 -c "import verl; print('verl OK')"
python3 -c "from verl.plugin.platform import get_platform;p = get_platform();print(f'device: {p.device_name}');print(f'vendor: {p.vendor_name}');print(f'available: {p.is_available()}')"
```
