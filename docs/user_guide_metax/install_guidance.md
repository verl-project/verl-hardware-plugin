## Prerequisites

- MetaX GPU hardware
- Docker environment
- Network access to pull images and download models

## 1. Pull the Base Image

Visit the MetaX Docker Hub page  (e.g., https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=verl:0.7.1-maca.ai3.5.3.3-torch2.8-py310-ubuntu22.04-amd64)
Copy the docker pull command and run it in your terminal.

Start a container (e.g., verl:0.7.1-maca.ai3.5.3.3-torch2.8-py310-ubuntu22.04-amd64):

```bash
docker_image=verl:0.7.1-maca.ai3.5.3.3-torch2.8-py310-ubuntu22.04-amd64
docker_name=verl_test
sudo docker run -itd \
    --name ${docker_name} \
    --net=host \
    --uts=host \
    --ipc=host \
    --privileged=true \
    --group-add video \
    --shm-size 100gb \
    --ulimit memlock=-1 \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    --device=/dev/dri \
    --device=/dev/mxcd \
    --device=/dev/infiniband \
    ${docker_image} \
    /bin/bash

docker exec -it verl_test bash
```

> **Note:** `/dev/mxcd` is the MetaX compute device and `/dev/dri` provides GPU rendering access — both are required for MetaX GPU workloads. Ensure `mx-smi` is available inside the container for hardware auto-detection. Add `-v` mounts for your data and model directories as needed (e.g., `-v /data/share/:/data/share/`).

## 2. Prepare Data and Models

```bash
cd /workspace

# Download model (example: Qwen3-8B)
modelscope download --model Qwen/Qwen3-8B --local_dir ./Qwen3-8B

# Download dataset (example: GSM8K)
mkdir gsm8k && cd gsm8k
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/train.parquet"
wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/rl/datasets/gsm8k/test.parquet"
```

## 3. Install verl and verl-hardware-plugin

verl is the RL training framework. For detailed installation options, see: [verl Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html).

verl-hardware-plugin provides the MetaX hardware platform integration for verl. For detailed information, see: [verl-hardware-plugin](https://github.com/verl-project/verl-hardware-plugin).

```bash
cd /workspace
git clone https://github.com/verl-project/verl verl_main
cd verl_main
pip install --no-build-isolation --no-dependencies -v -e .

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install --no-build-isolation -v -e .
```

## 4. Platform Configuration

Set the MetaX platform before launching training:

```bash
export VERL_PLATFORM=metax
```

Or let auto-detection handle it (requires `mx-smi` in PATH inside the container).

## Verification

After installation, verify the components are properly installed:

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import ray; ray.init(address='auto'); res=ray.cluster_resources(); print('Ray OK' if 'GPU' in res and res['GPU']>0 else 'Failed'); ray.shutdown()"
python3 -c "import transformer_engine; print('TransformerEngine OK')"
python3 -c "import megatron.core; print('Megatron-LM OK')"
python3 -c "import verl; print('verl OK')"
python3 -c "from verl.plugin.platform import get_platform;p = get_platform();print(f'device: {p.device_name}');print(f'vendor: {p.vendor_name}');print(f'available: {p.is_available()}')"
```

Verify MetaX hardware detection:

```bash
mx-smi -L                                         # List MetaX GPUs
python3 -c "
import subprocess, sys
ret = subprocess.run(['mx-smi', '-L'], capture_output=True, text=True)
if ret.returncode == 0:
    print('MetaX platform detected')
else:
    print('MetaX platform not detected')
    sys.exit(1)
"
```
