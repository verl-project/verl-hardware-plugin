# Google TPU Installation Guide

## Prerequisites

- A Cloud TPU VM or a GKE node pool with TPU v6e chips attached.
- A container image providing `torch`, `torch_tpu` (`pip install torch-tpu`, 0.2.0 or newer), and `libtpu`.
  `torch_tpu` registers both the `tpu` device type and the `tpu_dist` distributed backend; `torch.tpu`
  does not exist until it is imported. The `Dockerfile` in verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo) builds a tested image
  (Ray, `torch_tpu`, JAX/`libtpu`, `tpu-sync`, torchtitan, vLLM + vllm-torchtpu).
- Network access to download models and datasets.
- A verl checkout and this plugin checkout.

Note that `torch` in a TPU image is normally a **CPU build** — TPU execution goes through PJRT, not
CUDA. Do not install CUDA builds of torch alongside it.

## 1. Install verl and verl-hardware-plugin

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

## 2. Verify the TPU Runtime

```bash
python3 -c 'import torch, torch_tpu; print(torch.accelerator.is_available(), torch.accelerator.device_count())'
```

Expected: `True` followed by the number of chips visible to this process.

If this reports `False`, check that `PJRT_DEVICE=TPU` is set and that the chip devices are visible
to the container (`/dev/vfio` must be mounted, and the container typically needs `--privileged`).

## 3. Verify the Platform Resolves

```bash
VERL_PLATFORM=tpu python3 -c '
from verl.plugin.platform.platform_manager import get_platform
p = get_platform()
print(p.device_name, p.vendor_name, p.communication_backend_name())
'
```

Expected output: `tpu google tpu_dist`

## Environment Variables

The platform sets these on every trainer worker: `get_worker_env_vars()` on a Ray worker, and
`configure_torchrun_tpu_env()` under `torchrun` when `TORCH_TPU_SLICEBUILDER_ADDRESSES` is not set yet.
They are listed here because they are useful when debugging a slice by hand.

| Variable | Purpose |
|----------|---------|
| `VERL_PLATFORM` | Set to `tpu` to select this platform explicitly |
| `PJRT_DEVICE` | Set to `TPU` (`torchrun` only) |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | `host:port` of every process of the slice in rank order, port `8471 + local_rank` |
| `TORCH_TPU_TOPOLOGY` | Mesh topology, e.g. `2,4,1` for `v6e-8`, or `x,y,z,cores` on TPU 7x |
| `TPU_WORKER_HOSTNAMES` | Unique worker hostnames in rank order |
| `LOCAL_RANK` / `LOCAL_WORLD_SIZE` | The process's index on its host and the processes per host (`torchrun` sets them itself) |
| `CHIPS_PER_HOST` | Chips per host of the pool; `VERL_TPU_CHIPS_PER_HOST` overrides |
| `TPU_ACCELERATOR_TYPE` | Read, not set (GKE sets it): tells TPU 7x apart from v6e |

From these and `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` (set by verl or `torchrun`), `torch_tpu`
(>= 0.2.0) derives libtpu's per-process variables when it starts, overwriting any value set before:
`CLOUD_TPU_TASK_ID` (the process's rank), `TPU_VISIBLE_DEVICES` / `TPU_VISIBLE_CHIPS` (its local rank, unless
`TPU_VISIBLE_DEVICES` already names one device), `TPU_PROCESS_ADDRESSES`, `TPU_PROCESS_PORT`, `TPU_HOST_BOUNDS` /
`TPU_PROCESS_BOUNDS` (the topology) and `TPU_CHIPS_PER_HOST_BOUNDS` / `TPU_CHIPS_PER_PROCESS_BOUNDS` (`1,1,1`, or
`1,1,1,1` on TPU 7x). The rollout's vLLM workers still get those from the platform: a TP=1 worker starts
`torch_tpu` in single-device mode, which derives only the visible device.

## Next Steps

Follow the [Quick Start](./quick_start.md).
