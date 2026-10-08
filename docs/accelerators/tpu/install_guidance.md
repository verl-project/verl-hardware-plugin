# Google TPU Installation Guide

## Prerequisites

- A Cloud TPU VM or a GKE node pool with TPU v6e chips attached.
- A container image providing `torch`, `torch_tpu`, and `libtpu`. `torch_tpu` registers both the
  `tpu` device type and the `tpu_dist` distributed backend; `torch.tpu` does not exist until it is
  imported.
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
python3 -c 'import torch, torch_tpu; print(torch.tpu.is_available(), torch.tpu.device_count())'
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

These are set automatically per worker by `get_worker_env_vars()` once verl core supports the hook.
They are listed here because they are useful when debugging a slice by hand.

| Variable | Purpose |
|----------|---------|
| `VERL_PLATFORM` | Set to `tpu` to select this platform explicitly |
| `PJRT_DEVICE` | Set to `TPU` for the PJRT runtime |
| `TPU_VISIBLE_CHIPS` | Per-worker chip index; also the source of the local rank |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | `host:port` list for the slice builder mesh |
| `TPU_PROCESS_ADDRESSES` | `host:port` list for the TPU process mesh |
| `TPU_PROCESS_PORT` | This process's port, `8471 + local_rank` |
| `CLOUD_TPU_TASK_ID` | Host index within the slice |
| `TPU_WORKER_HOSTNAMES` | Unique worker hostnames in rank order |
| `TORCH_TPU_TOPOLOGY` / `TPU_HOST_BOUNDS` | 3D mesh topology, e.g. `2,4,1` for `v6e-8` |
| `TPU_ACCELERATOR_TYPE` | Used to determine chip generation and HBM size |
| `TPU_CHIPS_PER_HOST_BOUNDS` | Chips-per-host bounds for the slice, always `1,1,1` |
| `CHIPS_PER_HOST` | Hardcoded to `"4"` regardless of detected topology |

## Next Steps

Follow the [Quick Start](./quick_start.md).
