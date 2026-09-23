# Google TPU Quick Start

> **TPU support is under active development and testing.** The TPU platform and the
> TorchTitan TPU training engine are registered and can be verified on any host, but the
> end-to-end training recipes are not published yet — see the status note in the
> [User Guide](./README.md). This page covers what you can verify today.

## 1. Select the Platform

Always select TPU explicitly:

```bash
export VERL_PLATFORM=tpu
export PJRT_DEVICE=TPU
```

Auto-detection also works on a TPU host, because each registered platform is probed in turn and only
the TPU probe succeeds. But that outcome depends on registration order, which is not a stable
contract. `VERL_PLATFORM=tpu` is the supported way to select TPU.

## 2. Verify Platform Resolution

```bash
python3 -c '
from verl.plugin.platform.platform_manager import get_platform
p = get_platform()
print("device:   ", p.device_name)
print("vendor:   ", p.vendor_name)
print("backend:  ", p.communication_backend_name())
print("ray res:  ", p.ray_resource_name())
print("ipc:      ", p.is_ipc_supported())
print("colocate: ", p.supports_colocated_worker_groups())
'
```

Expected output:

```text
device:    tpu
vendor:    google
backend:   tpu_dist
ray res:   TPU
ipc:       False
colocate:  False
```

## 3. Verify Ray Resource Requests

The platform requests chips as a custom Ray resource named `TPU`, not as `num_gpus`:

```bash
python3 -c '
from verl.plugin.platform.platform_manager import get_platform
print(get_platform().ray_resource_options(4))
'
```

Expected output: `{'resources': {'TPU': 4}}`

Your Ray cluster must advertise a `TPU` resource for scheduling to succeed. On KubeRay this comes
from the TPU node pool's resource annotations.

## 4. Verify TorchTitan TPU Engine Registration

```bash
pytest tests/test_plugin_registration.py -k tpu -v
pytest tests/test_tpu_engine.py -v
```

Expected: all TPU registration and engine utility cases pass on any host, with or without a TPU attached.
