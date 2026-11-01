# Google TPU Quick Start

This page verifies the TPU platform and the TorchTitan TPU training engine. For end-to-end GRPO
training (Qwen3-0.6B / Qwen3-4B on GSM8K, TorchTitan trainer + vLLM rollout on TPU v6e), follow verl's
[`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo): it has the image `Dockerfile`, a KubeRay `RayCluster`
and step-by-step commands (see also [Run GRPO](#6-run-grpo-on-tpu) below).

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
pytest tests/accelerators/tpu/test_tpu_engine.py -v
```

Expected: all TPU registration and engine utility cases pass on any host, with or without a TPU attached.

## 5. Run SFT with TorchTitan on TPU

Launch the Multi-Controller SFT trainer (`verl.trainer.sft_trainer`) via `torchrun` with `engine=torchtitan` and `engine.pad_to_length=True`:

```bash
torchrun --nnodes=1 --nproc_per_node=4 -m verl.trainer.sft_trainer \
    engine=torchtitan \
    engine.pad_to_length=True \
    engine.pad_to_length_bucket=256 \
    trainer.device=tpu
```

## 6. Run GRPO on TPU

verl's [`examples/tpu/grpo`](https://github.com/verl-project/verl/tree/main/examples/tpu/grpo) runs GRPO on GSM8K with a TorchTitan trainer on one TPU v6e
slice and a vLLM rollout on another, with `tpu-sync` weight sync between them (see
[vLLM Rollout](./rollout.md) and [Weight Sync](./weight_sync.md)). Its README covers the image, the GKE /
KubeRay cluster (2 v6e-4 slices by default) and the `ray job submit` commands:

- `run_qwen3_0_6b_torchtitan.sh`: 5-step smoke test (Qwen3-0.6B).
- `run_qwen3_4b_torchtitan.sh`: 100-step GRPO run (Qwen3-4B).

Ray ships this plugin to every worker with `py_modules` and `VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin`,
so no `pip install` is needed in the image.
