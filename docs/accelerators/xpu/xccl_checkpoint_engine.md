# XCCL checkpoint engine

Syncs weights from the trainer to the rollout servers over XCCL (oneCCL) on Intel GPUs, for disaggregated setups such as one-step-off or fully-async GRPO.

It is registered as `xccl` and also as `nccl`, the name those trainer configs default to, in the same way verl's HCCL engine does on Ascend. The stock configs therefore need no change on an XPU host.

Code: `verl_hardware_plugin/accelerators/xpu/engines/xccl_checkpoint_engine.py`, registered from `verl_hardware_plugin/accelerators/xpu/registration.py`.

## If upstream changes

- **Name resolution.** If verl picks the checkpoint-engine backend from the platform's `communication_backend_name()`, the `nccl` alias is no longer needed and can be removed. Left in place, it would take precedence over verl's own `nccl` engine once that engine works on XPU.
- **Ray or vLLM transport.** Only four methods touch the transport: `_start_store`, `_create_process_group`, `_destroy_process_group` and the module-level `_broadcast`. If Ray ships an XCCL backend in `ray.util.collective`, or vLLM provides an XPU communicator, a variant can replace those four methods and keep the bucketing and rank layout as they are.
