# Cambricon Quick Start

Last updated: 06/16/2026.

This guide walks you through running a GRPO training job with verl on the Cambricon platform. Make sure you have completed the [Installation Guide](./install_guidance.md) first.

## Environment Setup
1. Please use Cambricon release docker images to run verl and make sure you are in pytorch_infer env.

2. Start ray cluster:
```bash
ray start --head --dashboard-host=0.0.0.0
```
3. Run verl scripts

We recommend using Ray to lanuch the task to make sure all env vars are set correctly.

Add necessary environment variables in runtime_env.yaml
```bash
working_dir: ./
excludes: ["/.git/"]
env_vars:
  TORCH_NCCL_AVOID_RECORD_STREAMS: "1"
  RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO: "0"
  VERL_USE_EXTERNAL_MODULES: "verl_hardware_plugin"
```
Run scripts in [verl examples](https://github.com/verl-project/verl/tree/main/examples)


## Next Steps

- See the [Environment Variables Reference](./env_reference.md) for fine-grained control over operator dispatch and backend selection.
- See the [FAQ](./faq.md) for troubleshooting common issues.
