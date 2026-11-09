# AMD ROCm

ROCm GPUs (MI300X, MI355X) run on verl's built-in `amd` platform (`verl/plugin/platform/platform_rocm.py`). This
package adds ROCm-specific components that do not belong in verl core. It registers no platform.

## Directory Structure

```text
verl_hardware_plugin/accelerators/rocm/
├── __init__.py                       # no platform: the ROCm platform is built into verl
├── registration.py                   # registers the components below on import
└── engines/mori_checkpoint_engine.py # `mori` checkpoint engine (MoRI-IO RDMA + XGMI weight sync)
```

## Components

| Component | Registered as | Doc |
|-----------|---------------|-----|
| MoRI-IO checkpoint engine: trainer → rollout weight sync with one-sided RDMA WRITE striped over all NICs and XGMI fan-out inside each rollout node | checkpoint engine `mori` | [mori_checkpoint_engine.md](./mori_checkpoint_engine.md) |

## Installation

Install verl for ROCm as described in verl's [AMD tutorial](https://github.com/verl-project/verl/tree/main/docs/amd_tutorial),
then this plugin and the component's dependencies:

```bash
pip install --no-build-isolation -e .   # this repository
pip install amd-mori                    # MoRI-IO, for the mori checkpoint engine
```

verl discovers the plugin through the `verl.plugins` entry point, in the driver and in every Ray worker; no
configuration is needed beyond selecting the component (e.g. `checkpoint_engine.backend: mori`).

## Related Documentation

- [verl plugin system](../../development.md)
