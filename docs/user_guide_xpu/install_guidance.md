# Installation Guide

Intel XPU support ships as this external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree — see
[`verl/plugin/platform/README.md`](https://github.com/verl-project/verl/blob/main/verl/plugin/platform/README.md)
in verl-core for the plugin pattern this follows.

## Prerequisites

- PyTorch with XPU support (`torch.xpu.is_available() == True`)
- vLLM with XPU kernels — either the prebuilt wheel from vLLM's XPU wheel
  index (`pip install "vllm==<version>+xpu" --extra-index-url
  https://wheels.vllm.ai/<version>/xpu`; see
  [`docker/intel_gpu/Dockerfile.intel_gpu`](../../docker/intel_gpu/Dockerfile.intel_gpu)
  for the exact pinned command) or built from source with
  `VLLM_TARGET_DEVICE=xpu`. The default PyPI `pip install vllm` wheel does
  not ship XPU kernels.
- oneCCL runtime for the `xccl` distributed backend

**verl-core version note:** the `attention_utils_module()`,
`profiler_markers()`, and `dist_profiler_cls()` hooks this plugin implements
only exist on verl-core builds that include
[verl-project/verl#7917](https://github.com/verl-project/verl/pull/7917),
which has not merged yet. Against a `verl-project/verl:main` checkout
without it, this plugin still installs and runs, but those three hooks
silently no-op: attention padding falls back to the generic implementation
and `profiler.tool: vtune` will not resolve to a profiler. The `xccl`
reduce_avg workaround is unaffected either way — it's applied entirely from
this plugin's own `fsdp_xpu.py` engine, not through a `PlatformBase` hook.
Use the pinned ref in [`docker/intel_gpu/`](../../docker/intel_gpu/) for a
known-good combination until #7917 merges.

## 1. Install verl and verl-hardware-plugin

```bash
# Install verl (or a checkout that already includes #7917 — see note above)
git clone https://github.com/verl-project/verl.git
cd verl
pip install -e .

# Install verl-hardware-plugin
git clone https://github.com/verl-project/verl-hardware-plugin.git
cd verl-hardware-plugin
pip install -e .
```

No environment variable is required beyond this. The plugin is
auto-discovered by verl through the `verl.plugins` setuptools entry_points
group declared in this repo's `pyproject.toml`
(`[project.entry-points."verl.plugins"] hardware = "verl_hardware_plugin"`),
which verl loads by default (`VERL_USE_EXTERNAL_PLUGINS=auto`). Set
`VERL_USE_EXTERNAL_PLUGINS=none` to disable discovery, e.g. to isolate a bug
to this plugin, or `VERL_PLATFORM=intel` to force platform selection instead
of relying on auto-detection.

## 2. Verify the Install

```bash
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print('device:', p.device_name, '/ vendor:', p.vendor_name)
"
```

Expected on Intel GPU: `device: xpu / vendor: intel`. If it instead falls
back to `nvidia`, the plugin was not discovered — check that `pip install`
completed without error and that no `VERL_USE_EXTERNAL_PLUGINS=none` is set
in the environment.

## Docker

A prebuilt image definition is at
[`docker/intel_gpu/`](../../docker/intel_gpu/) in this repo — it clones
verl-core at a pinned ref, installs vLLM from its prebuilt XPU wheel index,
and installs this plugin on top. See
[`docker/intel_gpu/README.md`](../../docker/intel_gpu/README.md)
for build/run instructions and the full software stack table. This is the
fastest way to get a known-good verl-core + plugin combination, since it
sidesteps the verl-core version note above.

## Next Steps

Follow the [Quick Start](./quick_start.md).
