# Installation Guide

Intel XPU support ships as this external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree — see
[`verl/plugin/platform/README.md`](https://github.com/verl-project/verl/blob/main/verl/plugin/platform/README.md)
in verl-core for the plugin pattern this follows.

## Prerequisites

- PyTorch with XPU support (`torch.xpu.is_available() == True`)
- vLLM with XPU kernels — either the prebuilt wheel from vLLM's XPU wheel
  index (`pip install "vllm==<version>+xpu" --extra-index-url
  https://wheels.vllm.ai/<version>/xpu`) or built from source with
  `VLLM_TARGET_DEVICE=xpu`. The default PyPI `pip install vllm` wheel does
  not ship XPU kernels.
- oneCCL runtime for the `xccl` distributed backend

**verl-core version note:** the one `PlatformBase` hook this plugin
implements, `profiler_markers()`, was merged into `verl-project/verl:main` on
2026-10-08. A checkout from before that date still installs and runs this
plugin, but the hook silently no-ops on it: ITT ranges are not emitted, and
the `tool_config` allowlist in `engine_workers.py` that the same change
lifted drops a `tool_config.vtune` entry. `VtuneProfiler` itself is
registered by a plugin-side monkeypatch and works either way. Attention
padding needs nothing from this plugin: verl core uses
its own pure-PyTorch `attention_padding_utils` whenever `flash_attn` is
unavailable. The `xccl` reduce_avg workaround is likewise unaffected — it's a
plugin-side monkeypatch over `torch.distributed.all_reduce`
(`accelerators/xpu/patches/reduce_avg_allreduce_patch.py`), not a
`PlatformBase` hook.

## 1. Install verl and verl-hardware-plugin

```bash
# Install verl (needs the profiler_markers() hook, on main since 2026-10-08 — see note above)
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

## Next Steps

Follow the [Quick Start](./quick_start.md).
