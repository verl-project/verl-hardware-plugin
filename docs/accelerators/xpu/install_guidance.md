# Installation Guide

Intel XPU support ships as this external plugin package
(`verl_hardware_plugin`), not baked into the verl source tree — see
[`verl/plugin/platform/README.md`](https://github.com/verl-project/verl/blob/main/verl/plugin/platform/README.md)
in verl-core for the plugin pattern this follows.

## Prerequisites

- An Intel GPU with the Level Zero driver stack on the host (compute-runtime >= 26.18 and Level Zero).
  **A system oneAPI install is not required**: PyTorch's XPU wheel and vLLM's XPU wheel bring their own SYCL
  runtime and oneCCL (the `xccl` backend). This was checked by hiding `/opt/intel/oneapi` and the oneAPI
  environment variables: platform detection, an `xccl` all_reduce and a GRPO training step all still worked.
- PyTorch with XPU support (`torch.xpu.is_available() == True`).
- vLLM with XPU kernels, from vLLM's XPU wheel index (`pip install "vllm==<version>+xpu"
  --extra-index-url https://wheels.vllm.ai/<version>/xpu`) or built from source with
  `VLLM_TARGET_DEVICE=xpu`. The default PyPI `pip install vllm` wheel does not ship XPU kernels.

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

Pick one. Options B and C install torch 2.15.0+xpu, vLLM 0.34.0+xpu and transformers 5.18.x. Option A builds verl's
Docker image, which on verl `main` still pins torch 2.14.0+xpu and vLLM 0.32.0 until verl PR #8863 (vLLM 0.34.0, torch
2.15.0) is merged; for the stack in this guide use Option B or C until then.

### Option A: Docker (versions follow verl #8863)

The image is [`docker/xpu/Dockerfile.stable.vllm`](https://github.com/verl-project/verl/blob/main/docker/xpu/Dockerfile.stable.vllm)
in verl-core. It installs verl and this plugin together, so nothing else is needed. Pin the two checkouts with
`--build-arg VERL_REF=<commit> --build-arg PLUGIN_REF=<commit>`; for release builds add `--no-cache` so the system
certificates, `pip`, `certifi` and `uv` are fetched fresh.

```bash
git clone https://github.com/verl-project/verl.git && cd verl
docker build -f docker/xpu/Dockerfile.stable.vllm -t verl:xpu-vllm .
docker run --rm -it --device /dev/dri --ipc=host --privileged verl:xpu-vllm
```

The image is built in stages (`drivers` -> `runtime-base` -> `builder` -> `runtime`); only the finished `/opt/venv` and
the verl / plugin checkouts (`/opt/verl`, `/opt/verl-hardware-plugin`) are shipped. It contains:

- the Intel GPU user-mode driver (compute-runtime, IGC, Level Zero). Once verl #8863 is merged it comes from the OMIX
  apt repo (`--build-arg OMIX_VERSION`, default 0.4.1, as in vLLM 0.34.0's XPU image); verl `main` today installs
  compute-runtime 26.27 from release packages;
- gcc/g++ and the Python headers, `ffmpeg`, `libnuma-dev` and `libze-dev` (see the system packages under Option C);
- `CCL_SYCL_ALLGATHERV_TMP_BUF`, `CCL_SYCL_ALLREDUCE_TMP_BUF`, `CCL_SYCL_REDUCE_SCATTER_TMP_BUF` and
  `CCL_SYCL_BROADCAST_TMP_BUF`, all set to `1`: they stop oneCCL from caching IPC handles of freed buffers, which
  leaked host memory on multi-GPU runs. Set them yourself when you install from source.

The build needs a verl checkout whose `requirements.txt` allows transformers >= 5.16.1 (verl `main` since PR #8390).
An older `VERL_REF` is rejected by a check at the end of the build.

### Option B: pip extras

For an existing Python 3.12 environment. The plugin's `xpu` extra pins the Intel build of torch and its runtime
packages, and `vllm-xpu` pins the vLLM XPU wheel; the pins live in
[`verl_hardware_plugin/accelerators/xpu/`](../../../verl_hardware_plugin/accelerators/xpu/) (`requirements.txt`,
`requirements-vllm.txt`). pip cannot read an index from package metadata, so the four index flags are required. Without
them pip stops with "no matching distribution found for torch==2.15.0+xpu" instead of installing a CUDA torch.

Install the system packages listed under Option C first (`libnuma-dev`, `libze-dev`, `g++`/`gcc` and the Python headers).
pip cannot install them. Then:

```bash
python3 -m venv /opt/venv && source /opt/venv/bin/activate
export LD_LIBRARY_PATH=/opt/venv/lib:/usr/local/lib   # without it oneCCL fails: "could not load any plugin"
export CCL_SYCL_ALLGATHERV_TMP_BUF=1 CCL_SYCL_ALLREDUCE_TMP_BUF=1 \
       CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1 CCL_SYCL_BROADCAST_TMP_BUF=1

git clone https://github.com/verl-project/verl-hardware-plugin.git && cd verl-hardware-plugin
pip install "verl[verl-core] @ git+https://github.com/verl-project/verl.git@<commit>" ".[xpu,vllm-xpu]" \
    --extra-index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://wheels.vllm.ai/0.34.0/xpu \
    --extra-index-url https://wheels.vllm.ai/xpu \
    --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl
```

- verl comes from a git commit because the verl on PyPI (0.9.1) pins `transformers<5.11`, while vLLM 0.34.0 needs
  `>=5.16.1,<5.23`; use a commit whose `requirements.txt` allows it (verl `main` since PR #8390). `verl[verl-core]` adds
  `TransferQueue`, `pyzmq` and `mathruler`, which `verl.trainer.main_ppo` imports and verl's base install omits.
- Extra names are case-insensitive: `[xpu]` and `[XPU]` are the same extra. Choose one vendor extra; a second vendor's
  torch pin would conflict.
- This path leaves both `triton` and `triton-xpu` installed. Option C removes `triton`; a 2-step GRPO run on 2 B60 cards
  worked with both present, so it is not required here.
- The XPU pins track vLLM 0.34.0 (`torch==2.15.0+xpu`). When vLLM moves, change `requirements.txt` and
  `requirements-vllm.txt` together, and keep them in step with `docker/xpu/Dockerfile.stable.vllm` in verl.

### Option C: From source

verl pins its own versions, and `pip install -e .` on a verl checkout can replace the XPU build of torch with a
different one (it also moves `transformers` and `tensordict` to verl's pins). Install the dependencies as below and
add verl and the plugin with `--no-deps`. The Dockerfile above is the reference for these steps.

System packages on the host: `g++`/`gcc` and the Python 3.12 headers (Triton compiles its launcher stubs at first
use), `libnuma-dev`, `libze-dev` (it provides the unversioned `libze_loader.so` that oneCCL opens; without it `xccl`
fails with "ze_data was not initialized"), and `ffmpeg` (the FFmpeg shared libraries that `torchcodec` needs to decode
video). The Docker image also installs `libhwloc15`, `libpciaccess0`, `libigsc1` and `intel-metrics-library` (from
the OMIX apt repo after verl #8863, `ppa:kobuk-team/intel-graphics` before it; see the Dockerfile) for the Sysman
PCI/NUMA query that pyzes makes; install them as well.

```bash
# Use pip >= 26.2.2 and uv >= 0.13.4 (both carry the corrected CA bundle).
uv venv --python 3.12 --seed /opt/venv && source /opt/venv/bin/activate
export UV_INDEX_URL=https://download.pytorch.org/whl/xpu UV_EXTRA_INDEX_URL=https://pypi.org/simple \
       UV_INDEX_STRATEGY=unsafe-best-match VLLM_TARGET_DEVICE=xpu
export LD_LIBRARY_PATH=/opt/venv/lib:/usr/local/lib   # without it oneCCL fails: "could not load any plugin"
export CCL_SYCL_ALLGATHERV_TMP_BUF=1 CCL_SYCL_ALLREDUCE_TMP_BUF=1 \
       CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1 CCL_SYCL_BROADCAST_TMP_BUF=1

# vLLM XPU wheel (the PyPI wheel has no XPU kernels). --extra-index-url on the command line replaces
# UV_EXTRA_INDEX_URL, so PyPI is repeated.
# vllm-xpu-kernels (pinned exactly by the vLLM wheel) is only on its own index.
uv pip install "vllm==0.34.0+xpu" \
    --extra-index-url https://wheels.vllm.ai/0.34.0/xpu --extra-index-url https://wheels.vllm.ai/xpu \
    --extra-index-url https://vllm-project.github.io/vllm-xpu-kernels/whl \
    --extra-index-url https://pypi.org/simple

git clone https://github.com/verl-project/verl.git && cd verl
# --no-config: run inside a verl checkout, uv would otherwise read verl's pyproject [tool.uv] overrides
# (for the CUDA stack) and they would beat the XPU pins below. transformers is not pinned: it comes from verl's
# requirements.txt window and vLLM's (>= 5.16.1, < 5.23), and the [kernels] extra brings the matching kernels release.
uv --no-config pip install -r requirements.txt \
    "torch==2.15.0+xpu" "transformers[kernels]" \
    dpctl triton-xpu "pip>=26.2.2" "certifi>=2026.10.19" qwen_vl_utils mathruler
uv --no-config pip install --no-deps -e .

# liger-kernel pulls the CUDA triton, which shares triton/ with triton-xpu: keep triton-xpu only.
TRITON_XPU="$(uv pip freeze | grep -i '^triton-xpu==')"
uv pip uninstall triton
uv pip install --reinstall --no-deps "${TRITON_XPU}" --extra-index-url https://download.pytorch.org/whl/xpu

# This plugin, on top of that environment:
cd .. && git clone https://github.com/verl-project/verl-hardware-plugin.git && cd verl-hardware-plugin
pip install --no-deps -e .
```

`--no-deps` keeps the plugin install from touching the torch and verl already in place. The plugin needs no
environment variable: it is auto-discovered by verl through the `verl.plugins` setuptools entry_points
group declared in this repo's `pyproject.toml`
(`[project.entry-points."verl.plugins"] hardware = "verl_hardware_plugin"`), which verl loads by default
(`VERL_USE_EXTERNAL_PLUGINS=auto`).

### Platform selection

With the plugin installed, verl picks the Intel platform by itself on a host where `torch.xpu` is available.
`VERL_PLATFORM=intel` is only an override, for forcing the choice (for example on a host with several kinds of
accelerator). `VERL_USE_EXTERNAL_PLUGINS=none` disables discovery, e.g. to isolate a bug to this plugin; verl then
falls back to `nvidia`.

## 2. Verify the Install

```bash
python3 -c "
from verl.plugin.platform import get_platform
p = get_platform()
print('device:', p.device_name, '/ vendor:', p.vendor_name)
"
```

Expected on Intel GPU: `device: xpu / vendor: intel`. If it instead falls
back to `nvidia`, either the plugin was not discovered (check that `pip install`
completed without error and that no `VERL_USE_EXTERNAL_PLUGINS=none` is set) or
torch lost its XPU build (check `python3 -c "import torch; print(torch.__version__, torch.xpu.is_available())"`;
a plain version like `2.15.1` without `+xpu` means a dependency install replaced it).

## Next Steps

Follow the [Quick Start](./quick_start.md).
