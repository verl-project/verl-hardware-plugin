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

Pick one. Both install the same set of packages (torch 2.14.0+xpu, vLLM 0.32.0+xpu, transformers 5.16.1): the Docker
image runs the commands of Option B.

### Option A: Docker (recommended)

The image is [`docker/xpu/Dockerfile.stable.vllm`](https://github.com/verl-project/verl/blob/main/docker/xpu/Dockerfile.stable.vllm)
in verl-core. It installs verl and this plugin together, so nothing else is needed. Pin the two checkouts with
`--build-arg VERL_REF=<commit> --build-arg PLUGIN_REF=<commit>`; for release builds add `--no-cache` so the system
certificates, `pip`, `certifi` and `uv` are fetched fresh.

```bash
git clone https://github.com/verl-project/verl.git && cd verl
docker build -f docker/xpu/Dockerfile.stable.vllm -t verl:xpu-vllm032 .
docker run --rm -it --device /dev/dri --ipc=host --privileged verl:xpu-vllm032
```

The image is built in stages (`drivers` -> `runtime-base` -> `builder` -> `runtime`); only the finished `/opt/venv` and
the verl / plugin checkouts (`/opt/verl`, `/opt/verl-hardware-plugin`) are shipped. It contains:

- the Intel GPU user-mode driver (compute-runtime 26.27, IGC 2.38.2, Level Zero 1.32), matching vLLM 0.32.0's XPU image;
- gcc/g++ and the Python headers, `ffmpeg`, `libnuma-dev` and `libze-dev` (see the system packages under Option B);
- `CCL_SYCL_ALLGATHERV_TMP_BUF`, `CCL_SYCL_ALLREDUCE_TMP_BUF`, `CCL_SYCL_REDUCE_SCATTER_TMP_BUF` and
  `CCL_SYCL_BROADCAST_TMP_BUF`, all set to `1`: they stop oneCCL from caching IPC handles of freed buffers, which
  leaked host memory on multi-GPU runs. Set them yourself when you install from source.

The build needs a verl checkout whose `requirements.txt` allows transformers >= 5.16.1 (verl `main` since PR #8390).
An older `VERL_REF` is rejected by a check at the end of the build.

### Option B: From source

verl pins its own versions, and `pip install -e .` on a verl checkout can replace the XPU build of torch with a
different one (it also moves `transformers` and `tensordict` to verl's pins). Install the dependencies as below and
add verl and the plugin with `--no-deps`. The Dockerfile above is the reference for these steps.

System packages on the host: `g++`/`gcc` and the Python 3.12 headers (Triton compiles its launcher stubs at first
use), `libnuma-dev`, `libze-dev` (it provides the unversioned `libze_loader.so` that oneCCL opens; without it `xccl`
fails with "ze_data was not initialized"), and `ffmpeg` (the FFmpeg shared libraries that `torchcodec` needs to decode
video). The Docker image also installs `libhwloc15`, `libpciaccess0`, `libigsc1` and `intel-metrics-library` (from
Intel's `ppa:kobuk-team/intel-graphics`) for the Sysman PCI/NUMA query that pyzes makes; install them as well.

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
uv pip install "vllm==0.32.0+xpu" \
    --extra-index-url https://wheels.vllm.ai/0.32.0/xpu --extra-index-url https://wheels.vllm.ai/xpu \
    --extra-index-url https://pypi.org/simple

git clone https://github.com/verl-project/verl.git && cd verl
echo 'pyzes>=0.1.2' > /tmp/override-pyzes.txt   # torch pins pyzes==0.1.1; the device-memory query needs >= 0.1.2

# --no-config: run inside a verl checkout, uv would otherwise read verl's pyproject [tool.uv] overrides
# (for the CUDA stack) and they would beat the XPU pins below. transformers is not pinned: it comes from verl's
# requirements.txt window and vLLM's (>= 5.16.1), and the [kernels] extra brings the matching kernels release.
uv --no-config pip install -r requirements.txt \
    "torch==2.14.0+xpu" "transformers[kernels]" \
    dpctl "pyzes>=0.1.2" triton-xpu "pip>=26.2.2" "certifi>=2026.10.19" qwen_vl_utils mathruler \
    --override /tmp/override-pyzes.txt
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
a plain version like `2.14.1` without `+xpu` means a dependency install replaced it).

## Next Steps

Follow the [Quick Start](./quick_start.md).
