#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
environment_root=${1:?Usage: setup_neuron_public.sh /absolute/environment/root}
[[ "$environment_root" == /* ]] || { echo "Use an absolute environment root." >&2; exit 1; }
uv_bin=${UV_BIN:-uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-"$environment_root/python"}
export UV_PYTHON_BIN_DIR=${UV_PYTHON_BIN_DIR:-"$environment_root/bin"}
mkdir -p "$environment_root/sources"
"$uv_bin" python install 3.12.13
python_bin=$("$uv_bin" python find --managed-python 3.12.13)

checkout_source() {
    local name=$1 url=$2 revision=$3 directory="$environment_root/sources/$1"
    if [[ ! -d "$directory/.git" ]]; then
        git init -q "$directory"
        git -C "$directory" remote add origin "$url"
        git -C "$directory" fetch -q --depth 1 origin "$revision"
        git -C "$directory" checkout -q --detach FETCH_HEAD
    fi
    [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || {
        echo "$name: existing checkout does not match $revision" >&2; exit 1;
    }
    git -C "$directory" diff --quiet HEAD || { echo "$name has local changes" >&2; exit 1; }
}

checkout_source torchtitan-neuron https://github.com/aws-neuron/torchtitan-neuron.git b7582e986586051d7be85d03956afe169624aeda
checkout_source vllm-neuron https://github.com/vllm-project/vllm-neuron.git eb6cb360f7cff6082e22a9437d8ce0ba3891304a

# Pin the runtime interfaces used by the plugin; resolve transitive dependencies
# from the published packages and the pinned source checkouts.
common_packages=(
    "verl[verl-core]==0.9.1" "ray==2.55.0" "TransferQueue==0.1.11"
    "transformers==5.10.4" "torchdata==0.11.0" "spmd-types==0.2.1"
    "triton==3.8.0" "protobuf==6.33.6"
    "neuronx-cc==2.28.7631.0+8f7b0cf5" "nki==0.7.0+37713544100.gfede04bb"
    "nixl==1.3.2" "nixl-cu12==1.3.2" "nvidia-cuda-runtime-cu12==12.9.79"
    "setuptools==79.0.1" wheel orjson
)
excluded_packages=(
    # NIXL's metapackage requests both CUDA variants. Neuron uses only CUDA 12.
    nixl-cu13
    # Unused audio/video and profiling dependencies in verl/vLLM metadata.
    # vLLM 0.29's torchaudio pin also conflicts with its own PyTorch version.
    torchaudio torchcodec pynvvideocodec nvtx
    # GPU kernels/loaders from vLLM's CUDA wheel, unused by the native backend.
    flashinfer-python tilelang nvidia-cudnn-frontend fastsafetensors instanttensor
    nvidia-cutlass-dsl quack-kernels tokenspeed-mla humming-kernels
)

for role in training serving; do
    environment="$environment_root/$role"
    if [[ ! -x "$environment/bin/python" ]]; then
        "$uv_bin" venv --python "$python_bin" "$environment"
    fi
    source_name=torchtitan-neuron
    runtime_packages=("torch==2.14.0+cpu" "torch-neuronx==2.14.3.0.281+dc70ffa")
    if [[ "$role" == serving ]]; then
        source_name=vllm-neuron
        runtime_packages=(
            "torch==2.13.0+cpu" "libtorch-neuronx-lite==2.13.0.1.0.3112+653bc48f"
            "vllm==0.29.0" "torch-xla==2.13.0" "torchvision==0.28.0+cpu"
        )
    fi
    # Resolve the entire environment together, retaining the CPU PyTorch wheels.
    # Exclude only the named unused dependencies, rather than disabling resolution.
    "$uv_bin" pip install --python "$environment/bin/python" --torch-backend cpu \
        --index https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match \
        --excludes <(printf '%s\n' "${excluded_packages[@]}") \
        "${common_packages[@]}" "${runtime_packages[@]}" \
        -e "$environment_root/sources/$source_name" -e "$ROOT"
    "$environment/bin/python" - <<'PY'
import sys
import sysconfig
from pathlib import Path
prefix = Path(sys.prefix)
runtime = prefix / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
links = prefix / "lib/nixl-cuda-stubs"
links.mkdir(exist_ok=True)
link = links / "libcudart.so.12"
if not link.exists():
    link.symlink_to(runtime)
# TorchXLA, imported by the public lite runtime, links against libpython.
shared_python = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
if shared_python.exists():
    for name in {shared_python.name, sysconfig.get_config_var("INSTSONAME")} - {None}:
        python_link = prefix / "lib" / name
        if not python_link.exists():
            python_link.symlink_to(shared_python)
PY
done

export VERL_NEURON_TRAIN_PYTHON="$environment_root/training/bin/python"
export VERL_NEURON_ROLLOUT_PYTHON="$environment_root/serving/bin/python"
"$VERL_NEURON_ROLLOUT_PYTHON" "$ROOT/scripts/accelerators/neuron/check_neuron_environment.py"
printf 'export VERL_NEURON_TRAIN_PYTHON=%q\nexport VERL_NEURON_ROLLOUT_PYTHON=%q\n' \
    "$VERL_NEURON_TRAIN_PYTHON" "$VERL_NEURON_ROLLOUT_PYTHON"
