#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${CONFIG_PATH:?Set CONFIG_PATH to the absolute directory containing your verl Hydra configuration}"
: "${CONFIG_NAME:?Set CONFIG_NAME to your verl Hydra configuration name}"
[[ "$CONFIG_PATH" == /* && -d "$CONFIG_PATH" ]] || {
    echo "CONFIG_PATH must be an existing absolute directory." >&2; exit 1;
}
export VERL_PLATFORM=aws
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_PLUGINS=""
export NEURON_RT_MAP_HBM=1
export NEURON_RT_ENABLE_P2P=0
export FI_EFA_ENABLE_SHM_TRANSFER=0
export NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-2}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

public_overrides=()
: "${VERL_NEURON_PYTHON:?Set the shared Neuron Python executable}"
export VERL_NEURON_PROCESS_ROLE=controller
# Persist NKI binaries with the environment. Cached vLLM graphs contain
# absolute paths to these files and may outlive a container's /var/tmp.
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-"$(dirname "$VERL_NEURON_PYTHON")/../../nki-cache"}
neuron_python=$VERL_NEURON_PYTHON
export LD_LIBRARY_PATH="$(dirname "$neuron_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Released verl does not merge PlatformBase.get_ray_init_kwargs(). Supply
# every runtime setting explicitly, including the shared interpreter path.
runtime_overrides=$("$neuron_python" - <<'PY'
import json
import sys
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
    from verl_hardware_plugin.accelerators.neuron.runtime import public_runtime_env
    runtime = public_runtime_env()
print("ray_kwargs.ray_init.runtime_env.py_executable=" + json.dumps(runtime["py_executable"]))
for name, value in runtime["env_vars"].items():
    print("++ray_kwargs.ray_init.runtime_env.env_vars." + name + "=" + json.dumps(value))
PY
)
mapfile -t public_overrides <<< "$runtime_overrides"
if pgrep -x raylet >/dev/null; then
    echo "An existing Ray cluster is running. Use an idle host; this script will not stop it." >&2
    exit 1
fi
export RUN_DIR=${RUN_DIR:-${TMPDIR:-/tmp}/verl-neuron-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
"$neuron_python" -m verl.trainer.main_ppo \
    --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" "${public_overrides[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
