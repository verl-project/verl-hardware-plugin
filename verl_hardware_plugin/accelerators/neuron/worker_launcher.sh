#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

if [[ "${VERL_NEURON_SERVING:-}" == 1 ]]; then
    export VERL_NEURON_PROCESS_ROLE=rollout
    neuron_python=${VERL_NEURON_ROLLOUT_PYTHON:?Set the serving Python executable}
elif [[ "${WG_BACKEND:-}" == ray ]]; then
    export VERL_NEURON_PROCESS_ROLE=trainer
    neuron_python=${VERL_NEURON_TRAIN_PYTHON:?Set the training Python executable}
else
    export VERL_NEURON_PROCESS_ROLE=controller
    neuron_python=${VERL_NEURON_ROLLOUT_PYTHON:?Set the controller Python executable}
fi
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PATH="$(dirname "$neuron_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$neuron_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$neuron_python" "$@"
