#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
set -euo pipefail

# Ray workers inherit the controller role. An explicit rollout role takes
# precedence over WG_BACKEND; otherwise worker-group processes are trainers.
if [[ "${VERL_NEURON_PROCESS_ROLE:-}" == rollout ]]; then
    export VERL_NEURON_PROCESS_ROLE=rollout
elif [[ "${WG_BACKEND:-}" == ray ]]; then
    export VERL_NEURON_PROCESS_ROLE=trainer
else
    export VERL_NEURON_PROCESS_ROLE=controller
fi
neuron_python=${VERL_NEURON_PYTHON:?Set the shared Neuron Python executable}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PATH="$(dirname "$neuron_python"):$PATH"
export LD_LIBRARY_PATH="$(dirname "$neuron_python")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$neuron_python" "$@"
