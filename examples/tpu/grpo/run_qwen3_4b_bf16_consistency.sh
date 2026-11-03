#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

# Apply PR #76's internal precision corrections with BF16 FSDP computation and logits.
# VERL_ROOT must contain the upstream Qwen3-4B TorchTitan TPU launcher.
set -euo pipefail

: "${VERL_ROOT:?Set VERL_ROOT to your verl checkout containing examples/tpu/grpo/run_qwen3_4b_torchtitan.sh}"
launcher="${VERL_ROOT}/examples/tpu/grpo/run_qwen3_4b_torchtitan.sh"
if [[ ! -f "${launcher}" ]]; then
    printf 'Missing upstream TPU launcher: %s\n' "${launcher}" >&2
    exit 1
fi

export VERL_TPU_VLLM_ATTN_FP32_STATS=1
export VERL_TPU_VLLM_RMSNORM_CUDA_ORDER=1
export VERL_TPU_FP32_LOGITS=0

options=()
for role in actor ref; do
    engine="++actor_rollout_ref.${role}.torchtitan"
    options+=("${engine}.use_torch_compile=True")
    for setting in mixed_precision_param=bfloat16 fp32_logits=False use_splash_attention=True \
        use_simple_fsdp=True tpu_eager_mode=DEFER_AND_FUSE use_fused_linear_cross_entropy=False; do
        options+=("${engine}.engine_kwargs.tpu.${setting}")
    done
done
exec bash "${launcher}" "${options[@]}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.head_dtype=bfloat16 \
    "$@"
