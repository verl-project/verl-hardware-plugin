# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for TPU rollout."""

import types
from unittest import mock

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def test_reset_encoder_cache_is_noop():
    """TPUWorker.reset_encoder_cache is a no-op for text models."""
    tpu_vllm_patches._reset_encoder_cache(None)


def test_patch_vllm_for_tpu_installs_reset_encoder_cache():
    """patch_vllm_for_tpu attaches reset_encoder_cache to TPUWorker."""
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    class TPUWorker:
        pass

    worker_mod.TPUWorker = TPUWorker
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }

    with mock.patch.dict("sys.modules", modules), mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False):
        tpu_vllm_patches.patch_vllm_for_tpu()
        assert TPUWorker.reset_encoder_cache is tpu_vllm_patches._reset_encoder_cache
