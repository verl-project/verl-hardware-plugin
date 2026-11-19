# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""vLLM / vllm-torchtpu runtime patches for TPU rollout."""

import os
import types
from unittest import mock

from verl_hardware_plugin.accelerators.tpu.rollout import tpu_vllm_patches


def test_patch_vllm_for_tpu_applies_precision_patches_and_leaves_tpu_worker_untouched():
    """patch_vllm_for_tpu installs precision patches once vllm_torchtpu is present, without patching TPUWorker."""
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")

    def upstream_reset_encoder_cache(self) -> None:
        pass

    class TPUWorker:
        reset_encoder_cache = upstream_reset_encoder_cache

    worker_mod.TPUWorker = TPUWorker
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }

    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "_apply_tpu_precision_patches") as apply_precision,
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()
        apply_precision.assert_called_once_with()
        assert tpu_vllm_patches._PATCHES_APPLIED is True
        assert TPUWorker.reset_encoder_cache is upstream_reset_encoder_cache


def _fake_tpu_worker_module(configure):
    worker_mod = types.ModuleType("vllm_torchtpu.worker.tpu_worker")
    worker_mod.TPUWorker = type("TPUWorker", (), {})
    worker_mod._configure_tpu_process_env = configure
    modules = {
        "vllm_torchtpu": types.ModuleType("vllm_torchtpu"),
        "vllm_torchtpu.worker": types.ModuleType("vllm_torchtpu.worker"),
        "vllm_torchtpu.worker.tpu_worker": worker_mod,
    }
    return worker_mod, modules


def _patch_in_fake_worker_process(modules):
    """Run patch_vllm_for_tpu as a vLLM worker would, without its process-global multiprocessing patch."""
    with (
        mock.patch.dict("sys.modules", modules),
        mock.patch.object(tpu_vllm_patches, "_PATCHES_APPLIED", False),
        mock.patch.object(tpu_vllm_patches, "patch_multiprocessing_for_tpu"),
    ):
        tpu_vllm_patches.patch_vllm_for_tpu()


def test_subhost_worker_hook_is_not_installed_for_other_engines():
    """A TP=8 or multi-host engine (no SUBHOST_MEGASCALE_PORT_ENV) keeps vllm-torchtpu's function as is."""

    def configure(rank, local_rank, world_size, local_world_size):
        pass

    worker_mod, modules = _fake_tpu_worker_module(configure)
    with mock.patch.dict(os.environ):
        os.environ.pop(tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV, None)
        _patch_in_fake_worker_process(modules)
    assert worker_mod._configure_tpu_process_env is configure


def test_subhost_worker_hook_sets_per_worker_ports_and_drops_pod_env():
    calls = []

    def configure(rank, local_rank, world_size, local_world_size):
        calls.append((rank, local_rank, world_size, local_world_size))

    worker_mod, modules = _fake_tpu_worker_module(configure)
    pod_env = {name: "pod" for name in tpu_vllm_patches._POD_TPU_ENV_VARS}
    with (
        mock.patch.dict(os.environ, {**pod_env, "MEGASCALE_PORT": "8081"}),
        mock.patch.object(os, "makedirs") as makedirs,
    ):
        # The second TP=4 engine of a host: device offset 4, MegaScale block from 19080.
        os.environ[tpu_vllm_patches.SUBHOST_MEGASCALE_PORT_ENV] = "19080"
        _patch_in_fake_worker_process(modules)
        with mock.patch.dict("sys.modules", modules):
            tpu_vllm_patches.patch_tpu_worker_subhost_env()  # a second call does not wrap twice
        worker_mod._configure_tpu_process_env(1, 5, 4, 8)

        assert calls == [(1, 5, 4, 8)]  # vllm-torchtpu's own variables are still set first
        assert os.environ["MEGASCALE_PORT"] == "19082"
        assert os.environ["MEGASCALE_DEBUG_PORT"] == "19083"
        assert os.environ["TPU_VISIBLE_DEVICES"] == os.environ["TPU_VISIBLE_CHIPS"] == "5"
        assert os.environ["TPU_LOG_DIR"] == "/tmp/tpu_logs/rank_5"
        assert not set(tpu_vllm_patches._POD_TPU_ENV_VARS) & set(os.environ)
    makedirs.assert_called_once_with("/tmp/tpu_logs/rank_5", exist_ok=True)
