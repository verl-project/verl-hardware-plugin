# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for plugin registration mechanism."""

import os
from contextlib import contextmanager
from unittest import mock

import pytest


@contextmanager
def _fresh_registries():
    """Reset platform manager singleton for isolated tests."""
    import verl.plugin.platform.platform_manager as pm

    old_platform = pm._current_platform
    pm._current_platform = None
    try:
        yield
    finally:
        pm._current_platform = old_platform


class TestPlatformRegistration:
    """Verify that all hardware platforms register correctly."""

    def test_xpu_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU  # noqa: F401

        assert "intel" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("intel")
        assert cls is PlatformXPU

    def test_mlu_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_mlu import PlatformMLU  # noqa: F401

        assert "cambricon" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("cambricon")
        assert cls is PlatformMLU

    def test_metax_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_cuda_metax import PlatformMetaX  # noqa: F401

        assert "metax" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("metax")
        assert cls is PlatformMetaX

    def test_xpu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "intel"}):
                assert _detect_platform_name() == "intel"

    def test_mlu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_mlu import PlatformMLU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cambricon"}):
                assert _detect_platform_name() == "cambricon"

    def test_metax_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_cuda_metax import PlatformMetaX  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "metax"}):
                assert _detect_platform_name() == "metax"


class TestEngineRegistration:
    """Verify that engine classes register correctly."""

    def test_fsdp_flagos_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_flagos import (
            FSDPFlagOSEngineWithLMHead,
            FSDPFlagOSEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("cuda", "flagos")] is FSDPFlagOSEngineWithLMHead
        assert EngineRegistry._engines["language_model"]["fsdp2"][("cuda", "flagos")] is FSDPFlagOSEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("cuda", "flagos")] is FSDPFlagOSEngineWithValueHead

    def test_fsdp_xpu_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_xpu import (
            FSDPXPUEngineWithLMHead,
            FSDPXPUEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("xpu", "intel")] is FSDPXPUEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("xpu", "intel")] is FSDPXPUEngineWithValueHead

    def test_fsdp_mlu_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_mlu import (
            FSDPMLUEngineWithLMHead,
            FSDPMLUEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("mlu", "cambricon")] is FSDPMLUEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("mlu", "cambricon")] is FSDPMLUEngineWithValueHead

    def test_fsdp_metax_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_metax import (
            FSDPMetaXEngineWithLMHead,
            FSDPMetaXEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("cuda", "metax")] is FSDPMetaXEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("cuda", "metax")] is FSDPMetaXEngineWithValueHead

    def test_megatron_flagos_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_flagos import MegatronFlagOSEngineWithLMHead

        assert (
            EngineRegistry._engines["language_model"]["megatron"][("cuda", "flagos")] is MegatronFlagOSEngineWithLMHead
        )

    def test_megatron_xpu_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_xpu import MegatronXPUEngineWithLMHead

        assert EngineRegistry._engines["language_model"]["megatron"][("xpu", "intel")] is MegatronXPUEngineWithLMHead

    def test_megatron_mlu_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_mlu import MegatronMLUEngineWithLMHead

        assert (
            EngineRegistry._engines["language_model"]["megatron"][("mlu", "cambricon")] is MegatronMLUEngineWithLMHead
        )

    def test_megatron_metax_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_metax import MegatronMetaXEngineWithLMHead

        assert EngineRegistry._engines["language_model"]["megatron"][("cuda", "metax")] is MegatronMetaXEngineWithLMHead


class TestFLEnvManager:
    """Test FLEnvManager utility."""

    def test_flaggems_disabled_by_default(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {}, clear=True):
            assert not FLEnvManager.is_flaggems_enabled()

    def test_flaggems_enabled(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "true"}):
            assert FLEnvManager.is_flaggems_enabled()

    def test_flaggems_enabled_with_1(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1"}):
            assert FLEnvManager.is_flaggems_enabled()

    def test_flaggems_disabled_with_false(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "false"}):
            assert not FLEnvManager.is_flaggems_enabled()

    def test_whitelist_parsing(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": "rmsnorm,layernorm,softmax"}):
            wl = FLEnvManager.get_training_whitelist()
            assert wl == ["rmsnorm", "layernorm", "softmax"]

    def test_blacklist_parsing(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGOS_BLACKLIST": "dropout,gelu"}):
            bl = FLEnvManager.get_training_blacklist()
            assert bl == ["dropout", "gelu"]

    def test_rollout_whitelist_parsing(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"VLLM_FL_FLAGOS_WHITELIST": "rmsnorm,softmax"}):
            wl = FLEnvManager.get_rollout_whitelist()
            assert wl == ["rmsnorm", "softmax"]

    def test_rollout_blacklist_parsing(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"VLLM_FL_FLAGOS_BLACKLIST": "dropout"}):
            bl = FLEnvManager.get_rollout_blacklist()
            assert bl == ["dropout"]

    def test_whitelist_empty_returns_none(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": ""}):
            assert FLEnvManager.get_training_whitelist() is None

    def test_summary(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1", "USE_FLAGCX": "0"}):
            summary = FLEnvManager.get_summary()
            assert "FlagGems=ON" in summary
            assert "FlagCX=OFF" in summary

    def test_env_snapshot_training(self):
        from verl_hardware_plugin.utils import FLEnvManager

        env = {
            "TRAINING_FL_FLAGGEMS_ENABLE": "true",
            "TE_FL_PLUGIN_MODULES": "my_module",
            "TE_FL_SKIP_CUDA": "1",
            "USE_FLAGCX": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            snapshot = FLEnvManager.get_env_snapshot(phase="training")
            assert snapshot["TRAINING_FL_FLAGGEMS_ENABLE"] == "true"
            assert snapshot["TE_FL_PLUGIN_MODULES"] == "my_module"
            assert snapshot["TE_FL_SKIP_CUDA"] == "1"
            assert snapshot["USE_FLAGCX"] == "1"

    def test_env_snapshot_rollout_excludes_training_keys(self):
        from verl_hardware_plugin.utils import FLEnvManager

        env = {
            "TRAINING_FL_FLAGGEMS_ENABLE": "true",
            "VLLM_FL_PREFER_ENABLED": "1",
            "USE_FLAGCX": "1",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            snapshot = FLEnvManager.get_env_snapshot(phase="rollout")
            assert "TRAINING_FL_FLAGGEMS_ENABLE" not in snapshot
            assert snapshot["VLLM_FL_PREFER_ENABLED"] == "1"
            assert snapshot["USE_FLAGCX"] == "1"


class TestMayEnableFlagGems:
    """Test may_enable_flag_gems function."""

    def test_noop_when_disabled(self):
        from verl_hardware_plugin.utils import may_enable_flag_gems

        with mock.patch.dict(os.environ, {}, clear=True):
            # Should not raise
            may_enable_flag_gems(phase="training")

    def test_skips_when_already_imported(self):
        import sys

        from verl_hardware_plugin.utils import may_enable_flag_gems

        # Simulate flag_gems already loaded
        fake_module = mock.MagicMock()
        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1"}):
            with mock.patch.dict(sys.modules, {"flag_gems": fake_module}):
                # Should return early without calling flag_gems.enable
                may_enable_flag_gems(phase="training")
                fake_module.enable.assert_not_called()
                fake_module.only_enable.assert_not_called()

    def test_enable_all_ops(self):
        import sys

        from verl_hardware_plugin.utils import may_enable_flag_gems

        fake_module = mock.MagicMock()
        fake_module.__version__ = "0.1.0"
        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1"}, clear=True):
            # Ensure flag_gems is NOT in sys.modules so the import path is taken
            with mock.patch.dict(sys.modules, {}, clear=False):
                sys.modules.pop("flag_gems", None)
                with mock.patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: fake_module
                    if name == "flag_gems"
                    else __import__(name, *a, **kw),
                ):
                    may_enable_flag_gems(phase="training")
                    fake_module.enable.assert_called_once()

    def test_raises_on_whitelist_and_blacklist(self):
        import sys

        from verl_hardware_plugin.utils import may_enable_flag_gems

        fake_module = mock.MagicMock()
        fake_module.__version__ = "0.1.0"
        env = {
            "TRAINING_FL_FLAGGEMS_ENABLE": "1",
            "TRAINING_FL_FLAGOS_WHITELIST": "rmsnorm",
            "TRAINING_FL_FLAGOS_BLACKLIST": "dropout",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.dict(sys.modules, {}, clear=False):
                sys.modules.pop("flag_gems", None)
                with mock.patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: fake_module
                    if name == "flag_gems"
                    else __import__(name, *a, **kw),
                ):
                    with pytest.raises(ValueError, match="Cannot set both whitelist and blacklist"):
                        may_enable_flag_gems(phase="training")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
