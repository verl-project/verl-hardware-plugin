# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for plugin registration mechanism."""

import os
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest import mock

import pytest


@pytest.fixture(scope="session", autouse=True)
def _stub_training_engine_runtimes():
    """Keep decorator tests independent of optional GPU training runtimes.

    These tests exercise registration metadata, not FSDP or Megatron execution.
    The lightweight bases let the complete registry suite run on a CPU-only CI host.
    """

    # BaseEngine must come from a real, unmocked import -- it's the base class
    # _StubEngine below needs, and importing it here (before any sys.modules
    # patching) runs verl/workers/engine/__init__.py for real, which resolves
    # cleanly on its own (its try/except ImportError guards handle the
    # genuinely-missing torchtitan/veomni/automodel/mindspeed/megatron
    # packages already).
    from verl.workers.engine.base import BaseEngine

    class _StubEngine(BaseEngine):
        pass

    # Populate every module attribute with _StubEngine *before* patching
    # sys.modules, not after. verl/workers/engine/__init__.py's mindspeed
    # import defines `class MindspeedEngineWithLMHead(MegatronEngineWithLMHead)`
    # with a live @EngineRegistry.register(...) decorator that asserts
    # issubclass(engine_class, BaseEngine) *at class-definition time*. If
    # that import gets re-triggered while these are still placeholders (as
    # they were here until the `yield`), the assertion fails for real --
    # confirmed via AssertionError at verl/workers/engine/base.py:376 on
    # real hardware. Only the plugin's own subsequent imports need to see
    # the stub; verl-core's internal mindspeed/megatron wiring never should.
    fsdp = ModuleType("verl.workers.engine.fsdp")
    fsdp.FSDPEngine = _StubEngine
    fsdp.FSDPEngineWithLMHead = _StubEngine
    fsdp.FSDPTurboEngineWithLMHead = _StubEngine

    fsdp_transformer = ModuleType("verl.workers.engine.fsdp.transformer_impl")
    fsdp_transformer.FSDPEngine = _StubEngine
    fsdp_transformer.FSDPEngineWithLMHead = _StubEngine
    fsdp_transformer.FSDPEngineWithValueHead = _StubEngine

    megatron_transformer = ModuleType("verl.workers.engine.megatron.transformer_impl")
    megatron_transformer.MegatronEngine = _StubEngine
    megatron_transformer.MegatronEngineWithLMHead = _StubEngine
    megatron_transformer.MegatronEngineWithValueHead = _StubEngine

    engine_modules = {
        "verl.workers.engine.fsdp": fsdp,
        "verl.workers.engine.fsdp.transformer_impl": fsdp_transformer,
        "verl.workers.engine.megatron.transformer_impl": megatron_transformer,
    }
    with (
        mock.patch.dict(os.environ, {"VERL_USE_EXTERNAL_PLUGINS": "none"}),
        mock.patch.dict(sys.modules, engine_modules),
    ):
        yield


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

    def test_iluvatar_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_cuda_iluvatar import PlatformIluvatar  # noqa: F401

        assert "iluvatar" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("iluvatar")
        assert cls is PlatformIluvatar

    def test_musa_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_musa import PlatformMUSA  # noqa: F401

        assert "musa" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("musa")
        assert cls is PlatformMUSA

    def test_tpu_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU  # noqa: F401

        assert "tpu" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("tpu")
        assert cls is PlatformTPU

    def test_xpu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "intel"}):
                assert _detect_platform_name() == "intel"

    def test_xpu_device_and_vendor_names(self):
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        platform = PlatformXPU()
        assert platform.device_name == "xpu"
        assert platform.vendor_name == "intel"
        assert platform.communication_backend_name() == "xccl"
        assert platform.visible_devices_envvar() == "ZE_AFFINITY_MASK"

    def test_xpu_ray_resource_options(self):
        """XPU reuses Ray's built-in GPU resource, unlike TPU's custom "TPU" resource.

        ray_resource_options() must return the num_gpus key Ray's built-in
        GPU scheduling expects, not a custom resources dict.
        """
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        platform = PlatformXPU()
        assert platform.ray_resource_name() == "GPU"
        assert platform.ray_resource_options(4) == {"num_gpus": 4}
        assert platform.ray_resource_options(0) == {"num_gpus": 0}

    def test_xpu_derives_from_platform_base(self):
        """PlatformXPU must not acquire another vendor's platform behaviour by inheritance."""
        from verl.plugin.platform.platform_base import PlatformBase
        from verl.plugin.platform.platform_cuda import PlatformCUDA
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        assert issubclass(PlatformXPU, PlatformBase)
        assert not issubclass(PlatformXPU, PlatformCUDA)
        assert PlatformXPU.__abstractmethods__ == frozenset()

    def test_xpu_no_cuda_collective_or_rollout_env(self):
        """Neither a CUDA collective module nor extra rollout env vars apply to XPU."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        platform = PlatformXPU()
        assert platform.get_collective_module() is None
        assert platform.rollout_env_vars() == {}

    def test_xpu_memory_and_capability_methods(self):
        """set_allocator_settings() is a no-op; XPU has no CUDA compute-capability model."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        platform = PlatformXPU()
        assert platform.set_allocator_settings("expandable_segments:True") is None
        assert platform.get_device_capability() == (None, None)

    def test_xpu_ray_noset_envvars(self):
        """XPU only manages ZE_AFFINITY_MASK -- it has no CUDA_VISIBLE_DEVICES aliasing to
        worry about, unlike TPU which deliberately keeps a CUDA entry too."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        assert PlatformXPU().ray_noset_envvars() == ["RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK"]

    def test_xpu_ipc_unsupported(self):
        """Intel XPU has no CUDA-style IPC handle, so verl must fall back to the
        serialized tensor transfer path instead of routing through CUDA IPC."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        assert PlatformXPU().is_ipc_supported() is False

    def test_xpu_cudart_returns_none(self):
        """There is no CUDA runtime on an XPU host; PlatformBase documents None as the answer."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        assert PlatformXPU().cudart() is None

    def test_xpu_profiler_is_noop(self):
        """XPU profiling is handled externally via Intel VTune/Advisor, not verl's profiler hooks."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        platform = PlatformXPU()
        assert platform.profiler_start() is None
        assert platform.profiler_stop() is None

    def test_xpu_nvtx_range_yields(self):
        """nvtx_range must yield immediately; XPU has no NVTX equivalent."""
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU

        entered = False
        with PlatformXPU().nvtx_range("xpu-test"):
            entered = True
        assert entered

    def test_mlu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_mlu import PlatformMLU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "cambricon"}):
                assert _detect_platform_name() == "cambricon"

    def test_enflame_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_enflame import PlatformENFLAME  # noqa: F401

        assert "enflame" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("enflame")
        assert cls is PlatformENFLAME

    def test_enflame_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_enflame import PlatformENFLAME  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "enflame"}):
                assert _detect_platform_name() == "enflame"

    def test_enflame_device_and_vendor_names(self):
        from verl_hardware_plugin.platforms.platform_enflame import PlatformENFLAME

        platform = PlatformENFLAME()
        assert platform.device_name == "gcu"
        assert platform.vendor_name == "enflame"

    def test_enflame_gcu_ipc_collect_shim(self):
        from types import ModuleType
        from unittest import mock

        import verl_hardware_plugin.platforms.platform_enflame as platform_enflame

        fake_gcu = ModuleType("gcu")
        old_patched = platform_enflame._gcu_runtime_patched
        try:
            platform_enflame._gcu_runtime_patched = False
            with mock.patch.object(platform_enflame, "_ensure_torch_gcu", return_value=True):
                with mock.patch.object(platform_enflame.torch, "gcu", fake_gcu, create=True):
                    module = platform_enflame._get_gcu_module()
                    assert module is fake_gcu
                    assert callable(module.ipc_collect)
                    module.ipc_collect()
        finally:
            platform_enflame._gcu_runtime_patched = old_patched

    def test_enflame_communication_backend(self):
        from verl_hardware_plugin.platforms.platform_enflame import PlatformENFLAME

        with mock.patch.dict(os.environ, {}, clear=True):
            assert PlatformENFLAME().communication_backend_name() == "eccl"
        with mock.patch.dict(os.environ, {"USE_FLAGCX": "1"}, clear=False):
            assert PlatformENFLAME().communication_backend_name() == "flagcx"

    def test_metax_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_cuda_metax import PlatformMetaX  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "metax"}):
                assert _detect_platform_name() == "metax"

    def test_iluvatar_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_cuda_iluvatar import PlatformIluvatar  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "iluvatar"}):
                assert _detect_platform_name() == "iluvatar"

    def test_musa_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_musa import PlatformMUSA  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "musa"}):
                assert _detect_platform_name() == "musa"

    def test_tpu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "tpu"}):
                assert _detect_platform_name() == "tpu"

    def test_musa_device_and_vendor_names(self):
        from verl_hardware_plugin.platforms.platform_musa import PlatformMUSA

        platform = PlatformMUSA()
        assert platform.device_name == "musa"
        assert platform.vendor_name == "moore_threads"
        assert platform.communication_backend_name() == "mccl"

    def test_tpu_device_and_vendor_names(self):
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.device_name == "tpu"
        assert platform.vendor_name == "google"
        assert platform.communication_backend_name() == "tpu_dist"

    def test_tpu_ray_resource_options(self):
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.ray_resource_name() == "TPU"
        assert platform.ray_resource_options(4) == {"resources": {"TPU": 4}}
        assert platform.ray_resource_options(0) == {}

    def test_tpu_core_hooks(self):
        """The three PlatformBase hooks TPU needs from verl core.

        Inert on verl 0.9.0 -- nothing calls them yet. They are defined
        unconditionally so the plugin needs no change when core lands the call sites.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.supports_colocated_worker_groups() is False
        assert "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS" in platform.ray_noset_envvars()
        with mock.patch.dict(os.environ, {"TPU_VISIBLE_CHIPS": "3"}):
            assert platform.ray_local_rank_override() == "3"

    def test_tpu_derives_from_platform_base(self):
        """PlatformTPU must not acquire another vendor's platform behaviour by inheritance.

        Deriving from a CUDA platform would silently supply CUDA answers for methods TPU
        never defined. PlatformBase keeps every answer explicit in the class, and makes a
        newly added abstract method fail loudly at instantiation instead.
        """
        from verl.plugin.platform.platform_base import PlatformBase
        from verl.plugin.platform.platform_cuda import PlatformCUDA
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        assert issubclass(PlatformTPU, PlatformBase)
        assert not issubclass(PlatformTPU, PlatformCUDA)
        assert PlatformTPU.__abstractmethods__ == frozenset()

    def test_tpu_no_cuda_collective_or_rollout_env(self):
        """Neither CUDA answer is usable on TPU, so both must be the vendor-neutral default.

        cupy's NCCL binding cannot drive a TPU interconnect, and NCCL_CUMEM_ENABLE has no
        meaning in a TPU rollout worker.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.get_collective_module() is None
        assert platform.rollout_env_vars() == {}

    def test_tpu_memory_and_capability_methods(self):
        """empty_cache() must reach the TPU device module, never torch.cuda.empty_cache()."""
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.empty_cache() is None
        assert platform.set_allocator_settings("expandable_segments:True") is None
        assert platform.get_device_capability() == (None, None)

    def test_tpu_ray_noset_envvars(self):
        """Both entries are load-bearing, including the CUDA one.

        visible_devices_envvar() deliberately returns CUDA_VISIBLE_DEVICES, so Ray must be
        told not to manage that variable either. Dropping it breaks rank mapping.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        assert PlatformTPU().ray_noset_envvars() == [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        ]

    def test_tpu_ipc_unsupported(self):
        """TPU has no CUDA-style IPC handle, so verl must fall back to shared memory.

        verl computes ``use_shm = not is_support_ipc()``. Reporting True would route weight
        transfer down the CUDA IPC path, which cannot work on TPU: torch in the TPU image is
        a CPU build and ``_share_cuda_()`` raises.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        assert PlatformTPU().is_ipc_supported() is False

    def test_tpu_keeps_cuda_visible_devices_envvar(self):
        """Intentional, not an oversight: TPU keeps CUDA_VISIBLE_DEVICES.

        Returning TPU_VISIBLE_CHIPS here would let vllm_async_server overwrite the
        chip index that get_worker_env_vars() writes and ray_local_rank_override()
        reads. See spec D4.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        assert PlatformTPU().visible_devices_envvar() == "CUDA_VISIBLE_DEVICES"

    def test_tpu_cudart_returns_none(self):
        """There is no CUDA runtime on a TPU host; PlatformBase documents None as the answer."""
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        assert PlatformTPU().cudart() is None

    def test_tpu_profiler_is_noop(self):
        """PlatformBase asks platforms without profiling support for no-ops.

        verl calls both from utils/profiler/nvtx_profile.py; a CUDA implementation would
        raise AssertionError ("Torch not compiled with CUDA enabled") on a TPU host.
        """
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        platform = PlatformTPU()
        assert platform.profiler_start() is None
        assert platform.profiler_stop() is None

    def test_tpu_nvtx_range_yields(self):
        """nvtx_range must yield immediately; inherited torch.cuda.nvtx raises without NVTX."""
        from verl_hardware_plugin.platforms.platform_tpu import PlatformTPU

        entered = False
        with PlatformTPU().nvtx_range("tpu-test"):
            entered = True
        assert entered

    def test_supa_detection_with_env(self):
        from verl.plugin.platform.platform_manager import _detect_platform_name
        from verl_hardware_plugin.platforms.platform_supa import PlatformSupa  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "biren"}):
                assert _detect_platform_name() == "biren"

    def test_supa_device_and_vendor_names(self):
        from verl_hardware_plugin.platforms.platform_supa import PlatformSupa

        platform = PlatformSupa()
        assert platform.device_name == "supa"
        assert platform.vendor_name == "biren"
        assert platform.communication_backend_name() == "bccl"


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

    def test_fsdp_iluvatar_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_iluvatar import (
            FSDPIluvatarEngineWithLMHead,
            FSDPIluvatarEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("cuda", "iluvatar")] is FSDPIluvatarEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("cuda", "iluvatar")] is FSDPIluvatarEngineWithValueHead

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

    def test_megatron_iluvatar_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_iluvatar import MegatronIluvatarEngineWithLMHead

        assert (
            EngineRegistry._engines["language_model"]["megatron"][("cuda", "iluvatar")]
            is MegatronIluvatarEngineWithLMHead
        )

    def test_megatron_musa_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_musa import (
            MegatronMUSAEngineWithLMHead,
            MegatronMUSAEngineWithValueHead,
        )

        assert (
            EngineRegistry._engines["language_model"]["megatron"][("musa", "moore_threads")]
            is MegatronMUSAEngineWithLMHead
        )
        assert (
            EngineRegistry._engines["value_model"]["megatron"][("musa", "moore_threads")]
            is MegatronMUSAEngineWithValueHead
        )

    def test_fsdp_musa_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_musa import (
            FSDPMUSAEngineWithLMHead,
            FSDPMUSAEngineWithValueHead,
        )

        for backend in ("fsdp", "fsdp2"):
            assert (
                EngineRegistry._engines["language_model"][backend][("musa", "moore_threads")]
                is FSDPMUSAEngineWithLMHead
            )
            assert (
                EngineRegistry._engines["value_model"][backend][("musa", "moore_threads")]
                is FSDPMUSAEngineWithValueHead
            )

    def test_fsdp_enflame_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_enflame import (
            FSDPEnflameEngineWithLMHead,
            FSDPEnflameEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"][("gcu", "enflame")] is FSDPEnflameEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"][("gcu", "enflame")] is FSDPEnflameEngineWithValueHead

    def test_megatron_enflame_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_enflame import MegatronEnflameEngineWithLMHead

        assert (
            EngineRegistry._engines["language_model"]["megatron"][("gcu", "enflame")] is MegatronEnflameEngineWithLMHead
        )


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

    def test_reuses_already_imported_module(self):
        from verl_hardware_plugin.utils import may_enable_flag_gems

        # Simulate flag_gems already loaded
        fake_module = mock.MagicMock()
        fake_module.__version__ = "0.1.0"
        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1"}):
            with mock.patch.dict(sys.modules, {"flag_gems": fake_module}):
                may_enable_flag_gems(phase="training")
                fake_module.enable.assert_called_once_with(record=True, once=True, path=None)
                fake_module.only_enable.assert_not_called()

    def test_enable_all_ops(self):
        from verl_hardware_plugin.utils import may_enable_flag_gems

        fake_module = mock.MagicMock()
        fake_module.__version__ = "0.1.0"
        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGGEMS_ENABLE": "1"}, clear=True):
            # Ensure flag_gems is NOT in sys.modules so the import path is taken
            with mock.patch.dict(sys.modules, {}, clear=False):
                sys.modules.pop("flag_gems", None)
                with mock.patch(
                    "builtins.__import__",
                    side_effect=lambda name, *a, **kw: (
                        fake_module if name == "flag_gems" else __import__(name, *a, **kw)
                    ),
                ):
                    may_enable_flag_gems(phase="training")
                    fake_module.enable.assert_called_once()

    def test_raises_on_whitelist_and_blacklist(self):
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
                    side_effect=lambda name, *a, **kw: (
                        fake_module if name == "flag_gems" else __import__(name, *a, **kw)
                    ),
                ):
                    with pytest.raises(ValueError, match="Cannot set both whitelist and blacklist"):
                        may_enable_flag_gems(phase="training")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
