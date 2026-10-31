# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the TPU options in ``options.py``: engine, ``tpu-sync`` weight sync and platform."""

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import TorchtitanEngineConfig
from verl_hardware_plugin.accelerators.tpu import options as tpu_options
from verl_hardware_plugin.accelerators.tpu.engines.tpu_utils import (
    extend_torchtitan_engine_config,
    get_tpu_seq_bucket_size,
    resolve_tpu_torchtitan_options,
    resolve_tpu_use_fused_linear_cross_entropy,
)
from verl_hardware_plugin.accelerators.tpu.options import (
    TPUEngineOptions,
    TPUSyncOptions,
    get_tpu_engine_options,
    get_tpu_sync_options,
    tpu_chips_per_host_override,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env_name in (*tpu_options._ENGINE_LEGACY_ENV_VARS.values(), *tpu_options._SYNC_LEGACY_ENV_VARS.values()):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv("VERL_TPU_CHIPS_PER_HOST", raising=False)
    monkeypatch.setattr(tpu_options, "_warned", set())


def _cfg(tpu=None, **fields):
    return SimpleNamespace(engine_kwargs={"tpu": tpu} if tpu is not None else {}, **fields)


def test_engine_kwargs_are_parsed_and_typed():
    options = get_tpu_engine_options(
        _cfg(
            {
                "use_splash_attention": "true",
                "use_simple_fsdp": False,
                "fuse_qkv": "false",
                "tpu_eager_mode": "DEFER_NEVER",
                "optim_impl": "for-loop",
                "seq_bucket_size": "512",
                "context_parallel_strategy": "ulysses",
                "ep_capacity_factor": "4.0",
            }
        )
    )
    assert options == TPUEngineOptions(
        use_splash_attention=True,
        use_simple_fsdp=False,
        fuse_qkv=False,
        tpu_eager_mode="DEFER_NEVER",
        optim_impl="for-loop",
        seq_bucket_size=512,
        context_parallel_strategy="ulysses",
        ep_capacity_factor=4.0,
    )
    assert get_tpu_engine_options(_cfg()) == TPUEngineOptions()
    assert get_tpu_engine_options(None) == TPUEngineOptions()


@pytest.mark.parametrize(
    ("tpu", "match"),
    [
        ({"use_splash_attn": True}, "Unknown TPU engine option"),
        ({"use_simple_fsdp": "maybe"}, "Invalid TPU engine option use_simple_fsdp"),
        ({"seq_bucket_size": 0}, "Invalid TPU engine option seq_bucket_size"),
    ],
)
def test_bad_engine_kwargs_raise(tpu, match):
    with pytest.raises(ValueError, match=match):
        get_tpu_engine_options(_cfg(tpu))


def test_resolvers_read_engine_kwargs():
    compiled = _cfg({"tpu_eager_mode": "NONE", "use_simple_fsdp": False}, use_torch_compile=True)
    assert resolve_tpu_torchtitan_options(compiled) == (True, False, None)
    assert resolve_tpu_use_fused_linear_cross_entropy(compiled) is True
    no_fused = _cfg({"use_fused_linear_cross_entropy": False}, use_torch_compile=True)
    assert resolve_tpu_use_fused_linear_cross_entropy(no_fused) is False
    assert get_tpu_seq_bucket_size(_cfg({"seq_bucket_size": 128}, pad_to_length=True, pad_to_length_bucket=512)) == 128
    assert get_tpu_seq_bucket_size(_cfg(pad_to_length=True, pad_to_length_bucket=512)) == 512


def test_legacy_field_and_env_still_work_with_a_warning(monkeypatch, caplog):
    legacy = _cfg(use_torch_compile=True, use_splash_attention=None, use_simple_fsdp=False, tpu_eager_mode=None)
    with caplog.at_level(logging.WARNING, logger=tpu_options.__name__):
        assert get_tpu_engine_options(legacy).use_simple_fsdp is False
    assert "torchtitan.use_simple_fsdp is deprecated" in caplog.text

    # engine_kwargs wins over the deprecated field; the environment still wins over both, as before.
    both = _cfg({"use_simple_fsdp": True}, use_simple_fsdp=False)
    assert get_tpu_engine_options(both).use_simple_fsdp is True
    monkeypatch.setenv("VERL_TPU_USE_SIMPLE_FSDP", "0")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=tpu_options.__name__):
        assert get_tpu_engine_options(both).use_simple_fsdp is False
    assert "VERL_TPU_USE_SIMPLE_FSDP='0'" in caplog.text and "deprecated" in caplog.text


def test_engine_kwargs_survive_verl_config_conversion():
    """Hydra/OmegaConf -> TorchtitanEngineConfig, as verl builds the engine config for every worker."""
    extend_torchtitan_engine_config()
    raw = OmegaConf.create(
        {
            "_target_": "verl.workers.config.TorchtitanEngineConfig",
            "use_torch_compile": True,
            "engine_kwargs": {"tpu": {"use_simple_fsdp": False, "tpu_eager_mode": "DEFER_NEVER"}},
        }
    )
    cfg = omega_conf_to_dataclass(raw)
    assert isinstance(cfg, TorchtitanEngineConfig)
    assert resolve_tpu_torchtitan_options(cfg) == (True, False, "DEFER_NEVER")
    assert omega_conf_to_dataclass(OmegaConf.create({"_target_": raw._target_})).engine_kwargs == {}


def test_tpu_sync_options_are_parsed_and_typed():
    assert get_tpu_sync_options(None) == TPUSyncOptions()
    assert get_tpu_sync_options({}) == TPUSyncOptions(
        parallelism=8, verify_parity="off", release_buffers_after_sync=True, tie_word_embeddings=None
    )
    # Command-line overrides may arrive as strings.
    options = get_tpu_sync_options(
        {"parallelism": "4", "verify_parity": "true", "release_buffers_after_sync": "0", "tie_word_embeddings": 1}
    )
    assert options == TPUSyncOptions(
        parallelism=4, verify_parity="norm", release_buffers_after_sync=False, tie_word_embeddings=True
    )


@pytest.mark.parametrize(
    ("value", "mode", "norms", "exact"),
    [
        (None, "off", False, False),
        (False, "off", False, False),
        ("false", "off", False, False),
        (True, "norm", True, False),
        ("True", "norm", True, False),
        ("norm", "norm", True, False),
        ("exact", "exact", False, True),
        (" ALL ", "all", True, True),
    ],
)
def test_tpu_sync_verify_parity_modes(value, mode, norms, exact):
    options = get_tpu_sync_options({} if value is None else {"verify_parity": value})
    assert (options.verify_parity, options.verify_norms, options.verify_exact) == (mode, norms, exact)


@pytest.mark.parametrize("kwargs", [{"paralelism": 4}, {"parallelism": 0}, {"verify_parity": "maybe"}])
def test_tpu_sync_options_reject_unknown_keys_and_bad_values(kwargs):
    with pytest.raises(ValueError, match="engine_kwargs.tpu-sync"):
        get_tpu_sync_options(kwargs)


def test_tpu_sync_options_name_the_configured_backend():
    with pytest.raises(ValueError, match="engine_kwargs.my-tpu-sync"):
        get_tpu_sync_options({"bogus": 1}, backend="my-tpu-sync")


@pytest.mark.parametrize(
    ("kwargs", "env", "expected"),
    [
        ({"release_buffers_after_sync": False}, "1", True),  # the deprecated variable still wins
        ({}, "false", False),
        ({"release_buffers_after_sync": False}, "", False),  # empty is unset
    ],
)
def test_release_buffers_env_var_is_deprecated(monkeypatch, caplog, kwargs, env, expected):
    monkeypatch.setenv("VERL_TPU_SYNC_RELEASE_BUFFERS", env)
    with caplog.at_level(logging.WARNING, logger=tpu_options.__name__):
        assert get_tpu_sync_options(kwargs).release_buffers_after_sync is expected
        get_tpu_sync_options(kwargs)
    warnings = [r.getMessage() for r in caplog.records if "VERL_TPU_SYNC_RELEASE_BUFFERS" in r.getMessage()]
    if env:
        # One warning per process, pointing at the config key that replaces it.
        assert len(warnings) == 1
        assert "deprecated" in warnings[0]
        assert "checkpoint_engine.engine_kwargs.tpu-sync.release_buffers_after_sync" in warnings[0]
    else:
        assert warnings == []


def test_tpu_chips_per_host_override(monkeypatch):
    assert tpu_chips_per_host_override() is None
    monkeypatch.setenv("VERL_TPU_CHIPS_PER_HOST", "4")
    assert tpu_chips_per_host_override() == 4
    monkeypatch.setenv("VERL_TPU_CHIPS_PER_HOST", "four")
    with pytest.raises(ValueError, match="VERL_TPU_CHIPS_PER_HOST"):
        tpu_chips_per_host_override()
