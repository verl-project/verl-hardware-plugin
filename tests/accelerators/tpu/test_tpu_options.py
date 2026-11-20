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
                "rollout_expert_parallel": "true",
                "param_dtype": "bfloat16",
                "optimizer_host_offload": "true",
                "fp32_master": "true",
                "ep_token_chunks": "2",
                "ep_paired_grouped_mm": "true",
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
        rollout_expert_parallel=True,
        param_dtype="bfloat16",
        optimizer_host_offload=True,
        fp32_master=True,
        ep_token_chunks=2,
        ep_paired_grouped_mm=True,
    )
    assert get_tpu_engine_options(_cfg()) == TPUEngineOptions()
    assert get_tpu_engine_options(None) == TPUEngineOptions()


@pytest.mark.parametrize(
    ("tpu", "match"),
    [
        ({"use_splash_attn": True}, "Unknown TPU engine option"),
        ({"use_simple_fsdp": "maybe"}, "Invalid TPU engine option use_simple_fsdp"),
        ({"seq_bucket_size": 0}, "Invalid TPU engine option seq_bucket_size"),
        ({"seq_bucket_size": True}, "Invalid TPU engine option seq_bucket_size"),
        ({"ep_capacity_factor": 0}, "Invalid TPU engine option ep_capacity_factor"),
        ({"ep_capacity_factor": "nan"}, "Invalid TPU engine option ep_capacity_factor"),
        ({"ep_capacity_factor": True}, "Invalid TPU engine option ep_capacity_factor"),
        ({"param_dtype": "float16"}, "param_dtype must be float32 or bfloat16"),
        ({"fp32_master": True}, "fp32_master needs param_dtype=bfloat16"),
        ({"param_dtype": "float32", "fp32_master": True}, "fp32_master needs param_dtype=bfloat16"),
        ({"context_parallel_strategy": "ring"}, "context_parallel_strategy must be one of"),
        (
            {"context_parallel_gdn_strategy": "all_to_all", "context_parallel_gdn_reduce_dtype": "bfloat16"},
            "context_parallel_gdn_reduce_dtype only applies when",
        ),
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


def test_loss_row_bucketing_and_static_micro_batch_num_seqs_options():
    assert get_tpu_engine_options(_cfg()).loss_row_bucketing is None
    assert get_tpu_engine_options(_cfg()).static_micro_batch_num_seqs is None

    opts = get_tpu_engine_options(_cfg({"loss_row_bucketing": "false", "static_micro_batch_num_seqs": "8"}))
    assert opts.loss_row_bucketing is False
    assert opts.static_micro_batch_num_seqs == 8

    opts_on = get_tpu_engine_options(_cfg({"loss_row_bucketing": True, "static_micro_batch_num_seqs": 1}))
    assert opts_on.loss_row_bucketing is True
    assert opts_on.static_micro_batch_num_seqs == 1

    for bad_bool in ("maybe", 2):
        with pytest.raises(ValueError, match="loss_row_bucketing"):
            get_tpu_engine_options(_cfg({"loss_row_bucketing": bad_bool}))

    for bad_seqs in (0, -1, -8, "0", "-3", True, False):
        with pytest.raises(ValueError, match="static_micro_batch_num_seqs"):
            get_tpu_engine_options(_cfg({"static_micro_batch_num_seqs": bad_seqs}))

    for bad_direct in (0, -1, True, False):
        with pytest.raises(ValueError, match="static_micro_batch_num_seqs"):
            TPUEngineOptions(static_micro_batch_num_seqs=bad_direct)  # type: ignore[arg-type]


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
        parallelism=8,
        group_size=16,
        verify_parity="off",
        release_buffers_after_sync=True,
        tie_word_embeddings=None,
        return_after_d2h=True,
        receive_into_vllm="auto",
        direct_install_quantized_moe=True,
    )
    # Command-line overrides may arrive as strings.
    options = get_tpu_sync_options(
        {
            "parallelism": "4",
            "group_size": "128",
            "verify_parity": "true",
            "release_buffers_after_sync": "0",
            "tie_word_embeddings": 1,
            "return_after_d2h": "false",
            "receive_into_vllm": "false",
            "direct_install_quantized_moe": "false",
        }
    )
    assert options == TPUSyncOptions(
        parallelism=4,
        group_size=128,
        verify_parity="norm",
        release_buffers_after_sync=False,
        tie_word_embeddings=True,
        return_after_d2h=False,
        receive_into_vllm="off",
        direct_install_quantized_moe=False,
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        (" AUTO ", "auto"),
        (True, "on"),
        ("true", "on"),
        ("on", "on"),
        (False, "off"),
        ("false", "off"),
        ("off", "off"),
    ],
)
def test_tpu_sync_receive_into_vllm_modes(value, expected):
    options = get_tpu_sync_options({} if value is None else {"receive_into_vllm": value})
    assert options.receive_into_vllm == expected


@pytest.mark.parametrize(
    ("value", "untied", "tied"),
    [
        (None, "auto", "off"),
        ("auto", "auto", "off"),
        (" AUTO ", "auto", "off"),
        ("off", "off", "off"),
        (False, "off", "off"),
        ("false", "off", "off"),
        ("on", "on", None),
        (True, "on", None),
        ("true", "on", None),
    ],
)
def test_tpu_sync_receive_into_vllm_with_tie_word_embeddings(caplog, value, untied, tied):
    """With tie_word_embeddings, only on (True) is refused: auto resolves to off, logged once, and off stays off.

    receive_into_vllm never fills lm_head from the embedding. tie_word_embeddings=False changes nothing.
    """
    kwargs = {} if value is None else {"receive_into_vllm": value}
    assert get_tpu_sync_options({**kwargs, "tie_word_embeddings": False}).receive_into_vllm == untied
    if tied is None:
        with pytest.raises(ValueError, match=r"receive_into_vllm=on \(True\) cannot be combined with tie_word_embed"):
            get_tpu_sync_options({**kwargs, "tie_word_embeddings": True})
        return
    with caplog.at_level(logging.INFO, logger=tpu_options.__name__):
        for _ in range(2):  # the driver parses the options at every sync
            options = get_tpu_sync_options({**kwargs, "tie_word_embeddings": "true"})
            assert (options.receive_into_vllm, options.tie_word_embeddings) == (tied, True)
    fallback = [r.getMessage() for r in caplog.records if "receive_into_vllm=auto falling back" in r.getMessage()]
    if untied == "auto":
        assert len(fallback) == 1
        assert "tie_word_embeddings is set" in fallback[0]
        assert fallback[0].startswith("checkpoint_engine.engine_kwargs.tpu-sync: ")
    else:
        assert fallback == []


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


@pytest.mark.parametrize(
    "kwargs", [{"paralelism": 4}, {"parallelism": 0}, {"group_size": 0}, {"verify_parity": "maybe"}]
)
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


def test_ep_dispatcher_and_topo_mesh_options():
    from types import SimpleNamespace

    opts = tpu_options.get_tpu_engine_options(
        SimpleNamespace(engine_kwargs={"tpu": {"ep_dispatcher": " RAGGED ", "enable_tpu_topo_mesh": True}})
    )
    assert opts.ep_dispatcher == "ragged"
    assert opts.enable_tpu_topo_mesh is True

    opts_static = tpu_options.get_tpu_engine_options(
        SimpleNamespace(engine_kwargs={"tpu": {"ep_dispatcher": "static", "enable_tpu_topo_mesh": False}})
    )
    assert opts_static.ep_dispatcher == "static"
    assert opts_static.enable_tpu_topo_mesh is False

    with pytest.raises(ValueError, match="engine_kwargs.tpu.ep_dispatcher"):
        tpu_options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"ep_dispatcher": "deepep"}}))


def test_ac_save_splash_and_routing_option():
    from types import SimpleNamespace

    opts = tpu_options.get_tpu_engine_options(
        SimpleNamespace(engine_kwargs={"tpu": {"ac_save_splash_and_routing": "true"}})
    )
    assert opts.ac_save_splash_and_routing is True
    opts_off = tpu_options.get_tpu_engine_options(
        SimpleNamespace(engine_kwargs={"tpu": {"ac_save_splash_and_routing": False}})
    )
    assert opts_off.ac_save_splash_and_routing is False


def test_rollout_fp8_moe_option():
    from types import SimpleNamespace

    # Default unset: neither flag is forced.
    opts = tpu_options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {}}))
    assert opts.rollout_expert_parallel is None and opts.rollout_fp8_moe is None

    # rollout_fp8_moe=True implies rollout_expert_parallel=True when unset.
    opts_on = tpu_options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"rollout_fp8_moe": True}}))
    assert opts_on.rollout_fp8_moe is True and opts_on.rollout_expert_parallel is True

    # Explicit rollout_fp8_moe=False leaves rollout_expert_parallel untouched.
    opts_off = tpu_options.get_tpu_engine_options(
        SimpleNamespace(engine_kwargs={"tpu": {"rollout_expert_parallel": True, "rollout_fp8_moe": False}})
    )
    assert opts_off.rollout_expert_parallel is True and opts_off.rollout_fp8_moe is False

    # Conflicting rollout_fp8_moe=True + rollout_expert_parallel=False fails loudly.
    with pytest.raises(ValueError, match="rollout_fp8_moe=True conflicts with rollout_expert_parallel=False"):
        tpu_options.get_tpu_engine_options(
            SimpleNamespace(engine_kwargs={"tpu": {"rollout_expert_parallel": False, "rollout_fp8_moe": True}})
        )


def test_context_parallel_strategy_and_gdn_option_validation():
    opts = tpu_options.get_tpu_engine_options(
        _cfg(
            {
                "context_parallel_strategy": " ALL_GATHER ",
                "context_parallel_gdn_strategy": " ALLGATHER_REDUCESCATTER ",
                "context_parallel_gdn_reduce_dtype": " BFLOAT16 ",
            }
        )
    )
    assert opts.context_parallel_strategy == "all_gather"
    assert opts.context_parallel_gdn_strategy == "allgather_reducescatter"
    assert opts.context_parallel_gdn_reduce_dtype == "bfloat16"

    for tpu_dict in (
        {"context_parallel_gdn_strategy": "all_to_all"},
        {"context_parallel_gdn_reduce_dtype": "float32"},
    ):
        with pytest.raises(ValueError, match="context_parallel_size > 1"):
            resolve_tpu_torchtitan_options(_cfg(tpu_dict, context_parallel_size=1, use_torch_compile=False))


def test_cp_gdn_slot_and_static_logprob_options():
    opts = tpu_options.get_tpu_engine_options(
        SimpleNamespace(
            engine_kwargs={
                "tpu": {
                    "cp_full_positions": "true",
                    "gdn_auto_row_slot_len": "32768",
                    "fuse_gdn_projections": True,
                    "sort_micro_batches_by_slot": "1",
                    "static_log_prob_micro_batch_len": 65536,
                }
            }
        )
    )
    assert opts.cp_full_positions is True
    assert opts.gdn_auto_row_slot_len == 32768
    assert opts.fuse_gdn_projections is True
    assert opts.sort_micro_batches_by_slot is True
    assert opts.static_log_prob_micro_batch_len == 65536

    with pytest.raises(ValueError, match="Invalid TPU engine option gdn_auto_row_slot_len"):
        tpu_options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": {"gdn_auto_row_slot_len": 0}}))

    with pytest.raises(ValueError, match="Invalid TPU engine option static_log_prob_micro_batch_len"):
        tpu_options.get_tpu_engine_options(
            SimpleNamespace(engine_kwargs={"tpu": {"static_log_prob_micro_batch_len": -4}})
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        tpu_options.get_tpu_engine_options(
            SimpleNamespace(engine_kwargs={"tpu": {"gdn_auto_row_slot_len": 16384, "gdn_row_slot_len": 16384}})
        )

    for bad_static_cfg in (
        {"gdn_auto_row_slot_len": 32768, "static_micro_batch_len": 16384},
        {"gdn_auto_row_slot_len": 32768, "static_micro_batch_len": 49152},
        {"gdn_auto_row_slot_len": 32768, "static_log_prob_micro_batch_len": 16384},
        {"gdn_auto_row_slot_len": 32768, "static_log_prob_micro_batch_len": 40000},
    ):
        with pytest.raises(ValueError, match="divisible by gdn_auto_row_slot_len"):
            tpu_options.get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": bad_static_cfg}))

    valid_static = tpu_options.get_tpu_engine_options(
        SimpleNamespace(
            engine_kwargs={
                "tpu": {
                    "gdn_auto_row_slot_len": 32768,
                    "static_micro_batch_len": 32768,
                    "static_log_prob_micro_batch_len": 98304,
                }
            }
        )
    )
    assert valid_static.static_micro_batch_len == 32768
    assert valid_static.static_log_prob_micro_batch_len == 98304
