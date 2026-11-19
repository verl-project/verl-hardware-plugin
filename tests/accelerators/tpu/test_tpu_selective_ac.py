# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for ``engines/tpu_selective_ac.py``: the ``engine_kwargs.tpu.selective_ac_*`` options and where the engine
finds torchtitan's ``TPUSelectiveAC``.

The tests at the end use the installed torchtitan, if any: each option either builds its ``TPUSelectiveAC`` with the
flag set and the save set the flag promises, or, with a torchtitan whose policy lacks the flag, fails closed.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import logging
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from verl_hardware_plugin.accelerators.tpu import options as tpu_options
from verl_hardware_plugin.accelerators.tpu.engines import tpu_selective_ac
from verl_hardware_plugin.accelerators.tpu.options import SELECTIVE_AC_FLAG_OPTIONS, get_tpu_engine_options

_FLAGS = tuple(SELECTIVE_AC_FLAG_OPTIONS.values())
_ALL_ON = {option: True for option in SELECTIVE_AC_FLAG_OPTIONS}


@dataclasses.dataclass(kw_only=True, slots=True)
class _FlagsConfig:
    """Stands in for torchtitan's ``TPUSelectiveAC.Config`` with every flag, plus an unrelated field."""

    preserve_rng_state: bool = True
    recompute_bmm: bool = False
    save_pallas_attention: bool = False
    save_kda_residuals: bool = False


@dataclasses.dataclass(kw_only=True, slots=True)
class _BmmAndSplashConfig:
    """A ``TPUSelectiveAC.Config`` with the flags of google-pytorch/torchtitan#168 only."""

    preserve_rng_state: bool = True
    recompute_bmm: bool = False
    save_pallas_attention: bool = False


@dataclasses.dataclass(kw_only=True, slots=True)
class _NoFlagsConfig:
    """A selective config without flags (core ``SelectiveAC.Config``, or ``TPUSelectiveAC`` before the flags)."""

    preserve_rng_state: bool = True


class _Op:
    """A hashable stand-in for a custom op overload, matched by its qualified name."""

    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


def _options(**tpu):
    return get_tpu_engine_options(SimpleNamespace(engine_kwargs={"tpu": tpu}))


def _flag_values(config) -> dict[str, bool]:
    return {flag: getattr(config, flag) for flag in _FLAGS}


def test_options_map_to_the_torchtitan_flags():
    assert SELECTIVE_AC_FLAG_OPTIONS == {
        "selective_ac_recompute_bmm": "recompute_bmm",
        "selective_ac_save_pallas_attention": "save_pallas_attention",
        "selective_ac_save_kda_residuals": "save_kda_residuals",
    }
    assert all(tpu_options._ENGINE_KINDS[option] is bool for option in SELECTIVE_AC_FLAG_OPTIONS)
    assert tpu_selective_ac.requested_flags(_options()) == {}
    assert tpu_selective_ac.requested_flags(_options(**_ALL_ON)) == dict.fromkeys(_FLAGS, True)


@pytest.mark.parametrize(
    "mode, config", [("selective", _FlagsConfig()), ("selective", _NoFlagsConfig()), ("full", object()), ("none", None)]
)
def test_without_the_options_every_config_is_left_alone(mode, config, caplog):
    all_off = {option: False for option in SELECTIVE_AC_FLAG_OPTIONS}
    with caplog.at_level(logging.WARNING, logger=tpu_selective_ac.logger.name):
        assert tpu_selective_ac.selective_ac_config(config, mode, _options()) is config
        assert tpu_selective_ac.selective_ac_config(config, mode, _options(**all_off)) is config
    assert "selective AC policy" not in caplog.text


@pytest.mark.parametrize("option", list(SELECTIVE_AC_FLAG_OPTIONS))
def test_each_option_sets_only_its_flag_on_a_copy(option, caplog):
    source = _FlagsConfig(preserve_rng_state=False)
    with caplog.at_level(logging.WARNING, logger=tpu_selective_ac.logger.name):
        config = tpu_selective_ac.selective_ac_config(source, "selective", _options(**{option: True}))
    assert config is not source and type(config) is _FlagsConfig
    assert _flag_values(config) == {flag: flag == SELECTIVE_AC_FLAG_OPTIONS[option] for flag in _FLAGS}
    assert config.preserve_rng_state is False  # every other field is kept
    assert not any(_flag_values(source).values())  # the input is untouched
    assert "selective AC policy: _FlagsConfig(" in caplog.text


def test_the_options_compose_and_the_policy_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=tpu_selective_ac.logger.name):
        config = tpu_selective_ac.selective_ac_config(_FlagsConfig(), "selective", _options(**_ALL_ON))
    assert all(_flag_values(config).values())
    assert "recompute_bmm=True, save_pallas_attention=True, save_kda_residuals=True" in caplog.text


@pytest.mark.parametrize("mode, config", [("full", object()), ("none", None)])
def test_the_options_need_selective_ac(mode, config):
    with pytest.raises(ValueError, match=r"engine_kwargs\.tpu\.selective_ac_recompute_bmm: only for activation_check"):
        tpu_selective_ac.selective_ac_config(config, mode, _options(selective_ac_recompute_bmm=True))


def test_a_requested_flag_that_the_policy_lacks_fails_closed():
    with pytest.raises(ValueError, match=r"has no field \['recompute_bmm'\]"):
        tpu_selective_ac.selective_ac_config(_NoFlagsConfig(), "selective", _options(selective_ac_recompute_bmm=True))
    with pytest.raises(ValueError, match="has no field"):
        tpu_selective_ac.selective_ac_config(object(), "selective", _options(selective_ac_save_pallas_attention=True))
    # A torchtitan with #168 but without #170: its two flags work, save_kda_residuals is refused by name.
    both = {"selective_ac_recompute_bmm": True, "selective_ac_save_pallas_attention": True}
    config = tpu_selective_ac.selective_ac_config(_BmmAndSplashConfig(), "selective", _options(**both))
    assert config.recompute_bmm is True and config.save_pallas_attention is True
    with pytest.raises(ValueError, match=r"has no field \['save_kda_residuals'\].*torchtitan#170"):
        tpu_selective_ac.selective_ac_config(_BmmAndSplashConfig(), "selective", _options(**_ALL_ON))


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_tpu_selective_ac_is_looked_up_in_components_then_in_the_older_module_then_core():
    new, old = tpu_selective_ac.TPU_SELECTIVE_AC_MODULES
    assert new == "torchtitan.experiments.tpu.components.activation_checkpoint"
    assert old == "torchtitan.experiments.tpu.activation_checkpoint"
    new_cls, old_cls = type("TPUSelectiveAC", (), {}), type("TPUSelectiveAC", (), {})
    core_cls = type("SelectiveAC", (), {})
    core = _module("torchtitan.distributed.activation_checkpoint", SelectiveAC=core_cls)
    cases = [
        ({new: _module(new, TPUSelectiveAC=new_cls), old: _module(old, TPUSelectiveAC=old_cls)}, new_cls),
        ({new: None, old: _module(old, TPUSelectiveAC=old_cls)}, old_cls),  # a torchtitan from before components/
        ({new: _module(new), old: None}, core_cls),  # a module without the class counts as absent
        ({new: None, old: None}, core_cls),  # no TPUSelectiveAC at all: core SelectiveAC, as before
    ]
    for modules, expected in cases:
        with mock.patch.dict(sys.modules, {**modules, "torchtitan.distributed.activation_checkpoint": core}):
            assert tpu_selective_ac.tpu_selective_ac_class() is expected


def test_the_module_imports_torchtitan_only_when_called():
    tree = ast.parse(open(tpu_selective_ac.__file__, encoding="utf-8").read())
    names = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    names += [node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("torchtitan") for name in names)


def _installed_policy():
    """The class the engine uses with the installed torchtitan, checked against torchtitan's own modules."""
    core = pytest.importorskip("torchtitan.distributed.activation_checkpoint")
    expected = core.SelectiveAC
    for module_name in (
        "torchtitan.experiments.tpu.components.activation_checkpoint",
        "torchtitan.experiments.tpu.activation_checkpoint",  # torchtitan before components/
    ):
        try:
            expected = importlib.import_module(module_name).TPUSelectiveAC
            break
        except ImportError:
            continue
    policy_cls = tpu_selective_ac.tpu_selective_ac_class()
    assert policy_cls is expected
    return policy_cls


def _policy_flags(policy_cls) -> set[str]:
    return {f.name for f in dataclasses.fields(policy_cls.Config)} & set(_FLAGS)


def test_with_the_installed_torchtitan_the_engine_finds_its_tpu_selective_ac():
    policy_cls = _installed_policy()
    if policy_cls.__name__ != "TPUSelectiveAC":
        pytest.skip("the installed torchtitan has no TPUSelectiveAC")


_SPLASH_FORWARD = _Op("pallas::splash_attn_fwd_0123456789abcdef")
_KDA_FORWARD = (_Op("tt_qwen35::gated_delta_fwd_res"), _Op("tt_qwen35::gated_delta_segments_fwd_res"))
_KDA_BACKWARD = _Op("tt_qwen35::gated_delta_bwd_res")


def _check_save_set(flag: str, save_ops, default_ops) -> None:
    """The save-set change each flag promises, against the policy built without flags."""
    torch = pytest.importorskip("torch")
    if flag == "recompute_bmm":
        assert torch.ops.aten.bmm.default in default_ops and torch.ops.aten.bmm.default not in save_ops
    elif flag == "save_pallas_attention":
        assert _SPLASH_FORWARD not in default_ops and _SPLASH_FORWARD in save_ops
    else:
        assert not any(op in default_ops for op in _KDA_FORWARD) and all(op in save_ops for op in _KDA_FORWARD)
        assert _KDA_BACKWARD not in save_ops


@pytest.mark.parametrize("option", list(SELECTIVE_AC_FLAG_OPTIONS))
def test_with_the_installed_torchtitan_each_option_builds_its_flag_or_fails_closed(option):
    policy_cls = _installed_policy()
    flag = SELECTIVE_AC_FLAG_OPTIONS[option]
    source = policy_cls.Config(preserve_rng_state=False)
    if flag not in _policy_flags(policy_cls):
        with pytest.raises(ValueError, match=rf"has no field \['{flag}'\]"):
            tpu_selective_ac.selective_ac_config(source, "selective", _options(**{option: True}))
        return
    config = tpu_selective_ac.selective_ac_config(source, "selective", _options(**{option: True}))
    assert type(config) is policy_cls.Config and getattr(config, flag) is True
    assert config.preserve_rng_state is False
    policy = config.build(dump_folder="")
    assert type(policy) is policy_cls
    _check_save_set(flag, policy.get_save_ops(), source.build(dump_folder="").get_save_ops())


def test_with_the_installed_torchtitan_all_options_compose():
    policy_cls = _installed_policy()
    if _policy_flags(policy_cls) != set(_FLAGS):
        pytest.skip("the installed torchtitan's TPUSelectiveAC does not have all three flags")
    source = policy_cls.Config()
    config = tpu_selective_ac.selective_ac_config(source, "selective", _options(**_ALL_ON))
    save_ops = config.build(dump_folder="").get_save_ops()
    default_ops = source.build(dump_folder="").get_save_ops()
    for flag in _FLAGS:
        _check_save_set(flag, save_ops, default_ops)
