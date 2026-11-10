# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the trainer-only libtpu flags (``VERL_TPU_TRAINER_LIBTPU_INIT_ARGS``)."""

from __future__ import annotations

import os
import types
from unittest import mock

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu import platform_tpu
from verl_hardware_plugin.accelerators.tpu.platform_tpu import PlatformTPU, trainer_libtpu_init_args

TRAINER_LIBTPU_ENV = platform_tpu.TRAINER_EXTRA_LIBTPU_INIT_ARGS_ENV

JOB = "--xla_a=1 --xla_tpu_dvfs_p_state=0 --xla_b=true"
TRAINER = "--xla_tpu_dvfs_p_state=7,--xla_enable_async_all_gather=true"


@pytest.fixture(autouse=True)
def _fresh_log_state(monkeypatch):
    monkeypatch.setattr(platform_tpu, "_trainer_libtpu_logged", set())


def test_unset_keeps_the_job_value_for_every_group():
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": JOB}, clear=True):
        for group in ("Ab12cd", "actor_rollout_ref", "rollout_standalone_0"):
            assert trainer_libtpu_init_args(group) is None


def test_rollout_groups_keep_the_job_value():
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": JOB, TRAINER_LIBTPU_ENV: TRAINER}, clear=True):
        assert trainer_libtpu_init_args("rollout_standalone_0") is None
        assert trainer_libtpu_init_args("rollout_colocate_0") is None


def test_trainer_groups_merge_the_flags_over_the_job_value(capsys):
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": JOB, TRAINER_LIBTPU_ENV: TRAINER}, clear=True):
        args = trainer_libtpu_init_args("Ab12cd")
        again = trainer_libtpu_init_args("Ab12cd")
    # The trainer value wins for a flag both name; the job's other flags stay, in order.
    assert args == "--xla_a=1 --xla_b=true --xla_tpu_dvfs_p_state=7 --xla_enable_async_all_gather=true"
    assert again == args
    # One line per worker group.
    assert capsys.readouterr().out.count("worker group Ab12cd: trainer LIBTPU_INIT_ARGS=") == 1


def test_space_separated_flags_and_no_job_value():
    env = {TRAINER_LIBTPU_ENV: "--xla_x=1 --xla_y=2"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert trainer_libtpu_init_args("trainer") == "--xla_x=1 --xla_y=2"


@pytest.mark.parametrize("bad", ["xla_x=1", "--xla_x", "--Xla_x=1", "--xla-x=1"])
def test_entries_must_be_flag_equals_value(bad):
    with mock.patch.dict(os.environ, {TRAINER_LIBTPU_ENV: f"--xla_ok=1,{bad}"}, clear=True):
        with pytest.raises(ValueError, match="entries must be --flag=value"):
            trainer_libtpu_init_args("trainer")


def _worker_env(name_prefix: str) -> dict[str, str]:
    tpu = types.ModuleType("torch.tpu")
    with mock.patch.object(torch, "tpu", tpu, create=True):
        platform = PlatformTPU()
    pool = mock.Mock()
    pool.get_placement_groups.return_value = []
    with mock.patch.object(PlatformTPU, "get_tpu_env_vars", return_value={"LOCAL_RANK": "0"}):
        return platform.get_worker_env_vars(
            pool,
            rank=0,
            world_size=8,
            local_rank=0,
            local_world_size=8,
            name_prefix=name_prefix,
            device_name="tpu",
        )


def test_worker_env_of_trainer_and_rollout_groups():
    env = {"LIBTPU_INIT_ARGS": JOB, TRAINER_LIBTPU_ENV: TRAINER, "VERL_PLATFORM": "tpu"}
    with mock.patch.dict(os.environ, env, clear=True):
        trainer = _worker_env("Ab12cd")
        rollout = _worker_env("rollout_standalone_0")
    assert trainer["LIBTPU_INIT_ARGS"] == (
        "--xla_a=1 --xla_b=true --xla_tpu_dvfs_p_state=7 --xla_enable_async_all_gather=true"
    )
    assert "LIBTPU_INIT_ARGS" not in rollout  # the job's value is inherited unchanged
    assert trainer["LOCAL_RANK"] == rollout["LOCAL_RANK"] == "0"
    with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": JOB, "VERL_PLATFORM": "tpu"}, clear=True):
        assert "LIBTPU_INIT_ARGS" not in _worker_env("Ab12cd")
