# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process rollout's switches: default off, validated, and scoped.

* With ``VERL_TPU_ROLLOUT_INPROC`` unset the plugin behaves as without the feature: the registration imports nothing of
  it and registers no rollout class, no worker group gets the colocated trainer's variables, and the platform import
  runs no check.
* With it set, only the trainer worker groups get the pins; a malformed switch value refuses; a ``VERL_TPU_INPROC_*``
  variable the package does not read is named in one warning; TP / DP / PP above 1 and a missing KV budget refuse.
"""

import importlib
import logging
import subprocess
import sys
from types import SimpleNamespace

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import common

PLATFORM = "verl_hardware_plugin.accelerators.tpu.platform_tpu"


@pytest.mark.parametrize(
    "value, on", [(None, False), ("", False), ("0", False), ("true", False), ("1", True), (" 1 ", True)]
)
def test_feature_switch(value, on):
    env = {} if value is None else {common.INPROC_ENV: value}
    assert common.inproc_enabled(env) is on
    if not on:
        with pytest.raises(RuntimeError, match="not switched on"):
            common.check_switches(env)


@pytest.mark.parametrize(
    "name, value",
    [
        (common.INTEGRITY_ENV, "flag"),
        (common.PARITY_EVERY_ENV, "five"),
        (common.PARITY_EVERY_ENV, "-1"),
        (common.LAST_STEP_ENV, "x"),
        (common.GC_RSS_GIB_ENV, "-2"),
        (common.GC_RSS_GIB_ENV, "lots"),
    ],
)
def test_malformed_switch_values_refuse(name, value):
    with pytest.raises(RuntimeError, match=name):
        common.check_switches({common.INPROC_ENV: "1", name: value})


def test_switch_defaults():
    env = {common.INPROC_ENV: "1"}
    assert common.check_switches(env) == []
    assert common.integrity_mode(env) == "fatal" and common.parity_every(env) == 5 and common.last_step(env) is None
    assert common.gc_rss_limit_bytes(env) == 8 * 2**30
    assert common.gc_rss_limit_bytes({common.GC_RSS_GIB_ENV: "0"}) == 0
    assert common.integrity_mode({common.INTEGRITY_ENV: "report"}) == "report"


def test_unread_variables_are_named():
    env = {common.INPROC_ENV: "1", "VERL_TPU_INPROC_SOMETHING": "1", common.LAST_STEP_ENV: "8", "VERL_TPU_OTHER": "1"}
    assert common.check_switches(env) == ["VERL_TPU_INPROC_SOMETHING"]


def _platform(monkeypatch, env: dict):
    for k in (common.INPROC_ENV, common.ROLE_ENV):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.import_module(PLATFORM)


def test_trainer_pins_only_with_the_switch(monkeypatch):
    p = _platform(monkeypatch, {})
    assert not p.inproc_on()
    p = _platform(monkeypatch, {common.INPROC_ENV: "1"})
    assert p.inproc_on()
    pins = p.inproc_trainer_env()
    assert pins["SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"] == "0" and pins[common.ROLE_ENV] == "trainer"
    monkeypatch.setenv("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    assert p.inproc_trainer_env()["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] == "false"


def _worker_env(p, name_prefix: str) -> dict:
    plat = p.PlatformTPU.__new__(p.PlatformTPU)
    plat.ray_noset_envvars = lambda: []
    plat.get_tpu_env_vars = lambda **kw: {"LOCAL_RANK": "0"}
    pool = SimpleNamespace(get_placement_groups=lambda device_name: [])
    return plat.get_worker_env_vars(pool, 0, 1, 0, 1, name_prefix, "TPU")


@pytest.mark.parametrize("prefix, pinned", [("actor_rollout_ref", True), ("ref", True), ("rollout_server", False)])
def test_worker_groups_get_the_pins(monkeypatch, prefix, pinned):
    p = _platform(monkeypatch, {common.INPROC_ENV: "1"})
    env = _worker_env(p, prefix)
    assert (env.get(common.ROLE_ENV) == "trainer") is pinned
    p = _platform(monkeypatch, {})
    assert common.ROLE_ENV not in _worker_env(p, prefix)  # default off: nothing added


def _run(code: str, env: dict) -> str:
    base = {k: v for k, v in __import__("os").environ.items() if not k.startswith("VERL_TPU_")}
    r = subprocess.run([sys.executable, "-B", "-c", code], env={**base, **env}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_default_off_registration_is_inert():
    """A fresh process that imports the plugin's TPU registration with the switch unset: nothing of the in-process
    package is imported, verl's ("vllm", "async") rollout is the upstream one, ActorRolloutRefWorker gains no method,
    and the
    environment is unchanged."""
    code = """
import json, os, sys
before = dict(os.environ)
from verl.workers.rollout import base
upstream = dict(base._ROLLOUT_REGISTRY)
from verl.workers.engine_workers import ActorRolloutRefWorker as W
methods = set(vars(W))
import verl_hardware_plugin.accelerators.tpu.registration
print(json.dumps({
    "imported": sorted(m for m in sys.modules if ".rollout.inproc" in m),
    "registry_same": dict(base._ROLLOUT_REGISTRY) == upstream,
    "methods_same": set(vars(W)) == methods,
    "env_same": dict(os.environ) == before,
}))
"""
    out = __import__("json").loads(_run(code, {}).strip().splitlines()[-1])
    assert out == {"imported": [], "registry_same": True, "methods_same": True, "env_same": True}


def test_registration_with_the_switch():
    code = """
import json
import verl_hardware_plugin.accelerators.tpu.registration
from verl.workers.rollout import base
from verl.workers.engine_workers import ActorRolloutRefWorker as W
direct = ("generate", "abort_all_requests", "resume_generation", "inproc_server_info")
wrapped = ("init_model", "update_actor", "compute_log_prob")
print(json.dumps({"rollout": base._ROLLOUT_REGISTRY[("vllm", "async")],
                  "methods": sorted(m for m in direct if m in vars(W)),
                  "wrapped": sorted(m for m in wrapped if getattr(vars(W)[m], "_inproc_method", False))}))
"""
    out = __import__("json").loads(_run(code, {"VERL_TPU_ROLLOUT_INPROC": "1"}).strip().splitlines()[-1])
    assert out["rollout"] == "verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout.TPUInProcRollout"
    assert out["methods"] == ["abort_all_requests", "generate", "inproc_server_info", "resume_generation"]
    assert out["wrapped"] == ["compute_log_prob", "init_model", "update_actor"]


def test_tp_above_one_refuses(monkeypatch):
    """Colocation builds one TP=1 engine per trainer rank: TP / DP / PP above 1 refuses before anything is built."""
    from verl.workers.config import RolloutConfig
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setenv(common.INPROC_ENV, "1")
    for tp, dp in ((2, 1), (1, 2), (4, 2)):  # (verl's RolloutConfig itself refuses pipeline parallelism for vLLM)
        cfg = RolloutConfig(
            name="vllm",
            mode="async",
            tensor_model_parallel_size=tp,
            data_parallel_size=dp,
            pipeline_model_parallel_size=1,
        )
        with pytest.raises(RuntimeError, match="needs TP=DP=PP=1 per replica"):
            TPUInProcRollout(cfg, None, None, defer_build=True)


def test_constructor_warns_about_unread_variables(monkeypatch, caplog):
    from verl.workers.config import RolloutConfig
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setenv(common.INPROC_ENV, "1")
    monkeypatch.setenv("VERL_TPU_INPROC_KV_BYTES", "10737418240")
    caplog.set_level(logging.WARNING, logger=common.logger.name)
    cfg = RolloutConfig(
        name="vllm", mode="async", tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=1
    )
    TPUInProcRollout(cfg, None, None, defer_build=True)
    assert any("['VERL_TPU_INPROC_KV_BYTES'] are set but not read" in m for m in caplog.messages)


@pytest.mark.parametrize("kv", [None, 0, -1])
def test_kv_budget_comes_from_the_config(monkeypatch, kv):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", [])
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.config = {"engine_kwargs": {"vllm": {} if kv is None else {"kv_cache_memory_bytes": kv}}}
    r.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3"))
    r.rank = 0
    with pytest.raises(RuntimeError, match="kv_cache_memory_bytes must be a positive byte count"):
        r._prepare()


@pytest.mark.parametrize(
    "config, env, want",
    [
        ({"enable_prefix_caching": True}, {}, "enable_prefix_caching=True"),
        ({"full_determinism": True}, {}, "full_determinism=True"),
        ({}, {"VERL_FULL_DETERMINISM": "1"}, "VERL_FULL_DETERMINISM='1'"),
    ],
)
def test_build_refuses_engine_settings_that_leak_into_the_trainer(monkeypatch, config, env, want):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", [])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.config = config
    r.rank = 0
    with pytest.raises(RuntimeError, match=want):
        r._prepare()


def test_build_refuses_after_a_failed_platform_import_check(monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", ["vllm_torchtpu.env_override would add [x]"])
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.config, r.rank = {}, 0
    with pytest.raises(RuntimeError, match="the platform-import libtpu check failed"):
        r._prepare()
