# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the Primus Megatron integration: YAML -> verl config, checks, launcher."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.workers.config")
from omegaconf import OmegaConf  # noqa: E402

from verl_hardware_plugin.accelerators.rocm import launch  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus import resolver, yaml_config  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus.megatron.config import (  # noqa: E402
    PrimusMcoreEngineConfig,
    apply_primus_params,
    validate_primus_capabilities,
)

PARAMS = {
    "tensor_model_parallel_size": 1,
    "context_parallel_size": 2,
    "expert_model_parallel_size": 4,
    "sequence_parallel": False,
    "bf16": True,
    "lr": 1.0e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "lr_warmup_iters": 0,
    "recompute_granularity": "full",
    "moe_permute_fusion": True,
    "use_packed_attention": True,
    "hf_path": "/models/qwen3",
}


@pytest.fixture
def fake_primus(monkeypatch):
    def load(config_path, module="pre_trainer", overrides=(), framework=None, primus_path=None):
        params = dict(PARAMS)
        for item in overrides:
            key, value = item.split("=", 1)
            params[key] = int(value) if value.isdigit() else value
        return params

    monkeypatch.setattr(yaml_config, "load_module_params", load)
    monkeypatch.setattr(yaml_config, "module_framework", lambda *a, **k: "megatron")
    resolver.register_resolver()


def _engine(**kwargs):
    defaults = dict(
        primus_config_path="/exp.yaml",
        context_parallel_size=2,
        expert_model_parallel_size=4,
        sequence_parallel=False,
        use_remove_padding=True,
    )
    defaults.update(kwargs)
    return PrimusMcoreEngineConfig(**defaults)


def test_resolver_reads_yaml_and_derived_fields(fake_primus):
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "megatron": {
                        "primus_config_path": "/exp.yaml",
                        "primus_overrides": ["context_parallel_size=4"],
                        "cp": "${primus:context_parallel_size,1}",
                        "dtype": "${primus:dtype,float16}",
                        "missing": "${primus:pipeline_model_parallel_size,1}",
                    },
                    "optim": {
                        "betas": ["${primus:adam_beta1,0.9}", "${primus:adam_beta2,0.999}"],
                        "warmup": "${primus:lr_warmup_steps,-1}",
                    },
                }
            }
        }
    )
    node = cfg.actor_rollout_ref.actor
    assert node.megatron.cp == 4
    assert node.megatron.dtype == "bfloat16"
    assert node.megatron.missing == 1
    assert list(node.optim.betas) == [0.9, 0.98]
    assert node.optim.warmup == 0


def test_resolver_returns_default_without_primus_yaml(fake_primus):
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"megatron": {"primus_config_path": "", "cp": "${primus:context_parallel_size,1}"}}
            }
        }
    )
    assert cfg.actor_rollout_ref.actor.megatron.cp == 1


def test_config_keeps_megatron_strategy():
    config = PrimusMcoreEngineConfig()
    assert config.strategy == "megatron"
    assert not config.primus_enabled


def test_apply_folds_transformer_overrides():
    config = _engine()
    apply_primus_params(config, PARAMS)
    assert config.override_transformer_config["recompute_granularity"] == "full"
    assert config.override_transformer_config["moe_permute_fusion"] is True
    assert config.override_transformer_config["attention_backend"] == "fused"
    assert config._primus_params["context_parallel_size"] == 2


def test_apply_keeps_explicit_transformer_override():
    config = _engine(override_transformer_config={"recompute_granularity": "selective"})
    apply_primus_params(config, PARAMS)
    assert config.override_transformer_config["recompute_granularity"] == "selective"


def test_apply_rejects_parallel_mismatch():
    with pytest.raises(ValueError, match="context_parallel_size"):
        apply_primus_params(_engine(context_parallel_size=1), PARAMS)


def test_apply_mismatch_allowed_when_not_strict():
    apply_primus_params(_engine(context_parallel_size=1, primus_strict_config=False), PARAMS)


def test_capabilities_require_thd_for_packed_attention():
    config = _engine(use_remove_padding=False)
    apply_primus_params(config, PARAMS)
    model = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]))
    with pytest.raises(ValueError, match="use_remove_padding"):
        validate_primus_capabilities(config, PARAMS, model)


def test_capabilities_reject_turbo_attention():
    config = _engine()
    params = dict(PARAMS, use_turbo_attention=True)
    apply_primus_params(config, params)
    with pytest.raises(ValueError, match="use_turbo_attention"):
        validate_primus_capabilities(config, params, SimpleNamespace(hf_config=None))


def test_launcher_maps_yaml_to_verl_overrides(fake_primus, tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("modules: {}\n")
    overrides = launch.build_overrides(str(exp), "pre_trainer", ["trainer.nnodes=1"])
    assert overrides[0] == "model_engine=primus"
    assert f"actor_rollout_ref.actor.megatron.primus_config_path={exp.resolve()}" in overrides
    assert "actor_rollout_ref.model.path=/models/qwen3" in overrides
    assert "actor_rollout_ref.model.use_remove_padding=True" in overrides
    assert overrides[-1] == "trainer.nnodes=1"


def test_launcher_keeps_user_model_path(fake_primus, tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("modules: {}\n")
    overrides = launch.build_overrides(str(exp), "pre_trainer", ["actor_rollout_ref.model.path=/mine"])
    assert [item for item in overrides if item.startswith("actor_rollout_ref.model.path=")] == [
        "actor_rollout_ref.model.path=/mine"
    ]


def test_hydra_groups_compose_with_verl(fake_primus):
    hydra = pytest.importorskip("hydra")
    import verl
    from verl_hardware_plugin.accelerators.rocm.hydra_searchpath import register_search_path

    register_search_path()

    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    if not (config_dir / "actor" / "megatron_actor.yaml").exists():
        pytest.skip("installed verl has no megatron actor config")
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            "ppo_trainer",
            overrides=["model_engine=primus", "actor_rollout_ref.actor.megatron.primus_config_path=/exp.yaml"],
        )
    # main_ppo resolves the whole config in place before building workers.
    OmegaConf.resolve(cfg)
    actor = cfg.actor_rollout_ref.actor
    assert list(actor.optim.betas) == [0.9, 0.98]
    assert actor.strategy == "megatron"
    assert actor.megatron._target_.endswith("PrimusMcoreEngineConfig")
    assert actor.megatron.context_parallel_size == 2
    assert actor.megatron.expert_model_parallel_size == 4
    assert actor.optim.lr == pytest.approx(1.0e-6)
    assert actor.megatron.override_transformer_config.recompute_granularity == "full"
    assert actor.megatron.override_transformer_config.attention_backend == "fused"
    assert cfg.actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend == "fused"
    assert cfg.critic.megatron.override_transformer_config.recompute_granularity == "full"
    assert cfg.actor_rollout_ref.ref.megatron.context_parallel_size == 2
    assert cfg.critic.megatron.context_parallel_size == 2
