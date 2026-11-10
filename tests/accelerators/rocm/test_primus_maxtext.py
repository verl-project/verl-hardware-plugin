# Copyright 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the Primus MaxText integration: loss, layout conversion, configs, launcher."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("verl.workers.config")
torch = pytest.importorskip("torch")

from verl_hardware_plugin.accelerators.rocm import launch  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus import resolver, yaml_config  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus.maxtext import export, hf_import  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus.maxtext.batch import next_token_logprobs  # noqa: E402
from verl_hardware_plugin.accelerators.rocm.primus.maxtext.loss import clipped_policy_loss  # noqa: E402

PARAMS = {"learning_rate": 2.0e-6, "per_device_batch_size": 1, "max_target_length": 128, "hf_path": "allenai/c4"}


@pytest.fixture
def fake_primus(monkeypatch):
    monkeypatch.setattr(yaml_config, "load_module_params", lambda *a, **k: dict(PARAMS))
    monkeypatch.setattr(yaml_config, "module_framework", lambda *a, **k: "maxtext")
    resolver.register_resolver()


def test_loss_matches_verl_vanilla():
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
    from verl.workers.config import ActorConfig

    rng = np.random.default_rng(0)
    shape = (4, 16)
    log_prob = rng.normal(-1.0, 0.3, shape).astype(np.float32)
    old = log_prob + rng.normal(0, 0.3, shape).astype(np.float32)
    adv = rng.normal(0, 1, shape).astype(np.float32)
    mask = (rng.random(shape) > 0.3).astype(np.float32)
    loss, _ = clipped_policy_loss(np, log_prob, old, adv, mask, 0.2, 0.28, 3.0)

    config = ActorConfig(
        strategy="fsdp",
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        clip_ratio_c=3.0,
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
    )
    config.global_batch_info["dp_size"] = 1
    config.global_batch_info["batch_num_tokens"] = int(mask.sum())
    config.global_batch_info["global_batch_size"] = shape[0]
    want, _ = compute_policy_loss_vanilla(
        torch.from_numpy(old), torch.from_numpy(log_prob), torch.from_numpy(adv), torch.from_numpy(mask), config=config
    )
    assert float(loss) == pytest.approx(float(want), rel=1e-5)


def test_next_token_logprobs_align_with_torch():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(2, 6, 11)).astype(np.float32)
    tokens = rng.integers(0, 11, size=(2, 6))
    got = next_token_logprobs(logits, tokens)
    want = torch.log_softmax(torch.from_numpy(logits[:, :-1]), -1).gather(-1, torch.from_numpy(tokens[:, 1:, None]))
    np.testing.assert_allclose(got[:, :-1], want[..., 0].numpy(), rtol=1e-5, atol=1e-5)
    assert np.all(got[:, -1] == 0)


def _tiny_hf(layers=2, heads=4, kv_heads=2, head_dim=8, hidden=16, experts=4, inter=6, vocab=32):
    rng = np.random.default_rng(2)

    def w(*shape):
        return rng.normal(size=shape).astype(np.float32)

    hf = {
        "model.embed_tokens.weight": w(vocab, hidden),
        "model.norm.weight": w(hidden),
        "lm_head.weight": w(vocab, hidden),
    }
    for layer in range(layers):
        p = f"model.layers.{layer}"
        hf[f"{p}.input_layernorm.weight"] = w(hidden)
        hf[f"{p}.post_attention_layernorm.weight"] = w(hidden)
        hf[f"{p}.self_attn.q_proj.weight"] = w(heads * head_dim, hidden)
        hf[f"{p}.self_attn.k_proj.weight"] = w(kv_heads * head_dim, hidden)
        hf[f"{p}.self_attn.v_proj.weight"] = w(kv_heads * head_dim, hidden)
        hf[f"{p}.self_attn.o_proj.weight"] = w(hidden, heads * head_dim)
        hf[f"{p}.self_attn.q_norm.weight"] = w(head_dim)
        hf[f"{p}.self_attn.k_norm.weight"] = w(head_dim)
        hf[f"{p}.mlp.gate.weight"] = w(experts, hidden)
        for e in range(experts):
            hf[f"{p}.mlp.experts.{e}.gate_proj.weight"] = w(inter, hidden)
            hf[f"{p}.mlp.experts.{e}.up_proj.weight"] = w(inter, hidden)
            hf[f"{p}.mlp.experts.{e}.down_proj.weight"] = w(hidden, inter)
    return hf, dict(num_layers=layers, num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim, num_experts=experts)


class _DictReader:
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def get(self, name, dtype):
        self.reads.append(name)
        return self.tensors[name].astype(dtype)

    def get_many(self, names, dtype):
        return [self.get(name, dtype) for name in names]


def _blocks(shape):
    """Split the first two axes in half, like a 2x2 device mesh."""
    halves = [[slice(0, n // 2), slice(n // 2, n)] if n > 1 else [slice(0, n)] for n in shape[:2]]
    rest = tuple(slice(None) for _ in shape[2:])
    return [(a, b) + rest for a in halves[0] for b in (halves[1] if len(shape) > 1 else [()])]


def _assemble(old, read_block):
    out = np.empty(old.shape, dtype=old.dtype)
    for block in _blocks(old.shape) if old.ndim > 1 else [(slice(None),)]:
        block = tuple(b for b in block if b != ())
        out[block] = read_block(block)
    return out


def test_hf_import_blocks_match_stacked_layout():
    hf, dims = _tiny_hf()
    stacked = export.stack_qwen3_for_test(hf, **dims)
    reader = _DictReader(hf)
    loaded = hf_import.load_stacked_from_hf(reader, stacked, _assemble)

    def leaves(tree, prefix=()):
        for key, value in tree.items():
            if isinstance(value, dict):
                yield from leaves(value, prefix + (key,))
            else:
                yield prefix + (key,), value

    for path, want in leaves(stacked):
        np.testing.assert_array_equal(hf_import._get(loaded, path), want, err_msg="/".join(path))
    # A block of the expert leaves reads only its own experts.
    expert_reads = [name for name in reader.reads if ".experts." in name]
    assert len(expert_reads) == len(set(expert_reads))


def test_export_round_trips_hf_names():
    hf, dims = _tiny_hf()
    stacked = export.stack_qwen3_for_test(hf, **dims)
    back = export.hf_from_stacked(stacked, dims["num_heads"], dims["num_kv_heads"], dims["head_dim"])
    assert set(back) == set(hf)
    for name, value in hf.items():
        np.testing.assert_array_equal(back[name], value, err_msg=name)


def test_launcher_selects_maxtext(fake_primus, tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("modules: {}\n")
    overrides = launch.build_overrides(str(exp), "pre_trainer", [])
    assert overrides[0] == "model_engine=maxtext"
    assert f"actor_rollout_ref.actor.maxtext.primus_config_path={exp.resolve()}" in overrides
    # MaxText's hf_path is a dataset, never the model.
    assert not any(item.startswith("actor_rollout_ref.model.path=") for item in overrides)


def test_hydra_groups_compose_with_verl(fake_primus):
    hydra = pytest.importorskip("hydra")
    import verl
    from verl.utils.config import omega_conf_to_dataclass
    from verl_hardware_plugin.accelerators.rocm.hydra_searchpath import register_search_path
    from verl_hardware_plugin.accelerators.rocm.primus.maxtext.config import MaxTextActorConfig, MaxTextEngineConfig

    register_search_path()

    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(
            "ppo_trainer",
            overrides=[
                "model_engine=maxtext",
                "actor_rollout_ref.actor.maxtext.primus_config_path=/exp.yaml",
                "actor_rollout_ref.actor.ppo_mini_batch_size=8",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            ],
        )
    from omegaconf import OmegaConf

    # main_ppo resolves the whole config in place before building workers.
    OmegaConf.resolve(cfg)
    actor_node = cfg.actor_rollout_ref.actor
    assert actor_node.strategy == "maxtext"
    assert actor_node.optim.lr == pytest.approx(2.0e-6)
    assert cfg.actor_rollout_ref.ref.maxtext.primus_config_path == "/exp.yaml"
    actor = omega_conf_to_dataclass(actor_node)
    assert isinstance(actor, MaxTextActorConfig)
    assert isinstance(actor.engine, MaxTextEngineConfig)
    assert actor.engine.primus_config_path == "/exp.yaml"


def test_detach_worker_gets_maxtext_handlers():
    separation = pytest.importorskip("verl.experimental.separation.engine_workers")
    from types import SimpleNamespace

    from verl_hardware_plugin.accelerators.rocm.primus.maxtext.engine import install_detach_handlers

    install_detach_handlers()
    worker = SimpleNamespace(_strategy_handlers=None, config=SimpleNamespace(actor=SimpleNamespace(strategy="maxtext")))
    save, restore = separation.DetachActorWorker._get_strategy_handlers(worker)
    assert save(None) is None
    with pytest.raises(NotImplementedError, match="parameter_sync_step=1"):
        restore(None, None)
