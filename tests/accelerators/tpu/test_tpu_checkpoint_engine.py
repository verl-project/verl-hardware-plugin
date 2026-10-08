# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for TPUCheckpointEngine, TPUWeightRegistry, and worker weight sharding/fusion."""

import logging
import sys
from types import ModuleType

import torch
import torch.nn as nn

for _mod_name in ("uvicorn", "fastapi"):
    if _mod_name not in sys.modules:
        try:
            __import__(_mod_name)
        except ImportError:
            _stub = ModuleType(_mod_name)
            _stub.FastAPI = object  # type: ignore[attr-defined]
            _stub.Server = object  # type: ignore[attr-defined]
            _stub.Config = object  # type: ignore[attr-defined]
            sys.modules[_mod_name] = _stub

from verl_hardware_plugin.accelerators.tpu.engines.ray_weight_registry import RayWeightRegistryState  # noqa: E402
from verl_hardware_plugin.accelerators.tpu.engines.tpu_checkpoint_engine import (  # noqa: E402
    TPUCheckpointEngine,
    apply_tpu_checkpoint_engine_hooks,
    get_clean_name,
    get_layer_group,
    load_weights_on_worker,
)


def test_tpu_checkpoint_engine_registered():
    from verl.checkpoint_engine.base import CheckpointEngineRegistry

    assert CheckpointEngineRegistry.get("tpu") is TPUCheckpointEngine


def test_apply_tpu_checkpoint_engine_hooks_logs_importerror(caplog):
    with caplog.at_level(logging.DEBUG, logger="verl_hardware_plugin.accelerators.tpu.engines.tpu_checkpoint_engine"):
        apply_tpu_checkpoint_engine_hooks()
    assert any("caused ImportError" in rec.message for rec in caplog.records)


def test_ray_weight_registry_write_eviction():
    """Verify write-based eviction: step 0 of a new job evicts step 5 from a previous job."""
    reg = RayWeightRegistryState()
    reg.set_weights(5, ["old_ref"])
    assert reg.get_weights(5) == ["old_ref"]

    # New job writes step 0 -> step 5 must be evicted
    reg.set_weights(0, ["new_ref"])
    assert reg.get_weights(0) == ["new_ref"]
    assert reg.get_weights(5) is None

    reg.clear()
    assert reg.get_weights(0) is None


def test_clean_name_and_layer_group():
    raw = "_fsdp_wrapped_module._checkpoint_wrapped_module.module.model.layers.12.self_attn.q_proj.weight"
    assert get_clean_name(raw) == "model.layers.12.self_attn.q_proj.weight"
    assert get_layer_group(raw) == "layers.12"
    assert get_layer_group("model.tok_embeddings.weight") == "embeddings"
    assert get_layer_group("model.norm.weight") == "output"


class _DummyRolloutAttention(nn.Module):
    def __init__(self, tp_size: int = 2, flipped: bool = False):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.qkv_proj.weight.data.zero_()
        self.qkv_proj.tp_size = tp_size
        self.qkv_proj.num_kv_head_replicas = 1
        self.qkv_proj._tpu_weight_flipped = flipped


class _DummyRolloutModel(nn.Module):
    def __init__(self, tp_size: int = 2, flipped: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"self_attn": _DummyRolloutAttention(tp_size, flipped)})])
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)


def test_pack_and_load_weights_with_qkv_fusion_and_tp_sharding():
    q_w = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    k_w = (torch.arange(8, dtype=torch.bfloat16) + 100).reshape(2, 4)
    v_w = (torch.arange(8, dtype=torch.bfloat16) + 200).reshape(2, 4)
    emb_w = (torch.arange(32, dtype=torch.bfloat16) + 300).reshape(8, 4)

    def weight_gen():
        yield ("_fsdp_wrapped_module.layers.0.self_attn.q_proj.weight", q_w)
        yield ("_fsdp_wrapped_module.layers.0.self_attn.k_proj.weight", k_w)
        yield ("_fsdp_wrapped_module.layers.0.self_attn.v_proj.weight", v_w)
        yield ("_fsdp_wrapped_module.tok_embeddings.weight", emb_w)

    state_dict = TPUCheckpointEngine.pack_weights_to_grouped_dict(weight_gen())
    assert set(state_dict["grouped"].keys()) == {"layers.0", "embeddings"}

    model_r0 = _DummyRolloutModel(tp_size=2, flipped=False)
    model_r1 = _DummyRolloutModel(tp_size=2, flipped=False)

    keys_r0 = load_weights_on_worker(model_r0, state_dict, rank=0, target_device="cpu")
    keys_r1 = load_weights_on_worker(model_r1, state_dict, rank=1, target_device="cpu")
    assert keys_r0 > 0
    assert keys_r1 > 0

    expected_r0_qkv = torch.cat([q_w[:2], k_w[:1], v_w[:1]], dim=0)
    expected_r1_qkv = torch.cat([q_w[2:], k_w[1:], v_w[1:]], dim=0)

    assert torch.equal(model_r0.layers[0]["self_attn"].qkv_proj.weight.data, expected_r0_qkv)
    assert torch.equal(model_r1.layers[0]["self_attn"].qkv_proj.weight.data, expected_r1_qkv)
    assert torch.equal(model_r0.lm_head.weight.data, emb_w)

    # Verify _tpu_weight_flipped=True transposes fused 2D weights to (n_in, n_out)
    model_flipped = _DummyRolloutModel(tp_size=2, flipped=True)
    load_weights_on_worker(model_flipped, state_dict, rank=0, target_device="cpu")
    assert torch.equal(model_flipped.layers[0]["self_attn"].qkv_proj.weight.data, expected_r0_qkv.T)


def test_update_tpu_weights_clears_kv_cache_and_sets_global_steps(monkeypatch):
    """After loading new weights every replica must drop its KV cache and learn the weight version.

    upstream verl stamps each trajectory with the server's ``global_steps`` and converts it with
    ``dtype=int`` in ``_compute_metrics``; a server that was never told its version yields None
    there and the first training step dies.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import ray

    from verl_hardware_plugin.accelerators.tpu.engines import tpu_checkpoint_engine as engine_mod

    fake_registry = SimpleNamespace(
        get_weights=SimpleNamespace(remote=MagicMock(return_value="published")),
        clear=SimpleNamespace(remote=MagicMock(side_effect=lambda: AsyncMock()())),
    )
    monkeypatch.setattr(ray, "get_actor", MagicMock(return_value=fake_registry))
    monkeypatch.setattr(ray, "get", MagicMock(side_effect=lambda ref: ref))

    fake_server_handle = SimpleNamespace(
        collective_rpc=SimpleNamespace(remote=MagicMock(side_effect=lambda **kw: AsyncMock(return_value=[3])())),
        clear_kv_cache=SimpleNamespace(remote=MagicMock(side_effect=lambda: AsyncMock()())),
        set_global_steps=SimpleNamespace(remote=MagicMock(side_effect=lambda s: AsyncMock()())),
    )
    fake_manager = SimpleNamespace(
        backend="tpu",
        actor_wg=SimpleNamespace(update_weights=MagicMock(return_value=None)),
        replicas=[SimpleNamespace(server_handle=fake_server_handle)],
        abort_replicas=AsyncMock(),
        resume_generation_replicas=AsyncMock(),
    )

    asyncio.run(engine_mod.update_tpu_weights(fake_manager, global_steps=5))

    fake_manager.abort_replicas.assert_awaited_once()
    fake_server_handle.collective_rpc.remote.assert_called_once_with(method="load_weights_from_ray_registry", args=(5,))
    fake_server_handle.clear_kv_cache.remote.assert_called_once()
    fake_server_handle.set_global_steps.remote.assert_called_once_with(5)
    fake_manager.resume_generation_replicas.assert_awaited_once()
