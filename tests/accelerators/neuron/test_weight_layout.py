# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU layout oracles, not substitutes for real NIXL hardware transfers."""

import os

import pytest
import torch

from verl_hardware_plugin.accelerators.neuron.weight_sync.streaming import iter_weight_buckets, validate_box_coverage
from verl_hardware_plugin.accelerators.neuron.weight_sync.weight_layout import (
    apply_box,
    pack_box,
    plan_box,
    plan_buckets,
    slot_numel_for,
)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("bucket_bytes", [32, 128, 4096])
@pytest.mark.parametrize("tied", [False, True])
def test_dense_tp_boxes_match_independent_full_tensor_oracle(tp_size, bucket_bytes, tied):
    device = os.environ.get("NEURON_TEST_LAYOUT_DEVICE", "cpu")
    if device != "cpu":
        from importlib import import_module

        import_module("libtorch_neuronx_lite")
    prefix = "model.layers.0.self_attn."
    shapes = {
        "model.embed_tokens.weight": (16, 8),
        prefix + "q_proj.weight": (16, 8),
        prefix + "k_proj.weight": (4, 8),
        prefix + "v_proj.weight": (4, 8),
        prefix + "o_proj.weight": (8, 16),
        "model.layers.0.mlp.gate_proj.weight": (16, 8),
        "model.layers.0.mlp.up_proj.weight": (16, 8),
        "model.layers.0.mlp.down_proj.weight": (8, 16),
        "model.norm.weight": (8,),
    }
    if not tied:
        shapes["lm_head.weight"] = (16, 8)
    generator = torch.Generator().manual_seed(123)
    weights = {name: torch.randn(shape, dtype=torch.bfloat16, generator=generator) for name, shape in shapes.items()}
    schedule = [(name, shape, "bfloat16") for name, shape in shapes.items()]
    buckets = plan_buckets(schedule, bucket_bytes=bucket_bytes)
    for rank in range(tp_size):
        query = weights[prefix + "q_proj.weight"].chunk(tp_size, dim=0)[rank]
        kv_shards = min(4, tp_size)
        kv_rank = rank // max(1, tp_size // 4)
        key = weights[prefix + "k_proj.weight"].chunk(kv_shards, dim=0)[kv_rank]
        value = weights[prefix + "v_proj.weight"].chunk(kv_shards, dim=0)[kv_rank]
        expected = {
            "model.embed_tokens.weight": weights["model.embed_tokens.weight"].chunk(tp_size)[rank],
            "lm_head.weight": weights["model.embed_tokens.weight" if tied else "lm_head.weight"].chunk(tp_size)[rank],
            prefix + "qkv_proj_weight": torch.cat([query, key, value]).T.contiguous(),
            prefix + "o_proj_weight": weights[prefix + "o_proj.weight"].chunk(tp_size, dim=1)[rank].T.contiguous(),
            "model.norm.weight": weights["model.norm.weight"],
        }
        for projection, axis in (("gate", 0), ("up", 0), ("down", 1)):
            name = f"model.layers.0.mlp.{projection}_proj"
            expected[name + "_weight"] = weights[name + ".weight"].chunk(tp_size, dim=axis)[rank].T.contiguous()
        inventory = {name: tensor.shape for name, tensor in expected.items()}
        plans = [plan_box(bucket, shapes, inventory, rank, tp_size) for bucket in buckets]
        validate_box_coverage(plans, inventory)
        slot = torch.zeros(slot_numel_for(plans), dtype=torch.bfloat16)
        actual = {name: torch.full_like(tensor, torch.nan, device=device) for name, tensor in expected.items()}
        for bucket, plan, buffer in zip(
            buckets, plans, iter_weight_buckets(weights.items(), buckets, "cpu"), strict=True
        ):
            pack_box(slot, buffer, bucket, plan)
            apply_box(slot, plan, actual)
        for name in expected:
            torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("bad_schedule", [[], [("norm", (4,), "float32")], [("norm", (0,), "bfloat16")]])
def test_invalid_schedule_is_rejected(bad_schedule):
    with pytest.raises(ValueError):
        plan_buckets(bad_schedule, bucket_bytes=32)


def test_oversized_rows_and_duplicate_parameters_are_rejected():
    with pytest.raises(ValueError, match="One row"):
        plan_buckets([("weight", (8, 32), "bfloat16")], bucket_bytes=32)
    with pytest.raises(ValueError, match="duplicate"):
        plan_buckets([("weight", (4,), "bfloat16")] * 2, bucket_bytes=32)


def test_missing_and_incompatible_resident_weights_fail_closed():
    shapes = {"linear.weight": (8, 4)}
    bucket = plan_buckets([("linear.weight", (8, 4), "bfloat16")], bucket_bytes=64)[0]
    with pytest.raises(ValueError, match="cannot apply"):
        validate_box_coverage([plan_box(bucket, shapes, {}, 0, 2)], {})
    with pytest.raises(ValueError, match="Unsupported TP"):
        plan_box(bucket, shapes, {"linear_weight": (3, 3)}, 0, 2)


def test_fused_bias_layout_and_incomplete_qkv_are_rejected_or_mapped():
    prefix = "model.layers.0.self_attn."
    shapes = {prefix + kind + "_proj.bias": (8 if kind == "q" else 4,) for kind in "qkv"}
    schedule = [(name, shape, "bfloat16") for name, shape in shapes.items()]
    weights = [(name, torch.arange(shape[0], dtype=torch.bfloat16)) for name, shape in shapes.items()]
    inventory = {prefix + "qkv_proj_bias": (8,)}
    buckets = plan_buckets(schedule, bucket_bytes=4)
    plans = [plan_box(bucket, shapes, inventory, 1, 2) for bucket in buckets]
    validate_box_coverage(plans, inventory)
    actual = {prefix + "qkv_proj_bias": torch.zeros(8, dtype=torch.bfloat16)}
    slot = torch.zeros(slot_numel_for(plans), dtype=torch.bfloat16)
    for bucket, plan, buffer in zip(buckets, plans, iter_weight_buckets(weights, buckets, "cpu"), strict=True):
        pack_box(slot, buffer, bucket, plan)
        apply_box(slot, plan, actual)
    assert actual[prefix + "qkv_proj_bias"].tolist() == [4, 5, 6, 7, 2, 3, 2, 3]
    del shapes[prefix + "v_proj.bias"]
    with pytest.raises(ValueError, match="Incomplete QKV"):
        plan_box(buckets[0], shapes, inventory, 0, 2)
