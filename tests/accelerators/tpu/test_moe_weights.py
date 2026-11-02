# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the vLLM fused-MoE expert weight writers shared by the direct tpu-sync installs."""

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import moe_weights
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

E, INTER, HID, PAD = 3, 4, 6, 128  # experts, rank-local intermediate size, hidden size, vllm-torchtpu's padded half


def _experts(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(E, INTER, HID, generator=g) for _ in range(2)), torch.randn(E, HID, INTER, generator=g)


def _reference(gate, up, down, layout):
    """``(w13, w2)`` as vLLM's loaders (+ vllm-torchtpu's re-layout for ``MOE_LAYOUT_TPU``) leave them."""
    if layout == moe_weights.MOE_LAYOUT_VLLM:
        return torch.cat([gate, up], dim=1), down.clone()
    w13 = torch.zeros(E, HID, 2 * PAD)
    w13[..., :INTER] = gate.transpose(1, 2)
    w13[..., PAD : PAD + INTER] = up.transpose(1, 2)
    w2 = torch.zeros(E, PAD, HID)
    w2[:, :INTER] = down.transpose(1, 2)
    return w13, w2


@pytest.mark.parametrize("layout", [moe_weights.MOE_LAYOUT_TPU, moe_weights.MOE_LAYOUT_VLLM])
def test_writers_match_vllm_layout(layout):
    (gate, up), down = _experts()
    ref_w13, ref_w2 = _reference(gate, up, down, layout)
    w13, w2 = torch.zeros_like(ref_w13), torch.zeros_like(ref_w2)
    moe_weights.write_w13(w13, gate, up, layout, check_padding=True)
    moe_weights.write_w2(w2, down, layout, check_padding=True)
    torch.testing.assert_close(w13, ref_w13, rtol=0, atol=0)
    torch.testing.assert_close(w2, ref_w2, rtol=0, atol=0)


def test_writers_refuse_wrong_shapes_layouts_and_nonzero_padding():
    (gate, up), down = _experts()
    w13, w2 = _reference(gate, up, down, moe_weights.MOE_LAYOUT_TPU)
    with pytest.raises(RuntimeError, match="w13_weight"):  # vLLM's own layout expected, vllm-torchtpu's found
        moe_weights.write_w13(w13.clone(), gate, up, moe_weights.MOE_LAYOUT_VLLM, check_padding=True)
    with pytest.raises(RuntimeError, match="w2_weight"):
        moe_weights.write_w2(w2.clone(), down, moe_weights.MOE_LAYOUT_VLLM, check_padding=True)
    with pytest.raises(RuntimeError, match="unknown MoE layout"):
        moe_weights.write_w13(w13.clone(), gate, up, "fp8", check_padding=True)
    for region, what in (((..., INTER), "w13 gate"), ((..., PAD + INTER), "w13 up")):
        bad = w13.clone()
        bad[region] = 1.0
        with pytest.raises(RuntimeError, match=f"{what} padding is not zero"):
            moe_weights.write_w13(bad, gate, up, moe_weights.MOE_LAYOUT_TPU, check_padding=True)
        moe_weights.write_w13(bad, gate, up, moe_weights.MOE_LAYOUT_TPU, check_padding=False)  # later syncs
    bad = w2.clone()
    bad[:, INTER] = 1.0
    with pytest.raises(RuntimeError, match="w2 padding is not zero"):
        moe_weights.write_w2(bad, down, moe_weights.MOE_LAYOUT_TPU, check_padding=True)


@pytest.mark.parametrize("layout", [moe_weights.MOE_LAYOUT_TPU, moe_weights.MOE_LAYOUT_VLLM])
def test_parity_hook_sees_every_written_region(layout):
    """``check`` compares each written region (not the padding) before it is overwritten."""
    (gate, up), down = _experts()
    w13, w2 = _reference(gate, up, down, layout)  # vLLM already holds the checkpoint (the step-0 case)
    parity = ExactParityCheck()
    moe_weights.write_w13(w13, gate, up, layout, True, check=parity.check, name="l0.w13_weight")
    moe_weights.write_w2(w2, down, layout, True, check=parity.check, name="l0.w2_weight")
    regions = 3 if layout == moe_weights.MOE_LAYOUT_TPU else 2  # w13 gate + up + w2, or w13 + w2
    assert parity.checked == regions and not parity.mismatched

    w2[1, 0, 0] += 1  # one misplaced expert value
    seen = []
    moe_weights.write_w13(w13, gate, up, layout, False, check=lambda n, t, r: seen.append(n), name="l0.w13_weight")
    parity = ExactParityCheck()
    moe_weights.write_w2(w2, down, layout, False, check=parity.check, name="l0.w2_weight")
    expected = ["l0.w13_weight[gate]", "l0.w13_weight[up]"] if regions == 3 else ["l0.w13_weight"]
    assert seen == expected
    assert [line.split(":")[0] for line in parity.mismatched] == ["l0.w2_weight"]


def test_expert_param_finds_and_caches_nested_keys():
    model_sd = {"model.layers.3.mlp.experts.routed_experts.w13_weight": torch.zeros(1)}
    cache: dict = {}

    def find(key):
        return key if key in model_sd else None

    key = moe_weights.expert_param(model_sd, find, "model.layers.3.mlp.experts", "w13_weight", cache)
    assert key == "model.layers.3.mlp.experts.routed_experts.w13_weight"
    assert cache == {("model.layers.3.mlp.experts", "w13_weight"): key}
    model_sd["language_model.model.layers.3.mlp.experts.w2_weight"] = torch.zeros(1)
    assert moe_weights.expert_param(model_sd, find, "model.layers.3.mlp.experts", "w2_weight", cache).startswith(
        "language_model."
    )
    with pytest.raises(RuntimeError, match="expected one vLLM parameter"):
        moe_weights.expert_param(model_sd, find, "model.layers.4.mlp.experts", "w13_weight", cache)
