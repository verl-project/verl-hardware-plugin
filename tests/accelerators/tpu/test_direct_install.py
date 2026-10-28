# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the table of direct tpu-sync install layouts that the vLLM worker extension walks."""

from types import SimpleNamespace

import pytest
import torch

from tests.accelerators.tpu import test_qwen3_5_layout as qwen3_5
from tests.accelerators.tpu import test_qwen3_moe_layout as qwen3_moe
from verl_hardware_plugin.accelerators.tpu.rollout import direct_install, qwen3_5_layout, qwen3_moe_layout


def test_table_order_and_entries():
    layouts = direct_install.direct_install_layouts()
    assert layouts == (qwen3_5_layout.LAYOUT, qwen3_moe_layout.LAYOUT)  # priority order
    assert [layout.name for layout in layouts] == ["qwen3_5", "qwen3_moe"]  # "install mode <name> direct"
    assert qwen3_5_layout.LAYOUT.fused_targets == qwen3_5_layout.FUSED_TARGETS


def test_select_layout():
    sel = direct_install.select_layout(qwen3_5._vllm_model(), qwen3_5.SHAPES, qwen3_5.CONFIG, 2, 0)
    assert sel.layout is qwen3_5_layout.LAYOUT and sel.plan.moe_layout is not None and sel.reasons == []

    sel = direct_install.select_layout(qwen3_moe._vllm_model(), qwen3_moe.SHAPES, qwen3_moe.CONFIG, 2, 0)
    assert sel.layout is qwen3_moe_layout.LAYOUT and sel.plan.local_experts == qwen3_moe.E // 2
    assert len(sel.reasons) == 1 and "not a Qwen3.5" in sel.reasons[0]

    # Qwen3 MoE without vLLM expert parallelism: no layout plans; every refusal is reported.
    model = qwen3_moe._vllm_model(use_ep=False)
    sel = direct_install.select_layout(model, qwen3_moe.SHAPES, qwen3_moe.CONFIG, 2, 0)
    assert sel.layout is None and sel.plan is None
    assert len(sel.reasons) == 2 and "not a Qwen3.5" in sel.reasons[0] and "no expert parallelism" in sel.reasons[1]


def test_loader_fallback_hooks():
    """With no layout, vLLM's loaders get HF's tensors back from every layout's tpu-sync wire format."""
    hf35 = qwen3_5._hf_tensors(torch.Generator().manual_seed(0))
    split = qwen3_5_layout.make_tpu_sync_local_layout(qwen3_5.CONFIG)(dict(hf35))
    hf_moe = qwen3_moe._hf_experts(torch.Generator().manual_seed(1))
    received = {**split, **qwen3_moe._folded(hf_moe)}
    out = dict(direct_install.loader_weights(iter(sorted(received.items())), qwen3_moe.CONFIG))
    assert set(out) == set(hf35) | set(hf_moe)
    for name, t in {**hf35, **hf_moe}.items():
        assert torch.equal(out[name], t), name

    shapes = {name: list(t.shape) for name, t in received.items()}
    assert direct_install.loader_error(shapes, qwen3_moe.CONFIG) == ""
    error = direct_install.loader_error(shapes, SimpleNamespace(model_type="qwen3_moe"))
    assert "received folded expert stacks" in error and "no num_experts" in error


def test_custom_table():
    """``select_layout`` takes any table: the first layout that plans wins."""

    def layout(name, plan):
        return direct_install.DirectInstallLayout(
            name=name,
            plan=lambda *args: (plan, "" if plan else f"{name} refused"),
            sharding_rule=lambda *args: None,
            install=lambda *args: (set(), set()),
        )

    table = (layout("a", None), layout("b", "plan-b"), layout("c", "plan-c"))
    sel = direct_install.select_layout(None, {}, None, 1, 0, layouts=table)
    assert (sel.layout.name, sel.plan, sel.reasons) == ("b", "plan-b", ["a refused"])
    with pytest.raises(AttributeError):  # frozen
        table[0].name = "x"
