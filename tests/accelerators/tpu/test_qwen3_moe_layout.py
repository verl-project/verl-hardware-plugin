# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the direct tpu-sync install of Qwen3 MoE experts into vLLM's fused-MoE weights (EP or TP-sliced)."""

from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import moe_weights, qwen3_moe_layout
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

LAYERS = ("model.layers.0", "model.layers.1")
E, INTER, HID = 4, 12, 8  # experts, expert intermediate size (splits over TP 1, 2, 4), hidden size
CONFIG = SimpleNamespace(model_type="qwen3_moe", num_experts=E, moe_intermediate_size=INTER, hidden_size=HID)
PROJ_SHAPES = {"gate_proj": (INTER, HID), "up_proj": (INTER, HID), "down_proj": (HID, INTER)}


def _hf_experts(generator) -> dict:
    """HF's per-expert tensors of every MoE layer, plus the routers."""
    out = {}
    for layer in LAYERS:
        out[f"{layer}.mlp.gate.weight"] = torch.randn(E, HID, generator=generator)
        for e in range(E):
            for proj, shape in PROJ_SHAPES.items():
                out[f"{layer}.mlp.experts.{e}.{proj}.weight"] = torch.randn(*shape, generator=generator)
    return out


def _stacked(hf: dict) -> dict:
    """What the trainer sends: each layer's expert stack as one 3-D tensor, the router as is."""
    out = {n: t for n, t in hf.items() if n.endswith(".mlp.gate.weight")}
    for layer in LAYERS:
        for proj in PROJ_SHAPES:
            stack = torch.stack([hf[f"{layer}.mlp.experts.{e}.{proj}.weight"] for e in range(E)])
            out[f"{layer}.mlp.experts.{proj}.weight"] = stack
    return out


def _received(full: dict, plan) -> dict:
    """Rank ``plan.tp_rank``'s staging tensors: the trainer's tensors sliced as ``sampler_sharding_rule`` registers
    them."""
    staging = {}
    tp_rank, tp_size = plan.tp_rank, plan.tp_size
    for name, t in full.items():
        rule = qwen3_moe_layout.sampler_sharding_rule(name, list(t.shape), tp_size, plan)
        assert rule is not None, name
        spec, local = rule
        if "tp" in spec:
            dim = spec.index("tp")
            t = t.narrow(dim, tp_rank * local[dim], local[dim])
        staging[name] = t.clone()
    return staging


def _vllm_reference(hf: dict, layer: str, tp_rank: int, tp_size: int, padded: bool, expert_parallel: bool) -> tuple:
    """Rank ``tp_rank``'s (w13, w2) of ``layer`` as vLLM's loaders (+ vllm-torchtpu's re-layout) leave them: whole
    experts with EP, every expert's intermediate slice without."""
    experts = range(tp_rank * E // tp_size, (tp_rank + 1) * E // tp_size) if expert_parallel else range(E)
    gate, up, down = (
        torch.stack([hf[f"{layer}.mlp.experts.{e}.{proj}.weight"] for e in experts]) for proj in PROJ_SHAPES
    )
    inter = INTER
    if not expert_parallel:
        inter = INTER // tp_size
        gate, up = (t[:, tp_rank * inter : (tp_rank + 1) * inter] for t in (gate, up))
        down = down[..., tp_rank * inter : (tp_rank + 1) * inter]
    if not padded:  # vLLM's own unquantized layout
        return torch.cat([gate, up], dim=1), down
    aligned = 128
    w13 = torch.zeros(len(experts), HID, 2 * aligned)
    w13[..., :inter] = gate.transpose(1, 2)
    w13[..., aligned : aligned + inter] = up.transpose(1, 2)
    w2 = torch.zeros(len(experts), aligned, HID)
    w2[:, :inter] = down.transpose(1, 2)
    return w13, w2


def _plan(tp_rank, tp_size, padded, expert_parallel=True):
    layout = moe_weights.MOE_LAYOUT_TPU if padded else moe_weights.MOE_LAYOUT_VLLM
    return qwen3_moe_layout.DirectInstallPlan(
        qwen3_moe_layout.Qwen3MoeDims.from_config(CONFIG), tp_rank, tp_size, layout, expert_parallel=expert_parallel
    )


def test_expert_stack_wire_name():
    slots = [(f"model.layers.3.mlp.experts.{e}.down_proj.weight", (HID, INTER)) for e in range(E)]
    assert (
        qwen3_moe_layout.expert_stack_wire_name(slots, (E, HID, INTER)) == "model.layers.3.mlp.experts.down_proj.weight"
    )
    # Anything else keeps the per-expert export: wrong rank, slot count, index order, slot shape, mixed projections.
    assert qwen3_moe_layout.expert_stack_wire_name(slots, (E, HID * INTER)) is None
    assert qwen3_moe_layout.expert_stack_wire_name(slots[:-1], (E, HID, INTER)) is None
    assert qwen3_moe_layout.expert_stack_wire_name(slots[::-1], (E, HID, INTER)) is None
    assert qwen3_moe_layout.expert_stack_wire_name(slots, (E, INTER, HID)) is None
    mixed = [*slots[:-1], (f"model.layers.3.mlp.experts.{E - 1}.gate_proj.weight", (HID, INTER))]
    assert qwen3_moe_layout.expert_stack_wire_name(mixed, (E, HID, INTER)) is None
    fused = [(f"model.layers.3.mlp.experts.{e}.gate_up_proj", (HID, INTER)) for e in range(E)]
    assert qwen3_moe_layout.expert_stack_wire_name(fused, (E, HID, INTER)) is None


def test_split_expert_stacks_for_vllm_loaders():
    """The loader fallback gets HF's per-expert tensors back from the 3-D expert stacks."""
    hf = _hf_experts(torch.Generator().manual_seed(0))
    out = dict(qwen3_moe_layout.split_expert_stacks(iter(sorted(_stacked(hf).items()))))
    assert set(out) == set(hf)
    for name, t in hf.items():
        assert torch.equal(out[name], t), name
    # Qwen3.5's 3-D expert pieces (no ``.weight`` suffix), 2-D tensors and dense tensors pass through.
    other = {
        "model.layers.0.mlp.experts.gate_proj": torch.zeros(E, 2, 3),
        "model.layers.0.mlp.experts.up_proj.weight": torch.zeros(2, 3),
        "model.norm.weight": torch.zeros(3),
    }
    assert dict(qwen3_moe_layout.split_expert_stacks(iter(other.items()))).keys() == other.keys()


def test_sampler_sharding_rules():
    rule = qwen3_moe_layout.sampler_sharding_rule
    dims = qwen3_moe_layout.Qwen3MoeDims(num_experts=128, moe_intermediate_size=768, hidden_size=2048)
    ep = qwen3_moe_layout.DirectInstallPlan(dims, 0, 16, moe_weights.MOE_LAYOUT_TPU, expert_parallel=True)
    tp = qwen3_moe_layout.DirectInstallPlan(dims, 0, 16, moe_weights.MOE_LAYOUT_TPU, expert_parallel=False)
    gate, down = "model.layers.0.mlp.experts.gate_proj.weight", "model.layers.0.mlp.experts.down_proj.weight"
    # EP: whole experts per rank (dim 0); TP: every expert's intermediate slice (gate / up dim 1, down dim 2).
    assert rule(gate, [128, 768, 2048], 16, ep) == (["tp", "", ""], [8, 768, 2048])
    assert rule(down, [128, 2048, 768], 16, ep) == (["tp", "", ""], [8, 2048, 768])
    assert rule(gate, [128, 768, 2048], 16, tp) == (["", "tp", ""], [128, 48, 2048])
    assert rule(down, [128, 2048, 768], 16, tp) == (["", "", "tp"], [128, 2048, 48])
    for plan in (ep, tp):
        assert rule("model.layers.0.mlp.gate.weight", [128, 2048], 16, plan) == (["", ""], [128, 2048])
        assert rule("model.layers.0.self_attn.o_proj.weight", [2048, 4096], 16, plan) is None  # dense path
        assert rule("model.layers.0.mlp.down_proj.weight", [2048, 768], 16, plan) is None  # a dense MLP layer
    with pytest.raises(RuntimeError, match="does not split"):
        rule(gate, [100, 768, 2048], 16, ep)
    with pytest.raises(RuntimeError, match="does not split"):
        rule(gate, [128 * 768, 2048], 16, tp)  # not a 3-D stack


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("expert_parallel", [True, False])
def test_direct_install_matches_vllm_loaders(tp_size, padded, nested, expert_parallel):
    """Each rank's experts (whole with EP, intermediate slices without) land exactly where vLLM's loaders (+
    vllm-torchtpu's re-layout) put them, and the exact parity check compares every written region."""
    hf = _hf_experts(torch.Generator().manual_seed(0))
    full = _stacked(hf)
    leaf_prefix = "routed_experts." if nested else ""  # vLLM 0.29 keeps them in the MoE runner's routed_experts
    for tp_rank in range(tp_size):
        expected = {}
        for layer in LAYERS:
            w13, w2 = _vllm_reference(hf, layer, tp_rank, tp_size, padded, expert_parallel)
            expected[f"{layer}.mlp.experts.{leaf_prefix}w13_weight"] = w13
            expected[f"{layer}.mlp.experts.{leaf_prefix}w2_weight"] = w2
        model_sd = {name: torch.zeros_like(t) for name, t in expected.items()}
        plan = _plan(tp_rank, tp_size, padded, expert_parallel)
        staging = _received(full, plan)
        parity = ExactParityCheck()
        consumed = qwen3_moe_layout.install(
            staging, model_sd, lambda k, sd=model_sd: k if k in sd else None, plan, check=parity.check
        )
        assert plan.padding_checked
        assert consumed == {n for n in staging if ".experts." in n}  # the routers go through the dense path
        for name, ref in expected.items():
            torch.testing.assert_close(model_sd[name], ref, rtol=0, atol=0, msg=name)
        # The targets held zeros: every written region differs (padding is not compared).
        regions_per_layer = 3 if padded else 2  # w13 gate + up + w2, or w13 + w2
        assert parity.checked == len(parity.mismatched) == regions_per_layer * len(LAYERS)

        # The step-0 case: vLLM already holds the checkpoint, so the check finds no difference.
        parity = ExactParityCheck()
        qwen3_moe_layout.install(staging, model_sd, lambda k, sd=model_sd: k if k in sd else None, plan, parity.check)
        assert parity.checked == regions_per_layer * len(LAYERS) and not parity.mismatched


def test_direct_install_refuses_wrong_shapes_and_nonzero_padding():
    full = _stacked(_hf_experts(torch.Generator().manual_seed(0)))
    staging = _received(full, _plan(0, 2, padded=True))
    good = {
        f"{layer}.mlp.experts.{leaf}": torch.zeros(E // 2, *shape)
        for layer in LAYERS
        for leaf, shape in (("w13_weight", (HID, 256)), ("w2_weight", (128, HID)))
    }

    def find(sd):
        return lambda k: k if k in sd else None

    with pytest.raises(RuntimeError, match="on this rank"):  # a plan for TP=4 against TP=2 slices
        qwen3_moe_layout.install(staging, good, find(good), _plan(0, 4, padded=True))
    with pytest.raises(RuntimeError, match="w13_weight"):  # vLLM's own layout planned, vllm-torchtpu's found
        qwen3_moe_layout.install(staging, good, find(good), _plan(0, 2, padded=False))
    good["model.layers.0.mlp.experts.w2_weight"][:, INTER] = 1.0
    with pytest.raises(RuntimeError, match="padding is not zero"):
        qwen3_moe_layout.install(staging, good, find(good), _plan(0, 2, padded=True))
    good["model.layers.0.mlp.experts.w2_weight"].zero_()
    del staging["model.layers.1.mlp.experts.up_proj.weight"]
    with pytest.raises(RuntimeError, match="lacks"):
        qwen3_moe_layout.install(staging, good, find(good), _plan(0, 2, padded=True))


class VllmUnquantizedFusedMoEMethod:  # vllm-torchtpu's unquantized MoE method (matched by class name)
    pass


class Fp8MoEMethod:
    pass


class _FusedMoE(torch.nn.Module):
    """vLLM's MoE layer of TP rank ``tp_rank``: whole experts per rank with EP (every expert's intermediate slice
    without), the EP geometry on itself or (newer vLLM) on its ``moe_config``."""

    def __init__(
        self, tp_size=2, tp_rank=0, use_ep=True, method=VllmUnquantizedFusedMoEMethod, on_config=False, **attrs
    ):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(1))
        self.quant_method = method()
        geometry = dict(
            use_ep=use_ep,
            ep_size=tp_size if use_ep else 1,
            ep_rank=tp_rank if use_ep else 0,
            global_num_experts=E,
            local_num_experts=E // tp_size if use_ep else E,
            **attrs,
        )
        if on_config:
            self.moe_config = SimpleNamespace(**geometry)
        else:
            for k, v in geometry.items():
                setattr(self, k, v)


def _vllm_model(tp_size=2, tp_rank=0, router_rows=E, **moe_kwargs):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in LAYERS:
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.gate = torch.nn.Linear(HID, router_rows, bias=False)
        layer.mlp.experts = _FusedMoE(tp_size, tp_rank, **moe_kwargs)
        model.model.layers.append(layer)
    return model


SHAPES = {name: list(t.shape) for name, t in _stacked(_hf_experts(torch.Generator())).items()}


def test_plan_direct_install():
    for on_config in (False, True):
        plan, reason = qwen3_moe_layout.plan_direct_install(_vllm_model(on_config=on_config), SHAPES, CONFIG, 2, 0)
        assert reason == "" and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU and plan.expert_parallel
        assert (plan.tp_rank, plan.tp_size, plan.local_experts, plan.local_intermediate) == (0, 2, E // 2, INTER)
    # Without EP, vLLM slices every expert's intermediate dimension over TP.
    plan, reason = qwen3_moe_layout.plan_direct_install(_vllm_model(4, 3, use_ep=False), SHAPES, CONFIG, 4, 3)
    assert reason == "" and not plan.expert_parallel
    assert (plan.tp_rank, plan.local_experts, plan.local_intermediate) == (3, E, INTER // 4)
    # TP=1 needs no expert parallelism: the single rank holds every expert.
    plan, _ = qwen3_moe_layout.plan_direct_install(_vllm_model(1, 0, use_ep=False), SHAPES, CONFIG, 1, 0)
    assert plan is not None and plan.local_experts == E and plan.local_intermediate == INTER

    def reason(model=None, shapes=SHAPES, config=CONFIG, tp=2, rank=0):
        plan, why = qwen3_moe_layout.plan_direct_install(model or _vllm_model(tp, rank), shapes, config, tp, rank)
        assert plan is None
        return why

    assert "not a Qwen3 MoE" in reason(config=SimpleNamespace(**{**vars(CONFIG), "model_type": "qwen2_moe"}))
    assert "incomplete" in reason(config=SimpleNamespace(model_type="qwen3_moe"))
    per_expert = {**SHAPES, "model.layers.0.mlp.experts.0.gate_proj.weight": [INTER, HID]}
    assert "per-expert" in reason(shapes=per_expert)  # e.g. a trainer that cannot send the stacks
    assert "no expert stacks" in reason(shapes={"model.norm.weight": [HID]})
    without_up = {n: s for n, s in SHAPES.items() if n != "model.layers.1.mlp.experts.up_proj.weight"}
    assert "model.layers.1.mlp.experts lacks up_proj" in reason(shapes=without_up)
    transposed_w2 = {**SHAPES, "model.layers.0.mlp.experts.down_proj.weight": [E, INTER, HID]}
    assert "the config gives" in reason(shapes=transposed_w2)
    # A trainer that still sends v1's 2-D folded stacks is refused, not misinstalled.
    folded = {**SHAPES, "model.layers.0.mlp.experts.down_proj.weight": [E * HID, INTER]}
    assert "the config gives" in reason(shapes=folded)
    assert "experts do not split" in reason(_vllm_model(3, 0), tp=3)
    assert "moe_intermediate_size 12 does not split over TP=5" in reason(_vllm_model(5, 0, use_ep=False), tp=5)
    other_tp = _vllm_model(use_ep=False)
    for layer in other_tp.model.layers:
        layer.mlp.experts.tp_size = 4  # vLLM's MoE TP group is not the rollout's
    assert "MoE TP size 4" in reason(other_tp)
    mixed = _vllm_model()
    mixed.model.layers[1].mlp.experts = _FusedMoE(2, 0, use_ep=False)
    assert "mix" in reason(mixed)
    assert "EP size" in reason(_vllm_model(tp_rank=1), rank=0)
    assert "placement" in reason(_vllm_model(expert_placement_strategy="round_robin"))
    assert "swigluoai" in reason(_vllm_model(activation="swigluoai"))
    assert "Fp8MoEMethod" in reason(_vllm_model(method=Fp8MoEMethod))
    one_layer = _vllm_model()
    del one_layer.model.layers[1]
    assert "1 fused-MoE layers" in reason(one_layer)
    assert "router" in reason(_vllm_model(router_rows=E // 2))  # vLLM shards the router


def test_trainer_weights_sends_expert_stacks_whole_only_for_local_shards():
    """``local_shards()`` sends each stack as one 3-D tensor; plain iteration (verl, ray-plasma-store) keeps HF's
    experts."""
    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights

    hf = _hf_experts(torch.Generator().manual_seed(0))
    layer = LAYERS[0]

    def weights():
        stacks = [
            (
                torch.stack([hf[f"{layer}.mlp.experts.{e}.{proj}.weight"] for e in range(E)]),
                [(f"{layer}.mlp.experts.{e}.{proj}.weight", shape) for e in range(E)],
            )
            for proj, shape in PROJ_SHAPES.items()
        ]
        dense = {f"{layer}.mlp.gate.weight": hf[f"{layer}.mlp.gate.weight"]}
        return TrainerWeights(dense, stacks, "cpu", expert_stack_wire=qwen3_moe_layout.expert_stack_wire_name)

    stacked = {n: t for n, t in _stacked(hf).items() if n.startswith(layer + ".")}
    local = list(weights().local_shards())
    assert [n for n, _, _ in local] == list(stacked)
    for name, tensor, shard in local:
        assert shard is None and tensor.dtype == torch.bfloat16, name
        assert torch.equal(tensor, stacked[name].to(torch.bfloat16)), name
    plain = dict(weights())
    assert set(plain) == {n for n in hf if n.startswith(layer + ".")}


def test_trainer_expert_stack_wire_needs_trainer_ep_1():
    """Only Qwen3 MoE sends its expert stacks whole, and only with FSDP-sharded experts (trainer EP = 1) for now."""
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3_moe", 1) is qwen3_moe_layout.expert_stack_wire_name
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3", 1) is None
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3_5_moe", 4) is None  # other models are not affected
    with pytest.raises(ValueError, match="expert_parallel_size=2 is not supported"):
        qwen3_moe_layout.trainer_expert_stack_wire("qwen3_moe", 2)
