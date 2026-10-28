# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the direct tpu-sync install of Qwen3 MoE experts into vLLM's expert-parallel MoE weights."""

from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import moe_weights, qwen3_moe_layout
from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

LAYERS = ("model.layers.0", "model.layers.1")
E, INTER, HID = 4, 6, 8  # experts, expert intermediate size, hidden size
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


def _folded(hf: dict) -> dict:
    """What the trainer sends: each layer's expert stack folded to [E * rows, cols], the router as is."""
    out = {n: t for n, t in hf.items() if n.endswith(".mlp.gate.weight")}
    for layer in LAYERS:
        for proj in PROJ_SHAPES:
            stack = torch.stack([hf[f"{layer}.mlp.experts.{e}.{proj}.weight"] for e in range(E)])
            out[f"{layer}.mlp.experts.{proj}.weight"] = stack.reshape(-1, stack.shape[-1])
    return out


def _received(full: dict, tp_rank: int, tp_size: int) -> dict:
    """Rank ``tp_rank``'s staging tensors: the trainer's tensors sliced as ``sampler_sharding_rule`` registers them."""
    staging = {}
    for name, t in full.items():
        rule = qwen3_moe_layout.sampler_sharding_rule(name, list(t.shape), tp_size)
        assert rule is not None, name
        spec, local = rule
        if "tp" in spec:
            dim = spec.index("tp")
            t = t.narrow(dim, tp_rank * local[dim], local[dim])
        staging[name] = t.clone()
    return staging


def _vllm_reference(hf: dict, layer: str, tp_rank: int, tp_size: int, padded: bool) -> tuple:
    """Rank ``tp_rank``'s (w13, w2) of ``layer`` as vLLM's EP loaders (+ vllm-torchtpu's re-layout) leave them."""
    local = range(tp_rank * E // tp_size, (tp_rank + 1) * E // tp_size)
    gate, up, down = (
        torch.stack([hf[f"{layer}.mlp.experts.{e}.{proj}.weight"] for e in local]) for proj in PROJ_SHAPES
    )
    if not padded:  # vLLM's own unquantized layout
        return torch.cat([gate, up], dim=1), down
    aligned = 128
    w13 = torch.zeros(len(local), HID, 2 * aligned)
    w13[..., :INTER] = gate.transpose(1, 2)
    w13[..., aligned : aligned + INTER] = up.transpose(1, 2)
    w2 = torch.zeros(len(local), aligned, HID)
    w2[:, :INTER] = down.transpose(1, 2)
    return w13, w2


def _plan(tp_rank, tp_size, padded):
    layout = moe_weights.MOE_LAYOUT_TPU if padded else moe_weights.MOE_LAYOUT_VLLM
    return qwen3_moe_layout.DirectInstallPlan(
        qwen3_moe_layout.Qwen3MoeDims.from_config(CONFIG), tp_rank, tp_size, layout
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


def test_unfold_expert_stacks_for_vllm_loaders():
    """The loader fallback gets HF's per-expert tensors back from the folded stacks."""
    hf = _hf_experts(torch.Generator().manual_seed(0))
    out = dict(qwen3_moe_layout.unfold_expert_stacks(iter(sorted(_folded(hf).items())), E))
    assert set(out) == set(hf)
    for name, t in hf.items():
        assert torch.equal(out[name], t), name
    # Qwen3.5's 3-D expert pieces and dense tensors pass through.
    other = {"model.layers.0.mlp.experts.gate_proj": torch.zeros(E, 2, 3), "model.norm.weight": torch.zeros(3)}
    assert dict(qwen3_moe_layout.unfold_expert_stacks(iter(other.items()), None)).keys() == other.keys()
    with pytest.raises(RuntimeError, match="cannot unfold"):
        list(
            qwen3_moe_layout.unfold_expert_stacks(
                iter([("model.layers.0.mlp.experts.up_proj.weight", out[next(iter(out))])]), None
            )
        )


def test_sampler_sharding_rules():
    rule = qwen3_moe_layout.sampler_sharding_rule
    assert rule("model.layers.0.mlp.experts.gate_proj.weight", [128 * 768, 2048], 16) == (["tp", ""], [8 * 768, 2048])
    assert rule("model.layers.0.mlp.experts.down_proj.weight", [128 * 2048, 768], 16) == (["tp", ""], [8 * 2048, 768])
    assert rule("model.layers.0.mlp.gate.weight", [128, 2048], 16) == (["", ""], [128, 2048])
    assert rule("model.layers.0.self_attn.o_proj.weight", [2048, 4096], 16) is None  # dense path
    assert rule("model.layers.0.mlp.down_proj.weight", [2048, 768], 16) is None  # a dense MLP layer
    with pytest.raises(RuntimeError, match="does not split"):
        rule("model.layers.0.mlp.experts.gate_proj.weight", [100, 2048], 16)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_direct_install_matches_vllm_loaders(tp_size, padded, nested):
    """Each rank's whole experts land exactly where vLLM's EP loaders (+ vllm-torchtpu's re-layout) put them, and the
    exact parity check compares every written region."""
    hf = _hf_experts(torch.Generator().manual_seed(0))
    full = _folded(hf)
    leaf_prefix = "routed_experts." if nested else ""  # vLLM 0.29 keeps them in the MoE runner's routed_experts
    for tp_rank in range(tp_size):
        expected = {}
        for layer in LAYERS:
            w13, w2 = _vllm_reference(hf, layer, tp_rank, tp_size, padded)
            expected[f"{layer}.mlp.experts.{leaf_prefix}w13_weight"] = w13
            expected[f"{layer}.mlp.experts.{leaf_prefix}w2_weight"] = w2
        model_sd = {name: torch.zeros_like(t) for name, t in expected.items()}
        staging = _received(full, tp_rank, tp_size)
        plan = _plan(tp_rank, tp_size, padded)
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
    full = _folded(_hf_experts(torch.Generator().manual_seed(0)))
    staging = _received(full, 0, 2)
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
    """vLLM's MoE layer of TP rank ``tp_rank``: whole experts per rank with EP, the EP geometry on itself or (newer
    vLLM) on its ``moe_config``."""

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


SHAPES = {name: list(t.shape) for name, t in _folded(_hf_experts(torch.Generator())).items()}


def test_plan_direct_install():
    for on_config in (False, True):
        plan, reason = qwen3_moe_layout.plan_direct_install(_vllm_model(on_config=on_config), SHAPES, CONFIG, 2, 0)
        assert reason == "" and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU
        assert (plan.tp_rank, plan.tp_size, plan.local_experts) == (0, 2, E // 2)
    # TP=1 needs no expert parallelism: the single rank holds every expert.
    plan, _ = qwen3_moe_layout.plan_direct_install(_vllm_model(1, 0, use_ep=False), SHAPES, CONFIG, 1, 0)
    assert plan is not None and plan.local_experts == E

    def reason(model=None, shapes=SHAPES, config=CONFIG, tp=2, rank=0):
        plan, why = qwen3_moe_layout.plan_direct_install(model or _vllm_model(tp, rank), shapes, config, tp, rank)
        assert plan is None
        return why

    assert "not a Qwen3 MoE" in reason(config=SimpleNamespace(**{**vars(CONFIG), "model_type": "qwen2_moe"}))
    assert "incomplete" in reason(config=SimpleNamespace(model_type="qwen3_moe"))
    per_expert = {**SHAPES, "model.layers.0.mlp.experts.0.gate_proj.weight": [INTER, HID]}
    assert "per-expert" in reason(shapes=per_expert)  # e.g. a trainer that cannot fold the stacks
    assert "no folded" in reason(shapes={"model.norm.weight": [HID]})
    without_up = {n: s for n, s in SHAPES.items() if n != "model.layers.1.mlp.experts.up_proj.weight"}
    assert "model.layers.1.mlp.experts lacks up_proj" in reason(shapes=without_up)
    transposed_w2 = {**SHAPES, "model.layers.0.mlp.experts.down_proj.weight": [E * INTER, HID]}
    assert "the config gives" in reason(shapes=transposed_w2)
    assert "do not split" in reason(_vllm_model(3, 0), tp=3)
    assert "no expert parallelism" in reason(_vllm_model(use_ep=False))  # vLLM TP-slices every expert instead
    assert "EP size" in reason(_vllm_model(tp_rank=1), rank=0)
    assert "placement" in reason(_vllm_model(expert_placement_strategy="round_robin"))
    assert "swigluoai" in reason(_vllm_model(activation="swigluoai"))
    assert "Fp8MoEMethod" in reason(_vllm_model(method=Fp8MoEMethod))
    one_layer = _vllm_model()
    del one_layer.model.layers[1]
    assert "1 fused-MoE layers" in reason(one_layer)
    assert "router" in reason(_vllm_model(router_rows=E // 2))  # vLLM shards the router


def test_trainer_weights_folds_expert_stacks_only_for_local_shards():
    """``local_shards()`` sends each stack folded; plain iteration (verl, ray-plasma-store) keeps HF's experts."""
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

    folded = {n: t for n, t in _folded(hf).items() if n.startswith(layer + ".")}
    local = list(weights().local_shards())
    assert [n for n, _, _ in local] == list(folded)
    for name, tensor, shard in local:
        assert shard is None and tensor.dtype == torch.bfloat16, name
        assert torch.equal(tensor, folded[name].to(torch.bfloat16)), name
    plain = dict(weights())
    assert set(plain) == {n for n in hf if n.startswith(layer + ".")}


def test_trainer_expert_stack_wire_needs_trainer_ep_1():
    """Only Qwen3 MoE folds its expert stacks, and only with FSDP-sharded experts (trainer EP = 1) for now."""
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3_moe", 1) is qwen3_moe_layout.expert_stack_wire_name
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3", 1) is None
    assert qwen3_moe_layout.trainer_expert_stack_wire("qwen3_5_moe", 4) is None  # other models are not affected
    with pytest.raises(ValueError, match="expert_parallel_size=2 is not supported"):
        qwen3_moe_layout.trainer_expert_stack_wire("qwen3_moe", 2)
