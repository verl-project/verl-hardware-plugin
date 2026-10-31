# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the direct tpu-sync install of Qwen3.5 into vLLM's TP-sharded parameters."""

from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import qwen3_5_layout

GDN = "model.layers.0.linear_attn"
EXPERTS = "model.layers.0.mlp.experts"
NK, NV, HK, HV = 2, 4, 4, 4  # GatedDeltaNet key / value heads and head dims
K, V, HID, CONV_K = NK * HK, NV * HV, 6, 4  # key dim, value dim, hidden, conv kernel
E, INTER = 3, 4  # experts, expert intermediate size
CONFIG = SimpleNamespace(
    model_type="qwen3_5_moe_text",
    linear_num_key_heads=NK,
    linear_num_value_heads=NV,
    linear_key_head_dim=HK,
    linear_value_head_dim=HV,
    moe_intermediate_size=INTER,
)


def _hf_tensors(generator):
    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    return {
        f"{GDN}.in_proj_qkv.weight": rand(2 * K + V, HID),
        f"{GDN}.in_proj_z.weight": rand(V, HID),
        f"{GDN}.conv1d.weight": rand(2 * K + V, 1, CONV_K),
        f"{GDN}.A_log": rand(NV),
        f"{GDN}.dt_bias": rand(NV),
        f"{EXPERTS}.gate_up_proj": rand(E, 2 * INTER, HID),
        f"{EXPERTS}.down_proj": rand(E, HID, INTER),
    }


def _received(full: dict, tp_rank: int, tp_size: int) -> dict:
    """What rank ``tp_rank`` receives: the trainer's local layout, sliced as ``sampler_sharding_rule`` registers it."""
    staging = {}
    for name, t in qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)(dict(full)).items():
        rule = qwen3_5_layout.sampler_sharding_rule(name, list(t.shape), tp_size)
        if rule is None:  # dense path: 2-D column parallel split on dim 0, 1-D whole
            rule = (["tp", ""], [t.shape[0] // tp_size, t.shape[1]]) if t.ndim == 2 else ([""], list(t.shape))
        spec, local = rule
        if "tp" in spec:
            dim = spec.index("tp")
            t = t.narrow(dim, tp_rank * local[dim], local[dim])
        staging[name] = t.clone()
    return staging


def _vllm_reference(full: dict, tp_rank: int, tp_size: int, aligned: int) -> dict:
    """vLLM's rank-``tp_rank`` parameters as its loaders and vllm-torchtpu's processing leave them."""

    def rank_rows(t, start, length):
        per = length // tp_size
        return t[start + tp_rank * per : start + (tp_rank + 1) * per]

    qkv, conv = full[f"{GDN}.in_proj_qkv.weight"], full[f"{GDN}.conv1d.weight"]
    bounds = ((0, K), (K, K), (2 * K, V))
    qkvz = torch.cat([*(rank_rows(qkv, s, n) for s, n in bounds), rank_rows(full[f"{GDN}.in_proj_z.weight"], 0, V)])
    gate_up, down = full[f"{EXPERTS}.gate_up_proj"], full[f"{EXPERTS}.down_proj"]
    inter_r = INTER // tp_size
    gate = gate_up[:, tp_rank * inter_r : (tp_rank + 1) * inter_r]
    up = gate_up[:, INTER + tp_rank * inter_r : INTER + (tp_rank + 1) * inter_r]
    w13 = torch.zeros(E, HID, 2 * aligned)
    w13[..., :inter_r] = gate.transpose(1, 2)
    w13[..., aligned : aligned + inter_r] = up.transpose(1, 2)
    w2 = torch.zeros(E, aligned, HID)
    w2[:, :inter_r] = down[..., tp_rank * inter_r : (tp_rank + 1) * inter_r].transpose(1, 2)
    return {
        f"{GDN}.in_proj_qkvz.weight": qkvz.T.contiguous(),  # vllm-torchtpu keeps dense weights as [n_in, n_out]
        f"{GDN}.conv1d.weight": torch.cat([rank_rows(conv, s, n) for s, n in bounds]),
        f"{GDN}.A_log": rank_rows(full[f"{GDN}.A_log"], 0, NV),
        f"{GDN}.dt_bias": rank_rows(full[f"{GDN}.dt_bias"], 0, NV),
        f"{EXPERTS}.w13_weight": w13,
        f"{EXPERTS}.w2_weight": w2,
    }


def _plan(tp_rank, tp_size, moe_layout):
    return qwen3_5_layout.DirectInstallPlan(
        qwen3_5_layout.Qwen3_5Dims.from_config(CONFIG), tp_rank, tp_size, moe_layout
    )


def _copy_into(model_sd):
    def copy_into(key, src):
        target = model_sd[key]
        target.copy_(src.T if src.ndim == 2 and src.T.shape == target.shape and src.shape != target.shape else src)

    return copy_into


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("nested", [True, False])
def test_direct_install_matches_vllm_loaders(tp_size, padded, nested):
    """Each rank's slices land exactly where vLLM's loaders (+ vllm-torchtpu's re-layout) put them."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    for tp_rank in range(tp_size):
        aligned = 128 if padded else INTER // tp_size
        expected = _vllm_reference(full, tp_rank, tp_size, aligned)
        model_sd = {name: torch.full_like(t, float("nan")) for name, t in expected.items()}
        for name in (f"{EXPERTS}.w13_weight", f"{EXPERTS}.w2_weight"):
            model_sd[name].zero_()  # the padding stays as vllm-torchtpu wrote it
        if not padded:  # vLLM's own MoE layout: w13 [E, 2 I_r, H], w2 [E, H, I_r]
            ref = expected
            ref[f"{EXPERTS}.w13_weight"] = ref[f"{EXPERTS}.w13_weight"].transpose(1, 2).contiguous()
            ref[f"{EXPERTS}.w2_weight"] = ref[f"{EXPERTS}.w2_weight"].transpose(1, 2).contiguous()
            model_sd[f"{EXPERTS}.w13_weight"] = torch.zeros_like(ref[f"{EXPERTS}.w13_weight"])
            model_sd[f"{EXPERTS}.w2_weight"] = torch.zeros_like(ref[f"{EXPERTS}.w2_weight"])

        if nested:  # vLLM 0.29 keeps the expert weights in the MoE runner's ``routed_experts`` submodule
            for leaf in ("w13_weight", "w2_weight"):
                key = f"{EXPERTS}.{leaf}"
                nested_key = f"{EXPERTS}.routed_experts.{leaf}"
                model_sd[nested_key] = model_sd.pop(key)
                expected[nested_key] = expected.pop(key)

        staging = _received(full, tp_rank, tp_size)
        layout = qwen3_5_layout.MOE_LAYOUT_TPU if padded else qwen3_5_layout.MOE_LAYOUT_VLLM
        plan = _plan(tp_rank, tp_size, layout)
        consumed = qwen3_5_layout.install(
            staging, model_sd, lambda k, model_sd=model_sd: k if k in model_sd else None, _copy_into(model_sd), plan
        )
        assert plan.padding_checked
        assert consumed == set(staging)
        for name, ref in expected.items():
            torch.testing.assert_close(model_sd[name], ref, rtol=0, atol=0, msg=name)


def test_direct_install_refuses_a_mismatched_layout():
    """The MoE layout comes from the plan, not from shapes: a w13 of the other layout is an error, not a guess."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    staging = _received(full, 0, 1)
    model_sd = {k: torch.zeros_like(t) for k, t in _vllm_reference(full, 0, 1, 128).items()}

    def find(key):
        return key if key in model_sd else None

    with pytest.raises(RuntimeError, match="w13_weight"):
        qwen3_5_layout.install(
            staging, model_sd, find, _copy_into(model_sd), _plan(0, 1, qwen3_5_layout.MOE_LAYOUT_VLLM)
        )
    model_sd[f"{EXPERTS}.w13_weight"][..., INTER] = 1.0  # non-zero padding
    with pytest.raises(RuntimeError, match="padding is not zero"):
        qwen3_5_layout.install(
            staging, model_sd, find, _copy_into(model_sd), _plan(0, 1, qwen3_5_layout.MOE_LAYOUT_TPU)
        )


def test_direct_install_refuses_a_too_small_target():
    """``copy_`` would broadcast a source into a larger target; the install checks the element counts first."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    staging = _received(full, 0, 1)
    model_sd = {k: torch.zeros_like(t) for k, t in _vllm_reference(full, 0, 1, 128).items()}
    model_sd[f"{GDN}.A_log"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="A_log"):
        qwen3_5_layout.install(
            staging,
            model_sd,
            lambda k: k if k in model_sd else None,
            _copy_into(model_sd),
            _plan(0, 1, qwen3_5_layout.MOE_LAYOUT_TPU),
        )


def test_trainer_splits_fused_tensors():
    full = _hf_tensors(torch.Generator().manual_seed(0))
    out = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)(dict(full))
    qkv, conv, gate_up = (
        full[f"{GDN}.in_proj_qkv.weight"],
        full[f"{GDN}.conv1d.weight"],
        full[f"{EXPERTS}.gate_up_proj"],
    )
    assert torch.equal(out[f"{GDN}.in_proj_q.weight"], qkv[:K])
    assert torch.equal(out[f"{GDN}.in_proj_k.weight"], qkv[K : 2 * K])
    assert torch.equal(out[f"{GDN}.in_proj_v.weight"], qkv[2 * K :])
    assert torch.equal(out[f"{GDN}.conv1d_v.weight"], conv[2 * K :])
    assert torch.equal(out[f"{EXPERTS}.gate_proj"], gate_up[:, :INTER])
    assert torch.equal(out[f"{EXPERTS}.up_proj"], gate_up[:, INTER:])
    assert not any(n.endswith(("in_proj_qkv.weight", "conv1d.weight", "gate_up_proj")) for n in out)
    with pytest.raises(RuntimeError, match="does not split"):
        qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)({f"{GDN}.in_proj_qkv.weight": torch.zeros(K, HID)})


def test_sampler_sharding_rules():
    rule = qwen3_5_layout.sampler_sharding_rule
    assert rule(f"{GDN}.in_proj_q.weight", [2048, 2048], 8) == (["tp", ""], [256, 2048])
    assert rule(f"{GDN}.conv1d_v.weight", [4096, 1, 4], 8) == (["tp", "", ""], [512, 1, 4])
    assert rule(f"{GDN}.A_log", [32], 8) == ([""], [32])
    assert rule("model.layers.0.mlp.gate.weight", [256, 2048], 8) == (["", ""], [256, 2048])
    assert rule(f"{GDN}.out_proj.weight", [2048, 4096], 8) == (["", "tp"], [2048, 512])
    assert rule(f"{EXPERTS}.gate_proj", [256, 512, 2048], 8) == (["", "tp", ""], [256, 64, 2048])
    assert rule(f"{EXPERTS}.down_proj", [256, 2048, 512], 8) == (["", "", "tp"], [256, 2048, 64])
    assert rule(f"{GDN}.in_proj_z.weight", [4096, 2048], 8) is None  # dense path: column parallel
    with pytest.raises(RuntimeError, match="does not split"):
        rule(f"{EXPERTS}.gate_proj", [256, 500, 2048], 8)


class UnquantizedFusedMoEMethod:  # vLLM's own unquantized MoE method (by class name)
    pass


class VllmUnquantizedFusedMoEMethod:  # vllm-torchtpu's
    pass


class Fp8MoEMethod:
    pass


class _GDN(torch.nn.Module):
    def __init__(self, tp_rank=0, tp_size=2, **attrs):
        super().__init__()
        self.in_proj_qkvz = torch.nn.Linear(1, 1)
        self.in_proj_ba = torch.nn.Linear(1, 1)
        self.tp_rank, self.tp_size = tp_rank, tp_size
        for k, v in attrs.items():
            setattr(self, k, v)


class _Experts(torch.nn.Module):
    def __init__(self, method, use_ep=False):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(torch.zeros(1))
        self.quant_method = method
        self.use_ep = use_ep


def _vllm_model(gdn=None, experts=None):
    model = torch.nn.Module()
    model.gdn = gdn if gdn is not None else _GDN()
    model.experts = experts if experts is not None else _Experts(VllmUnquantizedFusedMoEMethod())
    return model


SHAPES = {
    name: list(t.shape)
    for name, t in qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)(_hf_tensors(torch.Generator())).items()
}


def test_plan_direct_install():
    plan, reason = qwen3_5_layout.plan_direct_install(_vllm_model(), SHAPES, CONFIG, 2, 0)
    assert reason == "" and plan.moe_layout == qwen3_5_layout.MOE_LAYOUT_TPU and (plan.tp_rank, plan.tp_size) == (0, 2)
    plan, _ = qwen3_5_layout.plan_direct_install(
        _vllm_model(experts=_Experts(UnquantizedFusedMoEMethod())), SHAPES, CONFIG, 2, 0
    )
    assert plan.moe_layout == qwen3_5_layout.MOE_LAYOUT_VLLM

    def reason(model=None, shapes=SHAPES, config=CONFIG, tp=2, rank=0):
        plan, why = qwen3_5_layout.plan_direct_install(model or _vllm_model(), shapes, config, tp, rank)
        assert plan is None
        return why

    assert "not a Qwen3.5" in reason(config=SimpleNamespace(**{**vars(CONFIG), "model_type": "qwen3_moe"}))
    assert "fused tensors" in reason(shapes={**SHAPES, f"{EXPERTS}.gate_up_proj": [E, 2 * INTER, HID]})
    assert "heads" in reason(tp=4)  # 2 key heads over 4 ranks
    assert "moe_intermediate_size" in reason(config=SimpleNamespace(**{**vars(CONFIG), "moe_intermediate_size": 5}))
    assert "rows" in reason(shapes={**SHAPES, f"{GDN}.in_proj_q.weight": [K + 1, HID]})
    assert "interleaved" in reason(_vllm_model(gdn=_GDN(gqa_interleaved_layout=True)))
    assert "in_proj_ba" in reason(_vllm_model(gdn=_GDN(disable_tp_for_ba_proj=True)))
    assert "TP rank" in reason(_vllm_model(gdn=_GDN(tp_rank=1)), rank=0)
    assert "expert parallelism" in reason(_vllm_model(experts=_Experts(VllmUnquantizedFusedMoEMethod(), use_ep=True)))
    assert "Fp8MoEMethod" in reason(_vllm_model(experts=_Experts(Fp8MoEMethod())))
    without_up = {n: s for n, s in SHAPES.items() if not n.endswith("up_proj")}
    assert "lacks ['.mlp.experts.up_proj']" in reason(shapes=without_up)  # refused before anything is written
    assert "no GatedDeltaNet" in reason(shapes={f"{EXPERTS}.gate_proj": [E, INTER, HID]})
    no_gdn = torch.nn.Module()
    no_gdn.experts = _Experts(VllmUnquantizedFusedMoEMethod())
    assert "in_proj_qkvz" in reason(no_gdn)  # e.g. LoRA: separate in_proj_qkv / in_proj_z


def test_trainer_weights_applies_local_transform_only_to_local_shards():
    """``local_shards()`` exports the tpu-sync layout; plain iteration keeps verl's HF layout."""
    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights

    def weights():
        chunks = [{f"{EXPERTS}.gate_up_proj": torch.ones(2, 4, 3)}, {"model.norm.weight": torch.ones(3)}]
        local_transform = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)
        return TrainerWeights(iter(chunks), [], "cpu", local_transform=local_transform)

    local = {name: tuple(t.shape) for name, t, _ in weights().local_shards()}
    assert local == {f"{EXPERTS}.gate_proj": (2, 2, 3), f"{EXPERTS}.up_proj": (2, 2, 3), "model.norm.weight": (3,)}
    assert {name: tuple(t.shape) for name, t in weights()} == {
        f"{EXPERTS}.gate_up_proj": (2, 4, 3),
        "model.norm.weight": (3,),
    }


def test_refuse_split_tensors_for_vllm_loaders():
    """The loader fallback gets HF's fused tensors back from the split pieces, in sorted-name order."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    split = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG)(dict(full))
    out = dict(qwen3_5_layout.refuse_split_tensors(iter(sorted(split.items()))))
    assert set(out) == set(full)
    for name, t in full.items():
        assert torch.equal(out[name], t), name
    partial = [(n, t) for n, t in sorted(split.items()) if not n.endswith("up_proj")]
    with pytest.raises(RuntimeError, match="missing pieces"):
        list(qwen3_5_layout.refuse_split_tensors(iter(partial)))


def test_trainer_expert_parallel_layout_and_refuse():
    """EP layout: the expert stacks are sent transposed ([E, H, 2I] / [E, I, H]); the loader fallback undoes it."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    out = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG, expert_parallel=True)(dict(full))
    assert torch.equal(out[f"{EXPERTS}.w13_ep"], full[f"{EXPERTS}.gate_up_proj"].transpose(1, 2))
    assert torch.equal(out[f"{EXPERTS}.w2_ep"], full[f"{EXPERTS}.down_proj"].transpose(1, 2))
    assert out[f"{EXPERTS}.w13_ep"].is_contiguous()
    assert not any(n.endswith(("gate_up_proj", "down_proj", "gate_proj", "up_proj")) for n in out)
    back = dict(qwen3_5_layout.refuse_split_tensors(iter(sorted(out.items()))))
    assert set(back) == set(full)
    for name, t in full.items():
        assert torch.equal(back[name], t), name
    rule = qwen3_5_layout.sampler_sharding_rule
    assert rule(f"{EXPERTS}.w13_ep", [512, 4096, 2048], 16) == (["tp", "", ""], [32, 4096, 2048])
    assert rule(f"{EXPERTS}.w2_ep", [512, 1024, 4096], 16) == (["tp", "", ""], [32, 1024, 4096])


EP_SHAPES = {
    name: list(t.shape)
    for name, t in qwen3_5_layout.make_tpu_sync_local_layout(CONFIG, expert_parallel=True)(
        _hf_tensors(torch.Generator())
    ).items()
}


def test_plan_expert_parallel():
    def ep_model(tp, method=VllmUnquantizedFusedMoEMethod, use_ep=True):
        return _vllm_model(gdn=_GDN(tp_size=tp), experts=_Experts(method(), use_ep=use_ep))

    plan, reason = qwen3_5_layout.plan_direct_install(ep_model(1), EP_SHAPES, CONFIG, 1, 0)
    assert reason == "" and plan.moe_layout == qwen3_5_layout.MOE_LAYOUT_TPU_EP
    # moe_intermediate_size need not split over the ranks under EP (the experts are not sliced).
    odd = SimpleNamespace(**{**vars(CONFIG), "moe_intermediate_size": 5})
    assert qwen3_5_layout.plan_direct_install(ep_model(1), EP_SHAPES, odd, 1, 0)[0] is not None
    assert "rollout_expert_parallel=True" in qwen3_5_layout.plan_direct_install(ep_model(2), SHAPES, CONFIG, 2, 0)[1]
    no_ep = ep_model(1, use_ep=False)
    assert "does not use expert parallelism" in qwen3_5_layout.plan_direct_install(no_ep, EP_SHAPES, CONFIG, 1, 0)[1]
    quant = ep_model(1, method=Fp8MoEMethod)
    assert "unquantized" in qwen3_5_layout.plan_direct_install(quant, EP_SHAPES, CONFIG, 1, 0)[1]
    # 3 experts over 2 ranks.
    assert "do not split over EP" in qwen3_5_layout.plan_direct_install(ep_model(2), EP_SHAPES, CONFIG, 2, 0)[1]


@pytest.mark.parametrize("nested", [True, False])
def test_zero_copy_targets_and_install(nested):
    """Each rank's expert slice is received straight into its vLLM w13 / w2; install() leaves them alone."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    sent = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG, expert_parallel=True)(dict(full))
    shapes = {n: list(t.shape) for n, t in sent.items()}
    plan = qwen3_5_layout.DirectInstallPlan(
        qwen3_5_layout.Qwen3_5Dims.from_config(CONFIG), 0, 1, qwen3_5_layout.MOE_LAYOUT_TPU_EP
    )
    mid = ".routed_experts" if nested else ""
    model_sd = {
        f"{EXPERTS}{mid}.w13_weight": torch.zeros(E, HID, 2 * INTER, dtype=torch.bfloat16),
        f"{EXPERTS}{mid}.w2_weight": torch.zeros(E, INTER, HID, dtype=torch.bfloat16),
    }
    targets = qwen3_5_layout.zero_copy_targets(plan, model_sd, lambda k: k if k in model_sd else None, shapes)
    assert set(targets) == {f"{EXPERTS}.w13_ep", f"{EXPERTS}.w2_ep"}
    assert targets[f"{EXPERTS}.w13_ep"].data_ptr() == model_sd[f"{EXPERTS}{mid}.w13_weight"].data_ptr()
    consumed = qwen3_5_layout.install(
        {n: t for n, t in sent.items() if n.endswith(("w13_ep", "w2_ep"))}, model_sd, lambda k: None, None, plan
    )
    assert consumed == {f"{EXPERTS}.w13_ep", f"{EXPERTS}.w2_ep"}
    bad = {f"{EXPERTS}{mid}.w13_weight": torch.zeros(E, HID, 2 * INTER), **model_sd}
    bad[f"{EXPERTS}{mid}.w13_weight"] = torch.zeros(E, HID, 2 * INTER)  # float32: not a valid receive buffer
    with pytest.raises(RuntimeError, match="expert-parallel layout needs"):
        qwen3_5_layout.zero_copy_targets(plan, bad, lambda k: k if k in bad else None, shapes)
    tp_plan = qwen3_5_layout.DirectInstallPlan(plan.dims, 0, 1, qwen3_5_layout.MOE_LAYOUT_TPU)
    assert qwen3_5_layout.zero_copy_targets(tp_plan, model_sd, lambda k: k, shapes) == {}
