# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CPU tests for the direct tpu-sync install of Qwen3.5 into vLLM's TP-sharded parameters."""

from types import SimpleNamespace

import pytest
import torch

from verl_hardware_plugin.accelerators.tpu.rollout import moe_weights, qwen3_5_layout

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
        layout = moe_weights.MOE_LAYOUT_TPU if padded else moe_weights.MOE_LAYOUT_VLLM
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
        qwen3_5_layout.install(staging, model_sd, find, _copy_into(model_sd), _plan(0, 1, moe_weights.MOE_LAYOUT_VLLM))
    model_sd[f"{EXPERTS}.w13_weight"][..., INTER] = 1.0  # non-zero padding
    with pytest.raises(RuntimeError, match="padding is not zero"):
        qwen3_5_layout.install(staging, model_sd, find, _copy_into(model_sd), _plan(0, 1, moe_weights.MOE_LAYOUT_TPU))


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
            _plan(0, 1, moe_weights.MOE_LAYOUT_TPU),
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
    assert reason == "" and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU and (plan.tp_rank, plan.tp_size) == (0, 2)
    plan, _ = qwen3_5_layout.plan_direct_install(
        _vllm_model(experts=_Experts(UnquantizedFusedMoEMethod())), SHAPES, CONFIG, 2, 0
    )
    assert plan.moe_layout == moe_weights.MOE_LAYOUT_VLLM

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


def test_direct_install_runs_the_exact_parity_check_on_the_expert_writes():
    """``check`` sees every region of w13 / w2 that the install overwrites (the step-0 exact parity check)."""
    from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

    full = _hf_tensors(torch.Generator().manual_seed(0))
    staging = _received(full, 0, 1)
    model_sd = {k: t.clone() for k, t in _vllm_reference(full, 0, 1, 128).items()}  # vLLM holds the checkpoint
    model_sd[f"{EXPERTS}.w2_weight"][1, 0, 0] += 1  # one misplaced expert value

    def find(key):
        return key if key in model_sd else None

    parity = ExactParityCheck()
    qwen3_5_layout.install(
        staging, model_sd, find, _copy_into(model_sd), _plan(0, 1, moe_weights.MOE_LAYOUT_TPU), check=parity.check
    )
    assert parity.checked == 3  # w13 gate, w13 up, w2 (copy_into's own tensors are checked by the worker)
    assert [line.split(":")[0] for line in parity.mismatched] == [f"{EXPERTS}.w2_weight"]


def test_trainer_agnostic_vllm_expert_parallel_install():
    """Standard trainer layout (rollout_expert_parallel=False) installs directly into vLLM EP (dim-0 expert shards)."""
    ep_cfg = SimpleNamespace(**{**vars(CONFIG), "num_experts": 4, "moe_intermediate_size": 5})  # INTER=5 need not / TP
    gen = torch.Generator().manual_seed(7)
    full = _hf_tensors(gen)
    full[f"{EXPERTS}.gate_up_proj"] = torch.randn(4, 10, HID, generator=gen)
    full[f"{EXPERTS}.down_proj"] = torch.randn(4, HID, 5, generator=gen)
    sent = qwen3_5_layout.make_tpu_sync_local_layout(ep_cfg)(dict(full))
    shapes = {n: list(t.shape) for n, t in sent.items()}

    for tp_rank in range(2):
        ep_mod = _Experts(VllmUnquantizedFusedMoEMethod(), use_ep=True)
        ep_mod.ep_size, ep_mod.ep_rank = 2, tp_rank
        model = _vllm_model(gdn=_GDN(tp_rank=tp_rank, tp_size=2), experts=ep_mod)
        plan, why = qwen3_5_layout.plan_direct_install(model, shapes, ep_cfg, 2, tp_rank)
        assert why == "" and plan is not None and plan.expert_parallel and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU

        staging = {}
        for name, t in sent.items():
            rule = qwen3_5_layout.sampler_sharding_rule(name, list(t.shape), 2, plan)
            if rule is None:
                rule = (["tp", ""], [t.shape[0] // 2, t.shape[1]]) if t.ndim == 2 else ([""], list(t.shape))
            spec, local = rule
            if "tp" in spec:
                dim = spec.index("tp")
                t = t.narrow(dim, tp_rank * local[dim], local[dim])
            staging[name] = t.clone()
        assert staging[f"{EXPERTS}.gate_proj"].shape == (2, 5, HID)
        assert staging[f"{EXPERTS}.down_proj"].shape == (2, HID, 5)

        model_sd = {
            f"{GDN}.in_proj_qkvz.weight": torch.zeros(HID, (2 * K + 2 * V) // 2),
            f"{GDN}.conv1d.weight": torch.zeros((2 * K + V) // 2, 1, CONV_K),
            f"{GDN}.A_log": torch.zeros(NV // 2),
            f"{GDN}.dt_bias": torch.zeros(NV // 2),
            f"{EXPERTS}.w13_weight": torch.zeros(2, HID, 256),
            f"{EXPERTS}.w2_weight": torch.zeros(2, 128, HID),
        }
        consumed = qwen3_5_layout.install(
            staging, model_sd, lambda k, sd=model_sd: k if k in sd else None, _copy_into(model_sd), plan
        )
        assert consumed == set(staging)
        exp_slice = slice(tp_rank * 2, (tp_rank + 1) * 2)
        torch.testing.assert_close(
            model_sd[f"{EXPERTS}.w13_weight"][..., :5], full[f"{EXPERTS}.gate_up_proj"][exp_slice, :5].transpose(1, 2)
        )
        torch.testing.assert_close(
            model_sd[f"{EXPERTS}.w13_weight"][..., 128:133],
            full[f"{EXPERTS}.gate_up_proj"][exp_slice, 5:].transpose(1, 2),
        )
        torch.testing.assert_close(
            model_sd[f"{EXPERTS}.w2_weight"][:, :5], full[f"{EXPERTS}.down_proj"][exp_slice].transpose(1, 2)
        )


def test_trainer_expert_parallel_layout_and_refuse():
    """Zero-copy EP layout: expert stacks are sent transposed ([E, H, 2I] / [E, I, H]); loader fallback undoes it."""
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
        exp = _Experts(method(), use_ep=use_ep)
        if use_ep:
            exp.ep_size, exp.ep_rank = tp, 0
        return _vllm_model(gdn=_GDN(tp_size=tp), experts=exp)

    plan, reason = qwen3_5_layout.plan_direct_install(ep_model(1), EP_SHAPES, CONFIG, 1, 0)
    assert reason == "" and plan is not None and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU_EP
    # moe_intermediate_size need not split over the ranks under EP (the experts are not sliced).
    odd = SimpleNamespace(**{**vars(CONFIG), "moe_intermediate_size": 5})
    assert qwen3_5_layout.plan_direct_install(ep_model(1), EP_SHAPES, odd, 1, 0)[0] is not None
    no_ep = ep_model(1, use_ep=False)
    assert "does not use expert parallelism" in qwen3_5_layout.plan_direct_install(no_ep, EP_SHAPES, CONFIG, 1, 0)[1]
    quant = ep_model(1, method=Fp8MoEMethod)
    assert "unquantized" in qwen3_5_layout.plan_direct_install(quant, EP_SHAPES, CONFIG, 1, 0)[1]
    # 3 experts over 2 ranks.
    assert "experts do not split" in qwen3_5_layout.plan_direct_install(ep_model(2), EP_SHAPES, CONFIG, 2, 0)[1]


@pytest.mark.parametrize("nested", [True, False])
def test_zero_copy_targets_and_install(nested):
    """Each rank's expert slice is received straight into its vLLM w13 / w2; install() leaves them alone."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    sent = qwen3_5_layout.make_tpu_sync_local_layout(CONFIG, expert_parallel=True)(dict(full))
    shapes = {n: list(t.shape) for n, t in sent.items()}
    plan = qwen3_5_layout.DirectInstallPlan(
        qwen3_5_layout.Qwen3_5Dims.from_config(CONFIG), 0, 1, moe_weights.MOE_LAYOUT_TPU_EP, expert_parallel=True
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
    bad = {**model_sd, f"{EXPERTS}{mid}.w13_weight": torch.zeros(E, HID, 2 * INTER)}  # float32: invalid receive buffer
    with pytest.raises(RuntimeError, match="expert-parallel layout needs"):
        qwen3_5_layout.zero_copy_targets(plan, bad, lambda k: k if k in bad else None, shapes)
    tp_plan = qwen3_5_layout.DirectInstallPlan(plan.dims, 0, 1, moe_weights.MOE_LAYOUT_TPU)
    assert qwen3_5_layout.zero_copy_targets(tp_plan, model_sd, lambda k: k, shapes) == {}


def test_weight_export_owned_and_size1_mesh_dims():
    """bf16 views/aliases are cloned into owned buffers; size-1 mesh dims (e.g. efsdp=1) export locally."""
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor.placement_types import _StridedShard

    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights, _owned, fsdp_dim0_shard

    base = torch.randn(6, 4, dtype=torch.bfloat16)
    view = base.narrow(0, 2, 2)
    owned_view = _owned(view, aliases_param=False)
    assert owned_view._base is None and torch.equal(owned_view, view)
    owned_alias = _owned(base, aliases_param=True)
    assert owned_alias.data_ptr() != base.data_ptr() and torch.equal(owned_alias, base)

    class _FakeMesh:
        ndim = 2
        device_type = "cpu"

        def size(self, d: int) -> int:
            return (1, 4)[d]

        def get_coordinate(self):
            return [0, 2]

    local_exp = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    spec = DTensorSpec(
        _FakeMesh(),  # type: ignore[arg-type]
        (_StridedShard(0, split_factor=4), Shard(0)),
        tensor_meta=TensorMeta(torch.Size([8, 4, 8]), (32, 8, 1), torch.bfloat16),
    )
    dt = DTensor(local_exp.view_as(local_exp), spec, requires_grad=False)
    assert fsdp_dim0_shard(dt) == (4, 2)
    dt._tpu_sync_owned = True  # type: ignore[attr-defined]
    [(name, exported, shard)] = list(TrainerWeights({"w": dt}, [], "cpu").local_shards())
    assert name == "w" and shard == (4, 2) and exported.data_ptr() == local_exp.data_ptr()


def test_fsdp_dim0_shard_efsdp_times_ep_whole_experts():
    """SimpleFSDP + EP with efsdp > 1: (_StridedShard(0, ep), Shard(0)) holds block i_ep * efsdp + i_efsdp.

    Matches torchtitan's layout as observed on an 8-rank (efsdp=2, ep=4) Gloo mesh with 16 experts: rank
    (i_efsdp, i_ep) = (0, 1) holds experts 4-5 (block 2 of 8) and (1, 0) holds experts 2-3 (block 1 of 8).
    """
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
    from torch.distributed.tensor.placement_types import _StridedShard

    from verl_hardware_plugin.accelerators.tpu.engines.weight_export import TrainerWeights, fsdp_dim0_shard

    efsdp, ep, num_experts = 2, 4, 16

    def _dt(coord, placements, local_rows=num_experts // (efsdp * ep), sizes=(efsdp, ep)):
        class _FakeMesh:
            ndim = 2
            device_type = "cpu"

            def size(self, d: int) -> int:
                return sizes[d]

            def get_coordinate(self):
                return list(coord)

        local = torch.randn(local_rows, 4, 8, dtype=torch.bfloat16)
        spec = DTensorSpec(
            _FakeMesh(),  # type: ignore[arg-type]
            placements,
            tensor_meta=TensorMeta(torch.Size([num_experts, 4, 8]), (32, 8, 1), torch.bfloat16),
        )
        return DTensor(local.view_as(local), spec, requires_grad=False), local

    strided = (_StridedShard(0, split_factor=ep), Shard(0))
    # Expected blocks from the live 8-rank (efsdp=2, ep=4) torchtitan run: experts held = [2 * block, 2 * block + 1].
    observed_first_expert = {(0, 0): 0, (0, 1): 4, (0, 2): 8, (0, 3): 12, (1, 0): 2, (1, 1): 6, (1, 2): 10, (1, 3): 14}
    for coord, first_expert in observed_first_expert.items():
        dt, _ = _dt(coord, strided)
        assert fsdp_dim0_shard(dt) == (efsdp * ep, first_expert // 2), coord

    # Wrong split factor, swapped placements, or a non-whole-expert local block are not this layout: gathered.
    assert fsdp_dim0_shard(_dt((1, 2), (_StridedShard(0, split_factor=2), Shard(0)))[0]) is None
    assert fsdp_dim0_shard(_dt((1, 2), (Shard(0), _StridedShard(0, split_factor=ep)))[0]) is None
    assert fsdp_dim0_shard(_dt((1, 2), (Replicate(), Shard(0)))[0]) is None
    assert fsdp_dim0_shard(_dt((1, 2), strided, local_rows=3)[0]) is None

    dt, local = _dt((1, 3), strided)
    dt._tpu_sync_owned = True  # type: ignore[attr-defined]
    [(name, exported, shard)] = list(TrainerWeights({"w13_ep": dt}, [], "cpu").local_shards())
    assert name == "w13_ep" and shard == (8, 7) and exported.data_ptr() == local.data_ptr()


# ---------------------------------------------------------------------------------------------------------------
# Quantized (FP8) experts: requantized on the rollout from the received bf16 slices (moe_weights.write_requantized).
# ---------------------------------------------------------------------------------------------------------------

FP8 = torch.float8_e4m3fn


def reference_quantize_and_format(w, *, name, quant_dtype, block_size, activation=""):
    """A CPU copy of vllm-torchtpu's ``_quantize_and_format_single_moe_weight`` + ``quantize_tensor(axis=-1)``:
    128-padding of each w13 half / of w2's intermediate dim, per-row (or per-block) amax scales with the reciprocal
    multiply, transpose to the GMM layout, scale [E, K / block or 1, 1, N]."""
    assert w.dtype == torch.float32 and activation != "swigluoai"
    if name == "w13":
        half = w.shape[1] // 2
        aligned = (half + 127) // 128 * 128
        if aligned != half:
            pad = w.new_zeros(w.shape[0], aligned - half, w.shape[2])
            w = torch.cat([w[:, :half], pad, w[:, half:], pad], dim=1)
    else:
        inter = w.shape[2]
        aligned = (inter + 127) // 128 * 128
        if aligned != inter:
            w = torch.cat([w, w.new_zeros(w.shape[0], w.shape[1], aligned - inter)], dim=2)
    if block_size is not None and w.shape[-1] % block_size:
        raise ValueError(f"{name} contracting dimension {w.shape[-1]} is not divisible by block_size={block_size}")
    bs = block_size or w.shape[-1]
    blocked = w.reshape(*w.shape[:-1], w.shape[-1] // bs, bs)
    info = torch.finfo(quant_dtype)
    scale = blocked.abs().amax(dim=-1, keepdim=True) * torch.tensor(1.0 / float(info.max))
    scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = torch.clamp(blocked * scale_inv, min=float(info.min), max=float(info.max)).reshape_as(w).to(quant_dtype)
    scale = scale.squeeze(-1)  # [E, N, K / block]
    return q.transpose(1, 2).contiguous(), scale.transpose(1, 2).contiguous().to(torch.float32).unsqueeze(2)


@pytest.fixture
def fake_vllm_torchtpu_fp8(monkeypatch):
    """vllm-torchtpu's FP8 module, quantizing with ``reference_quantize_and_format``.

    Set ``.block_size`` for ``MOE_REQUANTIZE_BLOCK_SIZE`` and ``._REQUANT_CHUNK_BYTES`` for the chunk budget.
    """
    import sys
    import types

    full = moe_weights.VLLM_TORCHTPU_FP8_MODULE
    mod = types.ModuleType(full)
    mod.block_size = None
    mod._get_moe_quant_config = lambda log_prefix: ("float8_e4m3fn", FP8, mod.block_size)
    mod._quantize_and_format_single_moe_weight = reference_quantize_and_format
    parts = full.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        module = mod if i == len(parts) else sys.modules.get(name) or types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if i > 1:
            monkeypatch.setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], module, raising=False)
    return mod


class VllmFp8MoEMethodTPU:  # vllm-torchtpu's FP8 MoE method (matched by class name)
    def __init__(self, activation="silu"):
        self._tpu_activation_str = activation  # what process_weights_after_loading quantized with


class _Fp8Experts(torch.nn.Module):
    """vLLM's FP8 MoE layer as vllm-torchtpu leaves it: w13 [E, H, 2A], w2 [E, A, H] in FP8, FP32 GMM scales."""

    def __init__(self, num_experts, inter, hidden, block_size=None, method=None, **attrs):
        super().__init__()
        aligned = (inter + 127) // 128 * 128
        nb13, nb2 = (hidden // block_size, aligned // block_size) if block_size else (1, 1)
        shapes = {
            "w13_weight": ((num_experts, hidden, 2 * aligned), FP8),
            "w13_weight_scale_inv": ((num_experts, nb13, 1, 2 * aligned), torch.float32),
            "w2_weight": ((num_experts, aligned, hidden), FP8),
            "w2_weight_scale_inv": ((num_experts, nb2, 1, hidden), torch.float32),
        }
        for leaf, (shape, dtype) in shapes.items():
            setattr(self, leaf, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
        self.quant_method = method if method is not None else VllmFp8MoEMethodTPU()
        self.use_ep = False
        for k, v in attrs.items():
            setattr(self, k, v)

    def expert_sd(self, prefix=EXPERTS):
        return {f"{prefix}.{leaf}": t.data for leaf, t in self.named_parameters()}


def _fp8_model_sd(experts, tp_rank, tp_size):
    """vLLM's state dict of rank ``tp_rank``: the GatedDeltaNet tensors (NaN: must all be written) and the experts."""
    full = _hf_tensors(torch.Generator().manual_seed(0))
    sd = {
        name: torch.full_like(t, float("nan"))
        for name, t in _vllm_reference(full, tp_rank, tp_size, 128).items()
        if ".experts." not in name
    }
    return {**sd, **experts.expert_sd()}


def _requantized_reference(gate, up, down, block_size=None):
    """The rollout's quantizer applied to rank-local bf16 slices: what vLLM's load-time quantization holds."""
    kw = dict(quant_dtype=FP8, block_size=block_size, activation="silu")
    w13, s13 = reference_quantize_and_format(torch.cat([gate, up], dim=1).float(), name="w13", **kw)
    w2, s2 = reference_quantize_and_format(down.float(), name="w2", **kw)
    return {"w13_weight": w13, "w13_weight_scale_inv": s13, "w2_weight": w2, "w2_weight_scale_inv": s2}


def _assert_bits_equal(actual, expected, msg):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, msg
    if actual.dtype == FP8:
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    assert torch.equal(actual, expected), msg


def test_plan_fp8_moe(fake_vllm_torchtpu_fp8):
    """The FP8 plan resolves vllm-torchtpu's format once and validates the live weights and scales up front."""

    def plan(experts=None, tp=2, rank=0, **kw):
        experts = experts if experts is not None else _Fp8Experts(E, INTER // tp, HID)
        model = _vllm_model(gdn=_GDN(tp_rank=rank, tp_size=tp), experts=experts)
        return qwen3_5_layout.plan_direct_install(model, SHAPES, CONFIG, tp, rank, **kw)

    p, why = plan()
    assert why == "" and p.moe_layout == moe_weights.MOE_LAYOUT_TPU_FP8 and not p.expert_parallel
    assert (p.moe_requant.quant_dtype, p.moe_requant.block_size, p.moe_requant.activation) == (FP8, None, "silu")
    assert p.moe_requant.quantize_and_format is reference_quantize_and_format
    assert p.moe_requant.chunk_bytes == 1 << 30
    # The activation comes from the live module (what vllm-torchtpu quantized and runs with), not a constant.
    gelu = _Fp8Experts(E, INTER // 2, HID, method=VllmFp8MoEMethodTPU("gelu"))
    assert plan(gelu)[0].moe_requant.activation == "gelu"
    no_str = _Fp8Experts(E, INTER // 2, HID, method=type("VllmFp8MoEMethodTPU", (), {})(), activation="SiLU")
    assert plan(no_str)[0].moe_requant.activation == "silu"  # vLLM's layer.activation when the method has none

    def reason(*args, **kw):
        p, why = plan(*args, **kw)
        assert p is None
        return why

    assert "direct install of quantized experts is off" in reason(allow_quantized_moe=False)
    wrong_dtype = _Fp8Experts(E, INTER // 2, HID)
    wrong_dtype.w2_weight.data = torch.zeros(E, 128, HID, dtype=torch.bfloat16)
    assert "w2_weight is (3, 128, 6) torch.bfloat16, requantizing gives (3, 128, 6) torch.float8_e4m3fn" in reason(
        wrong_dtype
    )
    no_scale = _Fp8Experts(E, INTER // 2, HID)
    del no_scale.w2_weight_scale_inv
    assert "has no w2_weight_scale_inv" in reason(no_scale)
    fake_vllm_torchtpu_fp8.block_size = 4  # MOE_REQUANTIZE_BLOCK_SIZE=4 does not divide the hidden size 6
    assert "does not divide the contracting dimensions" in reason()
    fake_vllm_torchtpu_fp8.block_size = 2  # blockwise scales: [E, H / 2, 1, 2A] and [E, A / 2, 1, H]
    assert "w13_weight_scale_inv is (3, 1, 1, 256)" in reason()
    assert plan(_Fp8Experts(E, INTER // 2, HID, block_size=2))[0].moe_requant.block_size == 2
    for bad_bs in (0, -2, (128, 128), True):
        fake_vllm_torchtpu_fp8.block_size = bad_bs
        assert "not a positive int" in reason()
    fake_vllm_torchtpu_fp8.block_size = None
    # Non-persistent scale buffer (excluded from state_dict) is caught at plan time, not mid-install.
    non_persistent = _Fp8Experts(E, INTER // 2, HID)
    s2 = non_persistent.w2_weight_scale_inv.detach().clone()
    del non_persistent.w2_weight_scale_inv
    non_persistent.register_buffer("w2_weight_scale_inv", s2, persistent=False)
    assert "has no w2_weight_scale_inv" in reason(non_persistent)
    # swigluoai on _tpu_activation_str, expert w2_bias, and armed fused decode kernels are refused at plan time.
    assert "swigluoai" in reason(_Fp8Experts(E, INTER // 2, HID, method=VllmFp8MoEMethodTPU("swigluoai")))
    assert "bias" in reason(_Fp8Experts(E, INTER // 2, HID, w2_bias=torch.zeros(HID)))
    armed = VllmFp8MoEMethodTPU()
    armed._fused_moe_decode_op = object()
    assert "fused shared+routed decode kernel" in reason(_Fp8Experts(E, INTER // 2, HID, method=armed))
    # Current vllm-torchtpu main exposes envs.TPU_MOE_REQUANT_CHUNK_BYTES instead of _REQUANT_CHUNK_BYTES.
    fake_vllm_torchtpu_fp8.envs = SimpleNamespace(TPU_MOE_REQUANT_CHUNK_BYTES=256 * 1024)
    assert plan()[0].moe_requant.chunk_bytes == 256 * 1024
    fake_vllm_torchtpu_fp8.envs = SimpleNamespace(TPU_MOE_REQUANT_CHUNK_BYTES=0)
    assert "must be positive" in reason()
    del fake_vllm_torchtpu_fp8.envs
    fake_vllm_torchtpu_fp8._quantize_and_format_single_moe_weight = lambda w, *, name, dtype: None  # renamed kwargs
    assert "unexpected signature" in reason()
    del fake_vllm_torchtpu_fp8._quantize_and_format_single_moe_weight
    assert "lacks _quantize_and_format_single_moe_weight" in reason()


def test_plan_refuses_quantized_non_expert_layers_and_mismatched_expert_blocks(fake_vllm_torchtpu_fp8):
    """Direct install requires non-expert weights to stay unquantized (copy_into writes unscaled bf16), and validates
    both the MoE module count and per-block 3-D wire shapes on Qwen3.5."""
    VllmUnquantizedLinearMethod = type("VllmUnquantizedLinearMethod", (), {})
    UnquantizedEmbeddingMethod = type("UnquantizedEmbeddingMethod", (), {})
    VllmFp8LinearMethodTPU = type("VllmFp8LinearMethodTPU", (), {})

    # Unquantized linear / embedding methods on non-expert modules are accepted.
    ok_model = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    ok_model.gdn.in_proj_qkvz.quant_method = VllmUnquantizedLinearMethod()
    ok_model.embed_tokens = torch.nn.Embedding(16, HID)
    ok_model.embed_tokens.quant_method = UnquantizedEmbeddingMethod()
    assert qwen3_5_layout.plan_direct_install(ok_model, SHAPES, CONFIG, 2, 0)[0] is not None

    # Pre-quantized FP8 checkpoint / QUANTIZE_ON_LOAD_PREFIXES beyond experts: non-expert FP8 linears are refused.
    fp8_linear_model = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    fp8_linear_model.gdn.in_proj_qkvz.quant_method = VllmFp8LinearMethodTPU()
    plan, why = qwen3_5_layout.plan_direct_install(fp8_linear_model, SHAPES, CONFIG, 2, 0)
    assert plan is None and "VllmFp8LinearMethodTPU" in why

    scaled_model = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    scaled_model.shared_expert = torch.nn.Linear(HID, INTER)
    scaled_model.shared_expert.register_parameter("weight_scale", torch.nn.Parameter(torch.ones(INTER)))
    plan, why = qwen3_5_layout.plan_direct_install(scaled_model, SHAPES, CONFIG, 2, 0)
    assert plan is None and "has weight_scale" in why

    fp8_param_model = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    fp8_param_model.gdn.in_proj_qkvz.weight = torch.nn.Parameter(
        torch.zeros(2 * K + 2 * V, HID, dtype=FP8), requires_grad=False
    )
    plan, why = qwen3_5_layout.plan_direct_install(fp8_param_model, SHAPES, CONFIG, 2, 0)
    assert plan is None and "quantized dtype" in why

    # Mismatched MoE module count (e.g. an extra draft-model MoE layer) or bad 3-D expert wire shapes are refused.
    extra_moe = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    extra_moe.draft_experts = _Fp8Experts(E, INTER // 2, HID)
    plan, why = qwen3_5_layout.plan_direct_install(extra_moe, SHAPES, CONFIG, 2, 0)
    assert plan is None and "2 fused-MoE layers" in why

    folded_shapes = {**SHAPES, f"{EXPERTS}.down_proj": [E * HID, INTER]}
    plan, why = qwen3_5_layout.plan_direct_install(
        _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID)), folded_shapes, CONFIG, 2, 0
    )
    assert plan is None and "3-D expert tensors" in why

    transposed_down = {**SHAPES, f"{EXPERTS}.down_proj": [E, INTER, HID]}
    plan, why = qwen3_5_layout.plan_direct_install(
        _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID)), transposed_down, CONFIG, 2, 0
    )
    assert plan is None and "the config gives" in why

    # Rank-local shape check in install() rejects wrong TP slices before write_requantized pads them to 128.
    experts = _Fp8Experts(E, INTER // 2, HID)
    valid_model = _vllm_model(experts=experts)
    valid_plan, _ = qwen3_5_layout.plan_direct_install(valid_model, SHAPES, CONFIG, 2, 0)
    bad_staging = _received(_hf_tensors(torch.Generator().manual_seed(0)), 0, 2)
    bad_staging[f"{EXPERTS}.gate_proj"] = torch.zeros(E, INTER, HID, dtype=torch.bfloat16)  # TP=1 slice on TP=2 plan
    model_sd = _fp8_model_sd(experts, 0, 2)
    with pytest.raises(RuntimeError, match="on this rank, expected"):
        qwen3_5_layout.install(
            bad_staging, model_sd, lambda k: k if k in model_sd else None, _copy_into(model_sd), valid_plan
        )


def test_plan_fp8_moe_without_vllm_torchtpu_falls_back():
    """No importable vllm-torchtpu requantizer: the plan refuses (vLLM's loaders install), nothing is written."""
    model = _vllm_model(experts=_Fp8Experts(E, INTER // 2, HID))
    plan, why = qwen3_5_layout.plan_direct_install(model, SHAPES, CONFIG, 2, 0)
    assert plan is None and "cannot import vllm-torchtpu's FP8 MoE requantizer" in why


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("block_size", [None, 2])
def test_fp8_install_equals_requantizing_the_local_shard(fake_vllm_torchtpu_fp8, tp_size, block_size):
    """Each rank's live FP8 experts and scales are bit for bit the rollout quantizer applied to its own bf16 TP slice
    (I_r = 4 / tp is padded to 128, w2's per-H scales cover only the local slice), the GatedDeltaNet tensors are
    written as for bf16, and every received tensor is consumed."""
    fake_vllm_torchtpu_fp8.block_size = block_size
    full = _hf_tensors(torch.Generator().manual_seed(0))
    for name in (f"{EXPERTS}.gate_up_proj", f"{EXPERTS}.down_proj"):
        full[name] = full[name].bfloat16().float()  # the trainer sends bf16
    for tp_rank in range(tp_size):
        experts = _Fp8Experts(E, INTER // tp_size, HID, block_size=block_size)
        model = _vllm_model(gdn=_GDN(tp_rank=tp_rank, tp_size=tp_size), experts=experts)
        plan, why = qwen3_5_layout.plan_direct_install(model, SHAPES, CONFIG, tp_size, tp_rank)
        assert why == "", why
        staging = {n: t.bfloat16() if ".experts." in n else t for n, t in _received(full, tp_rank, tp_size).items()}
        model_sd = _fp8_model_sd(experts, tp_rank, tp_size)
        consumed = qwen3_5_layout.install(
            staging, model_sd, lambda k, sd=model_sd: k if k in sd else None, _copy_into(model_sd), plan
        )
        assert consumed == set(staging) and not plan.padding_checked  # requantized: no padding check ran

        inter_r = INTER // tp_size
        gate_up, down = full[f"{EXPERTS}.gate_up_proj"], full[f"{EXPERTS}.down_proj"]
        gate = gate_up[:, tp_rank * inter_r : (tp_rank + 1) * inter_r]
        up = gate_up[:, INTER + tp_rank * inter_r : INTER + (tp_rank + 1) * inter_r]
        local_down = down[..., tp_rank * inter_r : (tp_rank + 1) * inter_r]
        for leaf, ref in _requantized_reference(gate, up, local_down, block_size).items():
            _assert_bits_equal(model_sd[f"{EXPERTS}.{leaf}"], ref, f"rank {tp_rank} {leaf}")
        w2 = model_sd[f"{EXPERTS}.w2_weight"]
        assert not w2[:, inter_r:].float().any()  # fresh zero padding
        if block_size is None:  # w2's per-output-channel scales: amax over this rank's I_r slice only
            torch.testing.assert_close(
                model_sd[f"{EXPERTS}.w2_weight_scale_inv"][:, 0, 0], local_down.abs().amax(-1) / 448.0
            )
        reference = _vllm_reference(full, tp_rank, tp_size, 128)
        for name in (f"{GDN}.in_proj_qkvz.weight", f"{GDN}.conv1d.weight", f"{GDN}.A_log", f"{GDN}.dt_bias"):
            torch.testing.assert_close(model_sd[name], reference[name], rtol=0, atol=0, msg=name)


def test_fp8_install_exact_parity_compares_the_requantized_experts(fake_vllm_torchtpu_fp8):
    """verify_parity=exact: vLLM quantized the same bf16 checkpoint on load, so the requantized weights and scales
    must match the live ones bit for bit; a changed live value is reported, not silently overwritten."""
    from verl_hardware_plugin.accelerators.tpu.weight_sync.tpu_sync_parity import ExactParityCheck

    full = _hf_tensors(torch.Generator().manual_seed(0))
    experts = _Fp8Experts(E, INTER // 2, HID)
    model = _vllm_model(gdn=_GDN(tp_rank=0, tp_size=2), experts=experts)
    plan, _ = qwen3_5_layout.plan_direct_install(model, SHAPES, CONFIG, 2, 0)
    staging = {n: t.bfloat16() if ".experts." in n else t for n, t in _received(full, 0, 2).items()}
    model_sd = _fp8_model_sd(experts, 0, 2)

    def find(key):
        return key if key in model_sd else None

    qwen3_5_layout.install(staging, model_sd, find, _copy_into(model_sd), plan)  # vLLM's load-time state
    parity = ExactParityCheck()
    qwen3_5_layout.install(staging, model_sd, find, _copy_into(model_sd), plan, check=parity.check)
    assert parity.checked == 4 and parity.mismatched == []  # w13, its scale, w2, its scale
    model_sd[f"{EXPERTS}.w2_weight_scale_inv"][1, 0, 0, 2] *= 2
    parity = ExactParityCheck()
    qwen3_5_layout.install(staging, model_sd, find, _copy_into(model_sd), plan, check=parity.check)
    assert [line.split(":")[0] for line in parity.mismatched] == [f"{EXPERTS}.w2_weight_scale_inv[requantized]"]


def test_fp8_requantization_is_chunked_and_materialized(fake_vllm_torchtpu_fp8, monkeypatch):
    """Experts are requantized a chunk at a time (vllm-torchtpu's ``_REQUANT_CHUNK_BYTES``), each chunk and each
    written weight materialized; the result is bit-identical to one whole-stack quantization."""
    fake_vllm_torchtpu_fp8._REQUANT_CHUNK_BYTES = 1  # one expert per chunk
    synced = []
    monkeypatch.setattr(moe_weights, "tpu_synchronize", lambda tensors: synced.append(len(tensors)))
    full = _hf_tensors(torch.Generator().manual_seed(3))
    experts = _Fp8Experts(E, INTER, HID)
    model = _vllm_model(gdn=_GDN(tp_rank=0, tp_size=1), experts=experts)
    plan, _ = qwen3_5_layout.plan_direct_install(model, SHAPES, CONFIG, 1, 0)
    assert plan.moe_requant.chunk_bytes == 1
    staging = _received(full, 0, 1)
    model_sd = _fp8_model_sd(experts, 0, 1)
    qwen3_5_layout.install(staging, model_sd, lambda k: k if k in model_sd else None, _copy_into(model_sd), plan)
    # w13: E chunks of (weight, scale) then the written (weight, scale); w2 likewise.
    assert synced == [2] * (E + 1) * 2
    gate_up, down = full[f"{EXPERTS}.gate_up_proj"], full[f"{EXPERTS}.down_proj"]
    reference = _requantized_reference(gate_up[:, :INTER], gate_up[:, INTER:], down)
    for leaf, ref in reference.items():
        _assert_bits_equal(model_sd[f"{EXPERTS}.{leaf}"], ref, leaf)


def test_fp8_install_under_vllm_expert_parallelism(fake_vllm_torchtpu_fp8):
    """vLLM EP over the TP group: each rank requantizes its whole experts (dim-0 slices) into its FP8 weights."""
    ep_cfg = SimpleNamespace(**{**vars(CONFIG), "num_experts": 4, "moe_intermediate_size": 5})
    gen = torch.Generator().manual_seed(7)
    full = _hf_tensors(gen)
    full[f"{EXPERTS}.gate_up_proj"] = torch.randn(4, 10, HID, generator=gen)
    full[f"{EXPERTS}.down_proj"] = torch.randn(4, HID, 5, generator=gen)
    sent = qwen3_5_layout.make_tpu_sync_local_layout(ep_cfg)(dict(full))
    shapes = {n: list(t.shape) for n, t in sent.items()}
    for tp_rank in range(2):
        experts = _Fp8Experts(2, 5, HID, use_ep=True, ep_size=2, ep_rank=tp_rank)
        model = _vllm_model(gdn=_GDN(tp_rank=tp_rank, tp_size=2), experts=experts)
        plan, why = qwen3_5_layout.plan_direct_install(model, shapes, ep_cfg, 2, tp_rank)
        assert why == "" and plan.expert_parallel and plan.moe_layout == moe_weights.MOE_LAYOUT_TPU_FP8
        staging = {}
        for name, t in sent.items():
            rule = qwen3_5_layout.sampler_sharding_rule(name, list(t.shape), 2, plan)
            if rule is None:
                rule = (["tp", ""], [t.shape[0] // 2, t.shape[1]]) if t.ndim == 2 else ([""], list(t.shape))
            spec, local = rule
            if "tp" in spec:
                dim = spec.index("tp")
                t = t.narrow(dim, tp_rank * local[dim], local[dim])
            staging[name] = t.clone()
        model_sd = {
            f"{GDN}.in_proj_qkvz.weight": torch.zeros(HID, (2 * K + 2 * V) // 2),
            f"{GDN}.conv1d.weight": torch.zeros((2 * K + V) // 2, 1, CONV_K),
            f"{GDN}.A_log": torch.zeros(NV // 2),
            f"{GDN}.dt_bias": torch.zeros(NV // 2),
            **experts.expert_sd(),
        }
        consumed = qwen3_5_layout.install(
            staging, model_sd, lambda k, sd=model_sd: k if k in sd else None, _copy_into(model_sd), plan
        )
        assert consumed == set(staging)
        mine = slice(tp_rank * 2, (tp_rank + 1) * 2)
        gate_up, down = full[f"{EXPERTS}.gate_up_proj"][mine], full[f"{EXPERTS}.down_proj"][mine]
        for leaf, ref in _requantized_reference(gate_up[:, :5], gate_up[:, 5:], down).items():
            _assert_bits_equal(model_sd[f"{EXPERTS}.{leaf}"], ref, f"rank {tp_rank} {leaf}")
