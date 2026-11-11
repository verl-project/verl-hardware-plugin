# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process rollout's install verification (``rollout/inproc/weight_sync.py``): fail closed.

No vLLM, vllm-torchtpu or torch_tpu: the loader-path checks run on a small torch stand-in of a vLLM dense model (fused
``qkv_proj`` / ``gate_up_proj``, vLLM's stacked-shard mapper surface) whose parameters hold what vLLM's loaders write.
What the tests pin:

* the first sync refuses an install mode other than the tested layout's family;
* the export pin: any missing / extra / reshaped tensor after the first verified sync is named and refused before
  the install, at every sync; until a verified sync has passed, every sync is verified;
* the packed parity: a missing fused shard is unresolved (an incomplete parameter is never excused), vllm-torchtpu's
  transposed (n_in, n_out) dense layout compares exact, a tied LM head must equal the embedding it aliases, a received
  tensor no engine parameter takes is refused, an untied LM head the trainer did not send is unresolved;
* the coverage check: the direct install's LM head copied from the embedding counts only for a tied config; the only
  exemption is the tested layout's not-exported patterns, for parameters nothing was received for, and those patterns
  must match no received tensor; a verified direct install whose recorder recorded nothing fails;
* ``TPUInProcRollout.update_weights`` itself on a rollout shell (an idle engine, the real installer, a stand-in of the
  install's writes): the export pin, verify-until-pinned, the mode pin at every sync, the coverage check folded into
  the parity verdict, the staleness check with skipped optimizer steps, and that no environment variable weakens any
  of it.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compat
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import rollout as rollout_mod
from verl_hardware_plugin.accelerators.tpu.rollout.inproc import weight_sync as ws
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.engine import InProcAsyncEngine
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.rollout import TPUInProcRollout

H, IM, V, HKV = 8, 16, 32, 4  # hidden, intermediate, vocab, k / v rows (GQA: q has H rows, k and v HKV each)
LOADERS = compat.LayoutEntry("qwen3", "DenseForCausalLM", "tests.Dense", "vllm loaders", (), ())
DIRECT = compat.LayoutEntry("qwen3", "DenseForCausalLM", "tests.Dense", "dense-test direct", (), ())
LOADERS_TOWER = compat.LayoutEntry(
    "qwen3", "DenseForCausalLM", "tests.Dense", "vllm loaders", compat.QWEN3_5_NOT_EXPORTED, ()
)


class _StackedMapper:
    """The surface of vLLM's ``WeightsMapper`` that ``InProcInstaller.loader_expected`` reads: ``orig_to_new_stacked``
    and ``_map_name_with_shard`` (a dense Qwen3 / Llama table)."""

    orig_to_new_stacked = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def _map_name_with_shard(self, key: str):
        shard = None
        for sub, (new, sid) in self.orig_to_new_stacked.items():
            if sub in key:
                key, shard = key.replace(sub, new, 1), sid
        return key, shard


class _DenseVllmModel(nn.Module):
    """A vLLM-shaped dense model (one layer): fused qkv / gate_up, separate o / down, norms, an LM head tied or not."""

    hf_to_vllm_mapper = _StackedMapper()
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, tied: bool, mtp: bool = False):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, H)
        layer = nn.Module()
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        layer.self_attn.qkv_proj = nn.Linear(H, H + 2 * HKV, bias=False)
        layer.self_attn.o_proj = nn.Linear(H, H, bias=False)
        layer.mlp.gate_up_proj = nn.Linear(H, 2 * IM, bias=False)
        layer.mlp.down_proj = nn.Linear(IM, H, bias=False)
        layer.input_layernorm = nn.LayerNorm(H, bias=False)
        self.model.layers = nn.ModuleList([layer])
        self.model.norm = nn.LayerNorm(H, bias=False)
        self.lm_head = nn.Linear(H, V, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight
        if mtp:  # an MTP head's attention (a not-exported pattern of the Qwen3.5 table row)
            self.mtp = nn.Module()
            self.mtp.self_attn = nn.Module()
            self.mtp.self_attn.qkv_proj = nn.Linear(H, H + 2 * HKV, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)


def _export(seed: int = 0, tied: bool = False) -> dict:
    """The trainer's bf16 HF-named export (a tied trainer sends its LM head under both names)."""
    g = torch.Generator().manual_seed(seed)
    shapes = {
        "model.embed_tokens.weight": (V, H),
        "model.layers.0.self_attn.q_proj.weight": (H, H),
        "model.layers.0.self_attn.k_proj.weight": (HKV, H),
        "model.layers.0.self_attn.v_proj.weight": (HKV, H),
        "model.layers.0.self_attn.o_proj.weight": (H, H),
        "model.layers.0.mlp.gate_proj.weight": (IM, H),
        "model.layers.0.mlp.up_proj.weight": (IM, H),
        "model.layers.0.mlp.down_proj.weight": (H, IM),
        "model.layers.0.input_layernorm.weight": (H,),
        "model.norm.weight": (H,),
        "lm_head.weight": (V, H),
    }
    out = {k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in shapes.items()}
    if tied:
        out["lm_head.weight"] = out["model.embed_tokens.weight"].clone()
    return out


def _installed(export: dict, tied: bool, mtp: bool = False) -> _DenseVllmModel:
    """The engine model after a vLLM-loader install of ``export``."""
    m = _DenseVllmModel(tied, mtp).to(torch.bfloat16)
    if mtp:
        with torch.no_grad():
            for p, (a, b) in zip("qkv", ((0, H), (H, H + HKV), (H + HKV, H + 2 * HKV)), strict=True):
                if f"mtp.self_attn.{p}_proj.weight" in export:
                    m.mtp.self_attn.qkv_proj.weight[a:b].copy_(export[f"mtp.self_attn.{p}_proj.weight"])
    lay = m.model.layers[0]
    with torch.no_grad():
        m.model.embed_tokens.weight.copy_(export["model.embed_tokens.weight"])
        lay.self_attn.qkv_proj.weight.copy_(
            torch.cat([export[f"model.layers.0.self_attn.{p}_proj.weight"] for p in "qkv"])
        )
        lay.self_attn.o_proj.weight.copy_(export["model.layers.0.self_attn.o_proj.weight"])
        lay.mlp.gate_up_proj.weight.copy_(
            torch.cat([export[f"model.layers.0.mlp.{p}_proj.weight"] for p in ("gate", "up")])
        )
        lay.mlp.down_proj.weight.copy_(export["model.layers.0.mlp.down_proj.weight"])
        lay.input_layernorm.weight.copy_(export["model.layers.0.input_layernorm.weight"])
        m.model.norm.weight.copy_(export["model.norm.weight"])
        if not tied and "lm_head.weight" in export:
            m.lm_head.weight.copy_(export["lm_head.weight"])
    return m


def _installer(model: nn.Module, layout=LOADERS, cfg=None) -> ws.InProcInstaller:
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    return ws.InProcInstaller(worker, cfg or SimpleNamespace(model_type="qwen3"), 0, layout=layout)


# -------------------------------------------------------------------------------------------- the first sync's mode pin


def test_first_sync_refuses_a_mode_other_than_the_tested_family():
    export = _export()
    qwen3_5 = compat.LayoutEntry("qwen3", "X", "tests.X", "qwen3_5 direct", (), ())
    with pytest.raises(RuntimeError, match="tested layout's family is 'qwen3_5 direct'"):
        _installer(_installed(export, False), layout=qwen3_5).plan_once(export)
    inst = _installer(_installed(export, False))
    inst.plan_once(export)
    assert inst.mode == "vllm loaders"


# ---------------------------------------------------------------------------------------------------- the export pin


def test_export_pin_names_every_difference():
    full = _export()
    sig = ws.export_signature(full)
    assert sig["model.layers.0.self_attn.k_proj.weight"] == ((HKV, H), "torch.bfloat16")
    assert ws.export_pin_problem(None, sig, 0, 0) is None
    assert ws.export_pin_problem((0, sig), sig, 7, 0) is None
    drop = {k: v for k, v in full.items() if k != "model.layers.0.self_attn.k_proj.weight"}
    msg = ws.export_pin_problem((0, sig), ws.export_signature(drop), 7, 3)
    assert (
        "INTEGRITY FAILED rank 3 step 7: EXPORT CHANGED since the verified sync 0: 1 missing" in msg and "k_proj" in msg
    )
    extra = {**full, "model.layers.0.self_attn.q_proj.bias": torch.zeros(H, dtype=torch.bfloat16)}
    assert "1 extra ['model.layers.0.self_attn.q_proj.bias']" in ws.export_pin_problem(
        (0, sig), ws.export_signature(extra), 8, 0
    )
    recast = {**full, "model.norm.weight": full["model.norm.weight"].float()}
    assert "1 with another shape / dtype ['model.norm.weight']" in ws.export_pin_problem(
        (0, sig), ws.export_signature(recast), 9, 0
    )


# ---------------------------------------------------------------------------------------------------- the packed parity


@pytest.mark.parametrize("tied", [False, True])
def test_packed_parity_exact_and_coverage(tied):
    export = _export(tied=tied)
    inst = _installer(_installed(export, tied))
    par = inst._post_check_loaders(export)
    n = len(dict(inst._model().named_parameters()))
    assert par["post_mismatched"] == [] and par["unresolved"] == [] and par["post_checked"] == par["params"] == n
    assert par["tied_checked"] == (1 if tied else 0)
    cov = inst.coverage({}, export)
    assert (
        cov["checked"]
        and cov["ok"]
        and cov["written"] == n
        and cov["tied_copy"] == (["lm_head.weight"] if tied else [])
    )


def test_missing_fused_shard_is_unresolved():
    export = _export()
    model = _installed(export, False)
    drop = {k: v for k, v in export.items() if k != "model.layers.0.self_attn.k_proj.weight"}
    par = _installer(model)._post_check_loaders(drop)
    assert par["unresolved"] == ["model.layers.0.self_attn.qkv_proj.weight"]
    assert not _installer(model).coverage({}, drop)["ok"]


def test_all_shards_of_a_fused_parameter_missing_is_unresolved():
    export = _export()
    model = _installed(export, False)
    drop = {k: v for k, v in export.items() if not any(f".{p}_proj." in k for p in "qkv")}
    par = _installer(model)._post_check_loaders(drop)
    assert par["unresolved"] == ["model.layers.0.self_attn.qkv_proj.weight"]
    cov = _installer(model).coverage({}, drop)
    assert not cov["ok"] and cov["not_written"] == ["model.layers.0.self_attn.qkv_proj.weight"]


def test_corrupted_fused_parameter_is_detected():
    export = _export()
    model = _installed(export, False)
    with torch.no_grad():
        model.model.layers[0].self_attn.qkv_proj.weight.view(-1)[H * H + 1] += 1.0  # an element of the k rows
    assert _installer(model)._post_check_loaders(export)["post_mismatched"] == [
        "model.layers.0.self_attn.qkv_proj.weight"
    ]


def test_tpu_transposed_layout_compares_exact():
    """vllm-torchtpu keeps dense linear weights as (n_in, n_out) after vLLM's reload and marks the module"""
    export = _export()
    model = _installed(export, False)
    lay = model.model.layers[0]
    flipped = (lay.self_attn.qkv_proj, lay.self_attn.o_proj, lay.mlp.gate_up_proj, lay.mlp.down_proj)
    with torch.no_grad():
        for mod in flipped:
            mod.weight.data = mod.weight.data.t().contiguous()
            setattr(mod, ws.WEIGHT_FLIPPED_ATTR, True)
    par = _installer(model)._post_check_loaders(export)
    assert par["post_mismatched"] == [] and par["unresolved"] == [] and par["flipped"] == len(flipped)
    with torch.no_grad():
        lay.mlp.down_proj.weight.view(-1)[3] += 1.0
    assert _installer(model)._post_check_loaders(export)["post_mismatched"] == ["model.layers.0.mlp.down_proj.weight"]


def test_tied_head_must_equal_the_embedding():
    """An untied trainer's LM head into a tied engine: vLLM's loaders skip the alias, the engine would serve the
    embedding."""
    export = _export(tied=True)
    model = _installed(export, True)
    other = {**export, "lm_head.weight": (export["lm_head.weight"].float() + 0.5).to(torch.bfloat16)}
    par = _installer(model)._post_check_loaders(other)
    assert (
        len(par["post_mismatched"]) == 1
        and "lm_head.weight (tied to model.embed_tokens.weight" in par["post_mismatched"][0]
    )


def test_received_tensor_without_engine_parameter_is_refused():
    export = _export()
    extra = {**export, "model.layers.0.self_attn.q_proj.bias": torch.zeros(H, dtype=torch.bfloat16)}
    par = _installer(_installed(export, False))._post_check_loaders(extra)
    assert par["unresolved"] == ["received model.layers.0.self_attn.q_proj.bias: no engine parameter takes it"]


def test_untied_head_not_sent_is_unresolved():
    """vLLM's loaders never copy the embedding into an untied LM head (tpu-sync's direct install does)."""
    export = _export()
    no_head = {k: v for k, v in export.items() if k != "lm_head.weight"}
    par = _installer(_installed(export, False))._post_check_loaders(no_head)
    assert par["unresolved"] == ["lm_head.weight"]


# ---------------------------------------------------------------------------------------------------- coverage


ALL_PARAMS = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def _direct_result(written) -> dict:
    """A verified direct install's record as InProcInstaller.coverage reads it: the recorder's pairs."""
    rec = SimpleNamespace(pairs=[(n, None, None) for n in written])
    return {"_verify": {"direct": True, "recorder": rec, "pre": False, "unresolved": []}}


@pytest.mark.parametrize("tie, counted", [(True, True), (False, False), ("absent", False)])
def test_direct_coverage_counts_the_lm_head_copy_only_for_a_tied_config(tie, counted):
    """tpu-sync's direct install copies the embedding into a separate LM head when the trainer sent none: the right
    value for a tied config only. A config without the attribute reads untied."""
    cfg = SimpleNamespace(model_type="qwen3_5_text", **({} if tie == "absent" else {"tie_word_embeddings": tie}))
    inst = _installer(_DenseVllmModel(tied=False), layout=DIRECT, cfg=cfg)
    no_head = {k: v for k, v in _export().items() if k != "lm_head.weight"}
    cov = inst.coverage(_direct_result(ALL_PARAMS[:-1]), no_head)
    if counted:
        assert cov["ok"] and cov["tied_copy"] == ["lm_head.weight"] and cov["written"] == len(ALL_PARAMS)
    else:
        assert not cov["ok"] and cov["not_written"] == ["lm_head.weight"] and cov["tied_copy"] == []
    full = inst.coverage(_direct_result(ALL_PARAMS), _export())
    assert full["ok"] and full["tied_copy"] == [] and full["written"] == len(ALL_PARAMS)


def test_direct_coverage_without_recorded_writes_fails():
    inst = _installer(_DenseVllmModel(tied=False), layout=DIRECT)
    cov = inst.coverage({"_verify": {"direct": True, "recorder": None, "pre": False, "unresolved": []}}, _export())
    assert not cov["ok"] and cov["not_written_n"] == len(ALL_PARAMS)


@pytest.mark.parametrize("pairs", [None, []])
def test_verified_direct_install_with_no_recorded_write_fails_parity(pairs):
    """A direct install that did not go through the swapped ExactParityCheck would otherwise compare nothing."""
    inst = _installer(_DenseVllmModel(tied=False), layout=DIRECT)
    rec = None if pairs is None else ws._RecordingParity()
    ws._RecordingParity.current = None
    res = {"_verify": {"direct": True, "recorder": rec, "pre": False, "unresolved": []}}
    par = inst.finish_parity(res, _export())
    assert par["post_checked"] == 0 and par["unresolved"] == [
        "recorder not engaged: the direct install recorded no write"
    ]


def test_not_exported_parameters_are_excused_only_when_nothing_was_received():
    """The Qwen3.5 table row declares the MTP head as not exported by a text-only trainer: an MTP stack nothing was sent
    for is excused; one that received some of its shards is incomplete (never excused); one that received all of them
    contradicts the table (the entry is wrong for this trainer)."""
    base = _export()
    g = torch.Generator().manual_seed(9)
    mtp = {
        f"mtp.self_attn.{p}_proj.weight": torch.randn(*s, generator=g).to(torch.bfloat16)
        for p, s in (("q", (H, H)), ("k", (HKV, H)), ("v", (HKV, H)))
    }
    stack = "mtp.self_attn.qkv_proj.weight"
    m = _installed(base, False, mtp=True)
    par, cov = _installer(m, LOADERS_TOWER)._post_check_loaders(base), _installer(m, LOADERS_TOWER).coverage({}, base)
    assert par["unresolved"] == [] and cov["ok"] and cov["allowlisted_first"] == [stack]
    part = {**base, **{k: v for k, v in mtp.items() if ".k_proj" not in k}}
    m = _installed(part, False, mtp=True)
    par, cov = _installer(m, LOADERS_TOWER)._post_check_loaders(part), _installer(m, LOADERS_TOWER).coverage({}, part)
    assert par["unresolved"] == [stack] and not cov["ok"] and cov["not_written"] == [stack]
    whole = {**base, **mtp}
    m = _installed(whole, False, mtp=True)
    cov = _installer(m, LOADERS_TOWER).coverage({}, whole)
    assert (
        not cov["ok"]
        and cov["not_exported_received"]
        and all(n.startswith("mtp.") for n in cov["not_exported_received"])
    )


def test_not_exported_pattern_matching_a_text_parameter_fails():
    """A table entry whose not-exported pattern matches qkv_proj (which the trainer always sends) is wrong."""
    wrong = compat.LayoutEntry("qwen3", "X", "tests.X", "vllm loaders", (r"qkv_proj",), ())
    export = _export()
    cov = _installer(_installed(export, False), wrong).coverage({}, export)
    assert not cov["ok"] and "model.layers.0.self_attn.qkv_proj.weight" in cov["not_exported_received"]


def test_without_a_table_entry_nothing_is_excused():
    export = _export()
    model = _installed(export, False, mtp=True)
    inst = _installer(model, layout=None)
    assert inst._post_check_loaders(export)["unresolved"] == ["mtp.self_attn.qkv_proj.weight"]
    assert not inst.coverage({}, export)["ok"]


# ------------------------------------------------------------------------------------------------ update_weights itself


class _IdleEngine:
    """The vLLM engine surface update_weights reads between two rollouts (nothing in flight)."""

    model_config = SimpleNamespace()

    def has_unfinished_requests(self):
        return False

    def get_num_unfinished_requests(self):
        return 0


class _Recorder:
    """The surface of _RecordingParity that finish_parity / coverage read after a verified direct install."""

    def __init__(self, pairs):
        self.pairs, self.checked, self.mismatched = pairs, len(pairs), []

    def post_mismatches(self):
        return []


def _rollout(model, layout=LOADERS, cfg=None, mode=None):
    """A TPUInProcRollout without its build (the attributes update_weights reads): an idle engine and the real
    InProcInstaller (``mode`` preset = an already planned install mode)."""
    r = TPUInProcRollout.__new__(TPUInProcRollout)
    r.rank = 0
    r.engine = InProcAsyncEngine(_IdleEngine(), 0)
    r.installer = _installer(model, layout=layout, cfg=cfg)
    r.worker = r.installer.worker
    if mode is not None:
        r.installer.mode = mode
    r.server = SimpleNamespace(global_steps=None)
    r.global_steps, r.sync_log, r._pending_export_t0 = None, [], None
    r._syncs_started, r._last_wsum = 0, None
    r.integrity, r.last_step, r.rss_limit = "fatal", None, 0
    return r


FUSED = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
ROWS = {
    "q_proj": (0, H),
    "k_proj": (H, H + HKV),
    "v_proj": (H + HKV, H + 2 * HKV),
    "gate_proj": (0, IM),
    "up_proj": (IM, 2 * IM),
}


def _write(model: nn.Module, staging: dict) -> None:
    """What an install writes into the stand-in: each received shard into its rows of the fused parameter (a missing
    shard leaves its rows as they were, as vLLM's loaders do), an unfused tensor into its parameter."""
    lay = model.model.layers[0]
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, t in staging.items():
            proj = name.split(".")[-2]
            if proj in ROWS:
                fused = lay.self_attn.qkv_proj.weight if proj in FUSED["qkv_proj"] else lay.mlp.gate_up_proj.weight
                fused[ROWS[proj][0] : ROWS[proj][1]].copy_(t)
            elif name in params and params[name].shape == t.shape:
                params[name].copy_(t.to(params[name].dtype))


def _direct_targets(model: nn.Module, staging: dict) -> list:
    """The engine parameters a direct install writes from ``staging`` (a fused one only from all of its shards)."""
    out = []
    for n, _p in model.named_parameters():
        proj = n.split(".")[-2]
        if proj in FUSED:
            if all(n.replace(proj, s) in staging for s in FUSED[proj]):
                out.append(n)
        elif n in staging:
            out.append(n)
    return out


@pytest.fixture
def installs(monkeypatch):
    """InProcInstaller._install_once replaced by a stand-in of the install's writes: the loader path writes what vLLM's
    loaders would; the direct path what tpu-sync's direct install would (the LM head copied from the embedding when
    the trainer sent none), with the recorder pairs a verified direct install leaves. Every call recorded."""
    ctl = SimpleNamespace(calls=[], fail_direct=False)

    def _install_once(self, staging_hf, verify, pre):
        direct = self.mode != ws.LOADER_FAMILY
        if direct and ctl.fail_direct:
            raise RuntimeError("stand-in: the direct install raised")
        ctl.calls.append("direct" if direct else "loaders")
        m = self._model()
        _write(m, staging_hf)
        pairs = _direct_targets(m, staging_hf) if direct else None
        if (
            direct
            and "lm_head.weight" not in staging_hf
            and m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()
        ):
            with torch.no_grad():
                m.lm_head.weight.copy_(m.model.embed_tokens.weight)
        out = {"mode": self.mode, "tensors": len(staging_hf), "timing": {}}
        if verify:
            rec = _Recorder([(n, None, None) for n in pairs]) if direct else None
            out["_verify"] = {"direct": direct, "recorder": rec, "pre": pre, "unresolved": []}
        return out

    monkeypatch.setattr(ws.InProcInstaller, "_install_once", _install_once)
    monkeypatch.setattr(rollout_mod, "freeze_heap", lambda: ("test", 0.0))
    for k in ("VERL_TPU_INPROC_PARITY_EVERY", "VERL_TPU_INPROC_LAST_STEP", "VERL_TPU_INPROC_INTEGRITY"):
        monkeypatch.delenv(k, raising=False)
    return ctl


def _sync(r, export: dict, step: int) -> None:
    async def go():
        await r.update_weights(iter(list(export.items())), global_steps=step)

    asyncio.run(go())


def _snapshot(model: nn.Module) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _unchanged(model: nn.Module, snap: dict) -> bool:
    return all(torch.equal(p, snap[n]) for n, p in model.named_parameters())


def test_update_weights_refuses_a_changed_export_before_the_install(installs):
    """At unverified syncs an export without a fused shard, with an extra tensor or with another dtype is refused
    before the install writes anything; the full export installs again afterwards."""
    model = _installed(_export(0), False)
    r = _rollout(model)
    _sync(r, _export(1), 0)
    assert (
        r.sync_log[-1]["verified"] == "step 0" and r.sync_log[-1]["export_pin"] == 0 and installs.calls == ["loaders"]
    )
    k = "model.layers.0.self_attn.k_proj.weight"
    bad = [
        ("missing", {n: t for n, t in _export(2).items() if n != k}, f"1 missing ['{k}']"),
        (
            "extra",
            {**_export(2), "model.layers.0.self_attn.q_proj.bias": torch.zeros(H, dtype=torch.bfloat16)},
            "1 extra ['model.layers.0.self_attn.q_proj.bias']",
        ),
        (
            "recast",
            {**_export(2), "model.norm.weight": _export(2)["model.norm.weight"].float()},
            "1 with another shape / dtype ['model.norm.weight']",
        ),
    ]
    for step, (what, export, want) in zip((2, 3, 4), bad, strict=True):
        snap = _snapshot(model)
        with pytest.raises(RuntimeError, match="EXPORT CHANGED since the verified sync 0") as e:
            _sync(r, export, step)
        assert want in str(e.value), what
        assert installs.calls == ["loaders"] and _unchanged(model, snap), f"{what}: the install ran"
    _sync(r, _export(3), 6)
    assert r.sync_log[-1]["step"] == 6 and r.sync_log[-1]["verified"] is None and r.sync_log[-1]["export_pin"] == 0
    assert installs.calls == ["loaders", "loaders"]


def test_update_weights_pins_only_a_verified_sync_that_passed(installs):
    model = _installed(_export(0), False)
    r = _rollout(model)
    with pytest.raises(RuntimeError, match="parity FAILED at step 0"):
        _sync(r, {n: t for n, t in _export(1).items() if n != "model.layers.0.self_attn.k_proj.weight"}, 0)
    assert r._export_pin is None
    _sync(r, _export(2), 1)
    assert r.sync_log[-1]["verified"] == "step 1" and r.sync_log[-1]["export_pin"] == 1 and r._export_pin[0] == 1


def test_update_weights_loader_path_untied_head_not_sent(installs):
    model = _installed(_export(0), False)
    r = _rollout(model, cfg=SimpleNamespace(model_type="qwen3", tie_word_embeddings=False))
    with pytest.raises(RuntimeError, match="parity FAILED at step 0"):
        _sync(r, {n: t for n, t in _export(1).items() if n != "lm_head.weight"}, 0)
    assert r._export_pin is None


@pytest.mark.parametrize("tie", [False, True])
def test_update_weights_direct_path_untied_head_not_sent(installs, tie, caplog):
    """A fresh process's first sync without lm_head.weight on the direct path. Untied config: the coverage check fails,
    the parity line itself reads PARITY FAILED (not EXACT), nothing is pinned. Tied config: the LM head copied from the
    embedding is the right value; the sync passes."""
    model = _installed(_export(0), False)
    cfg = SimpleNamespace(model_type="qwen3_5_text", tie_word_embeddings=tie)
    r = _rollout(model, layout=DIRECT, cfg=cfg, mode="dense-test direct")
    no_head = {n: t for n, t in _export(1).items() if n != "lm_head.weight"}
    caplog.set_level(logging.INFO, logger="verl_hardware_plugin.accelerators.tpu.rollout.inproc")
    if tie:
        _sync(r, no_head, 0)
        assert r.sync_log[-1]["install_coverage"]["ok"] and r._export_pin[0] == 0
        assert any("PARITY EXACT" in m for m in caplog.messages)
    else:
        with pytest.raises(RuntimeError, match=r"coverage: 1 engine parameter\(s\) never written .*lm_head\.weight"):
            _sync(r, no_head, 0)
        assert r._export_pin is None
        parity = [m for m in caplog.messages if "parity step 0 rank 0" in m]
        assert (
            parity
            and "PARITY FAILED" in parity[0]
            and "PARITY EXACT" not in parity[0]
            and "lm_head.weight" in parity[0]
        )
    assert installs.calls == ["direct"]


def test_update_weights_install_mode_stays_the_tested_family_at_every_sync(installs, monkeypatch):
    """A direct install that raises at a later sync stops the run: there is no fallback to another install path."""
    model = _installed(_export(0), False)
    r = _rollout(
        model,
        layout=DIRECT,
        cfg=SimpleNamespace(model_type="qwen3", tie_word_embeddings=False),
        mode="dense-test direct",
    )
    _sync(r, _export(1), 0)
    assert r.sync_log[-1]["mode"] == "dense-test direct"
    installs.fail_direct = True
    with pytest.raises(RuntimeError, match="INTEGRITY FAILED rank 0: the dense-test direct install raised at step 2"):
        _sync(r, _export(2), 2)
    assert installs.calls == ["direct"]
    r.installer.mode = "vllm loaders"  # an installer that somehow switched modes
    installs.fail_direct = False
    with pytest.raises(RuntimeError, match="not the tested family 'dense-test direct'"):
        _sync(r, _export(3), 3)


def test_update_weights_verifies_every_sync_until_one_passes(installs):
    model = _installed(_export(0), False)
    r = _rollout(model)
    k = "model.layers.0.self_attn.k_proj.weight"
    with pytest.raises(RuntimeError, match="parity FAILED at step 0"):
        _sync(r, {n: t for n, t in _export(1).items() if n != k}, 0)
    with pytest.raises(RuntimeError, match="parity FAILED at step 2"):  # step 2: unverified by the cadence
        _sync(r, {n: t for n, t in _export(2).items() if n != k}, 2)
    _sync(r, _export(3), 3)
    assert r.sync_log[-1]["verified"] == "until a verified sync pins the export" and r.sync_log[-1]["export_pin"] == 3
    _sync(r, _export(4), 4)  # pinned: back to the cadence
    assert r.sync_log[-1]["verified"] is None and r.sync_log[-1]["export_pin"] == 3


def test_update_weights_refuses_requests_in_flight(installs):
    r = _rollout(_installed(_export(0), False))
    r.engine._collectors["x"] = object()
    with pytest.raises(RuntimeError, match="with requests in flight"):
        _sync(r, _export(1), 0)


WEAKENING = {
    "VERL_TPU_INPROC_COVERAGE": "report",
    "VERL_TPU_INPROC_COVERAGE_ALLOW": r"qkv_proj,.*,layers\.0\.self_attn\.qkv",
    "VERL_TPU_INPROC_LOADER_FALLBACK": "allow",
    "VERL_TPU_INPROC_REQUIRE_DIRECT": "0",
    "VERL_TPU_INPROC_LOADER_VERIFY": "names",
}


def test_no_environment_variable_weakens_the_install_checks(installs, monkeypatch):
    """Variables that once switched these checks off are not read: every refusal still fires with them set."""
    for k, v in WEAKENING.items():
        monkeypatch.setenv(k, v)
    k = "model.layers.0.self_attn.k_proj.weight"
    r = _rollout(_installed(_export(0), False))
    with pytest.raises(RuntimeError, match="parity FAILED at step 0"):  # a missing shard at a verified sync
        _sync(r, {n: t for n, t in _export(1).items() if n != k}, 0)
    _sync(r, _export(2), 1)
    with pytest.raises(RuntimeError, match="EXPORT CHANGED"):  # a missing shard at an unverified sync
        _sync(r, {n: t for n, t in _export(3).items() if n != k}, 2)
    d = _rollout(
        _installed(_export(0), False), layout=DIRECT, cfg=SimpleNamespace(model_type="x"), mode="dense-test direct"
    )
    with pytest.raises(RuntimeError, match="coverage: 1 engine parameter"):  # a whole tensor missing on the direct path
        _sync(d, {n: t for n, t in _export(1).items() if n != "model.norm.weight"}, 0)
    installs.fail_direct = True
    with pytest.raises(RuntimeError, match="install raised"):  # no loader fallback
        _sync(d, _export(2), 1)
    m = _installed(_export(0), False, mtp=True)  # the not-exported patterns come from the table, never from the env
    assert _installer(m, layout=LOADERS)._post_check_loaders(_export(0))["unresolved"] == [
        "mtp.self_attn.qkv_proj.weight"
    ]


def test_check_switches_names_unread_variables(monkeypatch):
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import common

    env = {"VERL_TPU_ROLLOUT_INPROC": "1", **WEAKENING, "VERL_TPU_INPROC_PARITY_EVERY": "5"}
    assert common.check_switches(env) == sorted(WEAKENING)


# ---------------------------------------------------------------------------------- staleness (skipped optimizer steps)


def test_staleness_unchanged_weights_fail_at_a_checksum_sync(installs):
    model = _installed(_export(0), False)
    r = _rollout(model)
    _sync(r, _export(1), 0)
    _sync(r, _export(2), 1)
    with pytest.raises(
        RuntimeError, match="INTEGRITY FAILED rank 0 step 5: the engine's weights are unchanged since sync 1"
    ):
        _sync(r, _export(2), 5)  # step 5 is verified: the checksum runs; the same export as step 1


def test_staleness_reported_only_in_report_mode(installs):
    model = _installed(_export(0), False)
    r = _rollout(model)
    r.integrity = "report"
    _sync(r, _export(1), 0)
    _sync(r, _export(2), 1)
    _sync(r, _export(2), 5)
    assert r.sync_log[-1]["stale"] is True


def test_staleness_not_judged_when_every_optimizer_step_was_skipped(installs, monkeypatch):
    """A non-finite grad norm skips the optimizer step: the trainer's weights, and so the served ones, stay the same.
    That is not staleness when the counter shows only skipped steps since the last checksum."""
    marks = iter([(1, 0, 0), (2, 0, 0), (2, 3, 0)])
    monkeypatch.setattr(ws.OPT_STEPS, "mark", lambda: next(marks))
    model = _installed(_export(0), False)
    r = _rollout(model)
    _sync(r, _export(1), 0)
    _sync(r, _export(2), 1)
    _sync(r, _export(2), 5)
    assert r.sync_log[-1]["stale"] is False and r.sync_log[-1]["unchanged_expected"] is True


def test_no_update_between():
    assert ws.no_update_between((3, 0, 0), (3, 2, 0))
    assert not ws.no_update_between((3, 0, 0), (4, 2, 0))  # a real step in between
    assert not ws.no_update_between((3, 0, 0), (3, 0, 0))  # no optimizer call at all: the plain rule
    assert not ws.no_update_between((3, 0, 0), (3, 1, 1))  # an unknown step
    assert not ws.no_update_between(None, (3, 1, 0))


def test_optimizer_steps_counter_classifies_returns():
    engine = SimpleNamespace(optimizer_step=lambda: next(rets))
    rets = iter([torch.tensor(1.5), float("inf"), None, float("nan"), 0.3])
    counter = ws.OptimizerSteps()
    counter.install(SimpleNamespace(actor=SimpleNamespace(engine=engine)))
    for _ in range(5):
        engine.optimizer_step()
    assert counter.mark() == (2, 2, 1)
    assert ws.OptimizerSteps().mark() is None


def test_stale_probe_verdicts():
    named = {"a": torch.arange(64, dtype=torch.float32), "b": torch.ones(8, dtype=torch.bfloat16)}
    probe = ws.StaleProbe(0)
    probe.dispatch(0, named, ["a", "b"], mark=(0, 0, 0))
    assert probe.resolve("test")["changed"] is None
    named["a"] = named["a"] + 1
    probe.dispatch(1, named, ["a", "b"], mark=(1, 0, 0))
    assert probe.resolve("test")["changed"] == 1
    probe.dispatch(2, named, ["a", "b"], mark=(1, 2, 0))  # unchanged, every step since skipped: expected
    rec = probe.resolve("test")
    assert rec["stale"] is False and rec["unchanged_expected"] is True
    probe.dispatch(3, named, ["a", "b"], mark=(2, 2, 0))  # unchanged after a real step: stale
    assert probe.resolve("test")["stale"] is True and probe.fatal_pending is not None
    with pytest.raises(RuntimeError, match="STALE WEIGHTS rank 0: sync step 3"):
        probe.check_fatal()


def test_stale_probe_is_reported_only_when_verl_barriers_after_the_sync():
    probe = ws.StaleProbe(0, mode="fatal", barrier_after=True)
    assert not probe.fatal and probe.demoted


def test_verify_reason(monkeypatch):
    monkeypatch.delenv("VERL_TPU_INPROC_PARITY_EVERY", raising=False)
    assert ws.verify_reason(0, False, None) == "step 0" and ws.verify_reason(1, False, None) == "step 1"
    assert ws.verify_reason(7, True, None) == "the first sync of this process"
    assert ws.verify_reason(10, False, None) == "every 5 steps" and ws.verify_reason(7, False, None) is None
    assert ws.verify_reason(7, False, 7) == "the run's last step"
    monkeypatch.setenv("VERL_TPU_INPROC_PARITY_EVERY", "0")
    assert ws.verify_reason(10, False, None) is None
    monkeypatch.setenv("VERL_TPU_INPROC_PARITY_EVERY", "x")
    with pytest.raises(RuntimeError, match="non-negative integer"):
        ws.verify_reason(10, False, None)
