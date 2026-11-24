# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Weight sync into the in-process engine, and the checks that make a silent divergence impossible.

The trainer's export (full tensors after the FSDP all-gathers, verl's ``naive`` checkpoint-engine path) is installed
in place through the tpu-sync worker extension's own install, with TP=1 and no transfer: the direct install of a
model family's layout, or vLLM's own loaders. Around it:

* the install mode must be the tested layout's family (:mod:`.compat`), at the first sync and at every later one;
  a direct install that raises stops the run (there is no fallback to another install path);
* the export (every name, shape and dtype) is pinned at the first verified sync that passed, and every later sync
  must send exactly it, checked before the install (:func:`export_pin_problem`);
* verified syncs (steps 0 and 1, the first sync of a process, every ``VERL_TPU_INPROC_PARITY_EVERY`` steps, the run's
  last step, and every sync until one has passed) compare every installed parameter bit for bit with what the
  trainer sent (the direct install's recorded writes; on the loader path every engine parameter rebuilt from the
  received tensors), and check that every engine parameter was written (:meth:`InProcInstaller.coverage`);
* every sync dispatches an exact on-device staleness probe, read back at the trainer's next entry
  (:class:`StaleProbe`), and the strided host checksum runs every few syncs.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections.abc import Mapping
from typing import Any, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import DEVICE_THREADS, engine_mode_ctx
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    INTEGRITY_TAG,
    log,
    matches_any,
    parity_every,
    warn,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compat import LayoutEntry

FULL_PARITY_STEPS = (0, 1)  # exact parity + the full sha256 at these global steps
CHECKSUM_MAX_BYTES = 4 * 2**20  # strided sample per tensor
CHECKSUM_EVERY = 5  # the strided sha256 at every 5th sync besides the verified ones (every sync while the probe is off)
WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"  # vllm-torchtpu's marker of a dense weight stored (n_in, n_out)
LOADER_FAMILY = "vllm loaders"


class _NoopSynchronizer:
    """tpu-sync's weight synchronizer slot of the install shim: in process the tensors are already on this device."""

    def h2d(self) -> None:
        return None


class _RecordingParity:
    """``ExactParityCheck``'s surface, swapped into the tpu-sync extension module for one install call.

    Records every (vLLM key, target, received-in-target-layout) the install writes through ``copy_into`` (and the
    regions a layout writes itself). With ``pre`` (global step 0, when the engine still holds the checkpoint vLLM
    loaded) it also compares before the overwrite; the post-install comparison (:meth:`post_mismatches`) runs at
    every verified step.
    """

    current: Optional[_RecordingParity] = None
    pre = False

    def __init__(self) -> None:
        self.pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self.checked = 0
        self.mismatched: list[str] = []
        _RecordingParity.current = self

    @torch.no_grad()
    def check(self, name: str, target: torch.Tensor, received: torch.Tensor) -> None:
        self.pairs.append((name, target, received))
        if not _RecordingParity.pre:
            return
        if tuple(target.shape) != tuple(received.shape):
            self.mismatched.append(f"{name}: shape target={tuple(target.shape)} received={tuple(received.shape)}")
            return
        self.checked += 1
        if not torch.equal(target, received.to(target.dtype)):
            diff = (target.float() - received.float()).abs().max().item()
            self.mismatched.append(f"{name}: max|diff|={diff:.4g}")

    def result(self, rank: int, unresolved: list[str]) -> dict:
        return {
            "rank": rank,
            "checked": self.checked,
            "num_mismatched": len(self.mismatched),
            "mismatched": self.mismatched[:60],
            "unresolved": list(unresolved),
        }

    @torch.no_grad()
    def post_mismatches(self) -> list[str]:
        bad = []
        for name, target, received in self.pairs:
            if tuple(target.shape) != tuple(received.shape) or not torch.equal(target, received.to(target.dtype)):
                bad.append(name)
        return bad


class _InstallShim:
    """The ``self`` of the extension's ``install_tpu_sync_weights`` for an in-process install (TP=1): the staging
    dict is the trainer's export and the synchronizer has nothing to transfer."""

    def __init__(self, worker: Any, staging: dict, layout: Any, plan: Any, use_loaders: bool, hf_text_config: Any):
        self.worker = worker
        self.rank = 0
        self._tpu_sync_ws = _NoopSynchronizer()
        self._tpu_sync_staging = staging
        self._tpu_sync_staging_sharded = dict.fromkeys(staging, False)
        self._tpu_sync_layout = layout
        self._tpu_sync_plan = plan
        self._tpu_sync_use_vllm_loaders = use_loaders
        self._tpu_sync_hf_text_config = hf_text_config
        # no vLLM parameter is a tpu-sync receive buffer here (receive_into_vllm): the install reads the staging dict
        self._tpu_sync_bound_names: set[str] = set()

    def _get_vllm_model(self) -> Any:
        return self.worker.model_runner.model

    def _install_with_vllm_loaders(self) -> None:
        from verl_hardware_plugin.accelerators.tpu.rollout import tpu_sync_worker_extension as ext

        return ext.vLLMTPUSyncWorkerExtension._install_with_vllm_loaders(self)


class InProcInstaller:
    """In-place install of the trainer's export into the in-process engine (tpu-sync's install, TP=1)."""

    def __init__(self, worker: Any, hf_text_config: Any, rank: int, layout: Optional[LayoutEntry] = None):
        self.worker = worker
        self.hf_text_config = hf_text_config
        self.rank = rank
        self.layout_entry = layout  # the tested layout: the family the first sync must plan, the not-exported patterns
        self.mode: Optional[str] = None
        self.layout: Any = None
        self.plan: Any = None
        self.reasons: list[str] = []
        self.qwen35 = getattr(hf_text_config, "model_type", None) in ("qwen3_5_text", "qwen3_5_moe_text")

    @property
    def family(self) -> Optional[str]:
        return self.layout_entry.family if self.layout_entry is not None else None

    @property
    def not_exported(self) -> tuple[str, ...]:
        return self.layout_entry.not_exported if self.layout_entry is not None else ()

    def _model(self):
        return self.worker.model_runner.model

    def _wire(self, staging_hf: dict) -> dict:
        if not self.qwen35:
            return staging_hf
        from verl_hardware_plugin.accelerators.tpu.rollout.qwen3_5_layout import make_tpu_sync_local_layout

        return make_tpu_sync_local_layout(self.hf_text_config, expert_parallel=False)(dict(staging_hf))

    def plan_once(self, staging_hf: dict) -> None:
        """The first sync plans the install mode once; it must be the tested layout's family."""
        if self.mode is not None:
            return
        from verl_hardware_plugin.accelerators.tpu.rollout import direct_install

        wire = self._wire(staging_hf)
        shapes = {k: list(v.shape) for k, v in wire.items()}
        sel = direct_install.select_layout(self._model(), shapes, self.hf_text_config, 1, 0)
        self.reasons = sel.reasons
        mode = f"{sel.layout.name} direct" if sel.layout is not None else LOADER_FAMILY
        if self.family is not None and mode != self.family:
            raise RuntimeError(
                f"{INTEGRITY_TAG} rank {self.rank}: the first sync planned install mode {mode!r}, but the tested "
                "layout's "
                f"family is {self.family!r}; direct layouts refused: {sel.reasons[:3]} (an install path without its "
                "own "
                "test is refused)"
            )
        if sel.layout is not None:
            self.layout, self.plan = sel.layout, sel.plan
        self.mode = mode

    @torch.no_grad()
    def install(self, staging_hf: dict, verify: bool, pre: bool, step: Optional[int] = None) -> dict:
        """The direct install of the planned layout, or vLLM's loaders; a direct install that raises stops the run
        (a silent switch of the install path mid-run is never taken). The verified comparison runs later, in
        :meth:`finish_parity`, outside the timed install."""
        self.plan_once(staging_hf)
        if self.mode == LOADER_FAMILY:
            return self._install_once(staging_hf, verify, pre)
        try:
            return self._install_once(staging_hf, verify, pre)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"{INTEGRITY_TAG} rank {self.rank}: the {self.mode} install raised at step {step}: "
                f"{type(e).__name__}: {e}"[:600]
            ) from e

    def _install_once(self, staging_hf: dict, verify: bool, pre: bool) -> dict:
        from verl_hardware_plugin.accelerators.tpu.rollout import tpu_sync_worker_extension as ext

        direct = self.mode != LOADER_FAMILY
        staging = self._wire(staging_hf) if direct else dict(staging_hf)
        shim = _InstallShim(self.worker, staging, self.layout, self.plan, not direct, self.hf_text_config)
        orig = ext.ExactParityCheck
        rec: Optional[_RecordingParity] = None
        try:
            if verify and direct:
                ext.ExactParityCheck = _RecordingParity  # type: ignore[misc,assignment]
                _RecordingParity.pre = pre
            result = ext.vLLMTPUSyncWorkerExtension.install_tpu_sync_weights(shim, exact_parity=verify and direct)
            rec = _RecordingParity.current if (verify and direct) else None
        finally:
            ext.ExactParityCheck = orig  # type: ignore[misc]
            _RecordingParity.pre = False
            _RecordingParity.current = None
        out = {
            "mode": self.mode,
            "tensors": len(staging),
            "timing": {k: v for k, v in (result or {}).items() if k != "exact_parity"},
        }
        if verify:
            out["_verify"] = {
                "direct": direct,
                "recorder": rec,
                "pre": pre,
                "unresolved": (result or {}).get("exact_parity", {}).get("unresolved", []),
            }
        return out

    @torch.no_grad()
    def finish_parity(self, res: dict, staging_hf: dict) -> Optional[dict]:
        """The verified install's comparison: the direct install's every recorded (target, received) pair after the
        copy, or the loader path's rebuilt parameters. A verified direct install whose recorder recorded nothing did
        not go through the swapped ``ExactParityCheck``: it fails (it would otherwise compare nothing)."""
        v = res.get("_verify")
        if v is None:
            return None
        if not v.get("direct"):
            res["parity"] = self._post_check_loaders(staging_hf)
            return res["parity"]
        rec = v.get("recorder")
        if rec is None or not rec.pairs:
            res["parity"] = {
                "pre_checked": None,
                "pre_mismatched": None,
                "post_checked": 0,
                "post_mismatched": [],
                "unresolved": ["recorder not engaged: the direct install recorded no write"],
            }
            return res["parity"]
        res["parity"] = {
            "pre_checked": rec.checked if v.get("pre") else None,
            "pre_mismatched": rec.mismatched if v.get("pre") else None,
            "post_checked": len(rec.pairs),
            "post_mismatched": rec.post_mismatches(),
            "unresolved": v.get("unresolved") or [],
        }
        return res["parity"]

    @torch.no_grad()
    def loader_expected(self, staging_hf: dict) -> dict:
        """The value every parameter of the vLLM model must hold after a vLLM-loader install of ``staging_hf`` (TP=1),
        rebuilt independently of the loaders.

        Each received tensor is mapped with the model's own ``hf_to_vllm_mapper`` stacked-parameter table
        ('.q_proj' -> ('.qkv_proj', 'q'), '.gate_proj' -> ('.gate_up_proj', 0), ...; a name it drops is skipped), else
        with its ``packed_modules_mapping``; a stacked parameter is the concatenation of all its shards along dim 0 in
        the table's order; an unstacked name maps 1:1. A received tensor whose engine name is a tied alias (one
        parameter under two names: vLLM's loaders skip the alias) is held to the canonical's received value; a
        received tensor that maps to no engine parameter is "unmapped". An engine parameter without a complete rebuild
        is unresolved: "incomplete" when it received some of its tensors (never excused), "unreceived" when none.

        Returns {"expected": {engine parameter: [received tensors to concatenate]}, "unresolved", "incomplete",
        "unmapped", "tied": [(alias, canonical, tensor)], "received": engine parameters some received tensor maps to,
        "via"}; the comparison builds each concatenation one parameter at a time.
        """
        model = self._model()
        alias_of: dict = {}
        first: dict = {}
        for n, p in model.named_parameters(remove_duplicate=False):
            if id(p) in first:
                alias_of[n] = first[id(p)]
            else:
                first[id(p)] = n
        params = dict(model.named_parameters())
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        stacked = dict(getattr(mapper, "orig_to_new_stacked", None) or {})
        packed = dict(getattr(model, "packed_modules_mapping", None) or {})
        order: dict = {}
        for i, (_suffix, (_new, sid)) in enumerate(stacked.items()):
            if not isinstance(sid, list):
                order.setdefault(sid, i)
        groups: dict = {}  # engine parameter -> {shard id (None = unstacked): received tensor}
        unmapped: list = []
        tied_in: list = []
        for name, t in staging_hf.items():
            if mapper is not None and hasattr(mapper, "_map_name_with_shard"):
                mapped = mapper._map_name_with_shard(name)
                if mapped is None:
                    continue
            else:
                mapped = (name, None)
                comps = name.split(".")
                for fused, parts in packed.items():
                    hits = [x for x in parts if x in comps]
                    if hits:
                        i = comps.index(hits[0])
                        mapped = (".".join(comps[:i] + [fused] + comps[i + 1 :]), parts.index(hits[0]))
                        break
            pname, sid = mapped
            if pname in alias_of and sid is None:
                tied_in.append((pname, alias_of[pname], t))
                continue
            if pname not in params:
                unmapped.append(name)
                continue
            groups.setdefault(pname, {})["__list__" if isinstance(sid, list) else sid] = t
        expected: dict = {}
        unresolved: list = []
        for pname in params:
            g = groups.get(pname)
            if g is None:
                unresolved.append(pname)
                continue
            if set(g) == {None}:
                expected[pname] = [g[None]]
                continue
            want = next((len(packed[c]) for c in pname.split(".") if c in packed), None)
            sids = [s for s in g if s is not None]
            if "__list__" in g or None in g or want is None or len(sids) != want:
                unresolved.append(pname)
                continue
            sids.sort(key=lambda s: order[s] if s in order else (s if isinstance(s, int) else 10**6))
            expected[pname] = [g[s] for s in sids]
        via = (
            "loaders, every engine parameter rebuilt from the received tensors ("
            + ("vLLM's stacked-shard table" if stacked else "packed_modules_mapping")
            + ", TP=1)"
        )
        tied = [(a, c, t) for a, c, t in tied_in if c in expected]
        incomplete = [n for n in unresolved if n in groups]
        received = sorted(set(groups) | {a for a, _c, _t in tied_in})
        return {
            "expected": expected,
            "unresolved": unresolved,
            "incomplete": incomplete,
            "unmapped": unmapped,
            "tied": tied,
            "received": received,
            "via": via,
        }

    @torch.no_grad()
    def _post_check_loaders(self, staging_hf: dict) -> dict:
        """The loader-path parity: every engine parameter compared with its rebuilt value (:meth:`loader_expected`),
        in the engine's own layout (a 2-D weight whose module carries vllm-torchtpu's ``_tpu_weight_flipped`` is held
        transposed, as the TPU linear method keeps it after vLLM's reload). A tied alias must have received the
        canonical's value; a received tensor no engine parameter takes is unresolved; an incomplete parameter is
        unresolved; an unreceived one only unless the tested layout declares it not exported."""
        model = self._model()
        le = self.loader_expected(staging_hf)
        exp = le["expected"]
        params = dict(model.named_parameters())
        modules = dict(model.named_modules())
        bad, flipped = [], 0
        for n, p in params.items():
            if n not in exp:
                continue
            e = exp[n][0] if len(exp[n]) == 1 else torch.cat(exp[n], dim=0)
            owner = modules.get(n.rsplit(".", 1)[0]) if "." in n else model
            if e.ndim == 2 and bool(getattr(owner, WEIGHT_FLIPPED_ATTR, False)):
                e = e.t()
                flipped += 1
            if tuple(p.shape) != tuple(e.shape) or not torch.equal(p, e.to(p.dtype)):
                bad.append(n)
            del e
        for alias, canon, t in le["tied"]:
            c = exp[canon][0]
            if tuple(t.shape) != tuple(c.shape) or not torch.equal(t, c.to(t.dtype)):
                bad.append(
                    f"{alias} (tied to {canon} in the engine): the trainer sent a different value, which the loaders "
                    "drop"
                )
        unres = [n for n in le["unresolved"] if n in le["incomplete"] or not matches_any(self.not_exported, n)]
        unres += [f"received {n}: no engine parameter takes it" for n in le["unmapped"]]
        return {
            "pre_checked": None,
            "pre_mismatched": None,
            "post_checked": len(exp),
            "post_mismatched": bad,
            "unresolved": unres,
            "params": len(params),
            "via": le["via"],
            "tied_checked": len(le["tied"]),
            "flipped": flipped,
        }

    def coverage(self, res: dict, staging_hf: dict) -> dict:
        """Did this install write every parameter of the engine model?

        The direct install: the parameters its recorded writes cover, plus the LM head tpu-sync copies from the
        embedding when the trainer sent none, counted only for a tied config (``tie_word_embeddings``; absent reads
        untied): for an untied model the trainer's own LM head is missing from the export. The loader path: the
        parameters a complete rebuild of the received tensors covers. A parameter never written keeps the
        checkpoint's value for the whole run while the trainer moves on. The only exemption is the tested layout's
        not-exported patterns, for an unreceived parameter; and those patterns must match nothing that was received
        (else the table entry is wrong for this trainer: ``not_exported_received``)."""
        model = self._model()
        try:
            names = [n for n, _ in model.named_parameters()]
        except Exception as e:  # noqa: BLE001
            return {"checked": False, "ok": False, "why": f"named_parameters: {type(e).__name__}: {e}"[:160]}
        v = res.get("_verify") or {}
        rec = v.get("recorder")
        incomplete: set = set()
        tie = bool(getattr(self.hf_text_config, "tie_word_embeddings", False))
        received = set(staging_hf)
        if v.get("direct"):
            written = {n for n, _t, _r in (rec.pairs if rec is not None else ())}
            via, tied = "direct install pairs", []
            sent_head = any(k == "lm_head.weight" or k.endswith(".lm_head.weight") for k in staging_hf)
            emb = [n for n in names if n.endswith("embed_tokens.weight") and n in written]
            if not sent_head and emb and tie:
                tied = [n for n in names if n.endswith("lm_head.weight") and n not in written]
                written |= set(tied)
            received |= written
        else:
            le = self.loader_expected(staging_hf)
            written, via, tied = set(le["expected"]), le["via"], [a for a, _c, _t in le["tied"]]
            incomplete = set(le["incomplete"])
            received |= set(le["received"])
        missing = [n for n in names if n not in written]
        allowed = [n for n in missing if n not in incomplete and matches_any(self.not_exported, n)]
        bad = [n for n in missing if n not in allowed]
        conflict = sorted(n for n in received if matches_any(self.not_exported, n))
        return {
            "checked": True,
            "via": via,
            "params": len(names),
            "written": len(names) - len(missing),
            "tied_copy": tied[:4],
            "allowlisted": len(allowed),
            "allowlisted_first": allowed[:6],
            "not_written": bad[:40],
            "not_written_n": len(bad),
            "not_exported_received": conflict[:8],
            "ok": not bad and not conflict,
            "tie_word_embeddings": tie,
        }


def export_signature(staging: dict) -> dict:
    """{name: (shape, dtype)} of one sync's export (names and metadata only: microseconds per sync)."""
    return {n: (tuple(t.shape), str(t.dtype)) for n, t in staging.items()}


def export_pin_problem(pin: Optional[tuple], sig: dict, step: int, rank: int) -> Optional[str]:
    """None when nothing is pinned yet or ``sig`` equals the pinned (step, signature); else the line naming the
    missing / extra / reshaped tensors (the caller refuses the install)."""
    if pin is None or sig == pin[1]:
        return None
    pin_step, pin_sig = pin
    missing = sorted(set(pin_sig) - set(sig))
    extra = sorted(set(sig) - set(pin_sig))
    changed = sorted(n for n in set(sig) & set(pin_sig) if sig[n] != pin_sig[n])
    return (
        f"{INTEGRITY_TAG} rank {rank} step {step}: EXPORT CHANGED since the verified sync {pin_step}: "
        f"{len(missing)} missing {missing[:4]}, {len(extra)} extra {extra[:4]}, {len(changed)} with another shape / "
        f"dtype {changed[:4]}; refusing to install (between verified syncs a missing shard or tensor would install "
        "silently)"
    )


def checksum_names(state_keys: list[str]) -> list[str]:
    """The fixed tensor subset of the per-sync checksum: embedding, LM head, layer 0's projections, the last MLP."""
    keys = sorted(state_keys)
    pick: list[str] = []

    def add(pred):
        for k in keys:
            if pred(k) and k not in pick:
                pick.append(k)

    add(lambda k: k.endswith("embed_tokens.weight"))
    add(lambda k: k.endswith("lm_head.weight"))
    add(
        lambda k: ".layers.0." in k
        and k.endswith(".weight")
        and any(s in k for s in ("in_proj", "qkv", "q_proj", "out_proj", "o_proj"))
    )
    layers = sorted(
        {
            int(k.split(".layers.")[1].split(".")[0])
            for k in keys
            if ".layers." in k and k.split(".layers.")[1].split(".")[0].isdigit()
        }
    )
    if layers:
        last = f".layers.{layers[-1]}."
        add(lambda k: last in k and ".mlp." in k and k.endswith(".weight") and ("down_proj" in k or "gate_up" in k))
    return pick


@torch.no_grad()
def tensor_checksums(named: dict, names: list[str], full: bool) -> tuple[str, int]:
    """sha256 over the named tensors' bytes: full tensors, or a strided sample of <= CHECKSUM_MAX_BYTES each."""
    h = hashlib.sha256()
    nbytes = 0
    for name in names:
        t = named[name]
        t = t.to_local() if hasattr(t, "to_local") else t
        flat = t.detach().reshape(-1)
        if not full:
            stride = max(1, math.ceil(flat.numel() * flat.element_size() / CHECKSUM_MAX_BYTES))
            flat = flat[::stride]
        host = flat.contiguous().to("cpu")
        if host.dtype == torch.bfloat16:
            host = host.view(torch.int16)
        b = host.numpy().tobytes()
        nbytes += len(b)
        h.update(name.encode())
        h.update(b)
    return h.hexdigest(), nbytes


def verify_reason(
    step: int, first_sync: bool, last: Optional[int], environ: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Why the sync at ``step`` runs the post-install parity and the coverage check, or None: steps 0 / 1, the first
    sync of this process (a resumed run), every ``VERL_TPU_INPROC_PARITY_EVERY`` steps (0 = never) and the run's last
    step (``last``)."""
    if step in FULL_PARITY_STEPS:
        return f"step {step}"
    if first_sync:
        return "the first sync of this process"
    every = parity_every(environ)
    if every > 0 and step > 0 and step % every == 0:
        return f"every {every} steps"
    if last is not None and step == last:
        return "the run's last step"
    return None


def _dispatch_nowait(t: torch.Tensor) -> None:
    """Materialize and dispatch a deferred device result without waiting for it (nothing on CPU)."""
    if t.device.type == "cpu":
        return
    try:
        from verl_hardware_plugin.accelerators.tpu.device_api import tpu_synchronize

        tpu_synchronize(t, wait=False)
    except Exception:  # noqa: BLE001 - left deferred: it is read back before the next install either way
        pass


@torch.no_grad()
def stale_reduction(named: dict, names: list) -> torch.Tensor:
    """One int32 per checksum tensor, on the tensors' device: the same strided sample as the host checksum, its bit
    patterns (16-bit floats viewed as int16, 32-bit as int32) widened to int32, times the odd position weights 2i+1,
    summed mod 2^32. Exact (wrapping integer arithmetic on bit patterns): any single changed element changes its sum,
    and so does a permutation inside the sample (up to 2^-32 collisions); a float sum would not see a bf16 weight that
    moved by one ulp among millions. No host transfer: the caller reads it back later."""
    vals = []
    for name in names:
        t = named[name]
        t = t.to_local() if hasattr(t, "to_local") else t
        flat = t.detach().reshape(-1)
        stride = max(1, math.ceil(flat.numel() * flat.element_size() / CHECKSUM_MAX_BYTES))
        flat = flat[::stride].contiguous()
        if flat.element_size() == 2:
            bits = flat.view(torch.int16).to(torch.int32)
        elif flat.element_size() == 4:
            bits = flat.view(torch.int32)
        else:
            bits = flat.to(torch.float32).view(torch.int32)
        w = torch.arange(bits.numel(), device=bits.device, dtype=torch.int32) * 2 + 1
        vals.append((bits * w).sum(dtype=torch.int32))
    return torch.stack(vals)


class OptimizerSteps:
    """The trainer engine's optimizer steps in this process, counted at their source: an instance-level wrap of
    ``worker.actor.engine.optimizer_step``. verl's engines call it once per mini-batch and the TPU engine skips the
    step when the global grad norm is not finite, returning that norm: a finite return is a real step, a non-finite
    one a skipped step, anything else unknown. A skipped step leaves the trainer's weights, and so the served ones,
    unchanged: that is not staleness. Fail-safe: no wrap, or an unknown return, keeps the plain staleness rule."""

    def __init__(self) -> None:
        self.real = 0
        self.skipped = 0
        self.unknown = 0
        self.installed: Optional[bool] = None

    def install(self, worker: Any) -> None:
        """Idempotent; never raises (a failure leaves the counter off)."""
        if self.installed is not None:
            return
        try:
            eng: Any = getattr(getattr(worker, "actor", None), "engine", None)
            fn = getattr(eng, "optimizer_step", None)
            if fn is None or not callable(fn):
                self.installed = False
                return
            if getattr(fn, "_inproc_counted", False):
                self.installed = True
                return
            counter = self

            def optimizer_step(*a, **k):
                ret = fn(*a, **k)
                counter.record(ret)
                return ret

            optimizer_step._inproc_counted = True  # type: ignore[attr-defined]
            eng.optimizer_step = optimizer_step  # an instance attribute: train_batch's self.optimizer_step() finds it
            self.installed = True
        except Exception:  # noqa: BLE001
            self.installed = False

    def record(self, ret: Any) -> None:
        try:
            v = float(ret.item() if hasattr(ret, "item") else ret)
        except Exception:  # noqa: BLE001 - a non-numeric return: unknown (judged as a real step)
            self.unknown += 1
            return
        if math.isfinite(v):
            self.real += 1
        else:
            self.skipped += 1

    def mark(self) -> Optional[tuple]:
        return (self.real, self.skipped, self.unknown) if self.installed else None


OPT_STEPS = OptimizerSteps()


def no_update_between(prev_mark: Optional[tuple], mark: Optional[tuple]) -> bool:
    """True when the trainer provably took no optimizer step between two syncs: both marks known, no real and no
    unknown step in between, and at least one skipped (an interval with no optimizer call at all keeps the plain
    staleness rule)."""
    return (
        prev_mark is not None
        and mark is not None
        and mark[0] == prev_mark[0]
        and mark[2] == prev_mark[2]
        and mark[1] > prev_mark[1]
    )


class StaleProbe:
    """The per-sync staleness check without a per-sync host checksum.

    Every sync calls :meth:`dispatch` right after the install: :func:`stale_reduction` on the device (no host wait).
    :meth:`resolve` reads it back later, off the generation's critical path: at the trainer's next entry
    (compute_log_prob / update_actor) or, failing that, at the next sync before its install. Each resolved probe is
    compared with the previous one: none of the sampled tensors changed although the trainer moved to a later step,
    with a real optimizer step in between -> ``STALE WEIGHTS``, logged at once and, in ``fatal`` mode, raised at the
    next sync right after its export (:meth:`check_fatal`): the trainer entries are collective, so the raise waits until
    every rank is past the step's collectives. The step guard refuses update_actor earlier on a stale verdict.
    ``barrier_after`` (verl's update_weights then barriers after this rollout's): a raise would strand ranks, so the
    verdict is only logged. The device lowering is cross-checked against a host reduction of the same sample at the
    full-sha syncs; a difference switches the probe off for the run (the strided sha256 then runs at every sync).
    """

    def __init__(self, rank: int, mode: str = "fatal", barrier_after: bool = False):
        self.fatal = mode == "fatal" and not barrier_after
        self.demoted = (
            "free_cache_engine: verl's update_weights barriers after the rollout's"
            if (mode == "fatal" and barrier_after)
            else None
        )
        self.rank = rank
        self.pending: Optional[tuple] = (
            None  # (step, device int32 vector, names, dispatch seconds, optimizer-step mark)
        )
        self.last: Optional[tuple] = None  # (step, names, [ints], optimizer-step mark) of the last resolved probe
        self.unavailable: Optional[str] = None
        self.fatal_pending: Optional[str] = None
        self.host_check: dict = {}
        self.cross_checks: list = []

    def on(self) -> bool:
        return self.unavailable is None

    def _off(self, what: str, e: BaseException) -> None:
        self.unavailable = f"{what}: {type(e).__name__}: {e}"[:240]
        self.pending = None
        warn(
            f"stale probe rank {self.rank}: OFF for this run ({self.unavailable}); the strided sha256 runs at every "
            "sync"
        )

    def dispatch(self, step: int, named: dict, names: list, mark: Optional[tuple] = None) -> Optional[float]:
        """After the install of sync ``step`` (inside the caller's engine scope): the reduction dispatched, not waited
        for. Seconds, or None when the probe is off (the caller then runs the strided sha256)."""
        if not self.on():
            return None
        if self.pending is not None:  # not read back since the last sync: in order, first
            self.resolve("sync")
            if not self.on():
                return None
        t0 = time.perf_counter()
        try:
            vec = stale_reduction(named, names)
            _dispatch_nowait(vec)
        except Exception as e:  # noqa: BLE001
            self._off("dispatch", e)
            return None
        dt = time.perf_counter() - t0
        self.pending = (int(step), vec, tuple(names), dt, mark)
        return dt

    def host_reference(self, step: int, named: dict, names: list) -> None:
        """At a full-sha sync, right after :meth:`dispatch`: the same reduction on a host copy of the same strided
        sample, kept for :meth:`resolve`'s comparison. Never raises."""
        if not self.on() or self.pending is None or self.pending[0] != int(step):
            return
        try:
            host = {}
            for name in names:
                t = named[name]
                t = t.to_local() if hasattr(t, "to_local") else t
                flat = t.detach().reshape(-1)
                stride = max(1, math.ceil(flat.numel() * flat.element_size() / CHECKSUM_MAX_BYTES))
                host[name] = flat[::stride].contiguous().to("cpu")
            self.host_check[int(step)] = [int(x) for x in stale_reduction(host, list(names)).tolist()]
        except Exception as e:  # noqa: BLE001
            self.host_check[int(step)] = f"host failed: {type(e).__name__}: {e}"[:160]

    def resolve(self, where: str) -> Optional[dict]:
        """Read the pending probe back and judge it against the previous one (never raises: a stale verdict is kept
        for :meth:`check_fatal`)."""
        p, self.pending = self.pending, None
        if p is None or not self.on():
            return None
        step, vec, names, disp, mark = p
        t0 = time.perf_counter()
        try:
            with engine_mode_ctx("stale"):
                DEVICE_THREADS.record("stale")
                vals = [int(x) for x in vec.to("cpu").tolist()]
        except Exception as e:  # noqa: BLE001
            self._off(f"read back at {where}", e)
            return None
        dt = time.perf_counter() - t0
        ref = self.host_check.pop(step, None)
        if ref is not None:
            if isinstance(ref, str):
                self.cross_checks.append((step, ref))
            elif ref != vals:
                self.cross_checks.append((step, "MISMATCH"))
                self._off(
                    f"cross-check at sync {step}",
                    RuntimeError(
                        f"the device reduction {vals[:3]}... != the host reduction {ref[:3]}... of the same sample"
                    ),
                )
                return None
            else:
                self.cross_checks.append((step, "equal"))
        digest = hashlib.sha256(struct.pack(f"<{len(vals)}q", *vals)).hexdigest()[:16]
        prev = self.last
        unchanged = (
            [n for n, a, b in zip(names, vals, prev[2], strict=False) if a == b]
            if prev is not None and prev[1] == names
            else None
        )
        all_unchanged = bool(
            prev is not None and unchanged is not None and len(unchanged) == len(names) and step > prev[0]
        )
        expected = all_unchanged and no_update_between(prev[3] if prev is not None else None, mark)
        stale = all_unchanged and not expected
        rec = {
            "step": step,
            "digest": digest,
            "tensors": len(names),
            "prev_step": prev[0] if prev is not None else None,
            "changed": (len(names) - len(unchanged)) if unchanged is not None else None,
            "stale": stale,
            "unchanged_expected": expected,
            "read_at": where,
        }
        self.last = (step, names, vals, mark)
        log(
            f"stale probe step {step} rank {self.rank}: digest {digest} over {len(names)} tensors; "
            + (f"host cross-check {self.cross_checks[-1][1]}; " if ref is not None else "")
            + (
                f"changed {rec['changed']}/{len(names)} vs step {rec['prev_step']}"
                if unchanged is not None
                else "first probe"
            )
            + (f"; UNCHANGED {unchanged}" if unchanged else "")
            + (
                f"; unchanged as expected: no optimizer step since step {rec['prev_step']} (optimizer steps "
                f"real/skipped/unknown {list(mark)})"
                if expected
                else ""
            )
            + f"; read back at {where} in {1e3 * dt:.3f} ms (dispatched in {1e3 * disp:.3f} ms at the sync)"
        )
        if stale:
            msg = (
                f"STALE WEIGHTS rank {self.rank}: sync step {step}: none of the {len(names)} sampled tensors changed "
                f"since step {rec['prev_step']} (the served weights are those of the previous sync)"
                + ("; the run stops at the next sync, after its export" if self.fatal else "; reported")
            )
            warn(msg)
            if self.fatal and self.fatal_pending is None:
                self.fatal_pending = msg
        return rec

    def check_fatal(self) -> None:
        """The next sync, right after its export: raise a kept stale verdict."""
        if self.fatal_pending is not None:
            msg, self.fatal_pending = self.fatal_pending, None
            raise RuntimeError(msg)
