# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""One model architecture through the real in-process rollout on CPU (one fresh interpreter per case). Test code.

A gloo world of one trainer rank. ``TPUInProcRollout`` is built in this process: verl's ``vLLMHttpServer.launch_server``
arguments, vLLM's real ``LLMEngine`` with ``InprocClient`` + ``UniProcExecutor``, the real scheduler and KV-cache
config, the package's KV capacity check; the worker is ``inproc_arch_worker.ArchModelWorker`` (vLLM's own model class
for the checkpoint's architecture, filled by its own loader; mocked forward). Two TPU-only steps are replaced: the
torch_tpu runtime check and the vllm-torchtpu worker patches (the logits-at-load hook goes on the test worker class).

The trainer stand-in is the HF model of the same checkpoint in fp32 with real AdamW steps, exporting bf16 HF-named
tensors (``named_parameters(remove_duplicate=False)``: a tied LM head is sent under both names). Every sync goes through
the real ``TPUInProcRollout.update_weights`` at the production parity cadence (syncs 0, 1, 5, 10 ... verified, the
others not). After every sync the engine's parameters are compared bitwise with an oracle: a second, independent vLLM
model of the same class filled by vLLM's own ``load_weights`` from the same full export.

Scenario (env ``INPROC_TEST_CASE``, JSON): ``{"model_dir", "key", "mode": "full" | "build", "add_layout": bool}``.
``full``: build, serve, syncs 0-5 (AdamW between), then the mutants, each followed by a good sync: 6 = a fused shard
missing at an unverified sync, 8 = a whole unfused tensor missing at an unverified sync, 10 = a fused shard missing at a
verified sync, 15 = (tied models) an LM head that differs from the embedding, 20 / 25 = variables that once weakened the
loader-path check set (full export, then a fused shard missing at an unverified sync), then the package's own parity on
a corrupted fused parameter and under vllm-torchtpu's transposed dense layout simulated in place, and fresh processes: a
first sync with a fused shard missing; an export without lm_head.weight twice, then the full export; a
coverage-weakening variable set with a whole tensor missing; an allowlist variable set with a fused shard missing; and
the weakening variables at the constructor (``defer_build``: nothing is built). ``build``: the build only (a layout the
package must refuse), and if it builds, one sync. Prints one line ``RESULT <json>``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import traceback

KV = 64 * 2**20
LOGGER = "verl_hardware_plugin.accelerators.tpu.rollout.inproc"
# the mutated tensors per family: (a fused shard at an unverified sync, a fused shard at a verified sync, a whole
# unfused tensor)
MUTANTS = {
    "qwen3_5": (
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.out_proj.weight",
    ),
    "dense": (
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ),
}
WEAKENING = {
    "VERL_TPU_INPROC_COVERAGE": "report",
    "VERL_TPU_INPROC_COVERAGE_ALLOW": "qkv_proj",
    "VERL_TPU_INPROC_LOADER_VERIFY": "names",
    "VERL_TPU_INPROC_LOADER_FALLBACK": "allow",
}


class _Lines(logging.Handler):
    """Every line the package logs, in order."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def mark(self) -> int:
        return len(self.lines)

    def since(self, i: int) -> list[str]:
        return self.lines[i:]


def family(key: str) -> str:
    return "qwen3_5" if key.startswith("qwen3_5") else "dense"


def rollout_config(max_num_seqs: int = 16, response_length: int = 16):
    from verl.workers.config import RolloutConfig
    from verl.workers.config.rollout import CheckpointEngineConfig

    return RolloutConfig(
        name="vllm",
        mode="async",
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        max_num_seqs=max_num_seqs,
        max_model_len=1024,
        prompt_length=256,
        response_length=response_length,
        seed=1,
        engine_kwargs={
            "vllm": {
                "kv_cache_memory_bytes": KV,
                "max_logprobs": 0,
                "seed": None,
                "no_enable_prefix_caching": True,
                "worker_cls": "inproc_arch_worker.ArchModelWorker",
                "distributed_executor_backend": "uni",
            }
        },
        calculate_log_probs=True,
        enable_prefix_caching=False,
        free_cache_engine=False,
        load_format="safetensors",
        gpu_memory_utilization=0.8,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        dtype="bfloat16",
        enforce_eager=False,
        max_num_batched_tokens=4096,
        checkpoint_engine=CheckpointEngineConfig(backend="naive"),
    )


def pick(lines: list[str], key: str):
    return next((x for x in lines if key in x), None)


def package_sha() -> str:
    from verl_hardware_plugin.accelerators.tpu.rollout import inproc

    d = os.path.dirname(inproc.__file__)
    h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(f.encode())
            h.update(open(os.path.join(d, f), "rb").read())
    return h.hexdigest()


def setup(case: dict, cap: _Lines):
    """The gloo world, verl (the CPU platform and the plugin's registration) and the two TPU-only replacements."""
    import torch.distributed as dist

    store = tempfile.NamedTemporaryFile(prefix="inproc-pg-", dir=os.environ["TMPDIR"], delete=False)
    dist.init_process_group("gloo", init_method=f"file://{store.name}", rank=0, world_size=1)
    logging.getLogger(LOGGER).addHandler(cap)
    import verl  # noqa: F401  (the CPU platform + verl_hardware_plugin through VERL_USE_EXTERNAL_MODULES: registration)
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compat, rollout

    logging.getLogger(LOGGER).setLevel(logging.INFO)  # (the package sets VERL_LOGGING_LEVEL's level at its import)

    import inproc_arch_worker as W

    if case.get("add_layout"):  # a layout added to the table without its test: the verification must still fail closed
        compat.TESTED_LAYOUTS = compat.TESTED_LAYOUTS + (
            compat.LayoutEntry(
                "qwen3_moe",
                "Qwen3MoeForCausalLM",
                "vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM",
                "vllm loaders",
                (),
                (),
            ),
        )
    rollout.TPUInProcRollout._check_runtime_up = lambda self: None

    def _patch_vllm_in_process(self):
        from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compile_guard import LOGITS_AT_LOAD

        if not LOGITS_AT_LOAD.install(W.ArchModelWorker):
            raise RuntimeError("compute_logits at load: ArchModelWorker has no load_model of its own to hook")

    rollout.TPUInProcRollout._patch_vllm_in_process = _patch_vllm_in_process
    return rollout, W


def main() -> dict:
    import torch

    case = json.loads(os.environ["INPROC_TEST_CASE"])
    model_dir, key = case["model_dir"], case["key"]
    out: dict = {"key": key, "mode": "full", "model_dir": model_dir}
    cap = _Lines()
    R, W = setup(case, cap)
    from verl.workers.config import HFModelConfig
    from verl_hardware_plugin.accelerators.tpu.rollout.inproc import weight_sync

    out["package_sha256"] = package_sha()
    try:
        r = R.TPUInProcRollout(rollout_config(), HFModelConfig(path=model_dir), None)
    except Exception as e:  # noqa: BLE001
        out["build_error"] = f"{type(e).__name__}: {e}"[:2500]
        out["build_lines"] = [x for x in cap.lines if "UNTESTED" in x or "FAILED" in x][-8:]
        return out
    eng = r.server._inproc_engine
    worker = eng.model_executor.driver_worker.worker
    vmodel = worker.model_runner.model
    lines = cap.lines
    out["engine_line"] = pick(lines, "in-process engine pid")
    out["compat_line"] = pick(lines, "compat rank 0: ")
    out["warm_line"] = pick(lines, "warm-up rank 0: ")
    kvl = pick(lines, "kv capacity rank 0: ")
    out["kv_capacity"] = json.loads(kvl.split("kv capacity rank 0: ", 1)[1]) if kvl else None
    params = dict(vmodel.named_parameters())
    out["engine"] = {
        "client": type(eng.engine_core).__name__,
        "executor": type(eng.model_executor).__name__,
        "worker": type(worker).__name__,
        "arch": worker.arch,
        "model_class": type(vmodel).__name__,
        "kv_spec_kinds": sorted(set(worker.kv_spec_kinds.values())),
        "kv_spec_layers": len(worker.kv_spec_kinds),
        "params": len(params),
        "param_names": sorted(params),
        "tied_lm_head": any(n.endswith("lm_head.weight") for n, _ in vmodel.named_parameters(remove_duplicate=False))
        and not any(n.endswith("lm_head.weight") for n in params),
        "installer_family": r.installer.family,
    }

    # verl's vLLMHttpServer.generate names its trace lane with ray.get_runtime_context(): a test-only stub of it
    import ray

    class _Ctx:
        def get_actor_name(self):
            return "inproc-trainer-rank0"

        def get_job_id(self):
            return "inproc"

    ray.get_runtime_context = lambda: _Ctx()

    async def serve(n: int, tag: str):
        res = []
        for i in range(n):
            prompt = [151644, 872, 198, 1000 + 37 * i, 2000 + i, 151645, 198]
            sp = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "logprobs": True,
                "max_tokens": 5 + i,
                "ignore_eos": True,
            }
            o = await r.generate(prompt, sp, f"{tag}-{i}")
            res.append(
                {
                    "ok": list(o.token_ids) == [W.mock_token(prompt, k) for k in range(5 + i)],
                    "logprobs": o.log_probs is not None and len(o.log_probs) == 5 + i,
                }
            )
        return res

    out["generate"] = asyncio.run(serve(4, "build"))

    from transformers import AutoModelForCausalLM
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.utils.torch_utils import set_default_torch_dtype

    torch.manual_seed(5)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    opt = torch.optim.AdamW(hf.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    vc_o = EngineArgs(
        model=model_dir,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        kv_cache_memory_bytes=KV,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    ).create_engine_config()
    cls_o, _ = ModelRegistry.resolve_model_cls(vc_o.model_config.architectures, vc_o.model_config)
    with set_current_vllm_config(vc_o), set_default_torch_dtype(vc_o.model_config.dtype), torch.device("cpu"):
        oracle = cls_o(vllm_config=vc_o, prefix="")
    losses: list = []

    def train_step() -> None:
        g = torch.Generator().manual_seed(100 + len(losses))
        ids = torch.randint(1000, 40000, (2, 24), generator=g)
        loss = hf(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(float(loss), 6))

    def export(drop: tuple = (), perturb: tuple = ()) -> list:
        items = []
        for n, p in hf.named_parameters(remove_duplicate=False):
            if n in drop:
                continue
            t = p.detach().to(torch.bfloat16).clone()
            if n in perturb:
                t = (t.float() + 0.5).to(torch.bfloat16)
            items.append((n, t))
        return items

    def engine_vs(items: list) -> dict:
        """the engine's parameters vs the oracle loaded with ``items`` (vLLM's own load_weights)"""
        with torch.no_grad(), set_current_vllm_config(vc_o):
            oracle.load_weights(iter([(n, t.clone()) for n, t in items]))
        pa, po = dict(vmodel.named_parameters()), dict(oracle.named_parameters())
        diff = sorted(n for n in pa if n not in po or pa[n].shape != po[n].shape or not torch.equal(pa[n], po[n]))
        return {"params": len(pa), "n_mismatched": len(diff), "mismatched": diff[:6]}

    def snapshot() -> dict:
        return {n: p.detach().clone() for n, p in vmodel.named_parameters()}

    def changed_since(snap: dict) -> list:
        return sorted(n for n, p in vmodel.named_parameters() if not torch.equal(p, snap[n]))

    def sync(items: list, step: int) -> dict:
        async def go():
            try:
                await r.update_weights(iter(items), global_steps=step)
                return None
            except Exception as e:  # noqa: BLE001
                return f"{type(e).__name__}: {e}"

        i0 = cap.mark()
        err = asyncio.run(go())
        ls = cap.since(i0)
        rec = dict(r.sync_log[-1]) if r.sync_log and r.sync_log[-1].get("step") == step and err is None else {}
        return {
            "step": step,
            "err": None if err is None else err[:900],
            "mode": rec.get("mode"),
            "verified": rec.get("verified"),
            "parity": rec.get("parity"),
            "coverage": rec.get("install_coverage"),
            "export_pin": rec.get("export_pin"),
            "tensors": len(items),
            "parity_line": pick(ls, f"parity step {step} rank 0"),
            "coverage_line": pick(ls, f"coverage step {step} rank 0"),
            "health_line": pick(ls, f"health step {step} rank 0"),
            "integrity_line": pick(ls, "INTEGRITY FAILED"),
        }

    syncs: list = []
    items = export()
    syncs.append({**sync(items, 0), "oracle": engine_vs(items)})
    for s in (1, 2, 3, 4, 5):
        train_step()
        items = export()
        syncs.append({**sync(items, s), "oracle": engine_vs(items)})
    out["syncs"] = syncs
    fam = family(key)
    shard_unv, shard_ver, whole = MUTANTS[fam]
    mut: dict = {}

    def mutant(label: str, step: int, **kw) -> None:
        train_step()
        snap = snapshot()
        bad = export(**kw)
        full = export()
        res = sync(bad, step)
        res["engine_vs_full_export"] = engine_vs(full)
        res["engine_changed_params"] = len(changed_since(snap))
        res["what"] = kw
        mut[label] = res
        train_step()
        good = export()
        mut[label + "_restore"] = {**sync(good, step + 1), "oracle": engine_vs(good)}

    mutant("m6_shard_unverified", 6, drop=(shard_unv,))
    mutant("m8_whole_unverified", 8, drop=(whole,))
    mutant("m10_shard_verified", 10, drop=(shard_ver,))
    if out["engine"]["tied_lm_head"]:
        mutant("m15_tie_mismatch", 15, perturb=("lm_head.weight",))
    os.environ.update(WEAKENING)
    try:
        mutant("m20_weakening_full", 20)
        mutant("m25_weakening_shard", 25, drop=(shard_unv,))
    finally:
        for k in WEAKENING:
            os.environ.pop(k, None)
    out["mutants"] = mut
    items = export()
    if fam == "dense":
        pa = dict(vmodel.named_parameters())
        sens = {}
        for name in (
            sorted(n for n in pa if "qkv_proj" in n)[0],
            sorted(n for n in pa if n.endswith("embed_tokens.weight"))[0],
        ):
            with torch.no_grad():
                old = pa[name].data.view(-1)[0].clone()
                pa[name].data.view(-1)[0] += 1.0
            par = r.installer._post_check_loaders(dict(items))
            sens[name] = {
                "post_checked": par.get("post_checked"),
                "detected": name in (par.get("post_mismatched") or []),
            }
            with torch.no_grad():
                pa[name].data.view(-1)[0] = old
        out["corrupt"] = sens
        from vllm.model_executor.layers.linear import LinearBase

        flipped = []
        with torch.no_grad():
            for mname, mod in vmodel.named_modules():
                w = getattr(mod, "weight", None)
                if isinstance(mod, LinearBase) and isinstance(w, torch.Tensor) and w.ndim == 2:
                    w.data = w.data.t().contiguous()
                    mod._tpu_weight_flipped = True
                    flipped.append(mname)
        par = r.installer._post_check_loaders(dict(items))
        out["tpu_flip_sim"] = {
            "flipped_modules": len(flipped),
            "post_checked": par.get("post_checked"),
            "post_mismatched_n": len(par.get("post_mismatched") or []),
            "unresolved_n": len(par.get("unresolved") or []),
            "first": (par.get("post_mismatched") or [])[:3],
        }
        with torch.no_grad():
            for mname, mod in vmodel.named_modules():
                if mname in flipped:
                    mod.weight.data = mod.weight.data.t().contiguous()
                    del mod._tpu_weight_flipped
        out["after_flip_restore"] = engine_vs(items)

    def fresh_process() -> None:
        """a fresh process (a resumed run): the export pin and the installer as the build makes them"""
        r._export_pin = None
        r._syncs_started = 0
        r.installer = weight_sync.InProcInstaller(r.worker, r.installer.hf_text_config, r.rank, layout=r.layout_entry)

    def bad_sync(step: int, drop: tuple) -> dict:
        train_step()
        snap = snapshot()
        res = sync(export(drop=drop), step)
        res["what"] = {"drop": list(drop)}
        res["engine_vs_full_export"] = engine_vs(export())
        res["engine_changed_params"] = len(changed_since(snap))
        return res

    def good_sync(step: int, drop: tuple = ()) -> dict:
        train_step()
        good = export(drop=drop)
        return {**sync(good, step), "oracle": engine_vs(good)}

    fresh_process()
    out["first_sync_shard"] = bad_sync(30, (shard_unv,))
    out["first_sync_restore"] = good_sync(31)
    head = "lm_head.weight"
    tied_engine = out["engine"]["tied_lm_head"]
    fresh_process()
    out["first_sync_no_head"] = {
        "first": bad_sync(40, (head,)),
        "second": bad_sync(41, (head,)),
        "full": good_sync(42, (head,) if tied_engine else ()),
    }
    for label, env, drop, s in (
        ("rt_coverage_report", {"VERL_TPU_INPROC_COVERAGE": "report"}, (whole,), 50),
        ("rt_allow_widened", {"VERL_TPU_INPROC_COVERAGE_ALLOW": "qkv_proj"}, (shard_unv,), 52),
    ):
        fresh_process()
        os.environ.update(env)
        try:
            bad = bad_sync(s, drop)
        finally:
            for k in env:
                os.environ.pop(k, None)
        out[label] = {"env": env, "bad": bad, "good": good_sync(s + 1)}
    out["generate_after"] = asyncio.run(serve(2, "after"))
    # the weakening variables at the constructor: not read; the constructor names them in one warning
    refusals = {}
    for env in (
        {"VERL_TPU_INPROC_COVERAGE": "report"},
        {"VERL_TPU_INPROC_COVERAGE_ALLOW": "qkv_proj"},
        {"VERL_TPU_INPROC_LOADER_VERIFY": "names"},
    ):
        tag = ",".join(f"{k}={v}" for k, v in env.items())
        os.environ.update(env)
        i0 = cap.mark()
        try:
            R.TPUInProcRollout(rollout_config(), HFModelConfig(path=model_dir), None, defer_build=True)
            refusals[tag] = {"constructed": True, "warning": pick(cap.since(i0), "are set but not read")}
        except Exception as e:  # noqa: BLE001
            refusals[tag] = {"constructed": False, "error": f"{type(e).__name__}: {e}"[:400]}
        finally:
            for k in env:
                os.environ.pop(k, None)
    out["constructor_switches"] = refusals
    out["losses"] = losses
    return out


def main_build() -> dict:
    """``mode: build``: the build of a layout the package must refuse; if it builds, one full-export sync and the "
    "oracle"""
    import torch

    case = json.loads(os.environ["INPROC_TEST_CASE"])
    out: dict = {
        "key": case["key"],
        "mode": "build",
        "model_dir": case["model_dir"],
        "add_layout": bool(case.get("add_layout")),
    }
    cap = _Lines()
    R, _W = setup(case, cap)
    from verl.workers.config import HFModelConfig

    out["package_sha256"] = package_sha()
    try:
        r = R.TPUInProcRollout(rollout_config(), HFModelConfig(path=case["model_dir"]), None)
    except Exception as e:  # noqa: BLE001
        out["build_error"] = f"{type(e).__name__}: {e}"[:2500]
        out["build_lines"] = [x for x in cap.lines if "UNTESTED" in x or "FAILED" in x][-8:]
        return out
    vmodel = r.server._inproc_engine.model_executor.driver_worker.worker.model_runner.model
    out["built"] = {"model_class": type(vmodel).__name__, "params": len(dict(vmodel.named_parameters()))}
    from transformers import AutoModelForCausalLM
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.utils.torch_utils import set_default_torch_dtype

    hf = AutoModelForCausalLM.from_pretrained(case["model_dir"], torch_dtype=torch.float32)
    items = [(n, p.detach().to(torch.bfloat16).clone()) for n, p in hf.named_parameters(remove_duplicate=False)]
    i0 = cap.mark()

    async def go():
        try:
            await r.update_weights(iter(items), global_steps=0)
            return None
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}"

    err = asyncio.run(go())
    ls = cap.since(i0)
    vc_o = EngineArgs(
        model=case["model_dir"],
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        kv_cache_memory_bytes=KV,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    ).create_engine_config()
    cls_o, _ = ModelRegistry.resolve_model_cls(vc_o.model_config.architectures, vc_o.model_config)
    with set_current_vllm_config(vc_o), set_default_torch_dtype(vc_o.model_config.dtype), torch.device("cpu"):
        oracle = cls_o(vllm_config=vc_o, prefix="")
    with torch.no_grad(), set_current_vllm_config(vc_o):
        oracle.load_weights(iter([(n, t.clone()) for n, t in items]))
    pa, po = dict(vmodel.named_parameters()), dict(oracle.named_parameters())
    diff = sorted(n for n in pa if n not in po or pa[n].shape != po[n].shape or not torch.equal(pa[n], po[n]))
    out["sync0"] = {
        "err": None if err is None else err[:1500],
        "parity_line": pick(ls, "parity step 0 rank 0"),
        "sync_line": pick(ls, "sync step 0 rank 0"),
        "oracle": {"params": len(pa), "n_mismatched": len(diff), "mismatched": diff[:6]},
    }
    return out


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    # run with ``python -I`` (no implicit script dir / PYTHONPATH): the test passes its import path explicitly
    sys.path[:0] = [p for p in os.environ.get("INPROC_TEST_PATH", "").split(os.pathsep) if p and p not in sys.path]
    try:
        res = main_build() if json.loads(os.environ["INPROC_TEST_CASE"]).get("mode", "full") == "build" else main()
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()[-4000:]}
    print("RESULT " + json.dumps(res, default=str))
