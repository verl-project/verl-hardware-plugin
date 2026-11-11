# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Feature detection for the in-process rollout, and the table of tested install layouts.

The in-process engine couples to vllm-torchtpu, vLLM, verl and this plugin at a few points. Each coupling is
probed instead of pinned to a file hash, with one rule: a coupling that correctness depends on refuses at build with a
clear message; an optional fast path disables itself; a feature that only labels what ran is reported. The decisions
are logged once per rank in the ``inproc compat rank r`` line. Probes read sources (``inspect`` / ``ast``) or state the
module already has; they never call the probed code.

The tested-layout table (:data:`TESTED_LAYOUTS`) is the install-layout policy: the engine builds only for a
(model type, instantiated vLLM model class, unquantized weights) key that a test of this package covers, and the
first weight sync must install it with the table's install family. Anything else is refused at build. To support
another model: add its CPU test first, then its table entry.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import sys
import textwrap
from collections.abc import Mapping
from typing import Any, Optional

MATERIALIZE_ENV = "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"
SC_ENV = "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"
R7_ENV = "VLLM_TPU_FAST_TOKEN_SUBSTITUTION"
DYNAMIC_SMEM_FLAG = "--xla_tpu_use_dynamic_smem_negotiation"
SC_FLAGS = (
    "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes",
)
_TPU_VMEM_BYTES_PER_TENSOR_CORE = {
    "v5p": 67108864,
    "v5litepod": 134217728,
    "v6e": 134217728,
    "tpu7x": 67108864,
    "v7x": 67108864,
}
RUNNER_MOD = "vllm_torchtpu.runner.tpu_runner"
AOUT_MOD = "vllm_torchtpu.runner.tpu_runner_async_output"
ENVS_MOD = "vllm_torchtpu.envs"

# ------------------------------------------------------------------------------------------------ env_override (libtpu)

# Import-time structures of vllm_torchtpu/env_override.py whose effects :func:`would_add_libtpu_flags` and
# :data:`REVIEWED_ENV_EFFECTS` replicate (every published vllm-torchtpu commit as of this change). A new structure
# refuses until it is reviewed and added here with a test.
REVIEWED_ENV_OVERRIDE = frozenset({"d4e58d167506d14c"})
# The import-time environment effects of the reviewed structure, per name: "read", "setdefault <value>" (a value in
# angle brackets is computed), "forced <value>", "assembled" (LIBTPU_INIT_ARGS); "conditional" ones may not happen.
REVIEWED_ENV_EFFECTS: dict[str, tuple[str, ...]] = {
    "LIBTPU_INIT_ARGS": ("assembled (conditional)", "read"),
    MATERIALIZE_ENV: ("setdefault 'false'",),
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE": ("setdefault 'tpu_tier2_cache' (conditional)",),
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT": ("setdefault <envs.TORCH_TPU_TIER3_COMPILATION_CACHE_R> "
    "(conditional)",),
    "TORCH_TPU_TIER2_COMPILATION_CACHE": ("setdefault 'tpu_tier2_cache' (conditional)",),
    "TPU_ACCELERATOR_TYPE": ("read (conditional)",),
    "TPU_KV_RESHARD_TRANSPORT": ("read (conditional)",),
    "TPU_PREMAPPED_BUFFER_SIZE": ("setdefault <str(16 << 30)> (conditional)",),
    "TPU_USE_RAIDEN_KV_CACHE_MANAGER": ("read (conditional)",),
    "VLLM_CACHE_ROOT": ("setdefault <os.path.expanduser('~/.cache/vllm')>",),
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": ("forced '1'",),
    "VLLM_USE_BREAKABLE_CUDAGRAPH": ("forced '0'",),
    "VLLM_USE_MEGA_AOT_ARTIFACT": ("setdefault '0'",),
}  # fmt: skip
_UNCHANGED_IN_PROCESS = ("LIBTPU_INIT_ARGS", MATERIALIZE_ENV)


def env_effect_names() -> tuple[str, ...]:
    """The environment names the build-time check compares: the reviewed effect table plus the SparseCore pin."""
    return tuple(sorted(set(REVIEWED_ENV_EFFECTS) | {SC_ENV}))


def would_add_libtpu_flags(libtpu_args: str, environ: Optional[Mapping[str, str]] = None) -> list[str]:
    """The ``LIBTPU_INIT_ARGS`` entries the reviewed env_override would add on import, given ``libtpu_args``.

    ``--xla_tpu_use_dynamic_smem_negotiation=true`` unless the user set it, then the two SparseCore offload
    thresholds (the per-TensorCore VMEM of ``TPU_ACCELERATOR_TYPE``) unless ``SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES``
    is set (0 = no flag) or the user set them. In the colocated process the trainer creates the PJRT client, so
    anything env_override would add arrives too late for libtpu: the trainer's own flags must already carry it.
    """
    env = os.environ if environ is None else environ

    def has(flag: str) -> bool:
        return any(tok.startswith(flag + "=") or tok == flag for tok in libtpu_args.split())

    out = []
    if not has(DYNAMIC_SMEM_FLAG):
        out.append(f"{DYNAMIC_SMEM_FLAG}=true")
    raw = env.get(SC_ENV)
    sc: Optional[int]
    if raw is not None and raw.strip().lower() not in ("", "auto"):
        value = int(raw)
        sc = value if value > 0 else None
    else:
        tpu_type = (env.get("TPU_ACCELERATOR_TYPE") or "").strip().lower()
        sc = (
            next((v for k, v in _TPU_VMEM_BYTES_PER_TENSOR_CORE.items() if tpu_type.startswith(k)), None)
            if tpu_type
            else None
        )
    if sc is not None:
        out.extend(f"{flag}={sc}" for flag in SC_FLAGS if not has(flag))
    return out


_SKIP_FIELDS = ("ctx", "type_comment", "type_params")


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (
            isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            n.body = body[1:] or [ast.Pass()]
    return node


def _canon(node: Any) -> Any:
    """A Python-version-stable canonical form of a syntax tree (no positions, no ctx / type comments)."""
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            tuple((f, _canon(getattr(node, f, None))) for f in node._fields if f not in _SKIP_FIELDS),
        )
    if isinstance(node, list):
        return tuple(_canon(x) for x in node)
    return repr(node)


def env_override_structure(src: Optional[str]) -> dict:
    """The import-time structure of env_override.py: every module-level statement that is not a def, a class or the
    docstring, plus the module-level functions those statements call (transitively), as one normalized syntax-tree
    digest (docstrings, comments and formatting ignored), with the flag literals and environment names that code
    touches. A function the module defines but never calls at import is outside the structure."""
    if src is None:
        return {"present": False}
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    top = [
        n
        for n in tree.body
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
    ]

    def called(nodes):
        return {
            c.func.id
            for x in nodes
            for c in ast.walk(x)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defs
        }

    closure, frontier = set(), called(top)
    while frontier:
        name = frontier.pop()
        if name in closure:
            continue
        closure.add(name)
        frontier |= called([defs[name]]) - closure
    scope = top + [defs[n] for n in sorted(closure)]
    flags, envs = set(), set()
    for x in scope:
        for c in ast.walk(x):
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.startswith("--"):
                flags.add(c.value)
            if (
                isinstance(c, ast.Call)
                and c.args
                and isinstance(c.args[0], ast.Constant)
                and isinstance(c.args[0].value, str)
            ):
                if ast.unparse(c.func) in ("os.getenv", "os.environ.get", "os.environ.setdefault", "os.environ.pop"):
                    envs.add(c.args[0].value)
            if (
                isinstance(c, ast.Subscript)
                and ast.unparse(c.value) == "os.environ"
                and isinstance(c.slice, ast.Constant)
                and isinstance(c.slice.value, str)
            ):
                envs.add(c.slice.value)
    digest = hashlib.sha256(
        repr(tuple(_canon(_strip_docstrings(copy.deepcopy(n))) for n in scope)).encode()
    ).hexdigest()[:16]
    fp = hashlib.sha256(json.dumps([digest, sorted(flags), sorted(envs)]).encode()).hexdigest()[:16]
    return {
        "present": True,
        "file_sha": hashlib.sha256(src.encode()).hexdigest()[:12],
        "fingerprint": fp,
        "import_time_functions": sorted(closure),
        "flags": sorted(flags),
        "env_names": sorted(envs),
    }


def env_override_file() -> Optional[str]:
    """Path of the installed vllm_torchtpu/env_override.py, found without importing vllm_torchtpu."""
    try:
        spec = importlib.util.find_spec("vllm_torchtpu")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0], "env_override.py")


def env_override_probe(path: Optional[str] = None) -> dict:
    """The installed env_override's import-time structure and whether it is a reviewed one. Never raises."""
    path = path or env_override_file()
    if path is None:
        return {"present": False, "why": "vllm_torchtpu not installed"}
    try:
        with open(path, encoding="utf-8") as f:
            st = env_override_structure(f.read())
    except Exception as e:  # noqa: BLE001 - unreadable counts as unreviewed
        return {
            "present": True,
            "reviewed": False,
            "fingerprint": None,
            "why": f"unreadable: {type(e).__name__}: {e}"[:200],
        }
    return {
        "present": True,
        "reviewed": st["fingerprint"] in REVIEWED_ENV_OVERRIDE,
        "fingerprint": st["fingerprint"],
        "file_sha": st["file_sha"],
    }


def env_after_build_problems(before: Optional[dict], after: dict) -> list[str]:
    """The environment names whose value after the engine import and build the reviewed effect table does not
    allow, given the platform-import snapshot ``before``. A read-only name must be unchanged; a setdefault name
    unchanged if it was set, else the reviewed default (any value when the default is computed); a forced name the
    reviewed value; ``LIBTPU_INIT_ARGS`` and the MATERIALIZE pin unchanged (the trainer created PJRT before the
    engine). ``before`` None (the platform-import check never ran) refuses."""
    if before is None:
        return ["the platform-import environment snapshot is missing (the trainer-side libtpu check did not run)"]
    bad = []
    for name in env_effect_names():
        b, a = before.get(name), after.get(name)
        if a == b:
            continue
        if name in _UNCHANGED_IN_PROCESS or name == SC_ENV:
            bad.append(f"{name} {b!r} -> {a!r}")
            continue
        ok = False
        for eff in (e.replace(" (conditional)", "") for e in REVIEWED_ENV_EFFECTS.get(name, ())):
            if eff.startswith("setdefault ") and b is None:
                value = eff[len("setdefault ") :]
                ok = ok or value.startswith("<") or a == ast.literal_eval(value)
            elif eff.startswith("forced "):
                ok = ok or a == ast.literal_eval(eff[len("forced ") :])
        if not ok:
            bad.append(f"{name} {b!r} -> {a!r}")
    return bad


# --------------------------------------------------------------------------------------------- runner features (report)


def _source_or_none(obj: Any) -> Optional[str]:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except Exception:  # noqa: BLE001
        return None


def runner_features(modules: Optional[dict] = None, environ: Optional[Mapping[str, str]] = None) -> dict:
    """Report only: vllm-torchtpu's sampler skip when no request sets top-k / top-p, its deferred
    sampled-logprob copy, the fast token substitution setting, and the installed version. Never raises."""
    mods = sys.modules if modules is None else modules
    env = os.environ if environ is None else environ
    out: dict = {}
    try:
        runner = mods.get(RUNNER_MOD)
        fn = getattr(getattr(runner, "TPUModelRunner", None), "_sampler_applies_topk_topp", None)
        src = _source_or_none(fn) if fn is not None else None
        if src is None:
            out["sampler_skip"] = "unknown" if runner is not None else "runner not imported"
        else:
            early = any(
                isinstance(n, ast.If)
                and "speculative_config is None" in ast.unparse(n.test)
                and any(isinstance(b, ast.Return) and ast.unparse(b) == "return True" for b in n.body)
                for n in ast.walk(ast.parse(src))
            )
            out["sampler_skip"] = "absent" if early else "present"
        aout = mods.get(AOUT_MOD)
        defer = getattr(runner, "_DEFER_PLAIN_LOGPROBS", "absent") if runner is not None else "absent"
        plain = callable(getattr(aout, "plain_logprobs_lists", None))
        out["deferred_logprobs"] = "absent" if not plain else ("off" if defer is False else "on")
        em = mods.get(ENVS_MOD)
        out["fast_token_substitution"] = {
            "env": env.get(R7_ENV),
            "envs": repr(getattr(em, R7_ENV, None)) if em else None,
        }
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
    try:
        out["vllm_torchtpu"] = importlib.metadata.version("vllm-torchtpu")
    except Exception:  # noqa: BLE001
        out["vllm_torchtpu"] = "unknown"
    return out


# -------------------------------------------------------------------------------------------- tpu-sync install contract

# The attributes the in-process install shim provides (weight_sync._InstallShim).
SHIM_ATTRS = frozenset(
    {
        "worker", "rank", "_tpu_sync_ws", "_tpu_sync_staging", "_tpu_sync_staging_sharded", "_tpu_sync_layout",
        "_tpu_sync_plan", "_tpu_sync_use_vllm_loaders", "_tpu_sync_hf_text_config", "_get_vllm_model",
        "_install_with_vllm_loaders", "_tpu_sync_bound_names",
    }
)  # fmt: skip
EXTENSION_MOD = "verl_hardware_plugin.accelerators.tpu.rollout.tpu_sync_worker_extension"


def tpu_sync_contract(ext: Any = None) -> Optional[str]:
    """The in-process install calls the plugin's ``install_tpu_sync_weights`` with a shim as ``self`` and swaps
    the extension module's ``ExactParityCheck`` for a recorder of every write. Both work only while the install is
    reachable, reads ``ExactParityCheck`` as a global of that same module, and reads no ``self`` attribute the shim
    lacks. None when the contract holds, else why (the build refuses: a recorder that does not engage would leave
    the direct install's parity check comparing nothing)."""
    try:
        ext = ext if ext is not None else importlib.import_module(EXTENSION_MOD)
        cls = ext.vLLMTPUSyncWorkerExtension
        install = inspect.unwrap(cls.install_tpu_sync_weights)
    except Exception as e:  # noqa: BLE001
        return f"install_tpu_sync_weights not reachable ({type(e).__name__}: {e})"[:240]
    if (
        "ExactParityCheck" not in vars(ext)
        or "ExactParityCheck" not in install.__code__.co_names
        or install.__globals__ is not vars(ext)
    ):
        return "install_tpu_sync_weights does not read ExactParityCheck as a global of the extension module"
    src = _source_or_none(install)
    if src is None:
        return "install_tpu_sync_weights source unreadable"
    reads = {
        n.attr
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    }
    reads |= {
        n.args[1].value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "getattr"
        and len(n.args) >= 2
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == "self"
        and isinstance(n.args[1], ast.Constant)
        and isinstance(n.args[1].value, str)
    }
    missing = sorted(reads - SHIM_ATTRS)
    if missing:
        return f"install_tpu_sync_weights reads self attributes the in-process shim lacks: {missing}"
    return None


# ------------------------------------------------------------------------------------------------------- verl contracts

# vLLMHttpServer.__init__ attributes the in-process path never reaches (multi-node / data-parallel sockets).
SERVER_UNUSED_ATTRS = frozenset({"_dp_master_sock", "_dp_rpc_sock", "_master_sock"})


def init_assigned_attrs(cls: Any) -> set[str]:
    """The ``self.X`` names ``cls.__init__`` assigns (from its source)."""
    src = _source_or_none(cls.__init__)
    if src is None:
        raise RuntimeError(f"{cls.__name__}.__init__ source unreadable")
    out = set()
    for n in ast.walk(ast.parse(src)):
        targets = (
            n.targets
            if isinstance(n, ast.Assign)
            else ([n.target] if isinstance(n, ast.AnnAssign | ast.AugAssign) else [])
        )
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    out.add(sub.attr)
    return out


def server_contract_problem(server_cls: Any, built: set[str]) -> Optional[str]:
    """The in-process server is built with ``object.__new__`` plus the attributes verl's ``__init__`` would set;
    a new attribute of ``vLLMHttpServer.__init__`` that it does not set refuses at build, naming it."""
    try:
        want = init_assigned_attrs(server_cls)
    except Exception as e:  # noqa: BLE001
        return f"cannot read {getattr(server_cls, '__name__', server_cls)}.__init__: {type(e).__name__}: {e}"[:240]
    missing = sorted(want - set(built) - SERVER_UNUSED_ATTRS)
    if missing:
        return f"vLLMHttpServer.__init__ sets {missing}, which the in-process server does not set"
    return None


WRAPPED_WORKER_METHODS = ("init_model", "update_actor", "compute_log_prob")


def worker_methods_problem(worker_cls: Any) -> Optional[str]:
    """The trainer worker methods the in-process rollout wraps must exist as synchronous methods."""
    bad = []
    for name in WRAPPED_WORKER_METHODS:
        fn = worker_cls.__dict__.get(name)
        if fn is None:
            bad.append(f"{name} absent")
        elif inspect.iscoroutinefunction(inspect.unwrap(fn)):
            bad.append(f"{name} is a coroutine")
    return f"{worker_cls.__name__}: {', '.join(bad)}" if bad else None


# ----------------------------------------------------------------------------------------------- tested install layouts

QWEN3_5_NOT_EXPORTED = (r"(^|\.)visual\.", r"(^|\.)vision_tower\.", r"(^|\.)multi_modal_projector\.", r"(^|\.)mtp\.")
_E2E = "tests/accelerators/tpu/inproc/test_inproc_cpu_e2e.py"


@dataclasses.dataclass(frozen=True)
class LayoutEntry:
    """One tested install layout: the key, the install family the first sync must plan, the engine parameters a
    text-only trainer does not export (the only exemption the coverage and parity checks accept: patterns that must
    match no received tensor), and the tests that cover it."""

    model_type: str  # the HF text config's model_type
    architecture: str  # the HF config's architectures[0]
    model_class: str  # the instantiated vLLM model class, module + qualified name
    family: str  # the install mode the first sync must plan
    not_exported: tuple[str, ...]
    tests: tuple[str, ...]


def _row(
    model_type: str, module: str, cls: str, family: str, key: str, not_exported: tuple[str, ...] = ()
) -> LayoutEntry:
    tests = (f"{_E2E}::test_tested_layout[{key}]", f"{_E2E}::test_tested_layout[{key}_tied]")
    return LayoutEntry(model_type, cls, f"vllm.model_executor.models.{module}.{cls}", family, not_exported, tests)


TESTED_LAYOUTS: tuple[LayoutEntry, ...] = (
    _row("qwen3_5_text", "qwen3_5", "Qwen3_5ForCausalLM", "qwen3_5 direct", "qwen3_5", QWEN3_5_NOT_EXPORTED),
    _row("qwen3", "qwen3", "Qwen3ForCausalLM", "vllm loaders", "qwen3"),
    _row("llama", "llama", "LlamaForCausalLM", "vllm loaders", "llama"),
    _row("gemma3_text", "gemma3", "Gemma3ForCausalLM", "vllm loaders", "gemma3"),
    _row("mistral", "mistral", "MistralForCausalLM", "vllm loaders", "mistral"),
    _row("qwen2", "qwen2", "Qwen2ForCausalLM", "vllm loaders", "qwen2"),
)
UNQUANTIZED_DTYPES = ("torch.bfloat16", "torch.float32")


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", None) or hf_config


def _untested(key: str) -> str:
    rows = ", ".join(f"{r.architecture} ({r.family})" for r in TESTED_LAYOUTS)
    return f"UNTESTED LAYOUT: {key}; tested: {rows}. Add a test and a table entry (inproc/compat.py TESTED_LAYOUTS)"


def layout_before_build(hf_config: Any) -> tuple[Optional[LayoutEntry], Optional[str]]:
    """The table row for the model's HF config, checked before the engine build (every rank reads the same config):
    (row, None), else (None, the refusal)."""
    archs = [str(a) for a in (getattr(hf_config, "architectures", None) or [])]
    model_type = getattr(_text_config(hf_config), "model_type", None)
    key = f"model_type {model_type!r}, architectures {archs}"
    if len(archs) != 1:
        return None, _untested(key + " (exactly one architecture is required)")
    row = next((r for r in TESTED_LAYOUTS if r.model_type == model_type and r.architecture == archs[0]), None)
    return (row, None) if row is not None else (None, _untested(key))


def layout_after_build(
    model: Any, model_config: Any, hf_text_config: Any
) -> tuple[Optional[LayoutEntry], Optional[str]]:
    """The full key after the build: the HF text config's model_type, the instantiated vLLM model class, unquantized
    weights (no quantization config, every parameter bfloat16 or float32). A row on vLLM's loaders also needs the
    model's stacked-shard table (``hf_to_vllm_mapper``) or its ``packed_modules_mapping``: the install's parity check
    rebuilds every fused parameter from them. (row, None), else (None, the refusal)."""
    cls = type(model)
    model_class = f"{cls.__module__}.{cls.__qualname__}"
    model_type = getattr(hf_text_config, "model_type", None)
    quant = getattr(model_config, "quantization", None)
    try:
        dtypes = sorted({str(p.dtype) for _, p in model.named_parameters()})
    except Exception as e:  # noqa: BLE001
        return None, _untested(f"{model_class}: parameters unreadable ({type(e).__name__})")
    key = f"model_type {model_type!r}, model class {model_class}, quantization {quant!r}, parameter dtypes {dtypes}"
    if quant is not None or any(d not in UNQUANTIZED_DTYPES for d in dtypes):
        return None, _untested(key + " (quantized weights are not tested)")
    row = next((r for r in TESTED_LAYOUTS if r.model_type == model_type and r.model_class == model_class), None)
    if row is None:
        return None, _untested(key)
    if row.family == "vllm loaders":
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        if not (getattr(mapper, "orig_to_new_stacked", None) or getattr(model, "packed_modules_mapping", None)):
            return None, _untested(
                key + " (a vLLM-loader layout needs the model's stacked-shard table or packed_modules_mapping, which "
                "its covering tests rely on)"
            )
    return row, None
