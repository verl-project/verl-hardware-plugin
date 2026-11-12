# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Compatibility guard for the in-process rollout's lazy final-only output path.

The lazy path (:class:`~.lazy_output.LazyFinalOutputs`) replaces
``vllm.v1.engine.output_processor.OutputProcessor.process_outputs`` on the in-process engine. For a FINAL_ONLY
request whose output is neither its first (prefilling) nor its last one, it runs part of vLLM's per-output loop
inline and keeps the step's sample logprobs on the request state; at the request's finish it applies them with
vLLM's own logprob code, so the final ``RequestOutput`` is built by vLLM from the same state. That is exact only
while the vLLM code it replicates, skips or defers is the code it was written against.

``check()`` verifies that before the lazy path is installed:

* the fields of ``EngineCoreOutput``, ``LogprobsLists``, ``LogprobsProcessor``, ``OutputProcessorOutput``,
  ``Logprob`` and ``FlatLogprobs``;
* the parameters of ``OutputProcessor.process_outputs``;
* the ``RequestState`` attributes the fast path reads, and ``RequestOutputKind.FINAL_ONLY``;
* a normalized digest (the syntax tree without docstrings, comments or formatting) of every vLLM function and class
  the fast path replicates, skips or defers, and that none of them has been wrapped or replaced at run time;
* optionally, that the engine's output processor is a plain ``OutputProcessor`` with no instance-level override.

It also requires the imported vLLM's version to be one the lazy path's trace test ran on (:data:`TRACE_TESTED_VLLM`),
so the path needs both: the reviewed code and a trace test.

Any difference, and any error while checking, makes ``check()`` return ``ok=False`` with the reasons; the caller then
leaves vLLM's own ``process_outputs`` in place (always correct, only slower). Nothing in this module raises.

The reviewed facts below are those of vLLM v0.32.0. To support another vLLM version, run the CPU equivalence test of
the lazy path (every final ``RequestOutput`` bitwise equal to vLLM's own on randomized traces) against it, add its
facts with ``collect_source_facts`` and its version to :data:`TRACE_TESTED_VLLM`.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import os
import textwrap
from typing import Any, Optional

# (module, qualified name): vLLM code the lazy path replicates (process_outputs' loop body, the plain detokenizer's
# update, the sample-logprob update and its container helpers), skips (stats with log_stats off, make_request_output
# for a non-final FINAL_ONLY output) or calls (track_weight_version). A change to any of them disables the lazy path.
DIGEST_TARGETS: tuple[tuple[str, str], ...] = (
    ("vllm.v1.engine.output_processor", "OutputProcessor.process_outputs"),
    ("vllm.v1.engine.output_processor", "OutputProcessor._update_stats_from_output"),
    ("vllm.v1.engine.output_processor", "RequestState.make_request_output"),
    ("vllm.v1.engine.output_processor", "RequestState.track_weight_version"),
    ("vllm.v1.engine.detokenizer", "IncrementalDetokenizer"),
    ("vllm.v1.engine.logprobs", "LogprobsProcessor._update_sample_logprobs"),
    ("vllm.v1.engine.logprobs", "LogprobsProcessor.update_from_output"),
    ("vllm.logprobs", "append_logprobs_for_next_position"),
    ("vllm.logprobs", "create_sample_logprobs"),
    ("vllm.logprobs", "Logprob"),
)

# (module, class): their fields, in order (msgspec Struct, NamedTuple or dataclass).
FIELD_TARGETS: tuple[tuple[str, str], ...] = (
    ("vllm.v1.engine", "EngineCoreOutput"),
    ("vllm.v1.outputs", "LogprobsLists"),
    ("vllm.v1.engine.logprobs", "LogprobsProcessor"),
    ("vllm.v1.engine.output_processor", "OutputProcessorOutput"),
    ("vllm.logprobs", "Logprob"),
    ("vllm.logprobs", "FlatLogprobs"),
)

PARAM_TARGETS: tuple[tuple[str, str], ...] = (("vllm.v1.engine.output_processor", "OutputProcessor.process_outputs"),)

# RequestState attributes the fast path reads or writes (it must find them assigned in RequestState.__init__).
REQUEST_STATE_ATTRS: tuple[str, ...] = (
    "detokenizer",
    "is_prefilling",
    "logprobs_processor",
    "output_kind",
    "parent_req",
    "queue",
    "request_id",
    "streaming_input",
    "weight_version_starts",
)
REQUEST_STATE_INIT = ("vllm.v1.engine.output_processor", "RequestState.__init__")
OUTPUT_KIND_ENUM = ("vllm.sampling_params", "RequestOutputKind")

# The facts of vLLM v0.32.0 (git tag v0.32.0, 5df63a7203cdb9328d824a27b682e77c7b9928be), Python 3.12 syntax trees.
REVIEWED: dict[str, Any] = {
    "fields:vllm.v1.engine:EngineCoreOutput": (
        "request_id",
        "new_token_ids",
        "new_logprobs",
        "new_prompt_logprobs_tensors",
        "pooling_output",
        "finish_reason",
        "stop_reason",
        "events",
        "kv_transfer_params",
        "ec_transfer_params",
        "trace_headers",
        "prefill_stats",
        "routed_experts",
        "num_nans_in_logits",
        "mm_cache_miss_hashes",
        "new_sampling_mask",
        "spec_decode_metrics",
        "prompt_token_id_logprobs",
    ),
    "fields:vllm.v1.outputs:LogprobsLists": (
        "logprob_token_ids",
        "logprobs",
        "sampled_token_ranks",
        "cu_num_generated_tokens",
    ),
    "fields:vllm.v1.engine.logprobs:LogprobsProcessor": (
        "tokenizer",
        "logprobs",
        "prompt_logprobs",
        "cumulative_logprob",
        "num_logprobs",
        "num_prompt_logprobs",
        "prompt_token_id_logprobs",
    ),
    "fields:vllm.v1.engine.output_processor:OutputProcessorOutput": ("request_outputs", "reqs_to_abort"),
    "fields:vllm.logprobs:Logprob": ("logprob", "rank", "decoded_token"),
    "fields:vllm.logprobs:FlatLogprobs": (
        "start_indices",
        "end_indices",
        "token_ids",
        "logprobs",
        "ranks",
        "decoded_tokens",
    ),
    "params:vllm.v1.engine.output_processor:OutputProcessor.process_outputs": (
        "self",
        "engine_core_outputs",
        "engine_core_timestamp",
        "iteration_stats",
        "weight_version",
    ),
    # collect_source_facts on the v0.32.0 sources (the tests recompute and compare them)
    "digest:vllm.v1.engine.output_processor:OutputProcessor.process_outputs": "c5dee7892b1aefce",
    "digest:vllm.v1.engine.output_processor:OutputProcessor._update_stats_from_output": "40fe8c1c9bc926b4",
    "digest:vllm.v1.engine.output_processor:RequestState.make_request_output": "b900bdd01f935ffb",
    "digest:vllm.v1.engine.output_processor:RequestState.track_weight_version": "4afefa0622b7adf2",
    "digest:vllm.v1.engine.detokenizer:IncrementalDetokenizer": "0e205d7ac21c4781",
    "digest:vllm.v1.engine.logprobs:LogprobsProcessor._update_sample_logprobs": "bd8ccdcca802538a",
    "digest:vllm.v1.engine.logprobs:LogprobsProcessor.update_from_output": "c51e5cf34098b5f4",
    "digest:vllm.logprobs:append_logprobs_for_next_position": "897cef334ddc46e5",
    "digest:vllm.logprobs:create_sample_logprobs": "321b3616ebf3edc1",
    "digest:vllm.logprobs:Logprob": "bfbcd9f691c18478",
}
REVIEWED_VLLM = "0.32.0"
# vLLM versions the lazy path's bitwise trace test ran on (tests/accelerators/tpu/inproc/test_lazy_output_compat.py).
TRACE_TESTED_VLLM: tuple[str, ...] = ("0.32.0",)


@dataclasses.dataclass(frozen=True)
class CompatResult:
    ok: bool
    reasons: tuple[str, ...]
    vllm_version: Optional[str]
    checks: dict

    def summary(self) -> str:
        if self.ok:
            return f"vLLM {self.vllm_version}: the output-processing code matches the reviewed vLLM {REVIEWED_VLLM}"
        return f"vLLM {self.vllm_version}: " + "; ".join(self.reasons[:4]) + (" ..." if len(self.reasons) > 4 else "")


# ------------------------------------------------------------------------------------------------ normalized digests

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
    """a Python-version-stable canonical form of a syntax tree (no positions, no ctx / type comments / type params)"""
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            tuple((name, _canon(getattr(node, name, None))) for name in node._fields if name not in _SKIP_FIELDS),
        )
    if isinstance(node, list):
        return tuple(_canon(x) for x in node)
    return repr(node)


def node_digest(node: ast.AST) -> str:
    import copy

    tree = _strip_docstrings(copy.deepcopy(node))
    return hashlib.sha256(repr(_canon(tree)).encode()).hexdigest()[:16]


def source_digest(source: str) -> str:
    """the digest of the first definition in ``source`` (dedented)"""
    tree = ast.parse(textwrap.dedent(source))
    return node_digest(tree.body[0])


# ----------------------------------------------------------------------------------------- static facts (a source tree)


def _find(tree: ast.Module, qualname: str) -> Optional[ast.AST]:
    body = tree.body
    node = None
    for part in qualname.split("."):
        node = next(
            (
                n
                for n in body
                if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and n.name == part
            ),
            None,
        )
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


def _class_fields(node: ast.ClassDef) -> tuple:
    out = []
    for n in node.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            ann = ast.unparse(n.annotation)
            if ann.startswith(("ClassVar", "typing.ClassVar")):
                continue
            out.append(n.target.id)
    return tuple(out)


def _init_attrs(node: ast.FunctionDef) -> tuple:
    names = set()
    for n in ast.walk(node):
        targets = (
            n.targets
            if isinstance(n, ast.Assign)
            else ([n.target] if isinstance(n, ast.AnnAssign | ast.AugAssign) else [])
        )
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    names.add(sub.attr)
    return tuple(sorted(names))


def _enum_members(node: ast.ClassDef) -> tuple:
    out: list[str] = []
    for n in node.body:
        if isinstance(n, ast.Assign):
            out.extend(t.id for t in n.targets if isinstance(t, ast.Name))
    return tuple(out)


def _module_file(root: str, module: str) -> str:
    rel = module.replace(".", os.sep)
    pkg = os.path.join(root, rel, "__init__.py")
    return pkg if os.path.exists(pkg) else os.path.join(root, rel + ".py")


def collect_source_facts(root: str, version: Optional[str] = None) -> dict:
    """the facts of a vLLM source tree (``root`` holds the ``vllm`` package directory), without importing it; a module
    file that is absent leaves its facts out (``compare`` reports them as unavailable)"""
    trees: dict[str, Optional[ast.Module]] = {}

    def tree(module: str) -> Optional[ast.Module]:
        if module not in trees:
            path = _module_file(root, module)
            try:
                with open(path, encoding="utf-8") as f:
                    trees[module] = ast.parse(f.read(), filename=path)
            except (OSError, SyntaxError, ValueError):
                trees[module] = None
        return trees[module]

    facts: dict[str, Any] = {"vllm_version": version}
    for module, qual in DIGEST_TARGETS:
        t = tree(module)
        node = _find(t, qual) if t is not None else None
        if node is not None:
            facts[f"digest:{module}:{qual}"] = node_digest(node)
            facts[f"wrapped:{module}:{qual}"] = False
    for module, qual in FIELD_TARGETS:
        t = tree(module)
        node = _find(t, qual) if t is not None else None
        if isinstance(node, ast.ClassDef):
            facts[f"fields:{module}:{qual}"] = _class_fields(node)
    for module, qual in PARAM_TARGETS:
        t = tree(module)
        node = _find(t, qual) if t is not None else None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            a = node.args
            facts[f"params:{module}:{qual}"] = tuple(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
    module, qual = REQUEST_STATE_INIT
    t = tree(module)
    node = _find(t, qual) if t is not None else None
    if isinstance(node, ast.FunctionDef):
        facts[f"attrs:{module}:{qual}"] = _init_attrs(node)
    module, qual = OUTPUT_KIND_ENUM
    t = tree(module)
    node = _find(t, qual) if t is not None else None
    if isinstance(node, ast.ClassDef):
        facts[f"members:{module}:{qual}"] = _enum_members(node)
    return facts


# ------------------------------------------------------------------------------------ runtime facts (the imported vLLM)


def _resolve(module: str, qual: str) -> tuple[Any, Any]:
    """(the object as stored on its owner, the owner): a class attribute is read from the class ``__dict__``"""
    obj: Any = importlib.import_module(module)
    owner: Any = obj
    for part in qual.split("."):
        owner = obj
        obj = vars(obj)[part] if isinstance(obj, type) else getattr(obj, part)
    return obj, owner


def _plain(member: Any) -> Any:
    if isinstance(member, property):
        return member.fget
    if isinstance(member, staticmethod | classmethod):
        return member.__func__
    return member


def _defined_in(fn: Any, path: str, first: int, last: int) -> bool:
    """a plain function (no wrapper) whose code lies in ``path`` between lines ``first`` and ``last``"""
    if not inspect.isfunction(fn) or hasattr(fn, "__wrapped__"):
        return False
    code = fn.__code__
    return os.path.realpath(code.co_filename) == path and first <= code.co_firstlineno <= last


def _is_replaced(module: str, qual: str, obj: Any) -> bool:
    """True when the object the module (or class) holds now is not vLLM's own definition from its source file: a
    wrapper, or code from another file or another place. For a class, every method it defines in its source is checked
    the same way (a monkeypatched method leaves the class's source, and so its digest, unchanged); methods the class
    does not define in its source (e.g. dataclass-generated ones) are not checked."""
    path = os.path.realpath(inspect.getsourcefile(importlib.import_module(module)) or "")
    if isinstance(obj, type):
        if obj.__module__ != module or obj.__qualname__ != qual:
            return True
        lines, start = inspect.getsourcelines(obj)
        node = ast.parse(textwrap.dedent("".join(lines))).body[0]
        last = start + len(lines) - 1
        for n in getattr(node, "body", []):
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                member = vars(obj).get(n.name)
                if member is None or not _defined_in(_plain(member), path, start, last):
                    return True
        return False
    fn = _plain(obj)
    if getattr(fn, "__qualname__", None) != qual:
        return True
    try:
        lines, start = inspect.getsourcelines(inspect.unwrap(fn))
    except (OSError, TypeError):
        return True
    return not _defined_in(fn, path, start, start + len(lines) - 1)


def collect_runtime_facts() -> dict:
    """the facts of the vLLM this process imports (raises on an import error: ``check`` turns that into a reason)"""
    import vllm

    facts: dict[str, Any] = {"vllm_version": str(getattr(vllm, "__version__", None))}
    for module, qual in DIGEST_TARGETS:
        obj, _owner = _resolve(module, qual)
        fn = obj.__func__ if isinstance(obj, staticmethod | classmethod) else obj
        facts[f"wrapped:{module}:{qual}"] = _is_replaced(module, qual, obj)
        facts[f"digest:{module}:{qual}"] = source_digest(inspect.getsource(inspect.unwrap(fn)))
    for module, qual in FIELD_TARGETS:
        cls, _owner = _resolve(module, qual)
        if hasattr(cls, "__struct_fields__"):
            names = tuple(cls.__struct_fields__)
        elif hasattr(cls, "_fields"):
            names = tuple(cls._fields)
        elif dataclasses.is_dataclass(cls):
            names = tuple(f.name for f in dataclasses.fields(cls))
        else:
            raise TypeError(f"{module}.{qual}: neither a msgspec Struct, a NamedTuple nor a dataclass")
        facts[f"fields:{module}:{qual}"] = names
    for module, qual in PARAM_TARGETS:
        fn, _owner = _resolve(module, qual)
        facts[f"params:{module}:{qual}"] = tuple(inspect.signature(inspect.unwrap(fn)).parameters)
    module, qual = REQUEST_STATE_INIT
    fn, _owner = _resolve(module, qual)
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise TypeError(f"{module}.{qual}: not a function definition")
    facts[f"attrs:{module}:{qual}"] = _init_attrs(node)
    module, qual = OUTPUT_KIND_ENUM
    enum_cls, _owner = _resolve(module, qual)
    facts[f"members:{module}:{qual}"] = tuple(enum_cls.__members__)
    return facts


# ------------------------------------------------------------------------------------------------ the decision


def _diff_names(want: tuple, got: tuple) -> str:
    added = [x for x in got if x not in want]
    removed = [x for x in want if x not in got]
    if not added and not removed:
        return "reordered"
    return ", ".join(([f"added {added}"] if added else []) + ([f"removed {removed}"] if removed else []))


def compare(facts: dict, reviewed: Optional[dict] = None) -> dict:
    """per check: 'ok', 'mismatch: ...' or 'unavailable'"""
    rv = REVIEWED if reviewed is None else reviewed
    out: dict[str, str] = {}
    for key, want in rv.items():
        if key not in facts:
            out[key] = "unavailable"
        elif key.startswith("digest:"):
            got = facts[key]
            out[key] = "ok" if got == want else f"mismatch: digest {got} != reviewed {want}"
        else:
            got = tuple(facts[key])
            out[key] = "ok" if got == tuple(want) else f"mismatch: {_diff_names(tuple(want), got)}"
    for key, value in facts.items():
        if key.startswith("wrapped:"):
            out[key] = "mismatch: wrapped or replaced at run time" if value else "ok"
    module, qual = REQUEST_STATE_INIT
    key = f"attrs:{module}:{qual}"
    if key not in facts:
        out[key] = "unavailable"
    else:
        miss = [a for a in REQUEST_STATE_ATTRS if a not in facts[key]]
        out[key] = "ok" if not miss else f"mismatch: RequestState no longer assigns {miss}"
    module, qual = OUTPUT_KIND_ENUM
    key = f"members:{module}:{qual}"
    if key not in facts:
        out[key] = "unavailable"
    else:
        out[key] = "ok" if "FINAL_ONLY" in facts[key] else "mismatch: no RequestOutputKind.FINAL_ONLY"
    return out


def check_processor(proc: Any) -> list[str]:
    """the engine's output processor: a plain OutputProcessor (no subclass override of process_outputs) whose
    process_outputs is not already replaced on the instance"""
    try:
        from vllm.v1.engine.output_processor import OutputProcessor

        problems = []
        if not isinstance(proc, OutputProcessor):
            problems.append(f"output processor is a {type(proc).__name__}, not vLLM's OutputProcessor")
        elif type(proc).process_outputs is not OutputProcessor.process_outputs:
            problems.append(f"{type(proc).__name__} overrides process_outputs")
        if "process_outputs" in getattr(proc, "__dict__", {}):
            problems.append("process_outputs is already replaced on this output processor instance")
        if not isinstance(getattr(proc, "request_states", None), dict):
            problems.append("output processor has no request_states dict")
        return problems
    except Exception as e:  # noqa: BLE001 - a guard never raises
        return [f"output processor check failed: {type(e).__name__}: {e}"[:200]]


def check(facts: Optional[dict] = None, proc: Any = None, reviewed: Optional[dict] = None) -> CompatResult:
    """whether the lazy final-only output path may be installed; never raises"""
    version = None
    try:
        if facts is None:
            facts = collect_runtime_facts()
        version = facts.get("vllm_version")
        checks = compare(facts, reviewed)
    except Exception as e:  # noqa: BLE001 - an import / inspection failure disables the lazy path
        reason = f"cannot inspect vLLM's output processing ({type(e).__name__}: {e})"[:300]
        return CompatResult(False, (reason,), version, {"collect": reason})
    reasons = [f"{k.split(':', 1)[1]}: {v}" for k, v in sorted(checks.items()) if v != "ok"]
    if str(version or "").removeprefix("v") not in TRACE_TESTED_VLLM:
        reasons.append(f"vLLM {version} is not a trace-tested version {list(TRACE_TESTED_VLLM)}")
    if proc is not None:
        reasons.extend(check_processor(proc))
    return CompatResult(not reasons, tuple(reasons), version, checks)
