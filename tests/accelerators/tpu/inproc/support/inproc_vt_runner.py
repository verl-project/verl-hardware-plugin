# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""vllm-torchtpu's runner functions on CPU, executed from vllm-torchtpu's own source. Test code.

The source is located through ``INPROC_TEST_VLLM_TORCHTPU_SRC`` (the directory that holds the ``vllm_torchtpu``
package) or an installed ``vllm_torchtpu`` package (located, never imported). :func:`build_runner_module` registers
``vllm_torchtpu.runner.tpu_runner`` in ``sys.modules`` as a module that holds the runner's real module-level functions
and the requested ``TPUModelRunner`` methods. Each function is compiled from the source's own syntax tree, so its code
object carries the runner's file name and line numbers (the in-process guards key engine code by them); annotations
are dropped and ``torch.compile(backend="tpu")`` becomes ``backend="eager"`` (dynamo's guards, cache order and
recompile limit do not depend on the backend).
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
import sys
import types
from typing import Optional

VT_SRC_ENV = "INPROC_TEST_VLLM_TORCHTPU_SRC"
RUNNER_MOD = "vllm_torchtpu.runner.tpu_runner"


def vt_root() -> Optional[str]:
    """The ``vllm_torchtpu`` package directory, or None (the tests that need it skip)."""
    src = os.environ.get(VT_SRC_ENV)
    if src:
        root = os.path.join(src, "vllm_torchtpu")
        return root if os.path.isdir(root) else None
    try:
        spec = importlib.util.find_spec("vllm_torchtpu")  # locates the package without importing it
    except (ImportError, ValueError):
        return None
    return os.path.dirname(spec.origin) if spec is not None and spec.origin else None


def skip_reason() -> Optional[str]:
    """Why the runner source cannot be used here (None: it can)."""
    if vt_root() is None:
        return f"vllm-torchtpu's source is not available (set {VT_SRC_ENV})"
    if sys.version_info < (3, 12):
        return "vllm-torchtpu's runner source needs Python >= 3.12"
    return None


def runner_file(root: str) -> str:
    return os.path.join(root, "runner", "tpu_runner.py")


def platform_file(root: str) -> str:
    return os.path.join(root, "platforms", "tpu_platform.py")


def topk_file(root: str) -> str:
    return os.path.join(root, "layers", "adapter", "sample", "top_k_top_p.py")


class _Prepare(ast.NodeTransformer):
    """Annotations dropped (they may name types the stand-in module does not import); ``torch.compile``'s TPU backend
    replaced by dynamo's eager backend."""

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return ast.Pass(lineno=node.lineno, col_offset=node.col_offset)
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)

    def visit_keyword(self, node):
        if node.arg == "backend" and isinstance(node.value, ast.Constant) and node.value.value == "tpu":
            node.value = ast.copy_location(ast.Constant("eager"), node.value)
        return node


def _parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def function_node(tree: ast.Module, name: str, cls: Optional[str] = None) -> ast.FunctionDef:
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def exec_node(node: ast.stmt, path: str, ns: dict, keep_decorators: bool = True):
    """Compile one top-level statement of ``path`` (annotations dropped, the TPU backend replaced) into ``ns``."""
    node = _Prepare().visit(copy.deepcopy(node))
    if not keep_decorators and isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    mod = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(mod, path, "exec"), ns)
    return ns.get(getattr(node, "name", None)) if hasattr(node, "name") else None


def paddings(root: str, max_tokens: int, max_seqs: int, min_seqs: int = 8) -> tuple:
    """(num_tokens_paddings, num_reqs_paddings) with the runner's and the platform's own helpers."""
    import logging

    ns = {"logger": logging.getLogger("inproc_vt_runner"), "MIN_NUM_SEQS": min_seqs}
    for path, names in (
        (platform_file(root), ("_get_exponential_token_paddings",)),
        (runner_file(root), ("_get_req_paddings", "_get_padded_num_reqs_with_upper_limit")),
    ):
        tree = _parse(path)
        for name in names:
            exec_node(function_node(tree, name), path, ns, keep_decorators=False)
    return ns["_get_exponential_token_paddings"](16, max_tokens), ns["_get_req_paddings"](min_seqs, max_seqs)


def validate_request_fn(root: str):
    """The platform's ``validate_request`` (a classmethod there), as a plain function ``(cls, prompt, params)``."""
    path = platform_file(root)
    tree = _parse(path)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and any(isinstance(m, ast.FunctionDef) and m.name == "validate_request" for m in n.body)
    )
    return exec_node(function_node(tree, "validate_request", cls.name), path, {}, keep_decorators=False)


def _logprobs_tensors():
    try:
        from vllm.v1.outputs import LogprobsTensors  # the runner's own return type

        return LogprobsTensors
    except Exception:  # noqa: BLE001 - a NamedTuple with its fields
        from typing import NamedTuple

        import torch

        class LogprobsTensors(NamedTuple):
            logprob_token_ids: torch.Tensor
            logprobs: torch.Tensor
            selected_token_ranks: torch.Tensor
            cu_num_generated_tokens: Optional[list] = None
            cu_num_generated_tokens_tensor: Optional[torch.Tensor] = None

        return LogprobsTensors


def build_runner_module(
    root: str,
    methods: tuple = (),
    functions: tuple = ("_substitute_placeholder_token",),
    topk_blocks: int = 0,
) -> types.ModuleType:
    """``vllm_torchtpu.runner.tpu_runner`` with the runner's real module-level ``functions``, ``SAMPLING_EPS`` and
    ``TPUModelRunner`` carrying the real ``methods`` (each at its own line numbers), registered in ``sys.modules``.
    ``topk_blocks`` is ``envs.TPU_SAMPLER_TOPK_CANDIDATES``."""
    import contextlib
    import functools
    import logging
    import time
    import typing

    import numpy as np
    import torch

    path = runner_file(root)
    tree = _parse(path)
    for pkg in (
        "vllm_torchtpu",
        "vllm_torchtpu.runner",
        "vllm_torchtpu.layers",
        "vllm_torchtpu.layers.adapter",
        "vllm_torchtpu.layers.adapter.sample",
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    mod = types.ModuleType(RUNNER_MOD)
    mod.__file__ = path
    g = mod.__dict__
    logger = logging.getLogger("inproc_vt_runner")
    logger.info_once = logger.info  # type: ignore[attr-defined]
    logger.warning_once = logger.warning  # type: ignore[attr-defined]
    g.update(
        {
            "torch": torch,
            "np": np,
            "contextlib": contextlib,
            "functools": functools,
            "time": time,
            "logger": logger,
            "synchronize_tensors": lambda *a, **k: None,
            "LogprobsTensors": _logprobs_tensors(),
            "Any": typing.Any,
            "envs": types.SimpleNamespace(TPU_SAMPLER_TOPK_CANDIDATES=topk_blocks),
        }
    )
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SAMPLING_EPS" for t in n.targets):
            exec_node(n, path, g)
    tk_path = topk_file(root)
    if os.path.isfile(tk_path):
        tk = types.ModuleType("vllm_torchtpu.layers.adapter.sample.top_k_top_p")
        tk.__file__ = tk_path
        with open(tk_path, encoding="utf-8") as f:
            exec(compile(f.read(), tk_path, "exec"), tk.__dict__)
        sys.modules[tk.__name__] = tk
        g["apply_top_k_top_p"] = getattr(tk, "apply_top_k_top_p", None)
        g["CANDIDATE_BLOCK"] = getattr(tk, "CANDIDATE_BLOCK", None)
    for name in functions:
        exec_node(function_node(tree, name), path, g)
    body = {"__module__": RUNNER_MOD}
    for name in methods:
        body[name] = exec_node(function_node(tree, name, "TPUModelRunner"), path, g)
        g.pop(name, None)
    g["TPUModelRunner"] = type("TPUModelRunner", (), body)
    sys.modules[RUNNER_MOD] = mod
    sys.modules["vllm_torchtpu.runner"].tpu_runner = mod  # type: ignore[attr-defined]
    return mod


def compiled_code(fn):
    """The code object dynamo caches entries on, for a ``torch.compile``'d function (pinned or not)."""
    orig = getattr(fn, "_torchdynamo_orig_callable", None)
    return getattr(orig, "__code__", None)


def cache_entries(fn) -> int:
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    return len(_debug_get_cache_entry_list(compiled_code(fn)))


def stale_grad_garbage(n: int = 1) -> None:
    """generators suspended inside ``torch.no_grad()`` entered while the thread's grad mode was on, each left in a
    reference cycle: a collection finalizes them, and each block's exit sets grad mode ON in whatever code runs the
    collection"""
    import torch

    prev = torch.is_grad_enabled()
    for _ in range(n):

        def gen():
            with torch.no_grad():
                yield

        torch.set_grad_enabled(True)
        box = types.SimpleNamespace()
        box.it = gen()
        box.cycle = box
        next(box.it)
        del box
    torch.set_grad_enabled(prev)


def attach_step_standins(mod: types.ModuleType) -> None:
    """``execute_model`` / ``sample_tokens`` with the runner's real decorators around the two compiled calls: the
    step's bookkeeping (where a collection or a grad-setting call can run), then per chunk the async substitution of
    the previous step's tokens (the kernel) as the real execute_model does it. A step is
    ``SimpleNamespace(chunks=((num_tokens, decodes), ...), num_reqs, collect, flip)``."""
    import gc

    import numpy as np
    import torch

    _pp_step = mod._pp_step

    @torch.no_grad()
    @_pp_step
    def execute_model(self, sched):
        if sched.collect:
            gc.collect()
        if sched.flip:
            torch.set_grad_enabled(True)
        off = 0
        for t, n_dec in sched.chunks:
            self.input_ids = torch.zeros(t, dtype=torch.int32)
            cur = np.arange(n_dec, dtype=np.int32)
            pre = np.arange(off, off + n_dec, dtype=np.int32)
            self.grad_at_kernel.append(bool(torch.is_grad_enabled()))
            self._apply_async_token_substitution(self.input_ids, cur, pre)
            off += n_dec
        return None

    @_pp_step
    def sample_tokens(self, num_reqs):
        logits = (torch.arange(num_reqs * self.vocab_size, dtype=torch.float32).reshape(num_reqs, -1) + self.k).sin()
        tok = self.sample_from_logits(
            logits,
            torch.ones((num_reqs, 1)),
            torch.full((num_reqs, self.vocab_size), 0.5),
            torch.zeros((num_reqs, 1), dtype=torch.int32),
            torch.ones((num_reqs, 1)),
            all_greedy=True,
        )
        self.gather_logprobs(logits, tok)
        self._pre_async_results = types.SimpleNamespace(next_tokens_tpu=tok.reshape(-1).to(torch.int64))
        self.k += 1

    mod.TPUModelRunner.execute_model = execute_model
    mod.TPUModelRunner.sample_tokens = sample_tokens
