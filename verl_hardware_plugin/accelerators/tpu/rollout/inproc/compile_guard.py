# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Keep the engine's compiled programs valid while they share a thread with the trainer.

vllm-torchtpu compiles its runner's functions (``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``) once
for a grid of shapes at build time and expects them to serve every step after that. dynamo guards each program on the
calling thread's state; in process that thread is the trainer's. This module holds the pieces that prevent a guard
from failing:

* :class:`CallPins` pins grad mode off at the call of every compiled engine function (the build refuses unless the
  runner's required functions are pinned);
* :class:`GradFlipWatch` puts grad mode back off when a garbage collection inside an engine call flips it;
* :class:`LogitsAtLoad` installs verl's sampling mask on the vLLM model at model load, before anything compiles, and
  freezes verl's ``monkey_patch_model`` RPC afterwards (a mask set after the precompile recompiles every bucket);
* :class:`RecompileLedger` counts every recompile and fails the run closed when an engine function recompiles after
  warm-up beyond what its precompiled grid allows.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import re
import sys
import traceback
import types
import weakref
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import _SET_GRAD, ENGINE_TLS
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import (
    dist_rank_world,
    log,
    register_switch,
    register_switch_check,
    warn,
)
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.compat import RUNNER_MOD

# A dynamo-compiled code object is the engine's when its file lies under one of these.
ENGINE_CODE_MARKERS = ("/vllm_torchtpu/", "/vllm/")
PIN_MODULE_PREFIX = "vllm_torchtpu."  # the engine's own modules
# Not compiled themselves: they call programs vllm-torchtpu compiles at run time (cached torch.compile closures),
# which no module scan can reach; the dispatcher runs them under the pin instead.
PIN_DISPATCHERS = (("vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op", "copy_mamba_state_rows"),)
# The build refuses without these pinned: the placeholder-token substitution kernel and the two compiled sampler
# helpers, which run under the caller's grad mode.
PIN_REQUIRED = ("_substitute_placeholder_token", "TPUModelRunner.sample_from_logits", "TPUModelRunner.gather_logprobs")
_IS_COMPILING = getattr(getattr(torch, "compiler", None), "is_compiling", None) or (lambda: False)


class EngineCode:
    """Which compiled code objects are the engine's, and their stable keys (function@file:line)."""

    markers: tuple[str, ...] = ENGINE_CODE_MARKERS

    @classmethod
    def key(cls, code: Any) -> str:
        f = str(getattr(code, "co_filename", "?"))
        for m in cls.markers:
            i = f.rfind(m)
            if i >= 0:
                f = f[i + len(m) :]
                break
        else:
            f = os.path.basename(f)
        return f"{getattr(code, 'co_name', '?')}@{f}:{getattr(code, 'co_firstlineno', 0)}"

    @classmethod
    def is_engine(cls, code: Any) -> bool:
        f = str(getattr(code, "co_filename", ""))
        return any(m in f for m in cls.markers)


def engine_cache_entries() -> dict:
    """dynamo cache entries per engine function (every code object dynamo was asked to convert)."""
    out: dict = {}
    try:
        import torch._dynamo.convert_frame as cf
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        for ref in list(cf.input_codes.seen):
            code = ref()
            if code is None or not EngineCode.is_engine(code):
                continue
            n = len(_debug_get_cache_entry_list(code))
            if n:
                k = EngineCode.key(code)
                out[k] = out.get(k, 0) + n
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
    return dict(sorted(out.items()))


def guard_trip(what: str) -> str:
    """The recompile guard's marker line for a cause outside dynamo's recompile accounting, kept as the guard's trip
    when it is armed; returns the line (the caller raises)."""
    r = RECOMPILES
    line = f"RECOMPILE GUARD rank {r.rank if r.rank >= 0 else dist_rank_world()[0]}: {what}"
    if r.armed and r.tripped is None:
        r.tripped = line
    warn(line)
    return line


def _compiled_fn(obj: Any) -> bool:
    """A torch.compile'd function of engine code not pinned yet."""
    if not isinstance(obj, types.FunctionType) or getattr(obj, "_inproc_pin", None) is not None:
        return False
    orig = getattr(obj, "_torchdynamo_orig_callable", None)
    return orig is not None and EngineCode.is_engine(getattr(orig, "__code__", None))


class CallPins:
    """Grad mode pinned off at the call of every compiled engine function.

    ``install()`` scans every loaded ``vllm_torchtpu.*`` module: each module-level compiled function (one wrapper per
    function, its aliases in other engine modules included), each compiled function in the ``__dict__`` of a class
    defined there, and each dispatcher in :data:`PIN_DISPATCHERS` is replaced by a wrapper that calls through inside
    another compiled region and otherwise sets grad mode off for the call if it was on (and back after). Installed
    before the build (the precompile goes through the pins), then frozen: a later scan reports what it would pin but
    replaces nothing (replacing an attribute after the engine compiled changes the object identity its programs
    guard on).
    """

    def __init__(self):
        self.wrappers: dict = {}  # id(original) -> (original, wrapper)
        self.keys: dict = {}  # key -> [calls with grad off, calls with grad on], for functions with a patched site
        self._counters: dict = {}
        self.sites: dict = {}
        self.dispatch: set = set()
        self.scans = 0
        self.errors: list = []
        self.frozen = False
        self.unpinned_after_freeze: dict = {}

    def freeze(self) -> None:
        self.frozen = True

    def key_of(self, fn: Any, dispatch: bool = False) -> str:
        orig = getattr(fn, "_torchdynamo_orig_callable", None) or fn
        code = getattr(orig, "__code__", None)
        k = EngineCode.key(code) if code is not None else f"{getattr(fn, '__qualname__', '?')}@?"
        return ("dispatch:" if dispatch else "") + k

    def _wrap(self, fn: Callable, key: str) -> Callable:
        st = self._counters.setdefault(key, [0, 0])
        grad_on, setg, compiling = torch.is_grad_enabled, _SET_GRAD, _IS_COMPILING

        def pinned(*args, **kwargs):
            if compiling():
                return fn(*args, **kwargs)
            if grad_on():
                st[1] += 1
                setg(False)
                try:
                    return fn(*args, **kwargs)
                finally:
                    setg(True)
            st[0] += 1
            return fn(*args, **kwargs)

        for a in ("__module__", "__name__", "__qualname__", "__doc__"):
            try:
                setattr(pinned, a, getattr(fn, a))
            except Exception:  # noqa: BLE001
                pass
        pinned._inproc_pin = key  # type: ignore[attr-defined]
        pinned._inproc_pinned = fn  # type: ignore[attr-defined]
        orig = getattr(fn, "_torchdynamo_orig_callable", None)
        if orig is not None:  # introspection of the compiled function keeps working
            pinned._torchdynamo_orig_callable = orig  # type: ignore[attr-defined]
        return pinned

    def _wrapper_for(self, fn: Callable, dispatch: bool = False) -> Callable:
        got = self.wrappers.get(id(fn))
        if got is not None and got[0] is fn:
            return got[1]
        key = self.key_of(fn, dispatch)
        w = self._wrap(fn, key)
        self.wrappers[id(fn)] = (fn, w)
        if dispatch:
            self.dispatch.add(key)
        return w

    def _site(self, w: Any, site: str) -> int:
        self.keys.setdefault(w._inproc_pin, self._counters[w._inproc_pin])
        s = self.sites.setdefault(w._inproc_pin, [])
        if site not in s:
            s.append(site)
        return 1

    def install(self) -> dict:
        """The scan; never raises: a site that cannot be read or patched is recorded in ``errors``; the build's
        fail-closed decision is :meth:`required_missing` alone."""
        self.scans += 1
        new = 0
        patch = not self.frozen
        mods = [(n, m) for n, m in sorted(sys.modules.items()) if n.startswith(PIN_MODULE_PREFIX) and m is not None]
        for name, mod in mods:
            d = getattr(mod, "__dict__", None)
            if not isinstance(d, dict):
                continue
            for attr, val in list(d.items()):
                try:
                    if _compiled_fn(val):
                        if not patch:
                            self.unpinned_after_freeze.setdefault(self.key_of(val), f"{name}.{attr}")
                            continue
                        w = self._wrapper_for(val)
                        setattr(mod, attr, w)
                        new += self._site(w, f"{name}.{attr}")
                    elif isinstance(val, type) and getattr(val, "__module__", None) == name:
                        new += self._install_class(name, val, patch)
                except Exception as e:  # noqa: BLE001
                    self._error(f"{name}.{attr}", e)
        for modname, attr in PIN_DISPATCHERS:
            try:
                dmod = sys.modules.get(modname)
                fn = getattr(dmod, attr, None) if dmod is not None else None
                if callable(fn) and getattr(fn, "_inproc_pin", None) is None:
                    if not patch:
                        self.unpinned_after_freeze.setdefault(self.key_of(fn, dispatch=True), f"{modname}.{attr}")
                        continue
                    w = self._wrapper_for(fn, dispatch=True)
                    for name, m2 in mods:  # the dispatcher and its aliases (the runner imports it by name)
                        if getattr(m2, "__dict__", {}).get(attr) is fn:
                            setattr(m2, attr, w)
                            new += self._site(w, f"{name}.{attr}")
            except Exception as e:  # noqa: BLE001
                self._error(f"{modname}.{attr}", e)
        return {
            "pinned_functions": len(self.keys),
            "new_sites": new,
            "scans": self.scans,
            "errors": len(self.errors),
            "frozen": self.frozen,
            "unpinned_after_freeze": dict(sorted(self.unpinned_after_freeze.items())),
        }

    def _install_class(self, name: str, cls: type, patch: bool) -> int:
        new = 0
        for cattr, cval in list(vars(cls).items()):
            try:
                wrapped = isinstance(cval, staticmethod | classmethod)
                raw = cval.__func__ if wrapped else cval
                if not _compiled_fn(raw):
                    continue
                if not patch:
                    self.unpinned_after_freeze.setdefault(self.key_of(raw), f"{name}.{cls.__name__}.{cattr}")
                    continue
                w = self._wrapper_for(raw)
                setattr(cls, cattr, type(cval)(w) if wrapped else w)
                new += self._site(w, f"{name}.{cls.__name__}.{cattr}")
            except Exception as e:  # noqa: BLE001
                self._error(f"{name}.{cls.__name__}.{cattr}", e)
        return new

    def _error(self, site: str, e: BaseException) -> None:
        x = f"{site}: {type(e).__name__}: {e}"[:200]
        if x not in self.errors and len(self.errors) < 20:
            self.errors.append(x)

    def required_missing(self) -> list:
        """:data:`PIN_REQUIRED` not pinned in vllm-torchtpu's runner module ([] when that module is not loaded)."""
        mod = sys.modules.get(RUNNER_MOD)
        if mod is None:
            return []
        miss = []
        for path in PIN_REQUIRED:
            obj: Any = mod
            for part in path.split("."):
                obj = vars(obj).get(part) if isinstance(obj, type) else getattr(obj, part, None)
            if getattr(obj, "_inproc_pin", None) is None:
                miss.append(path)
        return miss

    def mark_warm(self) -> dict:
        """The warm-up record: a re-scan (the modules the build imported) and the pin set."""
        scan = self.install()
        return {
            "pinned_functions": len(self.keys),
            "required_missing": self.required_missing(),
            "runner_loaded": RUNNER_MOD in sys.modules,
            "frozen": self.frozen,
            "unpinned_after_freeze": scan["unpinned_after_freeze"],
            "scan_errors": list(self.errors),
        }


CALL_PINS = CallPins()


class GradFlipWatch:
    """A ``gc.callbacks`` hook: a finalizer that runs during a collection can set grad mode to a stale value in the
    middle of whatever code triggered the collection (an allocation, or dynamo's own post-compile collection inside the
    call being compiled). Inside an engine scope, a collection that ends with grad on puts it back off before the
    engine code continues; outside engine scopes the trainer's autograd state is never touched."""

    def __init__(self):
        self.installed = False

    def _cb(self, phase: str, info: dict) -> None:
        try:
            t = ENGINE_TLS
            if phase == "start":
                t.gc_grad = torch.is_grad_enabled()
                return
            g0, t.gc_grad = t.gc_grad, None
            if g0 is None or t.depth == 0 or t.audit:
                return
            if torch.is_grad_enabled() and not g0:
                _SET_GRAD(False)
        except Exception:  # noqa: BLE001 - a gc callback never raises (e.g. at interpreter shutdown)
            pass

    def install(self) -> bool:
        if not self.installed:
            import gc

            gc.callbacks.append(self._cb)
            self.installed = True
        return True


GRAD_WATCH = GradFlipWatch()


def install_call_pins_and_watch() -> dict:
    """Before anything of the engine compiles (after the in-process patches imported vllm-torchtpu's worker and its
    runner): the call-site pins, which must cover :data:`PIN_REQUIRED` whenever the runner is loaded (else refuse),
    and the GC-time grad watch. The pins are frozen once the runner's required functions are pinned."""
    scan = CALL_PINS.install()
    miss = CALL_PINS.required_missing()
    if miss:
        raise RuntimeError(
            f"the call-site grad pin does not cover {miss} of {RUNNER_MOD}: refusing to build without it"
        )
    GRAD_WATCH.install()
    if RUNNER_MOD in sys.modules:
        CALL_PINS.freeze()
    return scan


class LogitsAtLoad:
    """verl's sampling mask on the vLLM model's ``compute_logits``, installed once at model load and never again.

    verl's ``monkey_patch_model`` RPC (``monkey_patch_compute_logits``) shadows ``compute_logits`` with an instance
    attribute of the model. Sent after the build, it comes after vllm-torchtpu's precompile, whose logits programs are
    guarded on that attribute being absent: every bucket would recompile on its first call. So the build arms the
    RPC's own arguments before the engine exists, a class-level wrapper on vllm-torchtpu's ``TPUWorker.load_model``
    applies them right after the model loads (before profiling, the KV-cache precompile and graph capture) and moves
    the closure's banned-id tensor to the model's device, and :meth:`verify_built` refuses the build when the patch
    did not land on the served model. Afterwards the worker's ``monkey_patch_model`` is frozen with an
    instance-level stub: the same mask is a no-op, any other mask trips the recompile guard's marker and raises.
    """

    def __init__(self):
        self.hooked = False
        self.spec: Optional[dict] = None
        self.applied: Optional[dict] = None
        self.patched: Any = None
        self.model_ref: Any = None
        self.frozen = False
        self.repatch_noop = 0
        self.repatch_refused = 0

    def install(self, worker_cls: Any) -> bool:
        """The class-level hook on ``worker_cls.load_model`` (idempotent); False when the class has no load_model of
        its own."""
        cur = worker_cls.__dict__.get("load_model")
        if cur is None:
            return False
        if getattr(cur, "_inproc_logits", False):
            self.hooked = True
            return True
        logits = self

        @functools.wraps(cur)
        def load_model(worker, *a, **k):
            out = cur(worker, *a, **k)
            if logits.spec is not None and logits.applied is None:
                logits.apply(worker)
            return out

        load_model._inproc_logits = True  # type: ignore[attr-defined]
        worker_cls.load_model = load_model
        self.hooked = True
        return True

    def arm(self, spec: dict) -> None:
        self.spec = {
            "vocab_size": int(spec["vocab_size"]),
            "banned_token_ids": list(spec.get("banned_token_ids") or []),
        }

    def disarm(self) -> None:
        self.spec = None

    @staticmethod
    def _place_banned(cl: Any, model: Any) -> str:
        """verl's closure keeps the banned ids in a cell as a host tensor and moves it to the logits' device on its
        first call (a cell store a compiled program would replay and guard on): done here, before anything compiles."""
        fn: Any = getattr(cl, "__func__", None)
        code = getattr(fn, "__code__", None)
        if code is None or "banned_index" not in code.co_freevars:
            return "no banned_index cell"
        cell = fn.__closure__[code.co_freevars.index("banned_index")]
        t = cell.cell_contents
        if not isinstance(t, torch.Tensor):
            return "none (no banned ids)"
        try:
            dev = (
                next((p.device for n, p in model.named_parameters() if "lm_head" in n or "embed_tokens" in n), None)
                or next(iter(model.parameters())).device
            )
        except StopIteration:
            return f"kept on {t.device} (the model has no parameters)"
        if t.device == dev:
            return f"already on {dev}"
        cell.cell_contents = t.to(dev)
        return f"{t.device} -> {dev}"

    def apply(self, worker: Any) -> dict:
        model = getattr(getattr(worker, "model_runner", None), "model", None)
        if model is None:
            raise RuntimeError("compute_logits at load: the worker has no model_runner.model after load_model")
        spec = dict(self.spec or {})
        rpc = getattr(worker, "monkey_patch_model", None)
        if callable(rpc):
            rpc(**spec)  # verl's extension method itself
            via = "the worker's monkey_patch_model"
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
            from verl.workers.rollout.vllm_rollout.utils import monkey_patch_compute_logits

            monkey_patch_compute_logits(model, spec["vocab_size"], spec["banned_token_ids"])
            patch_vllm_moe_model_weight_loader(model)
            via = "verl's monkey_patch_compute_logits"
        cl = model.__dict__.get("compute_logits")
        if cl is None:
            raise RuntimeError(
                f"compute_logits at load: {via} set no instance attribute compute_logits on {type(model).__name__}"
            )
        moved = self._place_banned(cl, model)
        self.patched, self.model_ref = cl, weakref.ref(model)
        self.applied = {"via": via, "model": type(model).__name__, "banned_index": moved}
        self.applied["rpc_frozen"] = self.freeze_rpc(worker, spec)
        return self.applied

    @staticmethod
    def _mask_key(vocab_size: Any, banned_token_ids: Any) -> tuple:
        return (int(vocab_size), tuple(sorted({int(x) for x in (banned_token_ids or ())})))

    def freeze_rpc(self, worker: Any, spec: dict) -> str:
        """The instance-level stub on ``worker.monkey_patch_model`` (vLLM's collective_rpc resolves the method name on
        the worker instance first)."""
        want = self._mask_key(spec.get("vocab_size"), spec.get("banned_token_ids"))
        logits = self

        def monkey_patch_model(vocab_size: Optional[int] = None, banned_token_ids: Optional[list] = None, **kwargs):
            return logits.on_repatch(want, vocab_size, banned_token_ids, kwargs)

        monkey_patch_model._inproc_frozen = True  # type: ignore[attr-defined]
        try:
            worker.monkey_patch_model = monkey_patch_model
        except Exception as e:  # noqa: BLE001 - reported; the build still verifies the mask
            return f"not frozen ({type(e).__name__}: {e})"[:160]
        self.frozen = True
        return "frozen"

    def on_repatch(self, want: tuple, vocab_size: Any, banned_token_ids: Any, extra: dict) -> None:
        try:
            got = self._mask_key(vocab_size, banned_token_ids) if not extra else None
        except Exception:  # noqa: BLE001
            got = None
        if got == want:
            self.repatch_noop += 1
            return None
        self.repatch_refused += 1
        line = guard_trip(
            f"monkey_patch_model re-sent after the build with another mask (vocab_size={vocab_size!r}, "
            "banned_token_ids="
            f"{banned_token_ids!r}, extra {sorted(extra)}; set at load: vocab_size={want[0]}, banned_token_ids="
            f"{list(want[1])}): applying it would replace compute_logits on the served model and recompile every "
            "logits program; refused"
        )
        raise RuntimeError(line)

    def verify_built(self, worker: Any) -> None:
        """After the engine exists: the patch landed at load on the model the runner serves, else refuse."""
        model = getattr(getattr(worker, "model_runner", None), "model", None)
        bad = []
        if self.applied is None:
            bad.append("the load_model hook never applied it (load_model not called through the hook)")
        elif model is None or self.model_ref() is not model:
            bad.append("the runner's model is not the object patched at load")
        elif model.__dict__.get("compute_logits") is not self.patched:
            bad.append("the model's compute_logits is not the instance attribute set at load")
        if bad:
            raise RuntimeError(
                f"compute_logits at load: {'; '.join(bad)}: refusing to serve (a mask set after the precompile "
                "recompiles every logits program)"
            )


LOGITS_AT_LOAD = LogitsAtLoad()


# ------------------------------------------------------------------------------------------------ the recompile guard

RECOMPILE_GUARD_ENV = register_switch("VERL_TPU_INPROC_RECOMPILE_GUARD")  # "fatal" (default) | "report"
RECOMPILE_GUARD_N = 8  # an engine function recompiling more than 8 times after warm-up (non-shape classes) trips
REVIEWED_TORCH = "2.13"  # the torch release whose private dynamo hooks the ledger was reviewed on
SHAPE_CLASSES = ("size",)  # the reason class of a new input shape


def recompile_guard_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """``fatal`` (default): the guard trips and the run stops; ``report``: recompiles are counted and logged only."""
    env = os.environ if environ is None else environ
    raw = (env.get(RECOMPILE_GUARD_ENV) or "").strip()
    value = raw or "fatal"
    if value not in ("fatal", "report"):
        raise RuntimeError(f"{RECOMPILE_GUARD_ENV}={raw!r}: expected one of ['fatal', 'report']")
    return value


register_switch_check(recompile_guard_mode)


def _dynamo_limit() -> Optional[int]:
    """dynamo's recompile_limit as this context sees it (torch keeps config overrides in a ContextVar)."""
    try:
        import torch._dynamo.config as dc

        return int(dc.recompile_limit)
    except Exception:  # noqa: BLE001
        return None


def _reason_class(reason: str, grad_only: bool = False) -> str:
    """The class of one recompile: 'grad_mode' when an existing cache entry serves this very call under the other
    grad mode; otherwise from dynamo's first failed guard: a tensor size mismatch is 'size' (a new input shape), other
    global state 'global_state:<what>', a dict-contains guard 'dict_absent:<attr>' / 'dict_present:<attr>', a
    constant guard 'equals:<expr>'."""
    if grad_only:
        return "grad_mode"
    r = str(reason or "").strip()
    head, sep, tail = r.partition(": ")
    if sep and "/" in head and len(head) < 24:
        r = tail
    if r.startswith("GLOBAL_STATE changed:"):
        what = r[len("GLOBAL_STATE changed:") :].strip()
        return "new_input+grad_mode" if what == "grad_mode" else f"global_state:{what}"
    for key, cls in (
        ("size mismatch", "size"),
        ("stride mismatch", "stride"),
        ("dtype mismatch", "dtype"),
        ("device mismatch", "device"),
        ("requires_grad mismatch", "requires_grad"),
        ("dispatch key set mismatch", "dispatch_keys"),
        ("___check_obj_id", "id_match"),
        ("___check_type_id", "type_match"),
    ):
        if key in r:
            return cls
    m = re.match(r"(not )?___dict_contains\('([^']+)', ([^)]+?)\.__dict__\)", r)
    if m:
        return f"{'dict_absent' if m.group(1) else 'dict_present'}:{m.group(2)}@{m.group(3)}"
    m = re.match(r"([A-Za-z_][\w\[\]'.]*) == \S+", r)
    if m:
        return f"equals:{m.group(1)[:60]}"
    return r[:80] or "unknown"


def _grad_only(cache_entries: Any, frame: Any) -> bool:
    """True when an existing cache entry's guards pass for this call under the other grad mode. Evaluated only on a
    recompile: the thread's grad mode is flipped around the entries' guard checks and put back."""
    try:
        f_locals = frame.f_locals
        cur = torch.is_grad_enabled()
        t = ENGINE_TLS
        audit0, t.audit = t.audit, True  # a flip on purpose: the GC watch leaves it alone
        _SET_GRAD(not cur)
        try:
            for e in cache_entries or ():
                gm = getattr(e, "guard_manager", None)
                if gm is None or type(gm).__name__ == "DeletedGuardManagerWrapper":
                    continue
                try:
                    if gm.check(f_locals):
                        return True
                except Exception:  # noqa: BLE001
                    continue
            return False
        finally:
            _SET_GRAD(cur)
            t.audit = audit0
    except Exception:  # noqa: BLE001
        return False


class RecompileLedger:
    """dynamo recompile accounting per compiled function, and the fail-closed guard on the engine's functions.

    A wrapper around ``convert_frame.get_and_maybe_log_recompilation_reasons`` (called once per recompile attempt,
    before dynamo's own limit check) counts each recompile with its reason class; :meth:`mark_warm` (the end of the
    build) snapshots the engine functions' cache entries. After warm-up the guard trips on the first engine function
    whose recompiles exceed what its class allows: a shape-class recompile (its own class and every entry's: 'size')
    up to the function's precompiled count (its cache entries at warm-up), any other class up to
    :data:`RECOMPILE_GUARD_N`; or whose recompile meets dynamo's limit. It also trips on a fresh compile of an engine
    function after warm-up (no cache entry to recompile: dynamo's recompile accounting never sees it) and refuses
    ``torch._dynamo.reset`` after warm-up (it would discard every compiled engine program). A trip logs
    ``RECOMPILE GUARD rank r:``; the pump raises it after the step and the sync at the sync. Nothing is raised inside
    dynamo; the trainer's own recompiles are counted, never guarded.
    """

    def __init__(self):
        self.installed = False
        self.unavailable: Optional[str] = None
        self.armed = True  # set from VERL_TPU_INPROC_RECOMPILE_GUARD by install()
        self.n = RECOMPILE_GUARD_N
        self.rank = -1
        self.warm = False
        self.counts: dict = {}  # key -> recompile attempts, all phases
        self.post: dict = {}  # key -> recompile attempts after warm-up
        self.reasons: dict = {}  # key -> Counter(reason class) after warm-up
        self.grad_at: dict = {}  # key -> Counter('on' / 'off'): the grad mode of the call being recompiled
        self.engine_keys: set = set()
        self.warm_entries: dict = {}
        self.tripped: Optional[str] = None
        self.fresh_pre: Counter = Counter()  # engine key -> fresh compiles before warm-up (the build's)
        self.fresh_post: dict = {}  # engine key -> fresh compiles after warm-up
        self.resets: list = []  # {"warm": bool, "caller": site, "refused": bool} per torch._dynamo.reset call
        self.compile_hooked = False
        self.reset_hooked = False
        self.post_shape: dict = {}  # key -> post-warm-up shape-class recompiles
        self.post_other: dict = {}  # key -> post-warm-up recompiles of every other class
        self.shape_first: dict = {}  # key -> the one-off line of its first shape-class recompile

    @property
    def fresh_mode(self) -> str:
        return "fatal" if self.armed else "report"

    def install(self) -> bool:
        self.armed = recompile_guard_mode() == "fatal"
        if self.installed:
            return True
        try:
            import torch._dynamo.convert_frame as cf

            orig = cf.get_and_maybe_log_recompilation_reasons
        except Exception as e:  # noqa: BLE001
            self.unavailable = f"{type(e).__name__}: {e}"[:200]
            return False
        if not getattr(orig, "_inproc_ledger", False):
            ledger = self

            @functools.wraps(orig)
            def hooked(cache_entries, frame, *a, **k):
                reasons = orig(cache_entries, frame, *a, **k)
                try:
                    code = frame.f_code
                    eng = EngineCode.is_engine(code)  # the grad-only check only for the engine's code
                    ledger.on_recompile(
                        code,
                        reasons,
                        len(cache_entries) if cache_entries is not None else None,
                        _grad_only(cache_entries, frame) if eng else False,
                    )
                except Exception:  # noqa: BLE001 - the accounting never breaks a compile
                    pass
                return reasons

            hooked._inproc_ledger = True  # type: ignore[attr-defined]
            cf.get_and_maybe_log_recompilation_reasons = hooked
        self._install_fresh(cf)
        self.installed = True
        return True

    def _install_fresh(self, cf: Any) -> None:
        """Two more hooks, each never in the way of a compile: ``convert_frame._compile`` (a compile that is not a
        recompilation in dynamo's sense is a fresh compile, counted per engine function) and ``torch._dynamo.reset``
        (recorded; refused after warm-up)."""
        ledger = self
        orig_c = getattr(cf, "_compile", None)
        is_rc = getattr(cf, "is_recompilation", None)
        if orig_c is not None and is_rc is not None and not getattr(orig_c, "_inproc_fresh", False):
            try:
                sig = inspect.signature(orig_c)
            except (TypeError, ValueError):
                sig = None
            if sig is not None and "code" in sig.parameters and "cache_size" in sig.parameters:

                @functools.wraps(orig_c)
                def compile_hooked(*a, **k):
                    try:
                        b = sig.bind_partial(*a, **k).arguments
                        code, cs = b.get("code"), b.get("cache_size")
                        if code is not None and cs is not None and not is_rc(cs):
                            ledger.on_fresh(code)
                    except Exception:  # noqa: BLE001 - the accounting never breaks a compile
                        pass
                    return orig_c(*a, **k)

                compile_hooked._inproc_fresh = True  # type: ignore[attr-defined]
                cf._compile = compile_hooked
                self.compile_hooked = True
        elif orig_c is not None and getattr(orig_c, "_inproc_fresh", False):
            self.compile_hooked = True
        try:
            import torch._dynamo as tdyn

            orig_r = tdyn.reset
            if not getattr(orig_r, "_inproc_reset", False):

                @functools.wraps(orig_r)
                def reset_hooked(*a, **k):
                    ledger.on_reset()
                    return orig_r(*a, **k)

                reset_hooked._inproc_reset = True  # type: ignore[attr-defined]
                tdyn.reset = reset_hooked
            self.reset_hooked = True
        except Exception:  # noqa: BLE001
            pass

    def on_fresh(self, code: Any) -> None:
        if not EngineCode.is_engine(code):
            return
        key = EngineCode.key(code)
        self.engine_keys.add(key)
        if not self.warm:
            self.fresh_pre[key] += 1
            return
        self.fresh_post[key] = self.fresh_post.get(key, 0) + 1
        if self.armed:
            if self.tripped is None:
                self.tripped = (
                    f"RECOMPILE GUARD rank {self.rank}: engine function {key} compiled fresh after warm-up (no cache "
                    "entry "
                    "to recompile: a multi-second compile inside an engine call)"
                )
                warn(self.tripped)
        elif self.fresh_post[key] == 1:
            warn(f"fresh compile after warm-up rank {self.rank}: {key} (counted; {RECOMPILE_GUARD_ENV}=report)")

    def on_reset(self) -> None:
        """``torch._dynamo.reset`` was called: recorded with its caller; after warm-up refused (fatal: raises, so the
        caches stay) or performed (report)."""
        caller = "?"
        try:
            fr = [
                f
                for f in traceback.extract_stack(limit=8)[:-2]
                if "torch/_dynamo" not in f.filename and "torch/compiler" not in f.filename
            ]
            if fr:
                caller = f"{os.path.basename(fr[-1].filename)}:{fr[-1].lineno}:{fr[-1].name}"
        except Exception:  # noqa: BLE001
            pass
        refuse = self.warm and self.armed
        self.resets.append({"warm": self.warm, "caller": caller, "refused": refuse})
        if not self.warm:
            return
        line = (
            f"RECOMPILE GUARD rank {self.rank}: torch._dynamo.reset() called after warm-up by {caller}: it discards "
            "every "
            "compiled engine program (each would compile again inside the decode steps)"
            + ("; refused (the caches stay)" if refuse else f"; performed ({RECOMPILE_GUARD_ENV}=report)")
        )
        warn(line)
        if refuse:
            if self.tripped is None:
                self.tripped = line
            raise RuntimeError(line)

    def on_recompile(self, code: Any, reasons: Any, n_entries: Optional[int], grad_only: bool = False) -> None:
        key = EngineCode.key(code)
        eng = EngineCode.is_engine(code)
        cls = _reason_class(reasons[0] if reasons else "", grad_only)
        every = Counter(_reason_class(r) for r in (reasons or ()))  # every entry's first failed guard, classed
        self.counts[key] = self.counts.get(key, 0) + 1
        if eng:
            self.engine_keys.add(key)
        if not self.warm:
            return
        self.post[key] = self.post.get(key, 0) + 1
        self.reasons.setdefault(key, Counter())[cls] += 1
        self.grad_at.setdefault(key, Counter())["on" if torch.is_grad_enabled() else "off"] += 1
        limit = _dynamo_limit()
        at_limit = n_entries is not None and limit is not None and n_entries >= limit
        # shape-class recompiles are allowed up to the function's precompiled count, one more is fatal; any other class
        # keeps the plain count N; dynamo's limit stays fatal
        shape = not grad_only and cls in SHAPE_CLASSES and bool(every) and all(c in SHAPE_CLASSES for c in every)
        cap = int(self.warm_entries.get(key, 0) or 0) if isinstance(self.warm_entries, dict) else 0
        if shape:
            self.post_shape[key] = self.post_shape.get(key, 0) + 1
        else:
            self.post_other[key] = self.post_other.get(key, 0) + 1
        over = self.post_shape[key] > cap if shape else self.post_other[key] > self.n
        if eng and shape and not over and key not in self.shape_first:
            line = (
                f"size-class recompile after warm-up rank {self.rank}: engine function {key} (1 of its precompiled "
                "count "
                f"{cap}: reported only); this recompile's entries by reason class "
                f"{json.dumps(dict(every), sort_keys=True)}; "
                f"cache entries {n_entries}; dynamo recompile_limit {limit}"
            )
            self.shape_first[key] = line
            log(line)
        if eng and self.armed and self.tripped is None and (over or at_limit):
            pin = CALL_PINS.keys.get(key)
            self.tripped = (
                f"RECOMPILE GUARD rank {self.rank}: engine function {key} recompiled {self.post[key]} times after "
                "warm-up "
                f"(guard N {self.n}{'; at dynamo recompile_limit' if at_limit else ''}; "
                + (
                    f"size-class recompiles {self.post_shape[key]} > its precompiled count {cap}"
                    if shape and over
                    else f"non-shape recompiles {self.post_other.get(key, 0)}, "
                    f"size-class {self.post_shape.get(key, 0)} (allowed up to {cap})"
                )
                + f"); reasons {json.dumps(dict(self.reasons[key]), sort_keys=True)}; grad mode at compile "
                f"{json.dumps(dict(self.grad_at[key]), sort_keys=True)}; cache entries {n_entries}; dynamo "
                "recompile_limit "
                f"{limit}; call-site pin "
                + (f"yes (calls with the thread's grad on/off {pin[1]}/{pin[0]})" if pin is not None else "no")
                + f"; this recompile's entries by reason class {json.dumps(dict(every), sort_keys=True)}"
            )
            warn(self.tripped)

    def mark_warm(self, rank: int) -> dict:
        """The warm-up record: the engine functions' cache entries (the baseline every later recompile is held to)."""
        self.rank = rank
        self.warm_entries = engine_cache_entries()
        self.warm = True
        return {
            "installed": self.installed,
            "unavailable": self.unavailable,
            "armed": self.armed,
            "guard_n": self.n,
            "fresh_mode": self.fresh_mode,
            "shape_rule": {
                "mode": "cap",
                "classes": list(SHAPE_CLASSES),
                "cap": "the function's entries at warm-up",
                "other_classes_n": self.n,
            },
            "entries": self.warm_entries,
            "fresh_before_warm": dict(sorted(self.fresh_pre.items())),
            "resets_before_warm": len(self.resets),
        }

    def compile_record(self) -> dict:
        """The per-sync record: post-warm-up recompiles split by class, fresh compiles, resets, and the cache entries no
        hook accounts for (entries now > warm + recompiles + fresh compiles: a compile path that bypassed both hooks)
        or that are gone."""
        now = engine_cache_entries() if self.warm else {}
        extra, lost = {}, {}
        if isinstance(now, dict) and "error" not in now:
            for k, n in now.items():
                exp = int(self.warm_entries.get(k, 0) or 0) + int(self.post.get(k, 0)) + int(self.fresh_post.get(k, 0))
                if n > exp:
                    extra[k] = n - exp
            for k, n in (self.warm_entries or {}).items():
                if isinstance(n, int) and now.get(k, 0) < n:
                    lost[k] = n - now.get(k, 0)
        return {
            "post_warm_shape": {k: self.post_shape[k] for k in sorted(self.post_shape) if k in self.engine_keys},
            "post_warm_other": {k: self.post_other[k] for k in sorted(self.post_other) if k in self.engine_keys},
            "fresh_post_warm": dict(sorted(self.fresh_post.items())),
            "resets_after_warm": sum(1 for r in self.resets if r["warm"]),
            "entries": now,
            "entries_warm": self.warm_entries,
            "unaccounted": extra,
            "lost": lost,
            "tripped": self.tripped is not None,
        }

    def check(self) -> None:
        if self.tripped is not None:
            raise RuntimeError(self.tripped)


RECOMPILES = RecompileLedger()
