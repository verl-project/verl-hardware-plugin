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
  freezes verl's ``monkey_patch_model`` RPC afterwards (a mask set after the precompile recompiles every bucket).
"""

from __future__ import annotations

import functools
import os
import sys
import types
import weakref
from typing import Any, Callable, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc.boundary import _SET_GRAD, ENGINE_TLS
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import dist_rank_world, warn
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
    """The recompile guard's marker line for a cause outside dynamo's recompile accounting; returns the line (the
    caller raises)."""
    line = f"RECOMPILE GUARD rank {dist_rank_world()[0]}: {what}"
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
