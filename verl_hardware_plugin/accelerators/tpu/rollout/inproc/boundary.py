# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""The boundary between the trainer and the in-process engine.

The engine runs on the trainer's own thread, so every piece of thread-local and process-wide state the engine's
compiled programs are guarded on is the trainer's too. :class:`EngineScope` wraps every synchronous call into the
engine and normalizes that state for the call: grad mode off, autocast off, the thread's TorchFunctionModes popped,
inference mode off, a drifted process-wide setting put back to its warm-up value, and torch_tpu's thread-local
DEFER_AND_FUSE eager mode. Everything is put back on exit. The module also holds the build-time process guards:
the identity environment the engine rewrites, torch_tpu's global eager mode, the device-thread and collective
invariants, and the trainer-side libtpu environment check that runs at the platform import.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import threading
from collections import Counter
from collections.abc import Mapping
from typing import Any, Optional

import torch

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import compat
from verl_hardware_plugin.accelerators.tpu.rollout.inproc.common import cfg_get, log, warn

# The identity variables vllm-torchtpu's TPUWorker.init_device / TpuPlatform.check_and_update_config rewrite in the
# calling process: snapshotted before the build, restored and asserted after it.
IDENTITY_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TPU_VISIBLE_CHIPS",
    "TPU_VISIBLE_DEVICES",
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "ALLOW_MULTIPLE_LIBTPU_LOAD",
)
FULL_DETERMINISM_ENV = "VERL_FULL_DETERMINISM"


class IdentityEnv:
    """Snapshot / restore / assert the identity variables around the engine build."""

    def __init__(self, names: tuple[str, ...] = IDENTITY_VARS):
        self.names = names
        self.before: dict[str, Optional[str]] = {}
        self.changed: dict[str, tuple[Optional[str], Optional[str]]] = {}
        self.other_changed: list[str] = []
        self._full_before: dict[str, str] = {}

    def __enter__(self) -> IdentityEnv:
        self.before = {k: os.environ.get(k) for k in self.names}
        self._full_before = dict(os.environ)
        return self

    def __exit__(self, *exc) -> None:
        after = {k: os.environ.get(k) for k in self.names}
        self.changed = {k: (self.before[k], after[k]) for k in self.names if after[k] != self.before[k]}
        self.other_changed = sorted(
            k
            for k in set(os.environ) | set(self._full_before)
            if k not in self.names and os.environ.get(k) != self._full_before.get(k)
        )
        for k, v in self.before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def restored(self) -> bool:
        return {k: os.environ.get(k) for k in self.names} == self.before


# ------------------------------------------------------------------------------------------------ eager mode


def _execution_mode():
    """torch_tpu's public eager-mode API (``torch.tpu``: the global ``eager_mode``, the ``set_eager_mode`` context
    manager and ``EagerMode``), or None without torch_tpu."""
    tpu = getattr(torch, "tpu", None)
    return tpu if tpu is not None and hasattr(tpu, "set_eager_mode") else None


def eager_mode_name() -> str:
    em = _execution_mode()
    if em is None:
        return "n/a"
    try:
        mode = em.eager_mode
        return getattr(mode, "name", str(mode))
    except Exception as e:  # noqa: BLE001
        return f"error {type(e).__name__}"


_ENGINE_MODE: Any = None


def _resolve_engine_mode():
    """torch_tpu's thread-local eager-mode context manager (``torch.tpu.set_eager_mode``) and its DEFER_AND_FUSE
    value, resolved once per process: the engine scope runs on every engine step, and each access through
    ``torch.tpu`` resolves an experimental API. Not cached while torch_tpu is absent, so an early call never pins a
    no-op.
    """
    global _ENGINE_MODE
    if _ENGINE_MODE is not None:
        return _ENGINE_MODE
    em = _execution_mode()
    fn = getattr(em, "set_eager_mode", None)
    mode = getattr(getattr(em, "EagerMode", None), "DEFER_AND_FUSE", None)
    if fn is None or mode is None:
        return None
    _ENGINE_MODE = (fn, mode)
    return _ENGINE_MODE


try:  # the C switch torch.no_grad() itself flips; the public form is the fallback
    _SET_GRAD = torch._C._set_grad_enabled
except AttributeError:  # pragma: no cover - a torch without the private name
    _SET_GRAD = torch.set_grad_enabled


class _EngineThread(threading.local):
    """Per-thread engine state: the depth and kind of the innermost :class:`EngineScope`, ``audit`` while a guard
    re-evaluates cache entries under the other grad mode on purpose, and the grad mode at the start of a GC
    collection on this thread."""

    def __init__(self):
        self.depth = 0
        self.kind: Optional[str] = None
        self.audit = False
        self.gc_grad: Optional[bool] = None


ENGINE_TLS = _EngineThread()
_ANY_AUTOCAST = getattr(torch._C, "_is_any_autocast_enabled", None)
_AUTOCAST_DEVICES: Optional[tuple] = None


def _autocast_devices() -> tuple:
    """The device types autocast is available for in this process (resolved once)."""
    global _AUTOCAST_DEVICES
    if _AUTOCAST_DEVICES is None:
        cands = ["cpu", "cuda", "xpu", "hpu", "mps", "xla", "tpu"]
        try:
            cands.append(torch._C._get_privateuse1_backend_name())
        except Exception:  # noqa: BLE001
            pass
        ok = []
        for d in dict.fromkeys(cands):
            try:
                if torch.amp.is_autocast_available(d):
                    ok.append(d)
            except Exception:  # noqa: BLE001
                continue
        _AUTOCAST_DEVICES = tuple(ok)
    return _AUTOCAST_DEVICES


def autocast_off() -> list:
    """Enter autocast disabled for every device type that has it enabled on this thread; returns the contexts."""
    cms = []
    for d in _autocast_devices():
        try:
            if torch.is_autocast_enabled(d):
                cm = torch.autocast(d, enabled=False)
                cm.__enter__()
                cms.append(cm)
        except Exception:  # noqa: BLE001
            continue
    return cms


def _gs_deterministic() -> tuple:
    return (
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    )


def _gs_set_deterministic(v: tuple) -> None:
    torch.use_deterministic_algorithms(bool(v[0]), warn_only=bool(v[1]))


# dynamo's GLOBAL_STATE components a process-wide setter can put back for one engine call: (name, getter, setter).
# Grad mode and autocast are pinned by EngineScope itself.
_GS_COMPONENTS = (
    ("num_threads", torch.get_num_threads, torch.set_num_threads),
    ("default_dtype", torch.get_default_dtype, torch.set_default_dtype),
    ("deterministic_algorithms", _gs_deterministic, _gs_set_deterministic),
    ("float32_matmul_precision", torch.get_float32_matmul_precision, torch.set_float32_matmul_precision),
)


class GlobalStateDrift:
    """dynamo guards every compiled program on more global state than grad mode: the deterministic-algorithms flags,
    the float32 matmul precision, num_threads, the default dtype, torch_function, autocast. :meth:`arm` snapshots
    dynamo's own ``GlobalStateGuard`` and the values of the process-wide components at warm-up; an engine entry whose
    check fails sets the differing components back to their warm-up values for the call (:meth:`restore`), so a
    drift in the trainer is a no-op for the engine instead of a recompile of every compiled program."""

    def __init__(self):
        self.snap: Any = None
        self.unavailable: Optional[str] = None
        self.warm_values: Optional[dict] = None

    def arm(self) -> None:
        try:
            self.snap = torch._C._dynamo.guards.GlobalStateGuard()
        except Exception as e:  # noqa: BLE001
            self.snap, self.unavailable = None, f"{type(e).__name__}: {e}"[:160]
        vals = {}
        for name, get, _set in _GS_COMPONENTS:
            try:
                vals[name] = get()
            except Exception:  # noqa: BLE001
                continue
        self.warm_values = vals

    def restore(self) -> list:
        """The differing components set back to their warm-up values; returns the undo list [(setter, the caller's
        value)] that EngineScope applies on exit (never raises)."""
        undo: list = []
        wv = self.warm_values or {}
        for name, get, set_ in _GS_COMPONENTS:
            if name not in wv:
                continue
            try:
                cur = get()
                if cur != wv[name]:
                    set_(wv[name])
                    undo.append((set_, cur))
            except Exception:  # noqa: BLE001
                continue
        return undo


GS_DRIFT = GlobalStateDrift()

_LEN_FN_STACK: Any = getattr(torch._C, "_len_torch_function_stack", None)
_POP_FN_MODE: Any = getattr(torch._C, "_pop_torch_function_stack", None)
_PUSH_FN_MODE: Any = getattr(torch._C, "_push_on_torch_function_stack", None)
_INFERENCE_ON: Any = getattr(torch._C, "is_inference_mode_enabled", None) or getattr(
    torch, "is_inference_mode_enabled", None
)


class EngineEntry:
    """What the outermost engine entry puts right for the call: TorchFunctionModes on the calling thread's mode
    stack (a ``torch.device(...)`` context is one) fail dynamo's mode-stack guard on every compiled program and are
    popped for the call; inference mode is switched off (tensors made under it carry another dispatch key set)."""

    def __init__(self):
        self.on = _LEN_FN_STACK is not None and _POP_FN_MODE is not None and _PUSH_FN_MODE is not None

    def pop_modes(self) -> list:
        """Every TorchFunctionMode on this thread's stack, top first; the stack is never left half-popped."""
        modes: list = []
        try:
            while _LEN_FN_STACK() > 0:
                modes.append(_POP_FN_MODE())
        except BaseException:
            self.push_back(modes)
            raise
        return modes

    @staticmethod
    def push_back(modes: list) -> None:
        for m in reversed(modes or ()):
            _PUSH_FN_MODE(m)


ENGINE_ENTRY = EngineEntry()


class EngineScope:
    """The engine boundary: every synchronous call into the in-process engine runs inside one.

    On entry: grad mode off (dynamo guards every program on the thread's grad mode, and in process the engine's calls
    inherit the trainer thread's), an enabled autocast entered disabled, at the outermost entry the thread's
    TorchFunctionModes popped and inference mode off (left first: c10's InferenceMode(false) turns grad mode on, so
    grad is pinned off after it), a drifted global state restored for the call (:class:`GlobalStateDrift`), and,
    unless ``eager=False``, torch_tpu's thread-local DEFER_AND_FUSE scope. Everything is put back on exit, in reverse
    order, an exception included; re-entrant. Entered and left around synchronous calls only, never across an await:
    grad mode and torch_tpu's mode stack are per thread.
    """

    __slots__ = ("kind", "eager", "prev", "cm", "prev_kind", "ac", "fm", "im", "gs")

    def __init__(self, kind: str = "engine", eager: bool = True):
        self.kind = kind
        self.eager = eager
        self.prev: Optional[bool] = None
        self.cm: Any = None
        self.prev_kind: Optional[str] = None
        self.ac: Optional[list] = None
        self.fm: Optional[list] = None
        self.im: Any = None
        self.gs: Optional[list] = None

    def __enter__(self) -> EngineScope:
        t = ENGINE_TLS
        outer = t.depth == 0
        prev = torch.is_grad_enabled()
        if outer and ENGINE_ENTRY.on and _INFERENCE_ON is not None and _INFERENCE_ON():
            im = torch.inference_mode(False)
            im.__enter__()
            self.im = im
            prev = torch.is_grad_enabled()
        self.prev = prev
        if prev:
            _SET_GRAD(False)
        self.prev_kind = t.kind
        t.depth += 1
        t.kind = self.kind
        try:
            if _ANY_AUTOCAST is not None and _ANY_AUTOCAST():
                self.ac = autocast_off()
            if outer and ENGINE_ENTRY.on and _LEN_FN_STACK():
                self.fm = ENGINE_ENTRY.pop_modes()
            snap = GS_DRIFT.snap
            if snap is not None and not snap.check():
                self.gs = GS_DRIFT.restore()
            if self.eager:
                em = _resolve_engine_mode()
                if em is not None:
                    cm = em[0](em[1])
                    cm.__enter__()
                    self.cm = cm
        except BaseException:
            self._undo()
            raise
        return self

    def _undo(self) -> None:
        gs, self.gs = self.gs, None
        fm, self.fm = self.fm, None
        ac, self.ac = self.ac, None
        im, self.im = self.im, None
        try:
            for set_, old in reversed(gs or ()):
                with contextlib.suppress(Exception):
                    set_(old)
            if fm:
                EngineEntry.push_back(fm)
            for cm in reversed(ac or ()):
                cm.__exit__(None, None, None)
        finally:
            t = ENGINE_TLS
            t.depth -= 1
            t.kind = self.prev_kind
            _SET_GRAD(bool(self.prev))
            if im is not None:
                im.__exit__(None, None, None)

    def __exit__(self, et, ev, tb) -> None:
        try:
            cm, self.cm = self.cm, None
            if cm is not None:
                cm.__exit__(et, ev, tb)
        finally:
            self._undo()


def engine_mode_ctx(kind: str = "engine") -> EngineScope:
    """The scope every synchronous engine call runs in (grad off, the engine's eager mode)."""
    return EngineScope(kind)


def engine_grad_ctx(kind: str = "build") -> EngineScope:
    """Grad mode off only, no eager-mode scope: the engine build, which runs under vllm-torchtpu's own platform
    activation (its global DEFER_AND_FUSE, put back to the trainer's by :class:`GlobalEagerMode` afterwards)."""
    return EngineScope(kind, eager=False)


def snapshot_engine_global_state() -> None:
    """GlobalStateDrift's warm-up snapshot under the engine's own modes (grad off, autocast off)."""
    prev = torch.is_grad_enabled()
    _SET_GRAD(False)
    ac = autocast_off() if (_ANY_AUTOCAST is not None and _ANY_AUTOCAST()) else []
    try:
        GS_DRIFT.arm()
    finally:
        for cm in reversed(ac):
            cm.__exit__(None, None, None)
        _SET_GRAD(prev)


TRAINER_EAGER_AT_START: Any = None  # torch_tpu's global eager mode of this trainer process before any vLLM import
TRAINER_EAGER_AT_START_NAME = "n/a"
TRAINER_EAGER_SNAPSHOT_WHERE = "never"


def snapshot_trainer_eager_mode(where: str) -> str:
    """Record the trainer's own global eager mode once, as early as possible (the platform import on a trainer rank,
    before vllm_torchtpu can be imported). Later snapshots keep the first value. Returns its name."""
    global TRAINER_EAGER_AT_START, TRAINER_EAGER_AT_START_NAME, TRAINER_EAGER_SNAPSHOT_WHERE
    if TRAINER_EAGER_AT_START is not None:
        return TRAINER_EAGER_AT_START_NAME
    em = _execution_mode()
    if em is None:
        return "n/a"
    try:
        TRAINER_EAGER_AT_START = em.eager_mode
    except Exception as e:  # noqa: BLE001
        return f"error {type(e).__name__}"
    TRAINER_EAGER_AT_START_NAME = eager_mode_name()
    TRAINER_EAGER_SNAPSHOT_WHERE = where + (
        " (vllm_torchtpu already imported)" if "vllm_torchtpu" in sys.modules else ""
    )
    return TRAINER_EAGER_AT_START_NAME


class GlobalEagerMode:
    """Put torch_tpu's global eager mode back after the engine build (vllm-torchtpu's platform activation sets it to
    DEFER_AND_FUSE for the rest of the process; the trainer relies on its own scopes).

    The target is the trainer's mode recorded at the platform import (:func:`snapshot_trainer_eager_mode`), not the
    mode at ``__enter__``, which can already be vLLM's. Without an earlier snapshot it falls back to the ``__enter__``
    value and says so (``target_source``)."""

    def __init__(self):
        self.before = "n/a"
        self.after_build = "n/a"
        self.restored = "n/a"
        self.target = "n/a"
        self.target_source = "n/a"
        self._raw = None

    def __enter__(self) -> GlobalEagerMode:
        em = _execution_mode()
        if em is not None:
            self._raw = em.eager_mode
        self.before = eager_mode_name()
        return self

    def __exit__(self, *exc) -> None:
        self.after_build = eager_mode_name()
        em = _execution_mode()
        if em is None:
            return
        if TRAINER_EAGER_AT_START is not None:
            target, self.target_source = TRAINER_EAGER_AT_START, f"trainer snapshot at {TRAINER_EAGER_SNAPSHOT_WHERE}"
        else:
            target, self.target_source = self._raw, "late snapshot at the build (no platform-import snapshot)"
        if target is not None:
            em.eager_mode = target
        self.target = getattr(target, "name", str(target))
        self.restored = eager_mode_name()

    def ok(self) -> bool:
        return self.restored == self.target or self.restored == "n/a"


class EagerGuard:
    """Every trainer entry point checks torch_tpu's global eager mode against the trainer's start value; a mismatch is
    put back (and logged): the trainer then runs in its own mode."""

    def __init__(self):
        self.fixed = 0

    def check(self, where: str) -> None:
        if TRAINER_EAGER_AT_START is None:
            return
        em = _execution_mode()
        if em is None:
            return
        try:
            now = em.eager_mode
        except Exception:  # noqa: BLE001
            return
        if now == TRAINER_EAGER_AT_START:
            return
        self.fixed += 1
        em.eager_mode = TRAINER_EAGER_AT_START
        warn(
            f"eager guard pid {os.getpid()}: global eager mode {getattr(now, 'name', str(now))} at {where} != the "
            f"trainer's {TRAINER_EAGER_AT_START_NAME}: restored ({self.fixed} so far)"
        )


EAGER_GUARD = EagerGuard()


class DeviceThreads:
    """The OS thread of every device-work entry point on this rank (trainer and engine): all must be one thread."""

    def __init__(self):
        self.counts: Counter = Counter()
        self._lock = threading.Lock()
        self._seen: set = set()

    def record(self, kind: str) -> None:
        tid = threading.get_ident()
        if (tid, kind) in self._seen:  # the per-step hot path: no lock once this (thread, kind) is known
            self.counts[tid] += 1
            return
        with self._lock:
            self.counts[tid] += 1
            self._seen.add((tid, kind))

    def single(self) -> bool:
        return len(self.counts) == 1


DEVICE_THREADS = DeviceThreads()


class CollectiveCounter:
    """vLLM ``GroupCoordinator`` collectives with world_size > 1 while the engine steps (armed by the pump).

    Patches the coordinator's communication methods once per process (a pass-through when not armed). At TP=1 every
    engine-step collective goes to size-1 groups, which vLLM short-circuits; a call on a larger group (vLLM's world
    group spans the trainer's ranks) would be a cross-rank sync point inside one rank's decode loop.
    """

    METHODS = (
        "all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv", "gather", "broadcast",
        "broadcast_object", "broadcast_object_list", "barrier", "send", "recv", "send_object", "recv_object",
        "send_tensor_dict", "recv_tensor_dict", "broadcast_tensor_dict",
    )  # fmt: skip

    def __init__(self):
        self.armed = False
        self.big: Counter = Counter()
        self.small = 0
        self.installed = False
        self._armed_ctx: Optional[_Armed] = None

    def install(self) -> bool:
        if self.installed:
            return True
        try:
            from vllm.distributed import parallel_state as ps
        except Exception:  # noqa: BLE001
            return False
        cls = ps.GroupCoordinator
        for name in self.METHODS:
            orig = getattr(cls, name, None)
            if orig is None or getattr(orig, "_inproc_counted", False):
                continue
            setattr(cls, name, self._wrap(name, orig))
        self.installed = True
        return True

    def _wrap(self, name, orig):
        counter = self

        @functools.wraps(orig)
        def wrapper(self_, *args, **kwargs):
            if counter.armed:
                if getattr(self_, "world_size", 1) > 1:
                    counter.big[f"{getattr(self_, 'unique_name', '?')}.{name}"] += 1
                else:
                    counter.small += 1
            return orig(self_, *args, **kwargs)

        wrapper._inproc_counted = True  # type: ignore[attr-defined]
        return wrapper

    def arm(self) -> _Armed:
        if self._armed_ctx is None:
            self._armed_ctx = _Armed(self)
        return self._armed_ctx


class _Armed:
    """CollectiveCounter.arm()'s reusable scope: nests, restores the previous state on exit."""

    __slots__ = ("counter", "depth")

    def __init__(self, counter: CollectiveCounter):
        self.counter = counter
        self.depth = 0

    def __enter__(self) -> _Armed:
        self.depth += 1
        self.counter.armed = True
        return self

    def __exit__(self, *exc) -> None:
        self.depth -= 1
        self.counter.armed = self.depth > 0


class _NoArm:
    """The collective counter's scope on a pump step between two sampled ones: nothing armed."""

    __slots__ = ()

    def __enter__(self) -> _NoArm:
        return self

    def __exit__(self, *exc) -> None:
        return None


COLLECTIVES = CollectiveCounter()
NOT_ARMED = _NoArm()


def full_determinism_problem(config: Any = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """verl's worker extension of the engine (constructed in this trainer process) runs enable_full_determinism when
    VERL_FULL_DETERMINISM is set: the trainer's RNG streams would be reseeded and deterministic algorithms switched on
    for the trainer too. Any value but "" / "0", or rollout.full_determinism=True, is refused."""
    env = os.environ if environ is None else environ
    raw = (env.get(FULL_DETERMINISM_ENV) or "").strip()
    cfg = cfg_get(config, "full_determinism", False)
    if raw not in ("", "0") or bool(cfg):
        return (
            f"{FULL_DETERMINISM_ENV}={raw!r}, rollout.full_determinism={cfg!r}: verl's worker extension would reseed "
            "the "
            "trainer's RNG and switch deterministic algorithms on in this trainer process; refusing to build"
        )
    return None


def prefix_caching_problem(config: Any = None) -> Optional[str]:
    """The in-process adapter never resets vLLM's prefix cache at a weight update, so a prefix cache would serve KV
    computed under the previous weights after every sync: refused."""
    if bool(cfg_get(config, "enable_prefix_caching", False)):
        return (
            "rollout.enable_prefix_caching=True: the in-process engine's reset_prefix_cache is a no-op, so KV cached "
            "under the previous weights would be reused after a weight update; refusing to build"
        )
    return None


# ----------------------------------------------------------------------------------------------- the libtpu environment


LIBTPU_AT_CHECK: Optional[str] = None  # this process's LIBTPU_INIT_ARGS when check_trainer_libtpu_env ran (before PJRT)
ENV_OVERRIDE_PROBE: Optional[dict] = None  # the env_override probe of that check (the compat line reports it)
ENV_AT_CHECK: Optional[dict] = None  # the environment names env_override touches, at the same point
# Sticky problems of the platform-import check: an exception raised at the platform import never reaches verl (the
# plugin's backend loader swallows import errors), so the engine build raises on them instead, inside init_model.
LIBTPU_PROBLEMS: list = []


def check_trainer_libtpu_env(environ: Optional[Mapping[str, str]] = None, strict: bool = True) -> dict:
    """On a trainer rank of a colocated run, before vllm_torchtpu is imported and before the first device touch.

    In process the trainer creates the PJRT client and libtpu reads ``LIBTPU_INIT_ARGS`` once, so whatever
    vllm-torchtpu's ``env_override`` would add at its import must already be among the trainer's flags. The check
    fingerprints env_override's import-time code without importing it (:func:`compat.env_override_probe`): a reviewed
    structure gives the early prediction (:func:`compat.would_add_libtpu_flags`), an unreviewed one refuses. It also
    requires ``TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS`` pinned and vllm_torchtpu not yet imported, records
    the environment names env_override touches for the build-time comparison (:func:`compat.env_after_build_problems`)
    and snapshots the trainer's global eager mode.
    """
    global LIBTPU_AT_CHECK, ENV_AT_CHECK, ENV_OVERRIDE_PROBE
    env = os.environ if environ is None else environ
    args = env.get("LIBTPU_INIT_ARGS", "")
    LIBTPU_AT_CHECK = args
    ENV_AT_CHECK = {k: env.get(k) for k in compat.env_effect_names()}
    probe = ENV_OVERRIDE_PROBE = compat.env_override_probe()
    extra = compat.would_add_libtpu_flags(args, env) if probe.get("reviewed") else []
    rank = env.get("RANK", "?")
    eager_start = snapshot_trainer_eager_mode("platform import")
    problems = []
    if probe.get("present") and not probe.get("reviewed"):
        problems.append(
            f"vllm_torchtpu.env_override import-time structure {probe.get('fingerprint')} is not a reviewed one "
            f"({sorted(compat.REVIEWED_ENV_OVERRIDE)}): review its import-time code and add the fingerprint with a test"
        )
    if extra:
        problems.append(f"vllm_torchtpu.env_override would add {extra} after the trainer created PJRT")
    if env.get(compat.MATERIALIZE_ENV) is None:
        problems.append(f"{compat.MATERIALIZE_ENV} is not pinned")
    if "vllm_torchtpu" in sys.modules:
        problems.append(
            "vllm_torchtpu is already imported: the trainer's eager mode / libtpu flags may already be vLLM's"
        )
    info = {
        "rank": rank,
        "libtpu_init_args": args,
        "would_add": extra,
        "env_override": probe,
        compat.MATERIALIZE_ENV: env.get(compat.MATERIALIZE_ENV),
        compat.SC_ENV: env.get(compat.SC_ENV),
        "eager_at_start": eager_start,
        "problems": problems,
    }
    text = (
        f"libtpu rank {rank}: effective LIBTPU_INIT_ARGS={args} | env_override would add {extra or 'nothing'} | "
        f"{compat.MATERIALIZE_ENV}={env.get(compat.MATERIALIZE_ENV)} {compat.SC_ENV}={env.get(compat.SC_ENV)} | "
        f"env_override structure {probe.get('fingerprint') or probe.get('why')} "
        f"{'reviewed' if probe.get('reviewed') else 'not reviewed' if probe.get('present') else ''} | "
        f"trainer global eager mode at start {eager_start} | "
        f"{'OK' if not problems else 'FAILED: ' + '; '.join(problems)}"
    )
    (warn if problems else log)(text)
    LIBTPU_PROBLEMS[:] = problems
    if problems and strict:
        raise RuntimeError(f"trainer libtpu environment check failed: {'; '.join(problems)}")
    return info
