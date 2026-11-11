# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""Default-off tests for the TPU in-process (colocated) rollout (``rollout/inproc``).

The colocated rollout is opt-in: ``VERL_TPU_ROLLOUT_INPROC=1`` turns it on. With the switch unset, or set to any
other value, importing verl and this plugin must take no colocation path and must leave the process in the state it
would be in without the feature.

Each scenario runs in a fresh interpreter (``python <this file> --child <json>``), because the property is about
import-time and registration-time process state. The child records:

* every attempt to import the colocation package, through a finder on ``sys.meta_path`` (which can also block it);
* every ``os.environ`` write made by the plugin's own code (the writing frame's file is in the plugin's package);
* a fingerprint of the state the feature changes when it is on (at registration or at the engine build): verl's
  rollout registry and replica loader, ``ActorRolloutRefWorker``'s methods, ``CheckpointEngineManager.update_weights``,
  the TPU platform's worker environment, the dynamo hooks and config, vLLM's ``GroupCoordinator`` collectives,
  ``torch.distributed``'s collectives, ``gc``, torch's global state, ``os.environ`` and ``sys.modules``.

Comparing switch-off runs with the unset run cannot see a change that does not depend on the switch. Three checks need
no other plugin tree: with the switch off the plugin's own code writes no environment variable; no worker group gets a
variable of the colocated trainer's pin; and everything the plugin leaves alone without the feature equals a run that
never imports the plugin.

Optional differential against a plugin tree without the feature: set ``INPROC_DEFAULT_OFF_BASELINE`` to that tree's
root (the directory that contains ``verl_hardware_plugin``). Optional evidence dump: ``INPROC_TEST_EVIDENCE_DIR``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

INPROC_ENV = "VERL_TPU_ROLLOUT_INPROC"
ROLE_ENV = "VERL_TPU_INPROC_ROLE"
PARITY_EVERY_ENV = "VERL_TPU_INPROC_PARITY_EVERY"
SWITCHES = (INPROC_ENV, ROLE_ENV, PARITY_EVERY_ENV)

# The colocation code.
COLOC_MODULES = ("verl_hardware_plugin.accelerators.tpu.rollout.inproc",)
PLUGIN_PKG = "verl_hardware_plugin"
PLATFORM_MOD = "verl_hardware_plugin.accelerators.tpu.platform_tpu"
REGISTRATION_MOD = "verl_hardware_plugin.accelerators.tpu.registration"
WORKER_PREFIXES = ("actor_rollout", "actor", "ref", "critic", "rollout_0")
TRAINER_PREFIXES = ("actor_rollout", "actor", "ref", "critic")
TRAINER_ENV_KEYS = (
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
    "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES",
    ROLE_ENV,
)
# Environment writes the plugin's own (non-colocation) code may make with the switch off: none today.
ALLOWED_PLUGIN_ENV_WRITES: tuple = ()
# Fingerprint fields the plugin leaves alone when the feature is off: equal to a run that never imports the plugin
# (measured on a plugin tree without the feature). The rest the plugin sets up anyway (replica loader, its
# update_weights hook, the worker environment, its modules) or verl's imports change (os.environ through modules the
# plugin imports).
PLUGIN_FREE_KEYS = (
    "rollout_registry",
    "worker_methods",
    "worker_attr_names",
    "dynamo_config",
    "dynamo_hooks",
    "dist_functions",
    "vllm",
    "gc",
    "torch_state",
    "threads",
    "meta_path",
    "env_removed",
)
DIST_FUNCTIONS = (
    "barrier",
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "all_gather_into_tensor",
    "broadcast",
    "broadcast_object_list",
    "reduce_scatter_tensor",
    "new_group",
    "new_subgroups_by_enumeration",
)
COORDINATOR_METHODS = (
    "all_reduce",
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "gather",
    "broadcast",
    "broadcast_object",
    "broadcast_object_list",
    "barrier",
    "send",
    "recv",
    "send_object",
    "recv_object",
    "send_tensor_dict",
    "recv_tensor_dict",
    "broadcast_tensor_dict",
)

# Switch values that must leave the feature off. The plugin's registration and platform hooks enable it on the exact
# string "1" only.
OFF_ARMS = {
    "unset": {},
    "inproc_0": {INPROC_ENV: "0"},
    "inproc_empty": {INPROC_ENV: ""},
    "inproc_true": {INPROC_ENV: "true"},
    "inproc_yes": {INPROC_ENV: "yes"},
    "inproc_space_1": {INPROC_ENV: " 1"},
    "inproc_1_space": {INPROC_ENV: "1 "},
    "role_trainer_only": {ROLE_ENV: "trainer"},
    "run_switch_only": {PARITY_EVERY_ENV: "1", ROLE_ENV: "trainer"},
}
ON_ARMS = {
    "on_trainer": {INPROC_ENV: "1", ROLE_ENV: "trainer"},
}
# Keys that differ by construction when the colocation module is imported on purpose.
MODULE_SET_KEYS = ("sys_modules", "coloc_modules", "coloc_import_attempts")
# The environment every child starts from, taken before anything in this process imports verl, vLLM or the plugin
# (their imports set variables of their own, which would leak into the children started after them).
BASE_ENV = {k: v for k, v in os.environ.items() if k not in SWITCHES}


# --------------------------------------------------------------------------------------------- child side


class _ImportWatch(importlib.abc.MetaPathFinder):
    """Records (and optionally blocks) every import of the colocation code; optionally blocks the whole plugin."""

    def __init__(self, block: bool, block_plugin: bool = False):
        self.block = block
        self.block_plugin = block_plugin
        self.seen: list[str] = []

    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in COLOC_MODULES):
            self.seen.append(name)
            if self.block:
                raise ImportError(f"{name} is blocked by the default-off test")
        if self.block_plugin and (name == PLUGIN_PKG or name.startswith(PLUGIN_PKG + ".")):
            raise ImportError(f"{name} is blocked by the default-off test (no plugin)")
        return None


class _EnvWriteWatch:
    """Records every ``os.environ`` write whose writing frame is the plugin's own code. Writes made by the modules the
    plugin imports (verl, vLLM, torch) are attributed to them, not to the plugin."""

    # the mapping machinery, by module name (both can be frozen: co_filename "<frozen os>"); _collections_abc names
    # itself collections.abc
    MACHINERY = ("os", "_collections_abc", "collections.abc")

    def __init__(self, root):
        self.root = os.path.join(root, "") if root else None
        self.writes: list[str] = []

    def install(self) -> None:
        cls = type(os.environ)
        orig_set, orig_del = cls.__setitem__, cls.__delitem__
        watch = self

        def setitem(env, key, value):
            watch.note("set", key)
            orig_set(env, key, value)

        def delitem(env, key):
            watch.note("del", key)
            orig_del(env, key)

        cls.__setitem__, cls.__delitem__ = setitem, delitem

    def note(self, op: str, key) -> None:
        frame = sys._getframe(2)
        while frame is not None and frame.f_globals.get("__name__") in self.MACHINERY:
            frame = frame.f_back
        path = frame.f_code.co_filename if frame is not None else ""
        if self.root and path.startswith(self.root):
            self.writes.append(f"{op} {key} ({os.path.relpath(path, self.root)}:{frame.f_code.co_name})")

    def selftest(self) -> bool:
        """Positive control: a setdefault / pop pair made from this file is seen and attributed to it."""
        root, self.root = self.root, os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
        try:
            os.environ.setdefault("INPROC_TEST_ENV_WATCH", "1")
            os.environ.pop("INPROC_TEST_ENV_WATCH", None)
            seen = list(self.writes)
        finally:
            self.root, self.writes = root, []
        me = os.path.basename(__file__)
        return seen == [f"set INPROC_TEST_ENV_WATCH ({me}:selftest)", f"del INPROC_TEST_ENV_WATCH ({me}:selftest)"]


def _canon(value):
    if isinstance(value, set | frozenset):
        return "{" + ", ".join(sorted(_canon(v) for v in value)) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_canon(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{_canon(k)}: {_canon(v)}" for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])))
            + "}"
        )
    if isinstance(value, types.CodeType):
        return _code_digest(value)
    return repr(value)


def _code_digest(code: types.CodeType) -> str:
    h = hashlib.sha256()
    h.update(code.co_code)
    h.update(repr((code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars)).encode())
    h.update(_canon(tuple(code.co_consts)).encode())
    return h.hexdigest()[:16]


def _code_id(obj) -> str:
    fn = getattr(obj, "__func__", obj)
    code = getattr(fn, "__code__", None)
    if code is None:
        return f"<{type(obj).__name__}>"
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}@{_code_digest(code)}"


def _class_name(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _fingerprint(env0: dict) -> dict:
    import gc
    import threading

    import torch
    import torch._dynamo.config as dynamo_config
    import torch._dynamo.convert_frame as convert_frame
    import torch.distributed as dist

    out: dict = {}
    from verl.workers.rollout import base as rollout_base

    out["rollout_registry"] = {
        repr(k): str(v) for k, v in sorted(rollout_base._ROLLOUT_REGISTRY.items(), key=lambda kv: repr(kv[0]))
    }

    from verl.utils import device as verl_device
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    loaders = {}
    real_resource = verl_device.get_resource_name
    for label, resource in (("native", None), ("tpu", "TPU")):
        if resource is not None:
            verl_device.get_resource_name = lambda _r=resource: _r
        try:
            loaders[label] = _class_name(RolloutReplicaRegistry._registry["vllm"]())
        except BaseException as exc:  # noqa: BLE001 - recorded, compared across scenarios
            loaders[label] = f"raised {type(exc).__name__}"
        finally:
            verl_device.get_resource_name = real_resource
    out["replica_loader"] = loaders

    from verl.workers.engine_workers import ActorRolloutRefWorker

    out["worker_methods"] = {
        k: _code_id(v)
        for k, v in sorted(ActorRolloutRefWorker.__dict__.items())
        if callable(v) or isinstance(v, staticmethod | classmethod)
    }
    out["worker_attr_names"] = sorted(ActorRolloutRefWorker.__dict__)

    from verl.checkpoint_engine.base import CheckpointEngineManager

    uw = CheckpointEngineManager.update_weights
    out["ckpt_update_weights"] = [_code_id(uw), _code_id(getattr(uw, "__wrapped__", uw))]

    platform_tpu = sys.modules.get(PLATFORM_MOD)
    worker_env: dict = {}
    calls = {"n": 0}
    if platform_tpu is not None:
        real_trainer_env = getattr(platform_tpu, "inproc_trainer_env", None)
        if real_trainer_env is not None:

            def counted_trainer_env(*a, **k):
                calls["n"] += 1
                return real_trainer_env(*a, **k)

            platform_tpu.inproc_trainer_env = counted_trainer_env
        try:
            plat = platform_tpu.PlatformTPU.__new__(platform_tpu.PlatformTPU)
            plat.get_tpu_env_vars = lambda **kw: {"TPU_ENV_STUB": "1"}
            pool = types.SimpleNamespace(get_placement_groups=lambda device_name=None: [])
            for prefix in WORKER_PREFIXES:
                try:
                    worker_env[prefix] = plat.get_worker_env_vars(pool, 0, 8, 0, 8, prefix, "tpu")
                except BaseException as exc:  # noqa: BLE001
                    worker_env[prefix] = f"raised {type(exc).__name__}: {exc}"[:200]
        finally:
            if real_trainer_env is not None:
                platform_tpu.inproc_trainer_env = real_trainer_env
    out["worker_env"] = worker_env
    out["trainer_env_calls"] = calls["n"]

    out["dynamo_config"] = {k: _canon(v)[:300] for k, v in sorted(dynamo_config.get_config_copy().items())}
    out["dynamo_hooks"] = {
        "convert_frame._compile": _code_id(getattr(convert_frame, "_compile", None)),
        "convert_frame.get_and_maybe_log_recompilation_reasons": _code_id(
            getattr(convert_frame, "get_and_maybe_log_recompilation_reasons", None)
        ),
        "torch._dynamo.reset": _code_id(torch._dynamo.reset),
        "torch.compiler.reset": _code_id(torch.compiler.reset),
    }
    out["dist_functions"] = {n: _code_id(getattr(dist, n, None)) for n in DIST_FUNCTIONS}
    try:
        from vllm.distributed import parallel_state
        from vllm.v1.engine.output_processor import OutputProcessor

        gcls = parallel_state.GroupCoordinator
        out["vllm"] = {"GroupCoordinator." + n: _code_id(getattr(gcls, n, None)) for n in COORDINATOR_METHODS}
        out["vllm"]["OutputProcessor.process_outputs"] = _code_id(OutputProcessor.process_outputs)
    except Exception as exc:  # noqa: BLE001 - vLLM absent: the same in every scenario
        out["vllm"] = f"unavailable: {type(exc).__name__}"
    out["gc"] = {
        "enabled": gc.isenabled(),
        "threshold": list(gc.get_threshold()),
        "frozen": gc.get_freeze_count(),
        "callbacks": len(gc.callbacks),
    }
    autocast = getattr(torch, "is_autocast_enabled", None)
    out["torch_state"] = {
        "grad": torch.is_grad_enabled(),
        "inference": torch.is_inference_mode_enabled(),
        "dtype": str(torch.get_default_dtype()),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fn_mode_stack": torch._C._len_torch_function_stack(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_cpu": bool(autocast("cpu")) if autocast is not None else None,
    }
    out["env_changed"] = {k: os.environ[k] for k in sorted(os.environ) if env0.get(k) != os.environ[k]}
    out["env_removed"] = sorted(k for k in env0 if k not in os.environ)
    out["meta_path"] = [type(f).__name__ for f in sys.meta_path if not isinstance(f, _ImportWatch)]
    out["threads"] = threading.active_count()
    out["sys_modules"] = sorted(sys.modules)
    out["coloc_modules"] = sorted(m for m in sys.modules if any(m == p or m.startswith(p + ".") for p in COLOC_MODULES))
    return out


def _child_main(spec: dict) -> dict:
    found = importlib.util.find_spec(PLUGIN_PKG)  # located, not imported
    env_watch = _EnvWriteWatch(os.path.dirname(found.origin) if found is not None and found.origin else None)
    env_watch.install()
    env_watch_ok = env_watch.selftest()
    watch = _ImportWatch(block=bool(spec.get("block")), block_plugin=bool(spec.get("no_plugin")))
    sys.meta_path.insert(0, watch)
    env0 = dict(os.environ)
    import torch  # noqa: F401

    import verl  # noqa: F401

    errors = {}
    try:
        import verl_hardware_plugin
    except Exception as exc:  # noqa: BLE001
        errors["verl_hardware_plugin"] = f"{type(exc).__name__}: {exc}"[:300]
        verl_hardware_plugin = None
    for name in (PLATFORM_MOD, REGISTRATION_MOD):
        if name not in sys.modules:
            errors[name] = "not imported by the plugin"
    if spec.get("import_module"):
        try:
            import pkgutil

            pkg = importlib.import_module(COLOC_MODULES[0])
            for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
                importlib.import_module(f"{COLOC_MODULES[0]}.{info.name}")
        except Exception as exc:  # noqa: BLE001
            errors["import_module"] = f"{type(exc).__name__}: {exc}"[:300]
    out = _fingerprint(env0)
    out["plugin_errors"] = errors
    out["plugin_env_writes"] = sorted(set(env_watch.writes))
    out["env_watch_selftest"] = env_watch_ok
    out["coloc_import_attempts"] = list(watch.seen)
    out["plugin_root"] = (
        os.path.dirname(os.path.dirname(os.path.abspath(verl_hardware_plugin.__file__)))
        if verl_hardware_plugin
        else None
    )
    return out


# --------------------------------------------------------------------------------------------- parent side


def _plugin_root() -> str:
    """The tree this process would import the plugin from, located without importing (executing) it."""
    spec = importlib.util.find_spec("verl_hardware_plugin")
    assert spec is not None and spec.origin, "verl_hardware_plugin is not importable"
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def _run_child(label: str, spec: dict, cwd: str) -> dict:
    env = dict(BASE_ENV)
    env.update(spec.get("env", {}))
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    root = spec.get("plugin_root")
    if root:
        current = _plugin_root()
        rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.abspath(p) != current]
        env["PYTHONPATH"] = os.pathsep.join([root, *rest])
    proc = subprocess.run(
        [sys.executable, "-B", os.path.abspath(__file__), "--child", json.dumps(spec)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if not lines:
        return {"child_error": f"{label}: rc {proc.returncode}: {proc.stderr[-3000:]}"}
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cwd = str(tmp_path_factory.mktemp("default-off"))
    specs = {label: {"env": env} for label, env in {**OFF_ARMS, **ON_ARMS}.items()}
    specs["blocked"] = {"env": {}, "block": True}
    specs["module_imported"] = {"env": {}, "import_module": True}
    specs["no_plugin"] = {"env": {}, "no_plugin": True}
    baseline = os.environ.get("INPROC_DEFAULT_OFF_BASELINE")
    if baseline:
        specs["baseline_unset"] = {"env": {}, "plugin_root": os.path.abspath(baseline)}
        specs["baseline_role_trainer_only"] = {"env": {ROLE_ENV: "trainer"}, "plugin_root": os.path.abspath(baseline)}
    workers = max(1, int(os.environ.get("INPROC_TEST_WORKERS", "2")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {label: pool.submit(_run_child, label, spec, cwd) for label, spec in specs.items()}
        results = {label: f.result() for label, f in futures.items()}
    evidence = os.environ.get("INPROC_TEST_EVIDENCE_DIR")
    if evidence:
        os.makedirs(evidence, exist_ok=True)
        with open(os.path.join(evidence, "default_off_fingerprints.json"), "w") as f:
            json.dump({"specs": specs, "results": results}, f, indent=1, sort_keys=True, default=str)
    return results


def _diff(a: dict, b: dict, skip=()) -> dict:
    out = {}
    for key in sorted(set(a) | set(b)):
        if key in skip or key == "plugin_root":
            continue
        va, vb = a.get(key), b.get(key)
        if va == vb:
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            out[key] = {k: [va.get(k), vb.get(k)] for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)}
        elif isinstance(va, list) and isinstance(vb, list):
            out[key] = {
                "only_first": sorted(set(map(str, va)) - set(map(str, vb)))[:20],
                "only_second": sorted(set(map(str, vb)) - set(map(str, va)))[:20],
            }
        else:
            out[key] = [va, vb]
    return out


def _ok(res: dict) -> dict:
    assert "child_error" not in res, res.get("child_error")
    return res


@pytest.mark.parametrize("arm", [*OFF_ARMS, "blocked"])
def test_switch_off_takes_no_colocation_path(runs, arm):
    res = _ok(runs[arm])
    assert res["plugin_errors"] == {}, res["plugin_errors"]
    assert res["coloc_import_attempts"] == [], (
        f"{arm}: the colocation module was imported: {res['coloc_import_attempts']}"
    )
    assert res["coloc_modules"] == []
    assert res["trainer_env_calls"] == 0, f"{arm}: the colocated trainer environment was computed"
    for prefix, env in res["worker_env"].items():
        assert isinstance(env, dict), (prefix, env)


@pytest.mark.parametrize("arm", [a for a in OFF_ARMS if a != "unset"] + ["blocked"])
def test_switch_off_state_equals_unset(runs, arm):
    diff = _diff(_ok(runs["unset"]), _ok(runs[arm]))
    assert diff == {}, f"{arm} differs from the unset switch: {json.dumps(diff, default=str)[:3000]}"


def test_importing_the_package_alone_is_inert(runs):
    """Importing every module of the colocation package without its switch registers nothing and patches nothing: the
    hooks and patches are installed by registration (switch on) or by the engine build, never at import."""
    res = _ok(runs["module_imported"])
    assert res["plugin_errors"] == {}, res["plugin_errors"]
    assert f"{COLOC_MODULES[0]}.rollout" in res["coloc_modules"], res["coloc_modules"]
    diff = _diff(_ok(runs["unset"]), res, skip=MODULE_SET_KEYS)
    assert diff == {}, f"importing the module changed process state: {json.dumps(diff, default=str)[:3000]}"


def test_switch_on_is_detected(runs):
    """Positive control: the same fingerprint sees the feature when it is on."""
    off, on = _ok(runs["unset"]), _ok(runs["on_trainer"])
    assert on["plugin_errors"] == {}, on["plugin_errors"]
    assert on["coloc_import_attempts"], "the colocation package was not imported with the switch on"
    diff = _diff(off, on)
    keys = ["rollout_registry", "worker_methods", "worker_env"]
    if importlib.util.find_spec("vllm") is not None:  # without vLLM every replica loader raises its ImportError
        keys.append("replica_loader")
    for key in keys:
        assert key in diff, f"the switch-on run does not change {key}: the fingerprint cannot see the feature"
    assert on["trainer_env_calls"] > 0
    for prefix in ("actor_rollout", "actor", "ref"):
        assert set(TRAINER_ENV_KEYS) <= set(on["worker_env"][prefix]), (prefix, on["worker_env"][prefix])
    assert not set(TRAINER_ENV_KEYS) & set(on["worker_env"]["rollout_0"])


# --------------------------------------------------------------------------------------------- no baseline tree


@pytest.mark.parametrize("arm", [*OFF_ARMS, "blocked"])
def test_switch_off_plugin_code_writes_no_environment_variable(runs, arm):
    """A write the switch does not guard (at import, or in a code path every run takes) shows in every arm alike, so the
    comparison with the unset run cannot see it: with the switch off the plugin's own code writes no variable."""
    res = _ok(runs[arm])
    assert res["env_watch_selftest"] is True, "the environment-write watch did not see its own control write"
    writes = [w for w in res["plugin_env_writes"] if w not in ALLOWED_PLUGIN_ENV_WRITES]
    assert writes == [], f"{arm}: the plugin's own code wrote environment variables with the switch off: {writes}"


@pytest.mark.parametrize("arm", [*OFF_ARMS, "blocked"])
def test_switch_off_gives_no_worker_group_the_colocated_trainer_variables(runs, arm):
    """The colocated trainer's variables (TRAINER_ENV_KEYS, the switches, and whatever the switch-on run gives the
    trainer groups but not the rollout group) reach no worker group with the switch off, whatever sets them."""
    on = _ok(runs["on_trainer"])["worker_env"]
    names = set(TRAINER_ENV_KEYS) | {INPROC_ENV}
    for prefix in TRAINER_PREFIXES:
        names |= set(on[prefix]) - set(on["rollout_0"])
    res = _ok(runs[arm])
    leaked = {p: sorted(names & set(env)) for p, env in res["worker_env"].items() if names & set(env)}
    assert leaked == {}, f"{arm}: worker groups get the colocated trainer's variables with the switch off: {leaked}"


def test_switch_off_leaves_what_the_plugin_never_touches_as_without_the_plugin(runs):
    """Everything the plugin leaves alone without the feature (PLUGIN_FREE_KEYS) equals a run that never imports the
    plugin: verl's rollout registry and worker classes, dynamo, the collectives, gc, torch's global state, threads."""
    none, off = _ok(runs["no_plugin"]), _ok(runs["unset"])
    assert none["plugin_errors"].get(PLUGIN_PKG, "").startswith("ImportError"), none["plugin_errors"]
    assert not [m for m in none["sys_modules"] if m == PLUGIN_PKG or m.startswith(PLUGIN_PKG + ".")]
    diff = {k: v for k, v in _diff(none, off).items() if k in PLUGIN_FREE_KEYS}
    assert diff == {}, f"with the switch off the plugin changes {sorted(diff)}: {json.dumps(diff, default=str)[:3000]}"
    # positive control: the same comparison sees the feature when it is on
    on_diff = {k for k in _diff(none, _ok(runs["on_trainer"])) if k in PLUGIN_FREE_KEYS}
    assert {"rollout_registry", "worker_methods"} <= on_diff, on_diff


@pytest.mark.skipif(not os.environ.get("INPROC_DEFAULT_OFF_BASELINE"), reason="INPROC_DEFAULT_OFF_BASELINE not set")
@pytest.mark.parametrize("arm", ["unset", "role_trainer_only"])
def test_switch_off_equals_plugin_without_the_feature(runs, arm):
    base = _ok(runs[f"baseline_{arm}"])
    assert base["plugin_root"] == os.path.abspath(os.environ["INPROC_DEFAULT_OFF_BASELINE"]), base["plugin_root"]
    diff = _diff(base, _ok(runs[arm]))
    assert diff == {}, (
        f"the switch-off state differs from the plugin without the feature: {json.dumps(diff, default=str)[:3000]}"
    )


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    print("RESULT " + json.dumps(_child_main(json.loads(sys.argv[2])), sort_keys=True, default=str), flush=True)
