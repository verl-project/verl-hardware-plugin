# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for patches/xpu/numa_affinity_patch.py.

This module's correctness hinges almost entirely on *when* the patch is
applied relative to verl-core's own imports (see the patch module's docstring),
so these tests reconstruct that dispatch shape rather than exercising pyzes:

- `verl.utils.distributed` exposes a module-level `set_numa_affinity`.
- Caller modules bind it *by value*, some before the patch and some after.

verl itself is stubbed and the patch module is loaded by file path, so this
runs on a CPU-only host with neither verl nor pyzes installed. The pyzes ->
sysfs -> sched_setaffinity mechanism is verified on real hardware instead;
what's verified here is the wiring
around it, which hardware runs can't easily prove is *not* silently no-oping.
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

PATCH_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl_hardware_plugin"
    / "accelerators"
    / "xpu"
    / "patches"
    / "numa_affinity_patch.py"
)


class _Recorder:
    """Records which path a set_numa_affinity() call took."""

    def __init__(self):
        self.original_calls = 0
        self.pinned_ranks = []


@pytest.fixture
def stub_verl(monkeypatch):
    """Build a fake verl package mimicking the real dispatch shape.

    Returns a factory so each test can choose the selected platform, whether
    Ray is initialised, and what LOCAL_RANK is set to.
    """
    created = []

    def factory(
        platform_device_name="xpu", ray_initialized=False, accel_id="3", local_rank=None, platform_raises=False
    ):
        recorder = _Recorder()

        def mod(name):
            m = ModuleType(name)
            monkeypatch.setitem(sys.modules, name, m)
            created.append(name)
            return m

        verl = mod("verl")
        verl.__path__ = []
        utils = mod("verl.utils")
        utils.__path__ = []
        verl.utils = utils

        distributed = mod("verl.utils.distributed")

        def set_numa_affinity():
            """Stand-in for verl-core's pynvml implementation."""
            recorder.original_calls += 1

        distributed.set_numa_affinity = set_numa_affinity
        utils.distributed = distributed

        device = mod("verl.utils.device")
        device.get_resource_name = lambda: "GPU"
        utils.device = device

        plugin = mod("verl.plugin")
        plugin.__path__ = []
        verl.plugin = plugin
        platform_pkg = mod("verl.plugin.platform")
        platform_pkg.__path__ = []
        plugin.platform = platform_pkg

        class _FakePlatform:
            device_name = platform_device_name

        def get_platform():
            if platform_raises:
                raise RuntimeError("platform detection failed")
            return _FakePlatform()

        platform_pkg.get_platform = get_platform

        ray = mod("ray")
        ray.is_initialized = lambda: ray_initialized

        class _Ctx:
            def get_accelerator_ids(self):
                return {"GPU": [accel_id]}

        ray.get_runtime_context = _Ctx

        monkeypatch.delenv("LOCAL_RANK", raising=False)
        if local_rank is not None:
            monkeypatch.setenv("LOCAL_RANK", local_rank)

        return distributed, recorder

    return factory


@pytest.fixture
def load_patch(monkeypatch):
    """Load the patch module fresh by path, with the real pinning call stubbed out.

    Fresh per test because the module keeps a module-level `_applied` flag; by
    path because importing it as `verl_hardware_plugin.patches.*` would run the
    plugin's top-level `__init__`, which needs torch.
    """

    def loader(recorder):
        spec = importlib.util.spec_from_file_location("numa_affinity_patch_under_test", PATCH_PATH)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        # Never touch real pyzes or real sysfs from a unit test. The real
        # implementation stays reachable for the tests that need to assert it
        # swallows its own failures.
        module._real_set_numa_affinity_xpu = module._set_numa_affinity_xpu
        monkeypatch.setattr(module, "_set_numa_affinity_xpu", lambda rank: recorder.pinned_ranks.append(rank))
        return module

    return loader


def test_apply_replaces_core_function(stub_verl, load_patch):
    distributed, recorder = stub_verl(local_rank="0")
    patch = load_patch(recorder)
    original = distributed.set_numa_affinity

    patch.apply()

    assert distributed.set_numa_affinity is not original


def test_importer_after_patch_gets_xpu_path(stub_verl, load_patch):
    """The usual case: the caller imports after the patch lands, so no rebinding needed."""
    distributed, recorder = stub_verl(local_rank="1")
    patch = load_patch(recorder)
    patch.apply()

    late = ModuleType("verl.workers.some_late_module")
    late.set_numa_affinity = distributed.set_numa_affinity  # simulates a from-import
    late.set_numa_affinity()

    assert recorder.pinned_ranks == [1]
    assert recorder.original_calls == 0


def test_known_importer_before_patch_is_rebound(stub_verl, load_patch, monkeypatch):
    """A call site that from-imported before PlatformXPU was constructed still gets the patch."""
    distributed, recorder = stub_verl(local_rank="2")
    patch = load_patch(recorder)

    early = ModuleType("verl.workers.engine_workers")
    early.set_numa_affinity = distributed.set_numa_affinity
    monkeypatch.setitem(sys.modules, "verl.workers.engine_workers", early)

    patch.apply()
    early.set_numa_affinity()

    assert recorder.pinned_ranks == [2]


def test_rebind_leaves_unrelated_bindings_alone(stub_verl, load_patch, monkeypatch):
    """Only the two known importers are rebound, and only if still pointing at the original."""
    distributed, recorder = stub_verl(local_rank="0")
    patch = load_patch(recorder)

    sentinel = object()
    other = ModuleType("verl.workers.engine_workers")
    other.set_numa_affinity = sentinel  # already something else; must not be clobbered
    monkeypatch.setitem(sys.modules, "verl.workers.engine_workers", other)

    patch.apply()

    assert other.set_numa_affinity is sentinel


def test_idempotent(stub_verl, load_patch):
    distributed, recorder = stub_verl(local_rank="0")
    patch = load_patch(recorder)

    patch.apply()
    after_first = distributed.set_numa_affinity
    patch.apply()

    assert distributed.set_numa_affinity is after_first


@pytest.mark.parametrize("device_name", ["cuda", "npu", "mlu"])
def test_non_xpu_platform_delegates_to_core(stub_verl, load_patch, device_name):
    """A plugin install must not change NUMA behavior for anyone else."""
    distributed, recorder = stub_verl(platform_device_name=device_name, local_rank="0")
    patch = load_patch(recorder)
    patch.apply()

    distributed.set_numa_affinity()

    assert recorder.original_calls == 1
    assert recorder.pinned_ranks == []


def test_ray_resolves_rank_from_accelerator_ids(stub_verl, load_patch):
    """Under Ray, LOCAL_RANK is always 0, so the accelerator id is what matters."""
    distributed, recorder = stub_verl(ray_initialized=True, accel_id="3", local_rank="0")
    patch = load_patch(recorder)
    patch.apply()

    distributed.set_numa_affinity()

    assert recorder.pinned_ranks == [3]


def test_platform_probe_failure_falls_back_without_raising(stub_verl, load_patch):
    """Both call sites are constructors; this must never be the thing that kills a run."""
    distributed, recorder = stub_verl(platform_raises=True, local_rank="0")
    patch = load_patch(recorder)
    patch.apply()

    distributed.set_numa_affinity()  # must not raise

    assert recorder.original_calls == 1
    assert recorder.pinned_ranks == []


def test_missing_local_rank_skips_quietly(stub_verl, load_patch):
    """Core does int(os.environ["LOCAL_RANK"]) and lets the KeyError be swallowed."""
    distributed, recorder = stub_verl(local_rank=None)
    patch = load_patch(recorder)
    patch.apply()

    distributed.set_numa_affinity()  # must not raise

    assert recorder.pinned_ranks == []


@pytest.mark.parametrize(
    ("cpulist", "expected"),
    [
        ("0-3", {0, 1, 2, 3}),
        ("8", {8}),
        ("0-3,8,10-11", {0, 1, 2, 3, 8, 10, 11}),
        ("0-1, 4", {0, 1, 4}),
        ("", set()),
    ],
)
def test_parse_cpulist(stub_verl, load_patch, cpulist, expected):
    _, recorder = stub_verl()
    patch = load_patch(recorder)

    assert patch._parse_cpulist(cpulist) == expected


def test_parse_cpulist_matches_real_hardware_output(stub_verl, load_patch):
    """The cpulist observed on the B60 probe (rank 1, numa_node=1)."""
    _, recorder = stub_verl()
    patch = load_patch(recorder)

    cpus = patch._parse_cpulist("128-255,384-511")

    assert len(cpus) == 256
    assert min(cpus) == 128
    assert max(cpus) == 511


def test_old_pyzes_raises_actionable_error(stub_verl, load_patch, monkeypatch):
    """pyzes 0.1.1 has no PCI API at all; the message must say so.

    Observed for real on a B60 run against an image shipping 0.1.1, where the
    bare failure was `AttributeError: module 'pyzes' has no attribute
    'zes_pci_properties_t'` -- true but useless.
    """
    _, recorder = stub_verl()
    patch = load_patch(recorder)

    old_pyzes = ModuleType("pyzes")  # 0.1.1: zesInit etc. exist, no PCI family
    old_pyzes.zesInit = lambda flags: None
    monkeypatch.setitem(sys.modules, "pyzes", old_pyzes)

    with pytest.raises(RuntimeError, match=r"pyzes>=0\.1\.2"):
        patch._zes_device_pci_bdf(0)


def test_old_pyzes_is_reported_as_warning_not_crash(stub_verl, load_patch, monkeypatch):
    """That RuntimeError must still be swallowed by the real pinning wrapper."""
    _, recorder = stub_verl()
    patch = load_patch(recorder)

    old_pyzes = ModuleType("pyzes")
    old_pyzes.zesInit = lambda flags: None
    monkeypatch.setitem(sys.modules, "pyzes", old_pyzes)

    patch._real_set_numa_affinity_xpu(0)  # the unmocked one; must not raise


def test_read_helpers_return_none_for_missing_paths(stub_verl, load_patch, tmp_path):
    _, recorder = stub_verl()
    patch = load_patch(recorder)

    missing = str(tmp_path / "nope")
    assert patch._read_int(missing) is None
    assert patch._read_text(missing) is None

    numa_node = tmp_path / "numa_node"
    numa_node.write_text("-1\n")
    assert patch._read_int(str(numa_node)) == -1


def _names_in_code(node) -> set:
    # Names/attributes used in real code; docstrings and comments don't count.
    found = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            found.add(n.id)
        elif isinstance(n, ast.Attribute):
            found.add(n.attr)
        elif isinstance(n, ast.alias):
            found.add(n.name)
    return found


def _find_def(tree, name, cls=None):
    scope = tree
    if cls is not None:
        scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next((n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def test_applied_from_platform_xpu_init():
    """Platform-gated: only PlatformXPU construction (verl selected XPU) installs the patch.

    PlatformXPU.__init__ applies every entry of the module-level ``_PATCHES``
    tuple in a loop (see platform_xpu.py), rather than calling each patch's
    ``apply()`` by name inline. So the AST check here is two-part: the patch
    module is listed in ``_PATCHES``, and ``__init__`` actually loops over
    ``_PATCHES`` calling ``.apply()`` on each entry.
    """
    tree = ast.parse((PATCH_PATH.parents[1] / "platform_xpu.py").read_text())

    patches_assign = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "_PATCHES"
        ),
        None,
    )
    assert patches_assign is not None, "platform_xpu.py must define a module-level _PATCHES tuple"
    patch_names = {elt.id for elt in patches_assign.value.elts if isinstance(elt, ast.Name)}
    assert "numa_affinity_patch" in patch_names, "_PATCHES must include numa_affinity_patch"

    init = _find_def(tree, "__init__", cls="PlatformXPU")
    assert init is not None, "PlatformXPU must define __init__ that applies the patches"
    loops_over_patches = any(
        isinstance(n, ast.For) and isinstance(n.iter, ast.Name) and n.iter.id == "_PATCHES" for n in ast.walk(init)
    )
    assert loops_over_patches, "PlatformXPU.__init__ must loop over _PATCHES calling .apply() on each entry"


def test_not_referenced_from_shared_patches_init():
    """Must never be wired through the shared, vendor-agnostic patches/__init__.py."""
    tree = ast.parse(PATCH_PATH.parent.joinpath("__init__.py").read_text())

    assert "numa_affinity_patch" not in _names_in_code(tree)


def test_no_os_environ_leak(stub_verl, load_patch):
    """_zes_device_pci_bdf sets ZES_ENABLE_SYSMAN; make sure nothing else mutates env."""
    before = dict(os.environ)
    _, recorder = stub_verl(local_rank="0")
    patch = load_patch(recorder)
    patch.apply()
    patch_module_env_keys = set(os.environ) - set(before)

    assert patch_module_env_keys <= {"LOCAL_RANK"}
