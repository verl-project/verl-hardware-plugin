# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

"""CPU tests of the in-process rollout's feature detection (``rollout/inproc/compat.py``).

Optional input: ``INPROC_TEST_VLLM_TORCHTPU_SRC`` = a vllm-torchtpu source directory (holding
``vllm_torchtpu/env_override.py``); the tests that compare the installed structure with the reviewed one need it.
"""

import ast
import os
import re
import sys
import types

import pytest

from verl_hardware_plugin.accelerators.tpu.rollout.inproc import boundary, compat

# A small env_override-shaped module: import-time code (a flag default, an environment default, a forced value) and a
# function the module never calls at import.
ENV_OVERRIDE = '''"""docstring"""
import os

_libtpu_extra_args: list[str] = []


def _user_has_flag(flag_name: str) -> bool:
    for token in os.environ.get("LIBTPU_INIT_ARGS", "").split():
        if token.startswith(flag_name + "=") or token == flag_name:
            return True
    return False


if not _user_has_flag("--xla_tpu_use_dynamic_smem_negotiation"):
    _libtpu_extra_args.append("--xla_tpu_use_dynamic_smem_negotiation=true")
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"


def experimental_accessor() -> list[str]:
    return os.environ.get("LIBTPU_INIT_ARGS", "").split()
'''


def _fp(src: str) -> str:
    return compat.env_override_structure(src)["fingerprint"]


@pytest.mark.parametrize(
    "edit, same",
    [
        (
            lambda s: s.replace(
                "_libtpu_extra_args: list[str] = []", "_libtpu_extra_args: list[str] = []  # a comment"
            ),
            True,
        ),
        (lambda s: s.replace('"""docstring"""', '"""another docstring"""'), True),
        (lambda s: s + "\n\ndef another_accessor():\n    return 1\n", True),
        (
            lambda s: s.replace(
                'os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"', 'os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"'
            ),
            False,
        ),
        (lambda s: s + '\nos.environ.setdefault("TPU_SOMETHING_NEW", "1")\n', False),
        (lambda s: s + '\n_libtpu_extra_args.append("--xla_tpu_some_new_flag=true")\n', False),
        (lambda s: s.replace(" or token == flag_name", ""), False),
    ],
)
def test_env_override_fingerprint_follows_the_import_time_code_only(edit, same):
    assert (_fp(edit(ENV_OVERRIDE)) == _fp(ENV_OVERRIDE)) is same


def test_env_override_structure_lists_the_import_time_scope():
    st = compat.env_override_structure(ENV_OVERRIDE)
    assert st["import_time_functions"] == ["_user_has_flag"]
    assert st["flags"] == ["--xla_tpu_use_dynamic_smem_negotiation", "--xla_tpu_use_dynamic_smem_negotiation=true"]
    assert (
        "VLLM_USE_BREAKABLE_CUDAGRAPH" in st["env_names"]
        and "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS" in st["env_names"]
    )
    assert compat.env_override_structure(None) == {"present": False}


def _vt_src():
    d = os.environ.get("INPROC_TEST_VLLM_TORCHTPU_SRC")
    if not d:
        pytest.skip("INPROC_TEST_VLLM_TORCHTPU_SRC not set")
    return os.path.join(d, "vllm_torchtpu", "env_override.py")


def test_installed_env_override_structure_is_reviewed():
    probe = compat.env_override_probe(_vt_src())
    assert probe["present"] and probe["reviewed"], probe


def _env_effects(src: str) -> dict:
    """The import-time environment effects of env_override.py (the fingerprint's scope), recomputed from source."""
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    top = [
        n
        for n in tree.body
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
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
        nm = frontier.pop()
        if nm not in closure:
            closure.add(nm)
            frontier |= called([defs[nm]]) - closure
    effects: dict = {}

    def note(name, eff, cond):
        effects.setdefault(name, set()).add(eff + (" (conditional)" if cond else ""))

    def scan(node, cond):
        for c in ast.walk(node):
            if (
                isinstance(c, ast.Call)
                and c.args
                and isinstance(c.args[0], ast.Constant)
                and isinstance(c.args[0].value, str)
            ):
                f, name = ast.unparse(c.func), c.args[0].value
                if f in ("os.getenv", "os.environ.get"):
                    note(name, "read", cond)
                elif f == "os.environ.setdefault":
                    v = c.args[1] if len(c.args) > 1 else None
                    note(
                        name,
                        "setdefault " + (repr(v.value) if isinstance(v, ast.Constant) else f"<{ast.unparse(v)[:40]}>"),
                        cond,
                    )
            if isinstance(c, ast.Assign):
                for t in c.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and ast.unparse(t.value) == "os.environ"
                        and isinstance(t.slice, ast.Constant)
                    ):
                        v = c.value
                        note(
                            t.slice.value,
                            ("forced " + repr(v.value)) if isinstance(v, ast.Constant) else "assembled",
                            cond,
                        )
            if (
                isinstance(c, ast.Subscript)
                and isinstance(c.ctx, ast.Load)
                and ast.unparse(c.value) == "os.environ"
                and isinstance(c.slice, ast.Constant)
            ):
                note(c.slice.value, "read", cond)

    for n in top:
        scan(n, isinstance(n, ast.If | ast.Try))
    for nm in sorted(closure):
        scan(defs[nm], True)
    return {k: tuple(sorted(v)) for k, v in sorted(effects.items())}


def test_reviewed_effect_table_is_the_reviewed_source():
    with open(_vt_src(), encoding="utf-8") as f:
        assert _env_effects(f.read()) == {k: tuple(sorted(v)) for k, v in compat.REVIEWED_ENV_EFFECTS.items()}


@pytest.mark.parametrize(
    "args, env, want",
    [
        ("", {}, ["--xla_tpu_use_dynamic_smem_negotiation=true"]),
        ("--xla_tpu_use_dynamic_smem_negotiation=true", {}, []),
        ("--xla_tpu_use_dynamic_smem_negotiation=true", {"TPU_ACCELERATOR_TYPE": "tpu7x-8"},
         [f"{f}=67108864" for f in compat.SC_FLAGS]),
        ("--xla_tpu_use_dynamic_smem_negotiation=true", {"TPU_ACCELERATOR_TYPE": "tpu7x-8", compat.SC_ENV: "0"}, []),
        (
            "--xla_tpu_use_dynamic_smem_negotiation=true",
            {compat.SC_ENV: "4096"},
            [f"{f}=4096" for f in compat.SC_FLAGS],
        ),
    ],
)  # fmt: skip
def test_would_add_libtpu_flags(args, env, want):
    assert compat.would_add_libtpu_flags(args, env) == want


@pytest.fixture
def libtpu_state(monkeypatch):
    monkeypatch.setattr(boundary, "LIBTPU_PROBLEMS", [])
    monkeypatch.setattr(boundary, "TRAINER_EAGER_AT_START", None)
    monkeypatch.delitem(sys.modules, "vllm_torchtpu", raising=False)


def _trainer_env(**kw):
    env = {
        "LIBTPU_INIT_ARGS": "--xla_tpu_use_dynamic_smem_negotiation=true",
        compat.MATERIALIZE_ENV: "1",
        compat.SC_ENV: "0",
        "RANK": "0",
    }
    env.update(kw)
    return {k: v for k, v in env.items() if v is not None}


def test_libtpu_check_reviewed_structure_ok(libtpu_state, monkeypatch, tmp_path):
    p = tmp_path / "env_override.py"
    p.write_text(ENV_OVERRIDE)
    monkeypatch.setattr(compat, "REVIEWED_ENV_OVERRIDE", frozenset({_fp(ENV_OVERRIDE)}))
    monkeypatch.setattr(compat, "env_override_file", lambda: str(p))
    info = boundary.check_trainer_libtpu_env(_trainer_env(), strict=True)
    assert info["problems"] == [] and info["env_override"]["reviewed"]
    assert boundary.ENV_AT_CHECK[compat.MATERIALIZE_ENV] == "1"


@pytest.mark.parametrize(
    "env, reviewed, want",
    [
        (dict(LIBTPU_INIT_ARGS=""), True, "would add ['--xla_tpu_use_dynamic_smem_negotiation=true']"),
        ({}, False, "is not a reviewed one"),
        ({compat.MATERIALIZE_ENV: None}, True, "is not pinned"),
    ],
)
def test_libtpu_check_refuses(libtpu_state, monkeypatch, tmp_path, env, reviewed, want):
    p = tmp_path / "env_override.py"
    p.write_text(ENV_OVERRIDE)
    monkeypatch.setattr(compat, "REVIEWED_ENV_OVERRIDE", frozenset({_fp(ENV_OVERRIDE)} if reviewed else {"0" * 16}))
    monkeypatch.setattr(compat, "env_override_file", lambda: str(p))
    with pytest.raises(RuntimeError, match=re.escape(want)):
        boundary.check_trainer_libtpu_env(_trainer_env(**env), strict=True)
    boundary.check_trainer_libtpu_env(_trainer_env(**env), strict=False)
    assert boundary.LIBTPU_PROBLEMS and want in boundary.LIBTPU_PROBLEMS[0]


def test_libtpu_check_refuses_when_vllm_torchtpu_is_imported(libtpu_state, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_torchtpu", types.ModuleType("vllm_torchtpu"))
    monkeypatch.setattr(compat, "env_override_file", lambda: None)
    with pytest.raises(RuntimeError, match="vllm_torchtpu is already imported"):
        boundary.check_trainer_libtpu_env(_trainer_env(), strict=True)


BEFORE = {
    "LIBTPU_INIT_ARGS": "--xla_tpu_use_dynamic_smem_negotiation=true",
    compat.MATERIALIZE_ENV: "false",
    compat.SC_ENV: "0",
    "TPU_ACCELERATOR_TYPE": "tpu7x",
    "VLLM_CACHE_ROOT": "/cache",
}
REVIEWED_AFTER = dict(
    BEFORE, VLLM_DISABLE_SHARED_EXPERTS_STREAM="1", VLLM_USE_BREAKABLE_CUDAGRAPH="0", VLLM_USE_MEGA_AOT_ARTIFACT="0"
)


@pytest.mark.parametrize(
    "after, bad",
    [
        (REVIEWED_AFTER, []),
        (dict(REVIEWED_AFTER, TPU_PREMAPPED_BUFFER_SIZE=str(16 << 30)), []),  # a computed setdefault default
        (dict(REVIEWED_AFTER, LIBTPU_INIT_ARGS=BEFORE["LIBTPU_INIT_ARGS"] + " --xla_x=1"), ["LIBTPU_INIT_ARGS"]),
        (dict(REVIEWED_AFTER, **{compat.MATERIALIZE_ENV: "true"}), [compat.MATERIALIZE_ENV]),
        (dict(REVIEWED_AFTER, VLLM_USE_BREAKABLE_CUDAGRAPH="1"), ["VLLM_USE_BREAKABLE_CUDAGRAPH"]),
        (dict(REVIEWED_AFTER, VLLM_USE_MEGA_AOT_ARTIFACT="1"), ["VLLM_USE_MEGA_AOT_ARTIFACT"]),
        (dict(REVIEWED_AFTER, VLLM_CACHE_ROOT="/elsewhere"), ["VLLM_CACHE_ROOT"]),  # set before: must stay
        (dict(REVIEWED_AFTER, TPU_ACCELERATOR_TYPE="v6e"), ["TPU_ACCELERATOR_TYPE"]),  # read-only
        (dict(REVIEWED_AFTER, **{compat.SC_ENV: "4096"}), [compat.SC_ENV]),
        (
            dict(REVIEWED_AFTER, VLLM_ENABLE_V1_MULTIPROCESSING="0"),
            [],
        ),  # outside the table: the identity line reports it
    ],
)
def test_env_after_build(after, bad):
    got = compat.env_after_build_problems(BEFORE, after)
    assert [x.split(" ", 1)[0] for x in got] == bad


def test_env_after_build_needs_the_platform_import_snapshot():
    assert (
        compat.env_after_build_problems(None, {})
        and "snapshot is missing" in compat.env_after_build_problems(None, {})[0]
    )


# ---------------------------------------------------------------------------------------- runner features (report only)


def test_runner_features_report_only():
    runner = types.ModuleType("vllm_torchtpu.runner.tpu_runner")
    src = "class TPUModelRunner:\n    def _sampler_applies_topk_topp(self, x):\n        return any(x)\n"
    exec(src, runner.__dict__)
    aout = types.ModuleType("vllm_torchtpu.runner.tpu_runner_async_output")
    aout.plain_logprobs_lists = lambda: None
    mods = {compat.RUNNER_MOD: runner, compat.AOUT_MOD: aout}
    out = compat.runner_features(mods, {compat.R7_ENV: "1"})
    assert out["deferred_logprobs"] == "on" and out["fast_token_substitution"]["env"] == "1"
    assert out["sampler_skip"] in ("present", "unknown")  # the source of an exec'd class is not readable
    assert compat.runner_features({}, {})["sampler_skip"] == "runner not imported"
