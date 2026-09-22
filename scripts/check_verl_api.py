#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Statically verify the verl APIs imported by this plugin.

The check reads source files instead of importing verl. That keeps it useful on
CPU-only CI runners where optional training and vendor runtimes are unavailable.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Requirement:
    """One absolute import from verl found in plugin source."""

    source: str
    line: int
    module: str
    name: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}" if self.name else self.module


@dataclass(frozen=True)
class MissingRequirement:
    """A required module or symbol that is absent from a verl source tree."""

    requirement: Requirement
    reason: str


@dataclass(frozen=True)
class ModuleSurface:
    """Names bound by a module and names exported by ``import *``."""

    bindings: frozenset[str]
    star_exports: frozenset[str]


def collect_requirements(plugin_root: Path) -> list[Requirement]:
    """Collect every absolute verl import in the plugin package."""

    requirements: set[Requirement] = set()
    for source_path in sorted(plugin_root.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        source = source_path.relative_to(plugin_root.parent).as_posix()
        module_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "verl" or node.module.startswith("verl."):
                    for alias in node.names:
                        requirements.add(Requirement(source, node.lineno, node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "verl" or alias.name.startswith("verl."):
                        requirements.add(Requirement(source, node.lineno, alias.name))
                        if alias.asname:
                            module_aliases[alias.asname] = alias.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module = module_aliases.get(node.value.id)
                if module:
                    requirements.add(Requirement(source, node.lineno, module, node.attr))
    return sorted(requirements, key=lambda item: (item.source, item.line, item.module, item.name or ""))


def _module_path(verl_root: Path, module: str) -> Path | None:
    if module != "verl" and not module.startswith("verl."):
        raise ValueError(f"Expected a verl module, got {module!r}")

    relative_parts = module.split(".")[1:]
    target = verl_root.joinpath(*relative_parts)
    candidates = [target.with_suffix(".py"), target / "__init__.py"] if relative_parts else [verl_root / "__init__.py"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _nested_module_statements(statements: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
    """Yield module-scope statements, including imports behind conditions."""

    for statement in statements:
        yield statement
        nested: list[list[ast.stmt]] = []
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            nested.extend([statement.body, statement.orelse])
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested.append(statement.body)
        elif isinstance(statement, ast.Try):
            nested.extend([statement.body, statement.orelse, statement.finalbody])
            nested.extend(handler.body for handler in statement.handlers)
        elif isinstance(statement, ast.Match):
            nested.extend(case.body for case in statement.cases)
        for body in nested:
            yield from _nested_module_statements(body)


def _resolve_import_from(current_module: str, module_path: Path, imported_module: str | None, level: int) -> str | None:
    """Resolve an ``ImportFrom`` target without importing upstream code."""

    if level == 0:
        return imported_module

    package_parts = current_module.split(".")
    if module_path.name != "__init__.py":
        package_parts = package_parts[:-1]
    parents_to_remove = level - 1
    if parents_to_remove > len(package_parts):
        return None
    resolved = package_parts[: len(package_parts) - parents_to_remove]
    if imported_module:
        resolved.extend(imported_module.split("."))
    return ".".join(resolved)


def _literal_string_collection(node: ast.expr) -> set[str] | None:
    """Evaluate a static collection of strings used for ``__all__``."""

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: set[str] = set()
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.add(element.value)
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_collection(node.left)
        right = _literal_string_collection(node.right)
        return None if left is None or right is None else left | right
    return None


def _declared_all(statements: Iterable[ast.stmt]) -> set[str] | None:
    """Return a literal module-level ``__all__`` declaration when available."""

    declared: set[str] | None = None
    for statement in statements:
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            value = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "__all__"
        ):
            value = statement.value
        if value is not None:
            declared = _literal_string_collection(value)
    return declared


def _module_surface(
    verl_root: Path,
    module: str,
    cache: dict[str, ModuleSurface],
    visiting: set[str],
) -> ModuleSurface:
    """Return a module's bindings, recursively following relative star imports."""

    cached = cache.get(module)
    if cached is not None:
        return cached
    if module in visiting:
        return ModuleSurface(frozenset(), frozenset())

    module_path = _module_path(verl_root, module)
    if module_path is None:
        return ModuleSurface(frozenset(), frozenset())
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    bindings: set[str] = set()
    star_imports: set[str] = set()

    for statement in _nested_module_statements(tree.body):
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings.add(statement.name)
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                bindings.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.name == "*":
                    target = _resolve_import_from(module, module_path, statement.module, statement.level)
                    if target and (target == "verl" or target.startswith("verl.")):
                        star_imports.add(target)
                else:
                    bindings.add(alias.asname or alias.name)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bindings.add(target.id)
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            bindings.add(statement.target.id)

    visiting.add(module)
    for target in star_imports:
        bindings.update(_module_surface(verl_root, target, cache, visiting).star_exports)
    visiting.remove(module)

    declared_all = _declared_all(tree.body)
    star_exports = {name for name in bindings if not name.startswith("_")}
    if declared_all is not None:
        star_exports &= declared_all
    surface = ModuleSurface(frozenset(bindings), frozenset(star_exports))
    cache[module] = surface
    return surface


def module_exports(verl_root: Path, module: str, cache: dict[str, ModuleSurface] | None = None) -> set[str]:
    """Return names that are statically bound by an upstream verl module."""

    surfaces = cache if cache is not None else {}
    return set(_module_surface(verl_root, module, surfaces, set()).bindings)


def check_contract(plugin_root: Path, verl_root: Path) -> tuple[list[Requirement], list[MissingRequirement]]:
    """Check plugin imports against a checked-out verl package directory."""

    requirements = collect_requirements(plugin_root)
    missing: list[MissingRequirement] = []
    export_cache: dict[str, ModuleSurface] = {}

    for requirement in requirements:
        module_path = _module_path(verl_root, requirement.module)
        if module_path is None:
            missing.append(MissingRequirement(requirement, "module not found"))
            continue
        if requirement.name is None or requirement.name == "*":
            continue

        exports = module_exports(verl_root, requirement.module, export_cache)
        submodule_path = _module_path(verl_root, f"{requirement.module}.{requirement.name}")
        if requirement.name not in exports and submodule_path is None:
            missing.append(MissingRequirement(requirement, "symbol not found"))

    return requirements, missing


def render_report(
    requirements: list[Requirement], missing: list[MissingRequirement], plugin_root: Path, verl_root: Path
) -> str:
    """Render a concise Markdown report for CI and repair agents."""

    lines = [
        "# verl API compatibility report",
        "",
        f"- Plugin source: `{plugin_root.as_posix()}`",
        f"- verl source: `{verl_root.as_posix()}`",
        f"- Imports checked: {len(requirements)}",
        f"- Missing APIs: {len(missing)}",
        "",
    ]
    if missing:
        lines.extend(["## Missing APIs", ""])
        for item in missing:
            requirement = item.requirement
            lines.append(
                f"- `{requirement.qualified_name}` ({item.reason}), imported at "
                f"`{requirement.source}:{requirement.line}`"
            )
    else:
        lines.append("All statically imported verl modules and symbols are present.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, default=Path("verl_hardware_plugin"))
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    requirements, missing = check_contract(args.plugin_root, args.verl_root)
    report = render_report(requirements, missing, args.plugin_root, args.verl_root)
    print(report, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
