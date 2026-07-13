from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_does_not_depend_on_interfaces_or_infrastructure() -> None:
    core = Path("src/coal_kb/core")
    forbidden = ("coal_kb.api", "coal_kb.web", "coal_kb.infra", "fastapi", "sqlalchemy")
    violations: list[str] = []
    for path in core.rglob("*.py"):
        for module in _imports(path):
            if module.startswith(forbidden):
                violations.append(f"{path}: {module}")
    assert not violations, "Core dependency violations:\n" + "\n".join(violations)
