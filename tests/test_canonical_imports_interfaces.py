from __future__ import annotations

import ast
from pathlib import Path

MIGRATED_FILES = [
    Path("src/coal_kb/api/app.py"),
    Path("src/coal_kb/api/routes_chat.py"),
    Path("src/coal_kb/api/runtime_overrides.py"),
    Path("src/coal_kb/query/planner.py"),
    Path("src/coal_kb/retrieval/query_rewrite.py"),
]
LEGACY_MODULES = {
    "coal_kb.settings",
    "coal_kb.query.plan",
    "coal_kb.embeddings.factory",
    "coal_kb.llm.factory",
    "coal_kb.retrieval.rerank",
    "coal_kb.logging",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_migrated_interface_query_and_answering_files_use_canonical_imports() -> None:
    violations: list[str] = []
    for path in MIGRATED_FILES:
        for module in sorted(_imports(path) & LEGACY_MODULES):
            violations.append(f"{path}: {module}")
    assert not violations, "Legacy imports remain:\n" + "\n".join(violations)
