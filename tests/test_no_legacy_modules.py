"""确保已删除的模块、备份文件和 import 不会重新进入仓库。"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "coal_kb"

REMOVED_PATHS = [
    PACKAGE_ROOT / "api",
    PACKAGE_ROOT / "chat",
    PACKAGE_ROOT / "chunking",
    PACKAGE_ROOT / "embeddings",
    PACKAGE_ROOT / "eval",
    PACKAGE_ROOT / "generation",
    PACKAGE_ROOT / "llm",
    PACKAGE_ROOT / "loaders",
    PACKAGE_ROOT / "metadata",
    PACKAGE_ROOT / "parsing",
    PACKAGE_ROOT / "pipelines",
    PACKAGE_ROOT / "qa",
    PACKAGE_ROOT / "query",
    PACKAGE_ROOT / "store",
    PACKAGE_ROOT / "web",
    PACKAGE_ROOT / "settings.py",
    PACKAGE_ROOT / "logging.py",
    PACKAGE_ROOT / "cli_ui.py",
    PACKAGE_ROOT / "retrieval" / "bm25.py",
    PACKAGE_ROOT / "retrieval" / "constraint_policy.py",
    PACKAGE_ROOT / "retrieval" / "elastic_retriever.py",
    PACKAGE_ROOT / "retrieval" / "filter_parser.py",
    PACKAGE_ROOT / "retrieval" / "query_rewrite.py",
    PACKAGE_ROOT / "retrieval" / "rerank.py",
    PACKAGE_ROOT / "retrieval" / "retriever.py",
    PACKAGE_ROOT / "context" / "builder.py",
    PACKAGE_ROOT / "context" / "types.py",
    REPO_ROOT / "src" / "coal_expert_kb.egg-info",
    REPO_ROOT / ".idea",
    REPO_ROOT / "build",
]

REMOVED_MODULES = {
    "coal_kb.api",
    "coal_kb.chat",
    "coal_kb.chunking",
    "coal_kb.cli_ui",
    "coal_kb.embeddings",
    "coal_kb.eval",
    "coal_kb.generation",
    "coal_kb.llm",
    "coal_kb.loaders",
    "coal_kb.logging",
    "coal_kb.metadata",
    "coal_kb.parsing",
    "coal_kb.pipelines",
    "coal_kb.qa",
    "coal_kb.query",
    "coal_kb.settings",
    "coal_kb.store",
    "coal_kb.web",
    "coal_kb.context.builder",
    "coal_kb.context.types",
    "coal_kb.retrieval.bm25",
    "coal_kb.retrieval.constraint_policy",
    "coal_kb.retrieval.elastic_retriever",
    "coal_kb.retrieval.filter_parser",
    "coal_kb.retrieval.query_rewrite",
    "coal_kb.retrieval.rerank",
    "coal_kb.retrieval.retriever",
}


def _is_removed_module(module_name: str) -> bool:
    return any(
        module_name == removed or module_name.startswith(f"{removed}.")
        for removed in REMOVED_MODULES
    )


def _iter_python_files() -> list[Path]:
    roots = [PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests"]
    return [
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_removed_paths_do_not_exist() -> None:
    existing = [str(path.relative_to(REPO_ROOT)) for path in REMOVED_PATHS if path.exists()]
    assert existing == []


def test_python_imports_use_only_canonical_modules() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module_name = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_removed_module(alias.name):
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if _is_removed_module(module_name):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{module_name}")
    assert violations == []


def test_backup_sources_are_absent() -> None:
    backups = [
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".bak" in path.name
    ]
    assert backups == []
