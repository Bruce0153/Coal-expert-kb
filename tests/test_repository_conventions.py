"""确保仓库只保留正式结构、规范文件名和有效 import。"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "coal_kb"

DISALLOWED_PATHS = [
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

DISALLOWED_MODULES = {
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


def _is_disallowed_module(module_name: str) -> bool:
    return any(
        module_name == disallowed or module_name.startswith(f"{disallowed}.")
        for disallowed in DISALLOWED_MODULES
    )


def _iter_python_files() -> list[Path]:
    roots = [PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests"]
    return [
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_disallowed_paths_do_not_exist() -> None:
    existing = [str(path.relative_to(REPO_ROOT)) for path in DISALLOWED_PATHS if path.exists()]
    assert existing == []


def test_python_imports_use_only_canonical_modules() -> None:
    violations: list[str] = []
    for path in _iter_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module_name = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_disallowed_module(alias.name):
                        violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if _is_disallowed_module(module_name):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{module_name}")
    assert violations == []


def test_generated_and_backup_sources_are_absent() -> None:
    backups = [
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".bak" in path.name
    ]
    assert backups == []


def test_repository_file_names_are_normalized() -> None:
    import re

    standard = {"README.md", "LICENSE", "Dockerfile", "pyproject.toml", "docker-compose.yml", "__init__.py", "config.py", "config.sh", "conftest.py", "index.html", "app.js", "styles.css"}
    snake = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    kebab = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    state = re.compile(r"(?:^|_)(?:old|backup|bak|final|optimized|copy|stage\d+)(?:_|\.|$)")
    violations: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}.intersection(path.parts):
            continue
        name = path.name
        if name in standard or name.startswith("."):
            continue
        stem = path.stem
        if state.search(name.lower()):
            violations.append(str(path.relative_to(REPO_ROOT)))
        elif path.relative_to(REPO_ROOT).parts[:2] == (".github", "workflows"):
            if path.suffix not in {".yml", ".yaml"} or not kebab.fullmatch(stem):
                violations.append(str(path.relative_to(REPO_ROOT)))
        elif path.suffix in {".py", ".sh", ".md", ".txt", ".jsonl", ".yaml", ".yml"} and not snake.fullmatch(stem):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == []
