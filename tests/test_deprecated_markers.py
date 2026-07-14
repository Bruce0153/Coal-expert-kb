"""确保迁移期文本标记不会重新进入当前仓库树。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".jsonl", ".yaml", ".yml", ".toml"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def test_deprecated_migration_marker_is_absent() -> None:
    marker = "leg" + "acy"
    violations: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if IGNORED_PARTS.intersection(path.parts):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").lower()
        if marker in content:
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == []
