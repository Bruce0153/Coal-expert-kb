"""确保已停用的迁移标记和兼容入口不会重新进入仓库。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
}
IGNORED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
FORBIDDEN_MARKERS = {
    "leg" + "acy",
    "compat" + "_where",
    "where" + "_full",
    "use" + "_fuse",
    "coal_kb" + ".settings",
    "eval" + "_retrieval.py",
    "eval" + "_lora_extractor.py",
}


def test_retired_markers_are_absent() -> None:
    violations: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if IGNORED_PARTS.intersection(path.parts):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in FORBIDDEN_MARKERS:
            if marker in content:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}: {marker}"
                )
    assert violations == []
