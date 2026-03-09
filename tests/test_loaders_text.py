from __future__ import annotations

from pathlib import Path

from coal_kb.loaders import load_any


def test_text_loader_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("标题\n\n段落一。\n段落二。", encoding="utf-8")
    docs = load_any(str(file_path))
    assert docs
    assert docs[0].metadata.get("doc_type") == "txt"


def test_text_loader_md(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("# 标题\n\n段落一。\n", encoding="utf-8")
    docs = load_any(str(file_path))
    assert docs
    assert docs[0].metadata.get("doc_type") == "md"
