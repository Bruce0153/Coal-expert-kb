from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

from coal_kb.parsing.table_extractor import TableExtractor


class FakeTable:
    def __init__(self, data: str, page: int = 1):
        self.df = SimpleNamespace(to_csv=lambda index=False: data)
        self.page = page


def test_table_extractor_auto_fallback(monkeypatch, tmp_path: Path):
    calls = []

    def read_pdf(path, pages="all", flavor="lattice"):
        calls.append(flavor)
        if flavor == "lattice":
            return []
        return [FakeTable("a,b\n1,2\n")]

    fake_camelot = SimpleNamespace(read_pdf=read_pdf)
    monkeypatch.setitem(sys.modules, "camelot", fake_camelot)

    extractor = TableExtractor(flavor="auto")
    docs = extractor.extract(tmp_path / "fake.pdf")

    assert calls == ["lattice", "stream"]
    assert len(docs) == 1
    assert docs[0].metadata.get("section") == "table"
