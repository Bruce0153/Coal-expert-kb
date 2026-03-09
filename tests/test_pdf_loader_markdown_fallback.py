from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.loaders.pdf_loader import PDFLoader


def test_pdf_loader_fallback_to_text(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake")

    def _raise(*args, **kwargs):
        raise RuntimeError("fitz unavailable")

    import builtins
    original_import = builtins.__import__

    def _mock_import(name, *a, **k):
        if name == "fitz":
            _raise()
        return original_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _mock_import)

    called = {"ok": False}

    def _fallback(path):
        called["ok"] = True
        return [Document(page_content="fallback text", metadata={"source_file": str(path), "page": 1})]

    monkeypatch.setattr("coal_kb.loaders.pdf_loader.load_pdf_pages", _fallback)

    docs = PDFLoader().load(str(pdf))
    assert called["ok"] is True
    assert docs
    assert docs[0].metadata.get("format") == "text"
