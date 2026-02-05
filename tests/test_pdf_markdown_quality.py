from __future__ import annotations

from coal_kb.loaders.markdown_clean import collapse_repeated_headers, fix_hyphenation
from coal_kb.loaders.pdf_loader import PDFLoader
from coal_kb.settings import PDFMarkdownConfig


def test_fix_hyphenation() -> None:
    text = "gasifi-\ncation kinetics"
    assert "gasification" in fix_hyphenation(text)


def test_collapse_repeated_headers() -> None:
    pages = ["Journal X\ncontent 1\n10", "Journal X\ncontent 2\n10"]
    out = collapse_repeated_headers(pages)
    assert all("Journal X" not in x for x in out)


def test_heading_infer_from_dict(monkeypatch, tmp_path) -> None:
    class FakePage:
        def get_text(self, mode):
            assert mode == "dict"
            return {
                "blocks": [
                    {"type": 0, "lines": [{"bbox": [10, 10, 100, 20], "spans": [{"text": "Main Title", "size": 20}]}, {"bbox": [10, 40, 100, 50], "spans": [{"text": "Section", "size": 16}]}, {"bbox": [10, 70, 200, 80], "spans": [{"text": "body text", "size": 11}]}]},
                ]
            }

    class FakeDoc(list):
        def close(self):
            return None

    import types, sys
    fake_fitz = types.SimpleNamespace(open=lambda p: FakeDoc([FakePage()]))
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    docs = PDFLoader(PDFMarkdownConfig(enabled=True)).extract_pdf_markdown(pdf)
    assert docs and "# Main Title" in docs[0].page_content


def test_pdf_markdown_fallback_to_text(monkeypatch, tmp_path) -> None:
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr("coal_kb.loaders.pdf_loader.PDFLoader.extract_pdf_markdown", lambda self, p: (_ for _ in ()).throw(RuntimeError("x")))
    from langchain_core.documents import Document
    monkeypatch.setattr("coal_kb.loaders.pdf_loader.load_pdf_pages", lambda p: [Document(page_content="fallback", metadata={"source_file": str(p)})])
    docs = PDFLoader(PDFMarkdownConfig(enabled=True)).load(str(pdf))
    assert docs and docs[0].metadata.get("format") == "text"
