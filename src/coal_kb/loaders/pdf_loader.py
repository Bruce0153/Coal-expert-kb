from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language
from coal_kb.loaders.registry import register_loader
from coal_kb.parsing.pdf_loader import load_pdf_pages

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    def _load_markdown(self, path: Path) -> List[Document]:
        try:
            import fitz  # pymupdf

            pdf = fitz.open(path)
            pages: List[str] = []
            page_metas: List[dict] = []
            for idx, page in enumerate(pdf, start=1):
                md = page.get_text("markdown") or ""
                if md.strip():
                    pages.append(md)
                    page_metas.append({"page": idx})
            pdf.close()
            if not pages:
                return []
            content = "\n\n".join(pages)
            meta = {
                "source_file": str(path),
                "doc_type": "pdf",
                "parser": "pymupdf_markdown",
                "format": "markdown",
                "page_range": [1, len(pages)],
                "pages": page_metas,
                "language": detect_language(content),
            }
            return [Document(page_content=content, metadata=meta)]
        except Exception as exc:
            logger.warning("PDF markdown extraction failed, fallback to text: %s", exc)
            return []

    def load(self, path: str) -> List[Document]:
        p = Path(path)
        docs = self._load_markdown(p)
        if docs:
            return docs

        docs = load_pdf_pages(p)
        for doc in docs:
            lang = detect_language(doc.page_content or "")
            doc.metadata = {
                **(doc.metadata or {}),
                "doc_type": "pdf",
                "language": lang,
                "parser": "pdf",
                "format": "text",
            }
        return docs


register_loader("pdf", PDFLoader)
