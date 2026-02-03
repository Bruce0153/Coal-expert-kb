from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from coal_kb.loaders.base import BaseLoader, detect_language
from coal_kb.parsing.pdf_loader import load_pdf_pages
from coal_kb.loaders.registry import register_loader


class PDFLoader(BaseLoader):
    def load(self, path: str) -> List[Document]:
        docs = load_pdf_pages(Path(path))
        for doc in docs:
            lang = detect_language(doc.page_content or "")
            doc.metadata = {**(doc.metadata or {}), "doc_type": "pdf", "language": lang, "parser": "pdf"}
        return docs


register_loader("pdf", PDFLoader)
