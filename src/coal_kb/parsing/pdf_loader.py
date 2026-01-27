from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from ..utils.text_clean import (
    basic_clean,
    find_common_header_footer_lines,
    remove_common_header_footer,
)

logger = logging.getLogger(__name__)


def load_pdf_pages(pdf_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    # First pass: clean texts
    cleaned_pages: List[Document] = []
    page_texts: List[str] = []

    for d in docs:
        text = basic_clean(d.page_content or "")
        if not text:
            continue
        meta = dict(d.metadata or {})
        meta["source_file"] = str(pdf_path)
        cleaned_pages.append(Document(page_content=text, metadata=meta))
        page_texts.append(text)

    # Second pass: detect and remove common header/footer lines
    common = find_common_header_footer_lines(
        page_texts,
        head_n=3,
        tail_n=3,
        min_ratio=0.6,
        min_len=6,
        max_len=120,
    )

    if common:
        out: List[Document] = []
        for d in cleaned_pages:
            new_text = remove_common_header_footer(d.page_content, common, head_n=3, tail_n=3)
            if not new_text:
                continue
            out.append(Document(page_content=new_text, metadata=d.metadata))
        return out

    return cleaned_pages


def load_pdfs_from_dir(raw_dir: Path) -> List[Document]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {raw_dir}")

    pdfs = sorted(raw_dir.rglob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs found under %s", raw_dir)

    all_docs: List[Document] = []
    for p in pdfs:
        try:
            all_docs.extend(load_pdf_pages(p))
        except Exception as e:
            logger.exception("Failed to load PDF %s: %s", p, e)
    return all_docs
