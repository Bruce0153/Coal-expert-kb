from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .sectioner import infer_section


def split_page_docs(
    page_docs: List[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split page-level docs into chunk-level docs.
    Keeps metadata and adds section if missing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Pre-enrich section metadata
    enriched: List[Document] = []
    for d in page_docs:
        meta = dict(d.metadata or {})
        meta.setdefault("section", infer_section(d.page_content) or "unknown")
        enriched.append(Document(page_content=d.page_content, metadata=meta))

    chunks = splitter.split_documents(enriched)

    # Add chunk_index within same source/page if needed later
    # (We keep it simple here; chunk_id is generated in Part 2.)
    return chunks
