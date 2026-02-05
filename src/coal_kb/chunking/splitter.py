from __future__ import annotations

from typing import Any, List, Mapping

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .markdown_hierarchical_semantic import ChunkingParams, split_docs_markdown_hierarchical_semantic as _split_mhs
from .sectioner import infer_section


def split_page_docs(
    page_docs: List[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Legacy character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    enriched: List[Document] = []
    for d in page_docs:
        meta = dict(d.metadata or {})
        meta.setdefault("section", infer_section(d.page_content) or "unknown")
        enriched.append(Document(page_content=d.page_content, metadata=meta))

    return splitter.split_documents(enriched)


def split_docs_markdown_hierarchical_semantic(
    docs: List[Document],
    config: Mapping[str, Any] | ChunkingParams,
) -> List[Document]:
    """Split docs with markdown-aware parent/child semantic strategy."""
    if isinstance(config, ChunkingParams):
        params = config
    else:
        raw = dict(config)
        allowed = {k: raw[k] for k in ChunkingParams.__dataclass_fields__.keys() if k in raw}
        params = ChunkingParams(**allowed)
    return _split_mhs(docs, params)
