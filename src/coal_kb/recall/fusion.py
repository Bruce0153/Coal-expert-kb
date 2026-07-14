"""提供保持原顺序语义的 Reciprocal Rank Fusion。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from coal_kb.recall import config


def _document_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(
        metadata.get("chunk_id")
        or f'{metadata.get("source_file", "")}|{metadata.get("page", "")}|{doc.page_content[:60]}'
    )


def rrf_fuse(
    ranked_a: Sequence[Document],
    ranked_b: Sequence[Document],
    *,
    k: int = config.RRF_K,
) -> list[Document]:
    """使用 1-based rank 计算 RRF，公式与旧实现一致。"""
    scores: dict[str, float] = {}
    document_map: dict[str, Document] = {}

    for rank, doc in enumerate(ranked_a, start=1):
        key = _document_key(doc)
        document_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(ranked_b, start=1):
        key = _document_key(doc)
        document_map[key] = doc
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [document_map[key] for key, _ in fused]
