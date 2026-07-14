"""封装现有 reranker 的候选截断和顺序合并逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


def _document_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    return str(metadata.get("chunk_id") or f'{metadata.get("source_file", "")}|{metadata.get("page", "")}')


@dataclass
class RerankingService:
    """持有远程或本地 reranker，并保持未重排候选的原相对顺序。"""

    reranker: Any

    def process(self, query: str, docs: list[Document], *, top_k: int) -> list[Document]:
        if not docs or top_k <= 0:
            return docs
        candidate_k = min(top_k, len(docs))
        reranked = list(self.reranker.rerank(query, docs[:candidate_k], top_k=candidate_k))
        seen = {_document_key(doc) for doc in reranked}
        return reranked + [doc for doc in docs if _document_key(doc) not in seen]
