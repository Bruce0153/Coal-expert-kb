from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class CrossEncoderReranker:
    """
    Optional reranker using sentence-transformers CrossEncoder.

    Pros: improves precision notably for scientific QA.
    Cons: downloads model, slower.

    Usage:
      reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
      docs = reranker.rerank(query, docs, top_k=6)
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def rerank(self, query: str, docs: Sequence[Document], *, top_k: int = 6) -> List[Document]:
        if not docs:
            return []
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.warning("sentence-transformers CrossEncoder unavailable: %s", e)
            return list(docs)[:top_k]

        model = CrossEncoder(self.model_name)
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:top_k]]
