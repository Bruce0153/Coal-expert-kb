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

    model_name: str = "BAAI/bge-reranker-base"
    device: str = "auto"

    def rerank(self, query: str, docs: Sequence[Document], *, top_k: int = 6) -> List[Document]:
        if not docs:
            return []
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            logger.warning("transformers reranker unavailable: %s", e)
            return self._fallback_sentence_transformers(query, docs, top_k=top_k)

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, local_files_only=True
            )
        except Exception as e:
            logger.warning("Local reranker model unavailable: %s", e)
            return self._fallback_sentence_transformers(query, docs, top_k=top_k)
        model.to(device)
        model.eval()

        pairs = [(query, d.page_content) for d in docs]
        inputs = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()

        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:top_k]]

    def _fallback_sentence_transformers(
        self, query: str, docs: Sequence[Document], *, top_k: int
    ) -> List[Document]:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.warning("sentence-transformers CrossEncoder unavailable: %s", e)
            return list(docs)[:top_k]

        try:
            model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception as e:
            logger.warning("CrossEncoder local model unavailable: %s", e)
            return list(docs)[:top_k]
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:top_k]]
