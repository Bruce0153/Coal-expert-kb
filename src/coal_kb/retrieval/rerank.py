from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class DashScopeReranker:
    """
    DashScope(OpenAI-compatible) reranker.
    Default model: qwen3-rerank

    Notes:
    - Different deployments may expose rerank endpoint as /v1/rerank or /rerank.
      We try both.
    - If API key is missing or request fails, we fall back to "no rerank" behavior.
    """

    base_url: str
    api_key_env: str
    model: str = "qwen3-rerank"
    timeout: int = 60

    def _endpoint_candidates(self) -> List[str]:
        b = self.base_url.rstrip("/")
        return [f"{b}/v1/reranks", f"{b}/reranks"]

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            logger.warning("DashScope rerank skipped: env %s is empty.", self.api_key_env)
            return docs[:top_k]

        texts = [(d.page_content or "")[:4000] for d in docs]  # safety truncate

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": min(top_k, len(texts)),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[Exception] = None
        for url in self._endpoint_candidates():
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                data = r.json() if r.content else {}

                # Common shapes:
                # 1) {"results":[{"index":0,"relevance_score":...}, ...]}
                # 2) {"data":[{"index":...}, ...]}
                # 3) {"scores":[...]} aligned with docs
                results = data.get("results") or data.get("data") or []
                if results:
                    indices: List[int] = []
                    for item in results:
                        idx = item.get("index")
                        if isinstance(idx, int):
                            indices.append(idx)
                    if indices:
                        reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
                        return reranked[:top_k]

                scores = data.get("scores")
                if isinstance(scores, list) and len(scores) == len(docs):
                    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
                    return [docs[i] for i in order[:top_k]]

                # If response is valid but unrecognized, do not break the pipeline
                logger.warning("DashScope rerank response unrecognized. Using original order.")
                return docs[:top_k]
            except Exception as e:
                last_err = e
                continue

        logger.warning("DashScope rerank failed (all endpoints). Last error: %s", last_err)
        return docs[:top_k]


@dataclass
class CrossEncoderReranker:
    """
    Local reranker using transformers / sentence-transformers CrossEncoder.

    If DashScope rerank is configured, this is used only as fallback.

    Usage:
      reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
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
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=False)
        except Exception as e:
            logger.warning("Local reranker model unavailable: %s", e)
            return self._fallback_sentence_transformers(query, docs, top_k=top_k)

        model.to(device)
        model.eval()

        pairs = [(query, d.page_content or "") for d in docs]
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

    def _fallback_sentence_transformers(self, query: str, docs: Sequence[Document], *, top_k: int) -> List[Document]:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.warning("sentence-transformers CrossEncoder unavailable: %s", e)
            return list(docs)[:top_k]

        try:
            model = CrossEncoder(self.model_name, local_files_only=False)
        except Exception as e:
            logger.warning("CrossEncoder model unavailable: %s", e)
            return list(docs)[:top_k]

        pairs = [(query, d.page_content or "") for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in ranked[:top_k]]


def make_reranker(cfg) -> object:
    """
    Prefer API reranker (cfg.rerank.*) when provider=dashscope.
    Fall back to local CrossEncoderReranker using cfg.retrieval.rerank_model/device.
    """
    try:
        rr = getattr(cfg, "rerank", None)
        if rr and getattr(rr, "provider", "") == "dashscope":
            return DashScopeReranker(
                base_url=rr.base_url,
                api_key_env=rr.api_key_env,
                model=getattr(rr, "model", "qwen3-rerank"),
                timeout=getattr(rr, "timeout", 60),
            )
    except Exception:
        pass

    return CrossEncoderReranker(
        model_name=getattr(cfg.retrieval, "rerank_model", "BAAI/bge-reranker-base"),
        device=getattr(cfg.retrieval, "rerank_device", "auto"),
    )
