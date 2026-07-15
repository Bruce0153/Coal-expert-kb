"""使用本地 CrossEncoder 模型执行重排序。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from coal_kb.infra.providers.config import LocalProviderConfig


@dataclass
class LocalCrossEncoderReranker:
    """延迟加载并复用本地 CrossEncoder。"""

    config: LocalProviderConfig
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.config.model_path or self.config.model,
                device=None if self.config.device == "auto" else self.config.device,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._model

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """使用本地模型计算候选相关性。"""
        if not docs:
            return []
        scores = self._get_model().predict([(query, doc.page_content or "") for doc in docs])
        ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
