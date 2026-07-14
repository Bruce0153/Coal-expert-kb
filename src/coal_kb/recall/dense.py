"""封装现有向量召回工厂，不改变后端调用协议。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass
class DenseRecall:
    """使用现有 retriever factory 执行单阶段向量召回。"""

    retriever_factory: Any

    def process(self, query: str, *, k: int, where: dict[str, Any]) -> list[Document]:
        retriever = self.retriever_factory(k=k, where=where)
        if hasattr(retriever, "get_relevant_documents"):
            return list(retriever.get_relevant_documents(query))
        return list(retriever.invoke(query))
