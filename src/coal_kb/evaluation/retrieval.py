"""评估检索结果是否命中人工标注来源。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document

from coal_kb.evaluation.datasets import EvalItem


def _match_gold(item: EvalItem, docs: list[Document]) -> bool:
    for gold in item.gold_sources:
        source_contains = str(gold.get("source_contains", "")).lower()
        page = gold.get("page")
        for doc in docs:
            metadata = doc.metadata or {}
            source_file = str(metadata.get("source_file", "")).lower()
            if source_contains and source_contains not in source_file:
                continue
            if page is not None:
                if metadata.get("page") == page:
                    return True
            else:
                return True
    return False


@dataclass
class RetrievalEvaluator:
    """持有检索函数并计算旧版 recall、total 和 hit。"""

    retrieve_fn: Callable[[str], list[Document]]

    def evaluate(self, items: list[EvalItem]) -> dict[str, float]:
        hit = 0
        for item in items:
            docs = self.retrieve_fn(item.question)
            if _match_gold(item, docs):
                hit += 1
        total = max(len(items), 1)
        return {"recall": hit / total, "total": float(total), "hit": float(hit)}
