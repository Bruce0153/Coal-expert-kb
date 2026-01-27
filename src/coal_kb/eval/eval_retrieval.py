from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .datasets import EvalItem


def _match_gold(item: EvalItem, docs: List[Document]) -> bool:
    for gold in item.gold_sources:
        contains = str(gold.get("source_contains", "")).lower()
        page = gold.get("page", None)
        for d in docs:
            m = d.metadata or {}
            src = str(m.get("source_file", "")).lower()
            if contains and contains not in src:
                continue
            if page is not None:
                if m.get("page", None) == page:
                    return True
            else:
                return True
    return False


@dataclass
class RetrievalEvaluator:
    retrieve_fn: Any  # fn(question)->List[Document]

    def evaluate(self, items: List[EvalItem]) -> Dict[str, float]:
        hit = 0
        for it in items:
            docs = self.retrieve_fn(it.question)
            if _match_gold(it, docs):
                hit += 1
        total = max(len(items), 1)
        return {"recall": hit / total, "total": float(total), "hit": float(hit)}
