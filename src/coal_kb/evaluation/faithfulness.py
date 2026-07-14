"""提供保持原公式的轻量回答可审计性检查。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from coal_kb.evaluation import config


def simple_faithfulness_check(answer: str, docs: list[Document]) -> float:
    """按数字引用标记数量计算旧版启发式分数。"""
    del docs
    if not answer.strip():
        return 0.0
    citations = re.findall(r"\[\d+\]", answer)
    return min(len(citations) / config.FAITHFULNESS_CITATION_TARGET, 1.0)


@dataclass
class FaithfulnessEvaluator:
    """保留现有 evaluate() 接口。"""

    def evaluate(self, answer: str, docs: list[Document]) -> float:
        return simple_faithfulness_check(answer, docs)
