from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


def simple_faithfulness_check(answer: str, docs: List[Document]) -> float:
    """
    Very lightweight heuristic:
    - Count how many claims mention citation markers like [1], [2].
    - Not a real faithfulness metric, but useful as a sanity check.
    """
    if not answer.strip():
        return 0.0
    citations = re.findall(r"\[\d+\]", answer)
    # normalize: more citations => better "auditability"
    return min(len(citations) / 6.0, 1.0)


@dataclass
class FaithfulnessEvaluator:
    def evaluate(self, answer: str, docs: List[Document]) -> float:
        return simple_faithfulness_check(answer, docs)
