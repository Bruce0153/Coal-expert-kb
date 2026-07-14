"""定义回答层对外返回结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnswerResult:
    answer_text: str
    citations: dict[str, dict]
    used_chunks: list[str]
    debug: dict[str, Any]
    llm: dict | None = None
    referenced_labels: list[str] = field(default_factory=list)
    rendered_citations: list[str] = field(default_factory=list)
    evidence_items: list[dict[str, Any]] = field(default_factory=list)
    source_cards: list[dict[str, Any]] = field(default_factory=list)
    claim_items: list[dict[str, Any]] = field(default_factory=list)
    evidence_sufficiency: str = "insufficient"
    confidence_score: float = 0.0
