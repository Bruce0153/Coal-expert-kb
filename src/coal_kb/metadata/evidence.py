from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class FieldEvidence:
    field: str
    value: object
    confidence: float
    start: int
    end: int
    evidence_text: str

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": self.value,
            "confidence": float(self.confidence),
            "start": int(self.start),
            "end": int(self.end),
            "evidence_text": self.evidence_text,
        }


def clip_span(text: str, start: int, end: int, *, window: int = 50) -> Tuple[int, int, str]:
    """
    Build a readable evidence snippet around [start,end).
    """
    if start < 0:
        start = 0
    if end < start:
        end = start
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    snippet = text[lo:hi].replace("\n", " ").strip()
    return lo, hi, snippet
