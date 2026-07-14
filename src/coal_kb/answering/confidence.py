"""按既有证据数量规则计算充分性与置信度。"""

from __future__ import annotations


def assess_evidence(evidence_count: int, min_evidence: int) -> tuple[str, float]:
    if evidence_count == 0:
        return "insufficient", 0.0
    if evidence_count < min_evidence:
        return "insufficient", round(evidence_count / max(min_evidence, 1) * 0.5, 2)
    if evidence_count < min_evidence * 2:
        return "partial", 0.65
    return "sufficient", min(0.95, 0.7 + evidence_count * 0.02)
