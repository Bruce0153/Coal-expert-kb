"""根据案例指标生成稳定的失败归因。"""

from __future__ import annotations

from coal_kb.evaluation import config
from coal_kb.evaluation.models import EvaluationCase


def classify_failure(case: EvaluationCase, retrieval: dict[str, float], answer: dict[str, float]) -> str:
    """按照最靠前的失败环节进行归因。"""
    if retrieval.get("hit_at_10", retrieval.get("hit_at_5", 0.0)) == 0.0:
        return "RECALL"
    if retrieval.get("recall_at_10", retrieval.get("recall_at_5", 1.0)) < 1.0:
        return "EVIDENCE_COVERAGE"
    if answer:
        if answer.get("citation_precision", 1.0) < 1.0:
            return "CITATION"
        if answer.get("unsupported_claim_rate", 0.0) > 0.0:
            return "GENERATION"
        if answer.get("abstention_correct", 1.0) < 1.0:
            return "ABSTENTION"
    if case.expected_filters and retrieval.get("hit_at_1", 0.0) == 0.0:
        return "QUERY_PLAN"
    return config.FAILURE_NONE
