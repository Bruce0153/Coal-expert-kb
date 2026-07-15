"""根据案例指标生成稳定的复杂问答失败归因。"""

from __future__ import annotations

from coal_kb.evaluation import config
from coal_kb.evaluation.models import EvaluationCase


def classify_failure(
    case: EvaluationCase,
    retrieval: dict[str, float],
    answer: dict[str, float],
    complex_values: dict[str, float] | None = None,
) -> str:
    """按照最靠前的失败环节进行归因。"""
    complex_values = complex_values or {}
    if complex_values.get("route_accuracy", 1.0) < 1.0:
        return "ROUTE"
    if complex_values.get("subquery_recall", 1.0) < 1.0:
        return "DECOMPOSITION"
    if complex_values.get("aggregation_operation_accuracy", 1.0) < 1.0:
        return "AGGREGATION"
    if complex_values.get("table_id_recall", 1.0) < 1.0:
        return "TABLE"
    if complex_values.get("cross_document_coverage", 1.0) < 1.0:
        return "CROSS_DOCUMENT"
    if complex_values.get("chain_completeness", 1.0) < 1.0:
        return "MULTI_HOP"
    hit_values = [value for key, value in retrieval.items() if key.startswith("hit_at_")]
    recall_values = [value for key, value in retrieval.items() if key.startswith("recall_at_")]
    if case.expected_evidence and (not hit_values or max(hit_values) == 0.0):
        return "RECALL"
    if case.expected_evidence and recall_values and max(recall_values) < 1.0:
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
