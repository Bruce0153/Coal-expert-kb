"""计算检索、复杂路线、引用、Claim 和拒答指标。"""

from __future__ import annotations

import math
import re

from coal_kb.evaluation.models import (
    AnswerObservation,
    EvaluationCase,
    EvidenceReference,
    QueryType,
    RetrievedEvidence,
)


def evidence_matches(expected: EvidenceReference, actual: RetrievedEvidence | EvidenceReference) -> bool:
    """按最具体的可用字段判断证据是否匹配。"""
    if expected.chunk_id:
        return expected.chunk_id == actual.chunk_id
    if expected.document_id and expected.document_id != actual.document_id:
        return False
    if expected.source_file:
        source = (actual.source_file or "").lower()
        if expected.source_file.lower() not in source:
            return False
    if expected.page is not None and expected.page != actual.page:
        return False
    if expected.section:
        section = (actual.section or "").lower()
        if expected.section.lower() not in section:
            return False
    return bool(expected.document_id or expected.source_file or expected.page is not None)


def retrieval_metrics(case: EvaluationCase, retrieved: tuple[RetrievedEvidence, ...], k_values: tuple[int, ...]) -> dict[str, float]:
    """计算分层检索指标。"""
    expected = case.expected_evidence
    metrics: dict[str, float] = {}
    first_relevant_rank: int | None = None
    relevance = [1 if any(evidence_matches(gold, item) for gold in expected) else 0 for item in retrieved]
    for index, relevant in enumerate(relevance, start=1):
        if relevant and first_relevant_rank is None:
            first_relevant_rank = index
    metrics["mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    for k in k_values:
        prefix = retrieved[:k]
        matched_gold = {
            gold_index
            for gold_index, gold in enumerate(expected)
            if any(evidence_matches(gold, item) for item in prefix)
        }
        relevant_count = sum(1 for item in prefix if any(evidence_matches(gold, item) for gold in expected))
        metrics[f"hit_at_{k}"] = 1.0 if matched_gold else 0.0
        metrics[f"recall_at_{k}"] = len(matched_gold) / len(expected) if expected else 1.0
        metrics[f"precision_at_{k}"] = relevant_count / len(prefix) if prefix else 0.0
        gains = relevance[:k]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
        ideal_hits = min(len(expected), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"ndcg_at_{k}"] = dcg / ideal_dcg if ideal_dcg else 1.0
    expected_sources = {item.source_file.lower() for item in expected if item.source_file}
    actual_sources = {(item.source_file or "").lower() for item in retrieved if item.source_file}
    metrics["source_recall"] = (
        sum(1 for source in expected_sources if any(source in actual for actual in actual_sources)) / len(expected_sources)
        if expected_sources
        else 1.0
    )
    expected_pages = {((item.source_file or "").lower(), item.page) for item in expected if item.page is not None}
    actual_pages = {((item.source_file or "").lower(), item.page) for item in retrieved}
    metrics["page_recall"] = len(expected_pages & actual_pages) / len(expected_pages) if expected_pages else 1.0
    return metrics


def complex_question_metrics(case: EvaluationCase, trace: dict, retrieved: tuple[RetrievedEvidence, ...]) -> dict[str, float]:
    """计算路由、分解、证据链、表格和跨文档指标。"""
    route = trace.get("complex_route") or trace.get("plan") or {}
    execution = trace.get("complex_execution") or {}
    actual_type = str(route.get("query_type") or execution.get("query_type") or "fact")
    expected_type = "fact" if case.query_type == QueryType.CONDITION else case.query_type.value
    metrics = {"route_accuracy": float(actual_type == expected_type)}

    actual_subqueries = [str(item.get("query") or "") for item in route.get("subqueries") or []]
    if case.expected_subqueries:
        matched = sum(
            1
            for expected in case.expected_subqueries
            if any(_text_overlap(expected, actual) >= 0.5 for actual in actual_subqueries)
        )
        metrics["subquery_recall"] = matched / len(case.expected_subqueries)

    if case.expected_operation:
        actual_operation = ((route.get("aggregation") or {}).get("operation") or execution.get("operation"))
        metrics["aggregation_operation_accuracy"] = float(actual_operation == case.expected_operation)

    sources = {item.source_file for item in retrieved if item.source_file and item.source_file != "computed_aggregation"}
    metrics["source_diversity"] = float(len(sources))
    if case.query_type == case.query_type.CROSS_DOCUMENT:
        metrics["cross_document_coverage"] = min(1.0, len(sources) / case.expected_min_sources)
    if case.query_type == case.query_type.COMPARISON:
        sides = {str(item.metadata.get("complex_role")) for item in retrieved if item.metadata.get("complex_role")}
        metrics["comparison_side_coverage"] = min(1.0, len(sides) / 2)
    if case.query_type == case.query_type.MULTI_HOP:
        steps = execution.get("steps") or []
        metrics["chain_completeness"] = float(bool(steps) and all(int(step.get("hits", 0)) > 0 for step in steps))
    if case.query_type == case.query_type.TABLE:
        table_hits = [item for item in retrieved if item.metadata.get("table_id")]
        metrics["table_evidence_rate"] = len(table_hits) / len(retrieved) if retrieved else 0.0
        if case.expected_table_ids:
            actual_ids = {str(item.metadata.get("table_id")) for item in table_hits}
            metrics["table_id_recall"] = sum(1 for value in case.expected_table_ids if value in actual_ids) / len(case.expected_table_ids)
    return metrics


def answer_metrics(case: EvaluationCase, answer: AnswerObservation | None) -> dict[str, float]:
    """计算引用、Claim 支撑和拒答指标。"""
    if answer is None:
        return {}
    expected = case.expected_evidence
    matched_citations = sum(1 for citation in answer.citations if any(evidence_matches(gold, citation) for gold in expected))
    citation_precision = matched_citations / len(answer.citations) if answer.citations else (1.0 if not expected else 0.0)
    matched_gold = sum(1 for gold in expected if any(evidence_matches(gold, citation) for citation in answer.citations))
    citation_recall = matched_gold / len(expected) if expected else 1.0
    unsupported = sum(1 for claim in answer.claims if not claim.supported)
    unsupported_rate = unsupported / len(answer.claims) if answer.claims else 0.0
    abstention_correct = float(answer.abstained == (not case.answerable))
    return {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "claim_support_rate": 1.0 - unsupported_rate,
        "unsupported_claim_rate": unsupported_rate,
        "abstention_correct": abstention_correct,
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """对案例级指标做宏平均。"""
    keys = sorted({key for result in results for key in result})
    return {
        key: sum(result[key] for result in results if key in result) / sum(1 for result in results if key in result)
        for key in keys
    }


def _text_overlap(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[a-z0-9_.+-]+|[一-鿿]", left.lower()))
    right_terms = set(re.findall(r"[a-z0-9_.+-]+|[一-鿿]", right.lower()))
    return len(left_terms & right_terms) / max(1, len(left_terms))
