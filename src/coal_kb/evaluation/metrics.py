"""计算检索、引用、Claim 和拒答指标。"""

from __future__ import annotations

import math

from coal_kb.evaluation.models import AnswerObservation, EvidenceReference, EvaluationCase, RetrievedEvidence


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


def retrieval_metrics(
    case: EvaluationCase,
    retrieved: tuple[RetrievedEvidence, ...],
    k_values: tuple[int, ...],
) -> dict[str, float]:
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
    expected_pages = {
        ((item.source_file or "").lower(), item.page)
        for item in expected
        if item.page is not None
    }
    actual_pages = {((item.source_file or "").lower(), item.page) for item in retrieved}
    metrics["page_recall"] = len(expected_pages & actual_pages) / len(expected_pages) if expected_pages else 1.0
    return metrics


def answer_metrics(case: EvaluationCase, answer: AnswerObservation | None) -> dict[str, float]:
    """计算引用、Claim 支撑和拒答指标。"""
    if answer is None:
        return {}
    expected = case.expected_evidence
    matched_citations = sum(
        1 for citation in answer.citations if any(evidence_matches(gold, citation) for gold in expected)
    )
    citation_precision = matched_citations / len(answer.citations) if answer.citations else (1.0 if not expected else 0.0)
    matched_gold = sum(
        1 for gold in expected if any(evidence_matches(gold, citation) for citation in answer.citations)
    )
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
