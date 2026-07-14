"""将应用层字典转换为稳定的 HTTP 响应模型。"""

from typing import Any

from coal_kb.interfaces.api.models import (
    AskResponse,
    CitationResponse,
    ClaimResponse,
    SourceCardResponse,
)


def build_ask_response(payload: dict[str, Any]) -> AskResponse:
    citations = [CitationResponse.model_validate(item) for item in payload["citations"]]
    source_cards = [SourceCardResponse.model_validate(item) for item in payload["source_cards"]]
    claim_items = [ClaimResponse.model_validate(item) for item in payload["claim_items"]]
    return AskResponse(
        query=payload["query"],
        retrieval_query=payload["retrieval_query"],
        answer=payload["answer"],
        referenced_labels=payload["referenced_labels"],
        rendered_citations=payload["rendered_citations"],
        citations=citations,
        used_chunks=payload["used_chunks"],
        evidence_items=payload["evidence_items"],
        source_cards=source_cards,
        claim_items=claim_items,
        retrieval_trace_summary=payload["retrieval_trace_summary"],
        evidence_sufficiency=payload["evidence_sufficiency"],
        confidence_score=payload["confidence_score"],
        timings_ms=payload["timings_ms"],
        diagnostics=payload["diagnostics"],
    )
