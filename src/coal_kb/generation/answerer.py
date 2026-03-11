from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from coal_kb.context.types import ContextPackage
from coal_kb.llm.factory import LLMConfig, make_chat_llm
from coal_kb.query.plan import QueryPlan

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\[(E\d+)\]")


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, dict]
    used_chunks: List[str]
    evidence_items: List[dict]
    source_cards: List[dict]
    claim_items: List[dict]
    rendered_citations: List[str]
    referenced_labels: List[str]
    evidence_sufficiency: str
    confidence_score: float
    debug: Dict[str, Any]


def _extract_referenced_labels(answer_text: str, citations: Dict[str, dict]) -> List[str]:
    seen: List[str] = []
    for label in _CITATION_PATTERN.findall(answer_text or ""):
        if label in citations and label not in seen:
            seen.append(label)
    return seen


def _extract_json_object(content: str) -> Optional[dict]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _evidence_sufficiency(evidence_count: int, source_count: int) -> str:
    if evidence_count < 2:
        return "insufficient"
    if source_count <= 1:
        return "limited"
    if source_count == 2:
        return "grounded"
    return "multi_source"


def _confidence_score(*, evidence_count: int, referenced_count: int, source_count: int) -> float:
    score = 0.15 + min(0.35, evidence_count * 0.1) + min(0.25, referenced_count * 0.08) + min(0.2, source_count * 0.08)
    return round(min(score, 0.95), 2)


def _rendered_citations(citations: Dict[str, dict], labels: List[str]) -> List[str]:
    rendered: List[str] = []
    for label in labels:
        item = citations.get(label)
        if not item:
            continue
        page = item.get("page")
        heading = item.get("heading_path")
        page_text = f" | page {page}" if page is not None else ""
        heading_text = f" | {heading}" if heading else ""
        rendered.append(f"[{label}] {item.get('source_file', 'unknown')}{page_text}{heading_text}")
    return rendered


def _claims_from_evidence(context_package: ContextPackage, *, max_claims: int = 3) -> List[dict]:
    claims: List[dict] = []
    for index, evidence in enumerate(context_package.evidence_items[:max_claims], start=1):
        claims.append(
            {
                "claim_id": f"C{index}",
                "text": evidence.snippet,
                "citations": [evidence.label],
                "support": "direct",
            }
        )
    return claims


def _fallback_answer(claim_items: List[dict], sufficiency: str) -> str:
    if not claim_items:
        return (
            "## Answer\n"
            "Insufficient evidence: no supporting passages were retrieved.\n\n"
            "## Evidence Sufficiency\n"
            "The retriever did not return enough grounded material to answer."
        )

    lines = ["## Answer", "Evidence-only mode is active, so the answer is a structured extractive summary.", ""]
    for claim in claim_items:
        citations = " ".join(f"[{label}]" for label in claim["citations"])
        lines.append(f"- {claim['text']} {citations}".rstrip())
    lines.extend(["", "## Evidence Sufficiency", f"Evidence status: {sufficiency}."])
    return "\n".join(lines)


def _build_prompt(query: str, context_markdown: str, conversation_context: str | None) -> List[object]:
    system_prompt = (
        "You are a retrieval-grounded assistant for a cite-aware RAG system. "
        "Use only the supplied evidence catalog. "
        "Return strict JSON with keys: answer_overview, claims, uncertainty. "
        "Each claim must contain: text, citations, support. "
        "Citations must be valid evidence labels such as E1. "
        "If evidence is weak or conflicting, say so in uncertainty."
    )
    history_block = f"Conversation context:\n{conversation_context}\n\n" if conversation_context else ""
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"{history_block}"
        "Build a concise answer from the evidence catalog below.\n\n"
        f"{context_markdown}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


def _claims_from_llm_payload(payload: dict, citations: Dict[str, dict]) -> List[dict]:
    claims: List[dict] = []
    for index, item in enumerate(payload.get("claims") or [], start=1):
        if not isinstance(item, dict):
            continue
        labels = [label for label in item.get("citations") or [] if label in citations]
        text = str(item.get("text") or "").strip()
        if not text or not labels:
            continue
        claims.append(
            {
                "claim_id": f"C{index}",
                "text": text,
                "citations": labels,
                "support": str(item.get("support") or "supported"),
            }
        )
    return claims


def _render_answer_markdown(answer_overview: str, claim_items: List[dict], uncertainty: str, sufficiency: str) -> str:
    lines = ["## Answer"]
    if answer_overview:
        lines.append(answer_overview)
        lines.append("")
    for claim in claim_items:
        citations = " ".join(f"[{label}]" for label in claim["citations"])
        lines.append(f"- {claim['text']} {citations}".rstrip())
    lines.extend(["", "## Evidence Sufficiency", uncertainty or f"Evidence status: {sufficiency}."])
    return "\n".join(lines)


class Answerer:
    def answer(
        self,
        plan: QueryPlan,
        context_package: ContextPackage,
        *,
        query: Optional[str] = None,
        enable_llm: bool = False,
        llm_config: Optional[LLMConfig] = None,
        conversation_context: Optional[str] = None,
    ) -> AnswerResult:
        evidence_count = len(context_package.used_chunks)
        source_count = len(context_package.source_cards)
        citations = {key: value.model_dump() for key, value in context_package.citations.items()}
        evidence_items = [item.model_dump() for item in context_package.evidence_items]
        source_cards = [item.model_dump() for item in context_package.source_cards]
        sufficiency = _evidence_sufficiency(evidence_count, source_count)

        if evidence_count < plan.answer.min_evidence:
            answer_text = (
                "## Answer\n"
                "Insufficient evidence to answer reliably.\n\n"
                "## Evidence Sufficiency\n"
                "Narrow the question or improve retrieval constraints."
            )
            return AnswerResult(
                answer_text=answer_text,
                citations=citations,
                used_chunks=context_package.used_chunks,
                evidence_items=evidence_items,
                source_cards=source_cards,
                claim_items=[],
                rendered_citations=[],
                referenced_labels=[],
                evidence_sufficiency="insufficient",
                confidence_score=0.0,
                debug={"reason": "insufficient_evidence", "evidence_count": evidence_count},
            )

        if enable_llm and llm_config is not None and query:
            try:
                model = make_chat_llm(llm_config)
                response = model.invoke(_build_prompt(query, context_package.markdown, conversation_context))
                content = getattr(response, "content", None) or ""
                payload = _extract_json_object(content)
                if payload:
                    claim_items = _claims_from_llm_payload(payload, citations)
                    referenced_labels = []
                    for claim in claim_items:
                        for label in claim["citations"]:
                            if label not in referenced_labels:
                                referenced_labels.append(label)
                    if claim_items and referenced_labels:
                        answer_text = _render_answer_markdown(
                            str(payload.get("answer_overview") or "").strip(),
                            claim_items,
                            str(payload.get("uncertainty") or "").strip(),
                            sufficiency,
                        )
                        return AnswerResult(
                            answer_text=answer_text,
                            citations=citations,
                            used_chunks=context_package.used_chunks,
                            evidence_items=evidence_items,
                            source_cards=source_cards,
                            claim_items=claim_items,
                            rendered_citations=_rendered_citations(citations, referenced_labels),
                            referenced_labels=referenced_labels,
                            evidence_sufficiency=sufficiency,
                            confidence_score=_confidence_score(
                                evidence_count=evidence_count,
                                referenced_count=len(referenced_labels),
                                source_count=source_count,
                            ),
                            debug={"mode": "llm", "context_debug": context_package.debug},
                        )
                logger.warning("LLM answer omitted valid structured claims; falling back to evidence-only output.")
            except Exception as exc:
                logger.warning("LLM answer generation failed; falling back to evidence-only output: %s", exc)

        claim_items = _claims_from_evidence(context_package)
        referenced_labels = []
        for claim in claim_items:
            for label in claim["citations"]:
                if label not in referenced_labels:
                    referenced_labels.append(label)
        answer_text = _fallback_answer(claim_items, sufficiency)
        return AnswerResult(
            answer_text=answer_text,
            citations=citations,
            used_chunks=context_package.used_chunks,
            evidence_items=evidence_items,
            source_cards=source_cards,
            claim_items=claim_items,
            rendered_citations=_rendered_citations(citations, referenced_labels),
            referenced_labels=referenced_labels,
            evidence_sufficiency=sufficiency,
            confidence_score=_confidence_score(
                evidence_count=evidence_count,
                referenced_count=len(referenced_labels),
                source_count=source_count,
            ),
            debug={"mode": "fallback", "context_debug": context_package.debug},
        )
