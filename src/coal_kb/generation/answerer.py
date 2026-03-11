from __future__ import annotations

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
    referenced_labels: List[str]
    debug: Dict[str, Any]


def _extract_referenced_labels(answer_text: str, citations: Dict[str, dict]) -> List[str]:
    seen = []
    for label in _CITATION_PATTERN.findall(answer_text or ""):
        if label in citations and label not in seen:
            seen.append(label)
    return seen


def _fallback_answer(context_package: ContextPackage) -> str:
    if not context_package.evidence_items:
        return (
            "## Answer\n"
            "Insufficient evidence: no supporting passages were retrieved.\n\n"
            "## Evidence Sufficiency\n"
            "The retriever did not return enough grounded material to answer."
        )

    lines = [
        "## Answer",
        "Evidence-only mode is active, so the system is not synthesizing a prose answer.",
        "",
        "## Best Available Evidence",
    ]
    for citation in context_package.evidence_items:
        lines.append(f"- [{citation.label}] {citation.snippet}")
    lines.extend(
        [
            "",
            "## Evidence Sufficiency",
            f"Retrieved {len(context_package.evidence_items)} grounded evidence chunk(s).",
        ]
    )
    return "\n".join(lines)


def _build_prompt(query: str, context_markdown: str) -> List[object]:
    system_prompt = (
        "You are a retrieval-grounded assistant for a RAG system demo. "
        "Answer only from the supplied evidence catalog. "
        "Every factual claim must cite one or more evidence labels such as [E1]. "
        "If evidence is weak, missing, or conflicting, say so explicitly. "
        "Do not cite labels that do not exist. "
        "Return Markdown with exactly these sections: "
        "'## Answer', '## Evidence Sufficiency', and optionally '## Notes'."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        "Use the evidence catalog below. Keep citations inline with each material claim.\n\n"
        f"{context_markdown}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


class Answerer:
    def answer(
        self,
        plan: QueryPlan,
        context_package: ContextPackage,
        *,
        query: Optional[str] = None,
        enable_llm: bool = False,
        llm_config: Optional[LLMConfig] = None,
    ) -> AnswerResult:
        evidence_count = len(context_package.used_chunks)
        citations = {key: value.model_dump() for key, value in context_package.citations.items()}

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
                referenced_labels=[],
                debug={"reason": "insufficient_evidence", "evidence_count": evidence_count},
            )

        if enable_llm and llm_config is not None and query:
            try:
                model = make_chat_llm(llm_config)
                response = model.invoke(_build_prompt(query, context_package.markdown))
                content = getattr(response, "content", None) or ""
                referenced_labels = _extract_referenced_labels(content, citations)
                if content.strip() and referenced_labels:
                    return AnswerResult(
                        answer_text=content.strip(),
                        citations=citations,
                        used_chunks=context_package.used_chunks,
                        referenced_labels=referenced_labels,
                        debug={"mode": "llm", "context_debug": context_package.debug},
                    )
                logger.warning("LLM answer omitted valid evidence labels; falling back to evidence-only output.")
            except Exception as exc:
                logger.warning("LLM answer generation failed; falling back to evidence-only output: %s", exc)

        answer_text = _fallback_answer(context_package)
        referenced_labels = [citation.label for citation in context_package.evidence_items]
        return AnswerResult(
            answer_text=answer_text,
            citations=citations,
            used_chunks=context_package.used_chunks,
            referenced_labels=referenced_labels,
            debug={"mode": "fallback", "context_debug": context_package.debug},
        )
