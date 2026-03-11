from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from coal_kb.context.types import ContextPackage
from coal_kb.llm.factory import LLMConfig, make_chat_llm
from coal_kb.query.plan import QueryPlan

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, dict]
    used_chunks: List[str]
    debug: Dict[str, Any]


def _fallback_answer(context_package: ContextPackage) -> str:
    if not context_package.citations:
        return "Insufficient evidence: no supporting passages were retrieved."

    lines = ["LLM answering is disabled. Review the grounded evidence below:"]
    for sid in context_package.citations:
        marker = f"[{sid}]"
        snippet = ""
        for line in context_package.markdown.splitlines():
            if line.startswith(marker):
                snippet = line
                break
        detail = snippet[len(marker):].strip() if snippet else "See retrieved context."
        lines.append(f"- {marker} {detail}")
    return "\n".join(lines)


def _build_prompt(query: str, context_markdown: str) -> List[object]:
    system_prompt = (
        "You are a retrieval-grounded assistant. "
        "Answer only from the supplied evidence. "
        "Do not invent missing facts. "
        "Cite each key claim with references such as [S1]. "
        "If the evidence is insufficient or conflicting, say so clearly."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        "Answer from the evidence below and preserve citation markers:\n"
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
            return AnswerResult(
                answer_text="Insufficient evidence to answer reliably. Narrow the question or improve retrieval constraints.",
                citations=citations,
                used_chunks=context_package.used_chunks,
                debug={"reason": "insufficient_evidence", "evidence_count": evidence_count},
            )

        if enable_llm and llm_config is not None and query:
            try:
                model = make_chat_llm(llm_config)
                response = model.invoke(_build_prompt(query, context_package.markdown))
                content = getattr(response, "content", None) or ""
                if content.strip():
                    return AnswerResult(
                        answer_text=content.strip(),
                        citations=citations,
                        used_chunks=context_package.used_chunks,
                        debug={"mode": "llm", "context_debug": context_package.debug},
                    )
            except Exception as exc:
                logger.warning("LLM answer generation failed; falling back to evidence-only output: %s", exc)

        return AnswerResult(
            answer_text=_fallback_answer(context_package),
            citations=citations,
            used_chunks=context_package.used_chunks,
            debug={"mode": "fallback", "context_debug": context_package.debug},
        )
