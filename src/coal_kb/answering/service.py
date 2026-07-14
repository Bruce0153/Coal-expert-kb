"""编排证据判断、LLM 生成、引用解析和回退回答。"""

from __future__ import annotations

from typing import Any

from coal_kb.answering.citations import build_rendered_citations, extract_referenced_labels
from coal_kb.answering.claims import build_claim_items
from coal_kb.answering.confidence import assess_evidence
from coal_kb.answering.models import AnswerResult
from coal_kb.answering.prompts import build_answer_prompt
from coal_kb.context.models import ContextPackage
from coal_kb.core.models.query import QueryPlan
from coal_kb.infra.providers.llm import LLMConfig, make_chat_llm


class Answerer:
    """保持原 Answerer 初始化和 answer() 调用方式。"""

    def __init__(
        self,
        *,
        enable_llm: bool = False,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.enable_llm = enable_llm
        self.llm_config = llm_config
        self._llm = None
        if enable_llm and llm_config is not None:
            self._llm = make_chat_llm(llm_config)

    def answer(
        self,
        plan: QueryPlan,
        context_package: ContextPackage,
        *,
        enable_llm: bool | None = None,
    ) -> AnswerResult:
        evidence_count = len(context_package.used_chunks)
        citations = {key: value.model_dump() for key, value in context_package.citations.items()}
        evidence_items = [value.model_dump() for value in context_package.evidence_items]
        source_cards = [value.model_dump() for value in context_package.source_cards]
        all_labels = list(citations.keys())
        evidence_sufficiency, confidence_score = assess_evidence(evidence_count, plan.answer.min_evidence)
        common: dict[str, Any] = {
            "citations": citations,
            "used_chunks": context_package.used_chunks,
            "evidence_items": evidence_items,
            "source_cards": source_cards,
            "evidence_sufficiency": evidence_sufficiency,
            "confidence_score": confidence_score,
        }

        if evidence_count < plan.answer.min_evidence:
            return AnswerResult(
                answer_text="Insufficient evidence（证据不足）：请补充更明确的工况或目标污染物证据。",
                debug={"reason": "insufficient_evidence", "evidence": evidence_count},
                referenced_labels=all_labels,
                rendered_citations=build_rendered_citations(citations, all_labels),
                claim_items=[],
                **common,
            )

        use_llm = self.enable_llm if enable_llm is None else enable_llm
        if not use_llm or self._llm is None:
            references = " ".join(f"[{label}]" for label in all_labels)
            text = (
                "基于检索证据，已检索到与问题相关的文献片段。\n\n"
                "由于当前未启用 LLM 归纳，下面给出证据引用，请结合原文核验：\n\n"
                f"{references}"
            )
            return AnswerResult(
                answer_text=text,
                debug={"context_debug": context_package.debug, "mode": "non_llm_fallback"},
                referenced_labels=all_labels,
                rendered_citations=build_rendered_citations(citations, all_labels),
                claim_items=build_claim_items(text, all_labels),
                **common,
            )

        user_question = plan.query.raw or plan.query.normalized
        prompt = build_answer_prompt(user_question, context_package.markdown)
        try:
            response = self._llm.invoke(prompt)
            content = getattr(response, "content", None)
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            else:
                text = str(content or "").strip()
            if not text:
                raise RuntimeError("LLM returned empty answer")

            referenced_labels = extract_referenced_labels(text, all_labels)
            return AnswerResult(
                answer_text=text,
                debug={"context_debug": context_package.debug, "mode": "llm_answer"},
                referenced_labels=referenced_labels,
                rendered_citations=build_rendered_citations(citations, referenced_labels),
                claim_items=build_claim_items(text, referenced_labels),
                **common,
            )
        except Exception as exc:
            references = " ".join(f"[{label}]" for label in all_labels)
            fallback = (
                "已检索到相关证据，但 LLM 归纳失败。请先结合以下证据核验：\n\n"
                f"{references}\n\n"
                f"错误信息：{type(exc).__name__}: {exc}"
            )
            return AnswerResult(
                answer_text=fallback,
                debug={"context_debug": context_package.debug, "mode": "llm_error", "error": str(exc)},
                referenced_labels=all_labels,
                rendered_citations=build_rendered_citations(citations, all_labels),
                claim_items=[],
                **common,
            )
