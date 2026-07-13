from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coal_kb.context.types import ContextPackage
from coal_kb.core.models.query import QueryPlan
from coal_kb.infra.providers.llm import LLMConfig, make_chat_llm


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, dict]
    used_chunks: List[str]
    debug: Dict[str, Any]
    llm: Optional[dict] = None
    referenced_labels: List[str] = field(default_factory=list)
    rendered_citations: List[str] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    source_cards: List[Dict[str, Any]] = field(default_factory=list)
    claim_items: List[Dict[str, Any]] = field(default_factory=list)
    evidence_sufficiency: str = "insufficient"
    confidence_score: float = 0.0

class Answerer:
    def __init__(
        self,
        *,
        enable_llm: bool = False,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        self.enable_llm = enable_llm
        self.llm_config = llm_config
        self._llm = None

        if enable_llm and llm_config is not None:
            self._llm = make_chat_llm(llm_config)

    @staticmethod
    def _extract_referenced_labels(text: str, available_labels: List[str]) -> List[str]:
        """Parse [E1], [E2] style references from answer text."""
        found = re.findall(r"\[(E\d+)\]", text)
        seen = set()
        result = []
        for label in found:
            if label not in seen:
                seen.add(label)
                result.append(label)
        if not result:
            return available_labels
        return result

    @staticmethod
    def _build_rendered_citations(citations: Dict[str, dict], referenced_labels: List[str]) -> List[str]:
        """Build human-readable citation strings like '[E1] source_file (page X)'."""
        result = []
        for label in referenced_labels:
            item = citations.get(label)
            if item is None:
                continue
            source = item.get("source_file", "unknown")
            page = item.get("page")
            if page is not None:
                result.append(f"[{label}] {source} (page {page})")
            else:
                result.append(f"[{label}] {source}")
        return result

    @staticmethod
    def _assess_evidence(evidence_count: int, min_evidence: int) -> tuple:
        """Return (evidence_sufficiency, confidence_score) based on evidence count."""
        if evidence_count == 0:
            return "insufficient", 0.0
        if evidence_count < min_evidence:
            return "insufficient", round(evidence_count / max(min_evidence, 1) * 0.5, 2)
        if evidence_count < min_evidence * 2:
            return "partial", 0.65
        return "sufficient", min(0.95, 0.7 + evidence_count * 0.02)

    @staticmethod
    def _build_claim_items(text: str, referenced_labels: List[str]) -> List[Dict[str, Any]]:
        """Build basic claim items from answer text and referenced labels."""
        if not text:
            return []
        sentences = re.split(r"(?<=[。.!?！？])\s*", text)
        claims = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            labels_in_sentence = re.findall(r"\[(E\d+)\]", sentence)
            unique_labels = list(dict.fromkeys(labels_in_sentence))
            claims.append({
                "claim_id": f"C{i + 1}",
                "text": sentence,
                "citations": unique_labels,
                "support": "direct" if unique_labels else "contextual",
            })
        if not claims:
            claims.append({
                "claim_id": "C1",
                "text": text[:200],
                "citations": referenced_labels,
                "support": "contextual",
            })
        return claims[:20]

    def answer(
        self,
        plan: QueryPlan,
        context_package: ContextPackage,
        *,
        enable_llm: Optional[bool] = None,
    ) -> AnswerResult:
        ev_count = len(context_package.used_chunks)
        citations = {k: v.model_dump() for k, v in context_package.citations.items()}
        evidence_items = [v.model_dump() for v in context_package.evidence_items]
        source_cards = [v.model_dump() for v in context_package.source_cards]
        all_labels = list(citations.keys())
        evidence_sufficiency, confidence_score = self._assess_evidence(ev_count, plan.answer.min_evidence)
        common = dict(
            citations=citations,
            used_chunks=context_package.used_chunks,
            evidence_items=evidence_items,
            source_cards=source_cards,
            evidence_sufficiency=evidence_sufficiency,
            confidence_score=confidence_score,
        )

        if ev_count < plan.answer.min_evidence:
            return AnswerResult(
                answer_text="Insufficient evidence（证据不足）：请补充更明确的工况或目标污染物证据。",
                debug={"reason": "insufficient_evidence", "evidence": ev_count},
                referenced_labels=all_labels,
                rendered_citations=self._build_rendered_citations(citations, all_labels),
                claim_items=[],
                **common,
            )

        use_llm = self.enable_llm if enable_llm is None else enable_llm

        if not use_llm or self._llm is None:
            refs = " ".join(f"[{k}]" for k in all_labels)
            text = (
                "基于检索证据，已检索到与问题相关的文献片段。\n\n"
                "由于当前未启用 LLM 归纳，下面给出证据引用，请结合原文核验：\n\n"
                f"{refs}"
            )
            return AnswerResult(
                answer_text=text,
                debug={"context_debug": context_package.debug, "mode": "non_llm_fallback"},
                referenced_labels=all_labels,
                rendered_citations=self._build_rendered_citations(citations, all_labels),
                claim_items=self._build_claim_items(text, all_labels),
                **common,
            )

        user_question = plan.query.raw or plan.query.normalized
        context_md = context_package.markdown

        prompt = f"""你是一个面向煤热解/气化/燃烧领域的科研问答助手。

请严格基于下面提供的证据片段回答用户问题，要求：
1. 只能依据给出的证据回答，不要编造文献中没有的信息。
2. 尽量先给出直接结论，再给出机理解释。
3. 回答中必须保留引用标记，例如 [E1] [E2]，并把引用放在对应结论句末。
4. 如果证据之间存在阶段差异（如热解/气化/燃烧），要明确区分。
5. 如果证据不足以支持强结论，要明确说"现有证据只表明……"。
6. 输出用中文，采用 Markdown。
7. 不要输出"根据上下文""根据提供材料"这类空话，直接回答。
8. 不要捏造不存在的引用编号。

用户问题：
{user_question}

证据片段：
{context_md}

请输出：
- 先给出一句总括结论
- 再分"机理关系""阶段差异""证据局限"三部分作答
- 每条关键判断后带引用
"""

        try:
            rsp = self._llm.invoke(prompt)
            content = getattr(rsp, "content", None)

            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            else:
                text = str(content or "").strip()

            if not text:
                raise RuntimeError("LLM returned empty answer")

            referenced_labels = self._extract_referenced_labels(text, all_labels)
            return AnswerResult(
                answer_text=text,
                debug={"context_debug": context_package.debug, "mode": "llm_answer"},
                referenced_labels=referenced_labels,
                rendered_citations=self._build_rendered_citations(citations, referenced_labels),
                claim_items=self._build_claim_items(text, referenced_labels),
                **common,
            )

        except Exception as e:
            refs = " ".join(f"[{k}]" for k in all_labels)
            fallback = (
                "已检索到相关证据，但 LLM 归纳失败。请先结合以下证据核验：\n\n"
                f"{refs}\n\n"
                f"错误信息：{type(e).__name__}: {e}"
            )
            return AnswerResult(
                answer_text=fallback,
                debug={"context_debug": context_package.debug, "mode": "llm_error", "error": str(e)},
                referenced_labels=all_labels,
                rendered_citations=self._build_rendered_citations(citations, all_labels),
                claim_items=[],
                **common,
            )