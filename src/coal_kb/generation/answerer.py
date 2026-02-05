from __future__ import annotations

"""Citation-grounded answerer with uncertainty fallback."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from coal_kb.llm.factory import LLMConfig, make_chat_llm
from coal_kb.context.builder import ContextPackage


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, Dict[str, object]] = field(default_factory=dict)
    used_docs: int = 0
    uncertain: bool = False
    mode: str = "markdown"
    debug: Dict[str, object] = field(default_factory=dict)


class Answerer:
    def __init__(self, *, enable_llm: bool, llm_provider: str, llm_config: Optional[LLMConfig] = None, output_mode: str = "markdown") -> None:
        self.enable_llm = enable_llm
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self.output_mode = output_mode

    def answer(self, question: str, context_package: ContextPackage) -> AnswerResult:
        if not context_package.used_docs:
            return AnswerResult(
                answer_text="无法可靠回答：当前检索证据不足。请提供更具体条件（阶段、温度、气氛、目标污染物）后重试。",
                citations={},
                used_docs=0,
                uncertain=True,
                mode=self.output_mode,
                debug={"reason": "no_evidence"},
            )

        if not self.enable_llm or self.llm_provider == "none" or self.llm_config is None:
            # extractive fallback
            lines = [f"- [{cid}] {meta.get('source_file')} / {meta.get('heading_path')}" for cid, meta in context_package.citations.items()]
            snippet = "\n".join(lines[:6])
            return AnswerResult(
                answer_text=f"基于已检索证据，相关信息如下：\n{snippet}",
                citations=context_package.citations,
                used_docs=len(context_package.used_docs),
                uncertain=False,
                mode=self.output_mode,
                debug={"mode": "extractive"},
            )

        llm = make_chat_llm(self.llm_config)
        prompt = (
            "你是煤热解/气化专家助手。必须只根据给定证据回答。"
            "每个关键结论后加引用标签[Sx]。"
            "若证据不足，明确说无法可靠回答并说明缺失证据。\n\n"
            f"问题：{question}\n\n"
            f"证据：\n{context_package.prompt_context_text}\n"
        )
        rsp = llm.invoke(prompt)
        text = getattr(rsp, "content", "") if rsp is not None else ""

        # post-check: must have citation if non-empty
        if text and "[S" not in text:
            lines = [f"[{cid}] {(d.page_content or '')[:160]}" for cid, d in [(d.metadata.get('citation_id'), d) for d in context_package.used_docs if d.metadata.get('citation_id')]][:6]
            text = "证据复述模式：\n" + "\n".join(lines)
            return AnswerResult(
                answer_text=text,
                citations=context_package.citations,
                used_docs=len(context_package.used_docs),
                uncertain=True,
                mode=self.output_mode,
                debug={"post_check": "citation_missing"},
            )

        return AnswerResult(
            answer_text=text,
            citations=context_package.citations,
            used_docs=len(context_package.used_docs),
            uncertain=False,
            mode=self.output_mode,
            debug={},
        )
