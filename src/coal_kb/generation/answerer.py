from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from coal_kb.context.types import ContextPackage
from coal_kb.query.plan import QueryPlan
from coal_kb.llm.factory import LLMConfig, make_chat_llm


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, dict]
    used_chunks: List[str]
    debug: Dict[str, Any]


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

    def answer(self, plan: QueryPlan, context_package: ContextPackage) -> AnswerResult:
        ev_count = len(context_package.used_chunks)
        citations = {k: v.model_dump() for k, v in context_package.citations.items()}

        if ev_count < plan.answer.min_evidence:
            return AnswerResult(
                answer_text="无法可靠回答：证据不足。请补充更明确的工况/目标污染物证据。",
                citations=citations,
                used_chunks=context_package.used_chunks,
                debug={"reason": "insufficient_evidence", "evidence": ev_count},
            )

        # 如果没开 LLM，就退回一个更有用的非 LLM 摘要
        if not self.enable_llm or self._llm is None:
            refs = " ".join(f"[{k}]" for k in context_package.citations.keys())
            text = (
                "基于检索证据，已检索到与问题相关的文献片段。\n\n"
                "由于当前未启用 LLM 归纳，下面给出证据引用，请结合原文核验：\n\n"
                f"{refs}"
            )
            return AnswerResult(
                answer_text=text,
                citations=citations,
                used_chunks=context_package.used_chunks,
                debug={"context_debug": context_package.debug, "mode": "non_llm_fallback"},
            )

        # 开启 LLM：把问题 + 证据 markdown 发给模型总结
        user_question = plan.query.raw or plan.query.normalized
        context_md = context_package.markdown

        prompt = f"""你是一个面向煤热解/气化/燃烧领域的科研问答助手。

请严格基于下面提供的证据片段回答用户问题，要求：
1. 只能依据给出的证据回答，不要编造文献中没有的信息。
2. 尽量先给出直接结论，再给出机理解释。
3. 回答中必须保留引用标记，例如 [S1] [S2]，并把引用放在对应结论句末。
4. 如果证据之间存在阶段差异（如热解/气化/燃烧），要明确区分。
5. 如果证据不足以支持强结论，要明确说“现有证据只表明……”。
6. 输出用中文，采用 Markdown。
7. 不要输出“根据上下文”“根据提供材料”这类空话，直接回答。
8. 不要捏造不存在的引用编号。

用户问题：
{user_question}

证据片段：
{context_md}

请输出：
- 先给出一句总括结论
- 再分“机理关系”“阶段差异”“证据局限”三部分作答
- 每条关键判断后带引用
"""

        try:
            rsp = self._llm.invoke(prompt)
            content = getattr(rsp, "content", None)

            if isinstance(content, list):
                # 某些兼容模型会返回分段 content
                text = "\n".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            else:
                text = str(content or "").strip()

            if not text:
                raise RuntimeError("LLM returned empty answer")

            return AnswerResult(
                answer_text=text,
                citations=citations,
                used_chunks=context_package.used_chunks,
                debug={"context_debug": context_package.debug, "mode": "llm_answer"},
            )

        except Exception as e:
            refs = " ".join(f"[{k}]" for k in context_package.citations.keys())
            fallback = (
                "已检索到相关证据，但 LLM 归纳失败。请先结合以下证据核验：\n\n"
                f"{refs}\n\n"
                f"错误信息：{type(e).__name__}: {e}"
            )
            return AnswerResult(
                answer_text=fallback,
                citations=citations,
                used_chunks=context_package.used_chunks,
                debug={"context_debug": context_package.debug, "mode": "llm_error", "error": str(e)},
            )