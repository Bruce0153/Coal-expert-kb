from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm.factory import LLMConfig, make_chat_llm


def format_citation(d: Document) -> str:
    m = d.metadata or {}
    src = m.get("source_file", "unknown")
    page = m.get("page", None)
    chunk_id = m.get("chunk_id", "")
    if isinstance(page, int):
        return f"{src} (page {page}) #{chunk_id}"
    return f"{src} #{chunk_id}"


def build_evidence_block(docs: List[Document], *, max_chars: int = 900) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, 1):
        text = (d.page_content or "").strip()
        text = text[:max_chars]
        lines.append(f"[{i}] {format_citation(d)}\n{text}")
    return "\n\n".join(lines)


@dataclass
class RAGAnswerer:
    """
    Answering strategy:
    - If LLM configured: generate a structured answer with citations.
    - Else: output ranked evidence snippets (still useful & auditable).
    """

    enable_llm: bool = False
    llm_provider: str = "none"  # "openai"
    llm_config: Optional[LLMConfig] = None

    def answer(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return "未检索到相关证据。建议：放宽条件（温度/压力范围）或减少过滤条件（气化剂/阶段）。"

        if self.enable_llm and self.llm_provider != "none":
            out = self._answer_openai(query=query, docs=docs)
            if out:
                return out

        # fallback: evidence-only
        return (
            "（当前未启用LLM生成答案，以下为检索到的证据片段，供你人工总结/核对）\n\n"
            + build_evidence_block(docs)
        )

    def _answer_openai(self, *, query: str, docs: List[Document]) -> Optional[str]:
        sys_prompt = (
            "你是煤炭热解/气化污染物领域的严谨助理。"
            "你必须仅根据提供的证据回答，并在每条关键结论后标注证据编号，例如[1][2]。"
            "如果证据不足，明确说明不确定，并给出需要补充的条件。"
            "输出结构：\n"
            "1) 工况总结\n"
            "2) 结论（趋势/机理）\n"
            "3) 不确定性与建议\n"
        )

        evidence = build_evidence_block(docs, max_chars=1200)
        user = f"问题：{query}\n\n证据：\n{evidence}"

        if self.llm_config is None:
            return None

        model = make_chat_llm(self.llm_config)
        rsp = model.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user)])
        return getattr(rsp, "content", None) or None
