from __future__ import annotations

"""ContextBuilder organizes retrieved evidence into prompt-ready package."""

from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document

from coal_kb.chunking.tokenizer import count_tokens
from coal_kb.query.plan import QueryPlan


@dataclass
class ContextPackage:
    prompt_context_text: str
    citations: Dict[str, Dict[str, object]]
    used_docs: List[Document]
    token_estimate: int


class ContextBuilder:
    def build(self, plan: QueryPlan, docs: List[Document]) -> ContextPackage:
        # group by heading path / parent
        grouped: Dict[str, List[Document]] = {}
        for d in docs:
            m = d.metadata or {}
            key = str(m.get("heading_path") or m.get("parent_id") or "(root)")
            grouped.setdefault(key, []).append(d)

        citations: Dict[str, Dict[str, object]] = {}
        lines: List[str] = []
        used: List[Document] = []
        budget = plan.token_budget
        cite_id = 1

        for heading, members in grouped.items():
            section_lines = [f"### {heading}"]
            for d in members:
                m = d.metadata or {}
                label = f"S{cite_id}"
                chunk_text = (d.page_content or "").strip()
                unit = f"[{label}] {chunk_text}"
                if count_tokens("\n".join(lines + section_lines + [unit])) > budget:
                    break
                section_lines.append(unit)
                m["citation_id"] = label
                d.metadata = m
                citations[label] = {
                    "source_file": m.get("source_file"),
                    "page": m.get("page"),
                    "heading_path": m.get("heading_path"),
                    "chunk_id": m.get("chunk_id"),
                }
                used.append(d)
                cite_id += 1
            if len(section_lines) > 1:
                lines.extend(section_lines)

        text = "\n\n".join(lines).strip()
        return ContextPackage(
            prompt_context_text=text,
            citations=citations,
            used_docs=used,
            token_estimate=count_tokens(text),
        )
