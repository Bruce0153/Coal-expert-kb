from __future__ import annotations

from collections import defaultdict
from typing import List

from langchain_core.documents import Document

from coal_kb.query.plan import QueryPlan

from .types import CitationItem, ContextPackage


class ContextBuilder:
    def build(self, plan: QueryPlan, docs: List[Document]) -> ContextPackage:
        budget = plan.context.max_evidence_chunks
        selected = docs[:budget]

        seen = set()
        deduped: List[Document] = []
        dropped = 0
        for d in selected:
            text_key = (d.page_content or "").strip().lower()
            if plan.context.deduplicate and text_key in seen:
                dropped += 1
                continue
            seen.add(text_key)
            deduped.append(d)

        groups = defaultdict(list)
        for d in deduped:
            m = d.metadata or {}
            group_key = str(m.get("heading_path") or m.get("parent_id") or m.get("source_file") or "unknown")
            groups[group_key].append(d)

        lines = []
        citations = {}
        used_chunks = []
        for i, d in enumerate(deduped, start=1):
            sid = f"S{i}"
            m = d.metadata or {}
            used_chunks.append(str(m.get("chunk_id") or ""))
            citations[sid] = CitationItem(
                sid=sid,
                source_file=str(m.get("source_file") or "unknown"),
                page=m.get("page"),
                heading_path=m.get("heading_path"),
                chunk_id=str(m.get("chunk_id") or sid),
            )

        for heading, items in groups.items():
            lines.append(f"## {heading}")
            for d in items:
                idx = deduped.index(d) + 1
                lines.append(f"- [S{idx}] {(d.page_content or '').strip()[:500]}")

        return ContextPackage(
            markdown="\n".join(lines),
            citations=citations,
            used_chunks=used_chunks,
            debug={
                "sections": {k: len(v) for k, v in groups.items()},
                "dropped_dedup": dropped,
            },
        )
