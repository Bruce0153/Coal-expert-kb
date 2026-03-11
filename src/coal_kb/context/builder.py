from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from coal_kb.query.plan import QueryPlan

from .types import CitationItem, ContextPackage


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", stripped, flags=re.UNICODE)))


def _snippet(text: str, max_chars: int = 420) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _display_name(meta: Dict[str, object]) -> str:
    source_file = str(meta.get("source_file") or "unknown")
    title = str(meta.get("title") or "").strip() or Path(source_file).name
    parts = [title]
    page = meta.get("page")
    if page is not None:
        parts.append(f"page {page}")
    heading = str(meta.get("heading_path") or "").strip()
    if heading:
        parts.append(heading)
    return " | ".join(parts)


class ContextBuilder:
    def build(self, plan: QueryPlan, docs: List[Document]) -> ContextPackage:
        max_chunks = max(0, plan.context.max_evidence_chunks)
        max_tokens = max(0, plan.context.max_context_tokens)

        selected: List[Document] = []
        used_chunk_ids = set()
        used_texts = set()
        dropped_duplicates = 0
        dropped_budget = 0
        token_total = 0

        for doc in docs:
            if len(selected) >= max_chunks:
                dropped_budget += 1
                continue

            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or "")
            text_key = _snippet(doc.page_content or "", max_chars=240).lower()

            if plan.context.deduplicate:
                duplicate = bool(chunk_id and chunk_id in used_chunk_ids) or text_key in used_texts
                if duplicate:
                    dropped_duplicates += 1
                    continue

            doc_tokens = _estimate_tokens(doc.page_content or "")
            if selected and max_tokens and token_total + doc_tokens > max_tokens:
                dropped_budget += 1
                continue

            selected.append(doc)
            token_total += doc_tokens
            if chunk_id:
                used_chunk_ids.add(chunk_id)
            used_texts.add(text_key)

        grouped_docs: "OrderedDict[str, List[Document]]" = OrderedDict()
        for doc in selected:
            meta = doc.metadata or {}
            source = str(meta.get("source_file") or "unknown")
            heading = str(meta.get("heading_path") or meta.get("title") or "(ungrouped)")
            grouped_docs.setdefault(f"{source}::{heading}", []).append(doc)

        citations: Dict[str, CitationItem] = {}
        evidence_items: List[CitationItem] = []
        used_chunks: List[str] = []
        chunk_labels: Dict[str, str] = {}
        for index, doc in enumerate(selected, start=1):
            label = f"E{index}"
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or label)
            chunk_labels[chunk_id] = label
            used_chunks.append(chunk_id)
            citation = CitationItem(
                label=label,
                source_file=str(meta.get("source_file") or "unknown"),
                title=str(meta.get("title") or "").strip() or None,
                page=meta.get("page"),
                heading_path=str(meta.get("heading_path") or "").strip() or None,
                chunk_id=chunk_id,
                snippet=_snippet(doc.page_content or ""),
                source_display=_display_name(meta),
            )
            citations[label] = citation
            evidence_items.append(citation)

        lines: List[str] = ["# Evidence Catalog"]
        section_sizes = defaultdict(int)
        for group_key, items in grouped_docs.items():
            _, heading = group_key.split("::", 1)
            lines.append(f"## {heading}")
            for doc in items:
                meta = doc.metadata or {}
                chunk_id = str(meta.get("chunk_id") or "")
                label = chunk_labels.get(chunk_id, "E?")
                citation = citations[label]
                lines.append(f"[{label}] {citation.source_display}")
                lines.append(citation.snippet)
                section_sizes[heading] += 1

        return ContextPackage(
            markdown="\n\n".join(lines),
            citations=citations,
            evidence_items=evidence_items,
            used_chunks=used_chunks,
            debug={
                "selected_chunks": len(selected),
                "estimated_context_tokens": token_total,
                "dropped_duplicates": dropped_duplicates,
                "dropped_budget": dropped_budget,
                "sections": dict(section_sizes),
            },
        )
