from __future__ import annotations

import re
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from coal_kb.core.models.query import QueryPlan

from .types import CitationItem, ContextPackage, SourceCard


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


def _source_id(meta: Dict[str, object]) -> str:
    source_file = str(meta.get("source_file") or "unknown")
    title = str(meta.get("title") or "").strip() or Path(source_file).stem
    return f"{title}|{source_file}"


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


def _diversify_docs(docs: List[Document]) -> List[Document]:
    grouped: "OrderedDict[str, deque[Document]]" = OrderedDict()
    for doc in docs:
        meta = doc.metadata or {}
        grouped.setdefault(_source_id(meta), deque()).append(doc)

    diversified: List[Document] = []
    while grouped:
        empty_keys: List[str] = []
        for key, bucket in grouped.items():
            if bucket:
                diversified.append(bucket.popleft())
            if not bucket:
                empty_keys.append(key)
        for key in empty_keys:
            grouped.pop(key, None)
    return diversified


class ContextBuilder:
    def build(self, plan: QueryPlan, docs: List[Document]) -> ContextPackage:
        max_chunks = max(0, plan.context.max_evidence_chunks)
        max_tokens = max(0, plan.context.max_context_tokens)

        preselected: List[Document] = []
        used_chunk_ids = set()
        used_texts = set()
        dropped_duplicates = 0

        for doc in docs:
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or "")
            text_key = _snippet(doc.page_content or "", max_chars=240).lower()

            if plan.context.deduplicate:
                duplicate = bool(chunk_id and chunk_id in used_chunk_ids) or text_key in used_texts
                if duplicate:
                    dropped_duplicates += 1
                    continue

            preselected.append(doc)
            if chunk_id:
                used_chunk_ids.add(chunk_id)
            used_texts.add(text_key)

        diversified_docs = _diversify_docs(preselected)

        selected: List[Document] = []
        dropped_budget = 0
        token_total = 0
        for doc in diversified_docs:
            if len(selected) >= max_chunks:
                dropped_budget += 1
                continue
            doc_tokens = _estimate_tokens(doc.page_content or "")
            if selected and max_tokens and token_total + doc_tokens > max_tokens:
                dropped_budget += 1
                continue
            selected.append(doc)
            token_total += doc_tokens

        grouped_docs: "OrderedDict[str, List[Document]]" = OrderedDict()
        for doc in selected:
            meta = doc.metadata or {}
            source = str(meta.get("source_file") or "unknown")
            heading = str(meta.get("heading_path") or meta.get("title") or "(ungrouped)")
            grouped_docs.setdefault(f"{source}::{heading}", []).append(doc)

        citations: Dict[str, CitationItem] = {}
        evidence_items: List[CitationItem] = []
        used_chunks: List[str] = []
        source_card_map: Dict[str, SourceCard] = {}

        for index, doc in enumerate(selected, start=1):
            label = f"E{index}"
            meta = doc.metadata or {}
            chunk_id = str(meta.get("chunk_id") or label)
            source_id = _source_id(meta)
            citation = CitationItem(
                label=label,
                source_file=str(meta.get("source_file") or "unknown"),
                title=str(meta.get("title") or "").strip() or None,
                page=meta.get("page"),
                heading_path=str(meta.get("heading_path") or "").strip() or None,
                chunk_id=chunk_id,
                snippet=_snippet(doc.page_content or ""),
                source_display=_display_name(meta),
                source_id=source_id,
                rank=index,
            )
            citations[label] = citation
            evidence_items.append(citation)
            used_chunks.append(chunk_id)

            card = source_card_map.get(source_id)
            if card is None:
                card = SourceCard(
                    source_id=source_id,
                    source_file=citation.source_file,
                    title=citation.title or Path(citation.source_file).name,
                    snippet_preview=citation.snippet,
                )
                source_card_map[source_id] = card
            card.evidence_labels.append(label)
            card.evidence_count += 1
            if citation.page is not None and citation.page not in card.pages:
                card.pages.append(citation.page)
            if citation.heading_path and citation.heading_path not in card.headings:
                card.headings.append(citation.heading_path)

        lines: List[str] = ["# Evidence Catalog"]
        section_sizes = defaultdict(int)
        for group_key, items in grouped_docs.items():
            _, heading = group_key.split("::", 1)
            lines.append(f"## {heading}")
            for doc in items:
                meta = doc.metadata or {}
                chunk_id = str(meta.get("chunk_id") or "")
                citation = next((item for item in evidence_items if item.chunk_id == chunk_id), None)
                if citation is None:
                    continue
                lines.append(f"[{citation.label}] {citation.source_display}")
                lines.append(citation.snippet)
                section_sizes[heading] += 1

        return ContextPackage(
            markdown="\n\n".join(lines),
            citations=citations,
            evidence_items=evidence_items,
            source_cards=list(source_card_map.values()),
            used_chunks=used_chunks,
            debug={
                "selected_chunks": len(selected),
                "selected_sources": len(source_card_map),
                "estimated_context_tokens": token_total,
                "dropped_duplicates": dropped_duplicates,
                "dropped_budget": dropped_budget,
                "sections": dict(section_sizes),
            },
        )
