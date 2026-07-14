"""编排证据去重、预算、引用目录和来源卡片构建。"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

from langchain_core.documents import Document

from coal_kb.context import config
from coal_kb.context.budgeting import select_with_budget
from coal_kb.context.citations import display_name, snippet, source_id
from coal_kb.context.deduplication import deduplicate_docs, diversify_docs
from coal_kb.context.models import CitationItem, ContextPackage, SourceCard
from coal_kb.core.models.query import QueryPlan


class ContextBuilder:
    """保持原 ContextBuilder 接口和证据编号顺序。"""

    def build(self, plan: QueryPlan, docs: list[Document]) -> ContextPackage:
        max_chunks = max(0, plan.context.max_evidence_chunks)
        max_tokens = max(0, plan.context.max_context_tokens)
        unique_docs, dropped_duplicates = deduplicate_docs(docs, enabled=plan.context.deduplicate)
        diversified_docs = diversify_docs(unique_docs)
        selected, dropped_budget, token_total = select_with_budget(
            diversified_docs,
            max_chunks=max_chunks,
            max_tokens=max_tokens,
        )

        grouped_docs: OrderedDict[str, list[Document]] = OrderedDict()
        for doc in selected:
            metadata = doc.metadata or {}
            source = str(metadata.get("source_file") or "unknown")
            heading = str(metadata.get("heading_path") or metadata.get("title") or "(ungrouped)")
            grouped_docs.setdefault(f"{source}::{heading}", []).append(doc)

        citations: dict[str, CitationItem] = {}
        evidence_items: list[CitationItem] = []
        used_chunks: list[str] = []
        source_card_map: dict[str, SourceCard] = {}

        for index, doc in enumerate(selected, start=1):
            label = f"{config.EVIDENCE_LABEL_PREFIX}{index}"
            metadata = doc.metadata or {}
            chunk_id = str(metadata.get("chunk_id") or label)
            current_source_id = source_id(metadata)
            citation = CitationItem(
                label=label,
                source_file=str(metadata.get("source_file") or "unknown"),
                title=str(metadata.get("title") or "").strip() or None,
                page=metadata.get("page"),
                heading_path=str(metadata.get("heading_path") or "").strip() or None,
                chunk_id=chunk_id,
                snippet=snippet(doc.page_content or ""),
                source_display=display_name(metadata),
                source_id=current_source_id,
                rank=index,
            )
            citations[label] = citation
            evidence_items.append(citation)
            used_chunks.append(chunk_id)

            card = source_card_map.get(current_source_id)
            if card is None:
                card = SourceCard(
                    source_id=current_source_id,
                    source_file=citation.source_file,
                    title=citation.title or Path(citation.source_file).name,
                    snippet_preview=citation.snippet,
                )
                source_card_map[current_source_id] = card
            card.evidence_labels.append(label)
            card.evidence_count += 1
            if citation.page is not None and citation.page not in card.pages:
                card.pages.append(citation.page)
            if citation.heading_path and citation.heading_path not in card.headings:
                card.headings.append(citation.heading_path)

        lines: list[str] = ["# Evidence Catalog"]
        section_sizes: defaultdict[str, int] = defaultdict(int)
        for group_key, items in grouped_docs.items():
            _, heading = group_key.split("::", 1)
            lines.append(f"## {heading}")
            for doc in items:
                chunk_id = str((doc.metadata or {}).get("chunk_id") or "")
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
