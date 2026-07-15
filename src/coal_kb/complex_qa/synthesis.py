"""执行跨文档综合检索并控制单一来源占比。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.utils import clone_plan_for_subquery, deduplicate_documents, tag_documents
from coal_kb.core.models.query import QueryPlan


@dataclass
class CrossDocumentExecutor:
    """持有检索器并聚合不同文档中的支持、冲突和条件证据。"""

    retriever: Any
    min_sources: int
    max_per_source: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        candidates = []
        steps = []
        for subquery in plan.complex.subqueries:
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(clone_plan_for_subquery(plan, subquery), trace=local_trace)
            candidates.extend(tag_documents(hits, complex_route="cross_document", complex_role=subquery.subquery_id))
            steps.append({"subquery_id": subquery.subquery_id, "query": subquery.query, "hits": len(hits)})
        unique = deduplicate_documents(candidates)
        per_source: dict[str, int] = {}
        selected = []
        for document in unique:
            source = str((document.metadata or {}).get("source_file") or "unknown")
            if per_source.get(source, 0) >= self.max_per_source:
                continue
            per_source[source] = per_source.get(source, 0) + 1
            selected.append(document)
        return ComplexExecutionResult(
            documents=selected,
            trace={
                "query_type": "cross_document",
                "steps": steps,
                "source_count": len(per_source),
                "sources": sorted(per_source),
                "minimum_sources_met": len(per_source) >= self.min_sources,
            },
        )
