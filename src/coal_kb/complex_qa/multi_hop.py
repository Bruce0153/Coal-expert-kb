"""执行受限多跳检索并保存每一跳的查询和证据。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.utils import clone_plan_for_subquery, deduplicate_documents, tag_documents
from coal_kb.core.models.query import QueryPlan


@dataclass
class MultiHopExecutor:
    """持有检索器并顺序执行有依赖关系的子问题。"""

    retriever: Any
    max_steps: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        documents = []
        steps = []
        bridge_terms: list[str] = []
        for subquery in plan.complex.subqueries[: self.max_steps]:
            effective_query = subquery.query
            if subquery.depends_on and bridge_terms:
                effective_query = f"{effective_query} {' '.join(bridge_terms)}"
            effective = subquery.model_copy(update={"query": effective_query})
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(clone_plan_for_subquery(plan, effective), trace=local_trace)
            tagged = tag_documents(hits, complex_route="multi_hop", complex_role=subquery.subquery_id)
            documents.extend(tagged)
            bridge_terms = self._extract_bridge_terms(hits)
            steps.append(
                {
                    "subquery_id": subquery.subquery_id,
                    "query": effective_query,
                    "depends_on": subquery.depends_on,
                    "bridge_terms": bridge_terms,
                    "hits": len(hits),
                }
            )
        return ComplexExecutionResult(
            documents=deduplicate_documents(documents),
            trace={"query_type": "multi_hop", "steps": steps, "chain_complete": bool(steps) and all(step["hits"] > 0 for step in steps)},
        )

    @staticmethod
    def _extract_bridge_terms(documents: list[Any]) -> list[str]:
        """从上一跳证据中稳定抽取少量中间术语。"""
        text = " ".join(str(document.page_content or "")[:800] for document in documents[:3]).lower()
        terms = re.findall(r"[a-z][a-z0-9_-]{2,}|[一-鿿]{2,6}", text)
        blocked = {"研究", "结果", "实验", "条件", "影响", "过程", "表明", "通过", "进行"}
        counts = Counter(term for term in terms if term not in blocked)
        return [term for term, _ in counts.most_common(4)]
