"""执行比较问题的双侧独立检索和证据标记。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.planning import clone_plan_for_subquery
from coal_kb.core.models.query import QueryPlan
from coal_kb.utils.documents import copy_documents_with_metadata, deduplicate_documents


@dataclass
class ComparisonExecutor:
    """持有检索器并执行比较问题。"""

    retriever: Any
    k_per_side: int

    def process(self, plan: QueryPlan) -> ComplexExecutionResult:
        documents = []
        steps = []
        for index, subquery in enumerate(plan.complex.subqueries[:2]):
            local_trace: dict[str, Any] = {}
            hits = self.retriever.execute(
                clone_plan_for_subquery(plan, subquery),
                trace=local_trace,
            )[: self.k_per_side]
            entity = (
                plan.complex.comparison_entities[index]
                if index < len(plan.complex.comparison_entities)
                else subquery.subquery_id
            )
            documents.extend(
                copy_documents_with_metadata(
                    hits,
                    complex_route="comparison",
                    complex_role=subquery.subquery_id,
                    comparison_entity=entity,
                )
            )
            steps.append(
                {
                    "subquery_id": subquery.subquery_id,
                    "query": subquery.query,
                    "entity": entity,
                    "hits": len(hits),
                }
            )
        return ComplexExecutionResult(
            documents=deduplicate_documents(documents),
            trace={
                "query_type": "comparison",
                "steps": steps,
                "entities": plan.complex.comparison_entities,
            },
        )
