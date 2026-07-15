"""统一分派事实、比较、多跳、聚合、表格和跨文档路线。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.aggregation import AggregationExecutor, AggregationRepository
from coal_kb.complex_qa.comparison import ComparisonExecutor
from coal_kb.complex_qa.models import ComplexExecutionResult
from coal_kb.complex_qa.multi_hop import MultiHopExecutor
from coal_kb.complex_qa.synthesis import CrossDocumentExecutor
from coal_kb.complex_qa.tables import TableExecutor, TableRepository
from coal_kb.core.models.query import QueryPlan


@dataclass
class ComplexQuestionService:
    """持有各路线执行器并提供单一 process() 入口。"""

    retriever: Any
    sqlite_path: str
    table_records_path: str
    comparison_k_per_side: int
    max_multi_hop_steps: int
    aggregation_record_limit: int
    aggregation_evidence_limit: int
    table_top_k: int
    cross_document_min_sources: int
    cross_document_max_per_source: int

    def __post_init__(self) -> None:
        self._comparison = ComparisonExecutor(self.retriever, self.comparison_k_per_side)
        self._multi_hop = MultiHopExecutor(self.retriever, self.max_multi_hop_steps)
        self._aggregation = AggregationExecutor(
            AggregationRepository(self.sqlite_path, self.aggregation_record_limit),
            self.retriever,
            self.aggregation_evidence_limit,
        )
        self._table = TableExecutor(TableRepository(self.table_records_path), self.retriever, self.table_top_k)
        self._cross_document = CrossDocumentExecutor(
            self.retriever,
            self.cross_document_min_sources,
            self.cross_document_max_per_source,
        )

    def process(self, plan: QueryPlan, trace: dict[str, Any] | None = None) -> list[Any]:
        query_type = plan.complex.query_type
        if query_type == "unanswerable":
            result = ComplexExecutionResult(documents=[], trace={"query_type": query_type, "reason": plan.complex.reason})
        elif query_type == "comparison":
            result = self._comparison.process(plan)
        elif query_type == "multi_hop":
            result = self._multi_hop.process(plan)
        elif query_type == "aggregation":
            result = self._aggregation.process(plan)
        elif query_type == "table":
            result = self._table.process(plan)
        elif query_type == "cross_document":
            result = self._cross_document.process(plan)
        else:
            local_trace: dict[str, Any] = {}
            documents = self.retriever.execute(plan, trace=local_trace)
            result = ComplexExecutionResult(documents=documents, trace={"query_type": "fact", "retrieval": local_trace})
        if trace is not None:
            trace["complex_route"] = plan.complex.model_dump()
            trace["complex_execution"] = result.trace
        return result.documents
