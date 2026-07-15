"""验证比较、多跳和跨文档执行器复用正式 Retriever。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.complex_qa.service import ComplexQuestionService
from coal_kb.infra.config import AppConfig
from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.retrieval.query import FilterParser, QueryPlanner


class FakeRetriever:
    """根据子查询稳定返回不同来源的离线 Retriever。"""

    def execute(self, plan, trace=None):
        query = plan.query.normalized.lower()
        if "支持性证据" in query:
            source = "support.pdf"
        elif "相反结果" in query or "冲突证据" in query:
            source = "conflict.pdf"
        elif "实验条件差异" in query:
            source = "conditions.pdf"
        elif "co2" in query or "二氧化碳" in query:
            source = "co2.pdf"
        elif "蒸汽" in query:
            source = "steam.pdf"
        else:
            source = "default.pdf"
        if trace is not None:
            trace["fake"] = True
        return [
            Document(
                page_content=f"{query} 的证据和反应机理。",
                metadata={"source_file": source, "page": 1, "chunk_id": f"{source}:{len(query)}"},
            )
        ]


def _service(tmp_path) -> ComplexQuestionService:
    return ComplexQuestionService(
        retriever=FakeRetriever(),
        sqlite_path=str(tmp_path / "records.db"),
        table_records_path=str(tmp_path / "tables.jsonl"),
        comparison_k_per_side=4,
        max_multi_hop_steps=3,
        aggregation_record_limit=100,
        aggregation_evidence_limit=10,
        table_top_k=5,
        cross_document_min_sources=2,
        cross_document_max_per_source=2,
    )


def _plan(query: str):
    cfg = AppConfig()
    return QueryPlanner(FilterParser(Ontology.load("configs/schema.yaml"))).build_plan(query, cfg)


def test_comparison_returns_both_sides(tmp_path) -> None:
    trace = {}
    documents = _service(tmp_path).process(_plan("比较蒸汽气化与CO2气化的差异"), trace=trace)
    roles = {document.metadata.get("complex_role") for document in documents}
    assert roles == {"comparison_1", "comparison_2"}
    assert trace["complex_execution"]["query_type"] == "comparison"


def test_multi_hop_trace_is_complete(tmp_path) -> None:
    trace = {}
    documents = _service(tmp_path).process(_plan("高温为什么通过水煤气反应提高H2产率"), trace=trace)
    assert documents
    assert trace["complex_execution"]["chain_complete"] is True
    assert len(trace["complex_execution"]["steps"]) == 3


def test_cross_document_controls_source_diversity(tmp_path) -> None:
    trace = {}
    documents = _service(tmp_path).process(_plan("多篇文献对压力影响的主要共识和冲突是什么"), trace=trace)
    sources = {document.metadata.get("source_file") for document in documents}
    assert len(sources) >= 2
    assert trace["complex_execution"]["minimum_sources_met"] is True
