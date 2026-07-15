"""验证统计聚合由 Python 基于结构化记录确定性计算。"""

from __future__ import annotations

from coal_kb.complex_qa.aggregation import AggregationExecutor, AggregationRepository
from coal_kb.core.models.query import (
    AggregationSpec,
    ComplexQuestionSpec,
    QueryPlan,
    QueryUnderstanding,
)
from coal_kb.infra.persistence.sql.records import SQLiteStore


class EmptyRetriever:
    def execute(self, plan, trace=None):
        return []


def test_average_and_top_k_are_computed_from_sqlite(tmp_path) -> None:
    path = tmp_path / "records.db"
    store = SQLiteStore(str(path))
    for index, temperature in enumerate((900.0, 1100.0, 1300.0), start=1):
        record_id = f"r{index}"
        store.upsert_record(
            record_id=record_id,
            source_file=f"paper_{index}.pdf",
            stage="gasification",
            T_K=temperature,
            pollutants={"H2": {"value": index * 10.0, "unit": "%"}},
        )
        store.add_evidence(record_id=record_id, source_file=f"paper_{index}.pdf", page=index, chunk_id=f"c{index}")

    executor = AggregationExecutor(AggregationRepository(str(path), 100), EmptyRetriever(), 10)
    average_plan = QueryPlan(
        query=QueryUnderstanding(raw="平均气化温度", normalized="平均气化温度"),
        complex=ComplexQuestionSpec(
            query_type="aggregation",
            aggregation=AggregationSpec(operation="average", field="T_K"),
        ),
    )
    average = executor.process(average_plan)
    assert average.trace["value"] == 1100.0
    assert average.trace["sample_size"] == 3

    top_plan = QueryPlan(
        query=QueryUnderstanding(raw="H2最高前2条", normalized="H2最高前2条"),
        complex=ComplexQuestionSpec(
            query_type="aggregation",
            aggregation=AggregationSpec(operation="top_k", field="pollutants.H2", top_k=2),
        ),
    )
    top = executor.process(top_plan)
    assert [item["value"] for item in top.trace["value"]] == [30.0, 20.0]
