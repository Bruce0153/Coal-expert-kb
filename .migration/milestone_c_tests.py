"""补齐 Milestone C 的脚本、单元测试和质量门禁。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def process() -> None:
    _write(
        "scripts/validate_complex_dataset.py",
        '''"""验证复杂科学问答 JSONL 的格式、唯一性和类型覆盖。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from coal_kb.evaluation.models import EvaluationCase, QueryType


@dataclass
class ValidateComplexDataset:
    """持有数据集路径并执行完整格式验证。"""

    dataset_path: Path
    require_all_types: bool = False

    def process(self) -> dict[str, object]:
        lines = self.dataset_path.read_text(encoding="utf-8").splitlines()
        cases = []
        for row_number, line in enumerate(tqdm(lines, total=len(lines), desc=self.__class__.__name__), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(EvaluationCase.from_dict(payload, row_number=row_number))
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("复杂问答评估集中的 id 必须唯一")
        counts: dict[str, int] = {}
        for case in cases:
            counts[case.query_type.value] = counts.get(case.query_type.value, 0) + 1
        if self.require_all_types:
            required = {
                QueryType.COMPARISON.value,
                QueryType.MULTI_HOP.value,
                QueryType.AGGREGATION.value,
                QueryType.TABLE.value,
                QueryType.CROSS_DOCUMENT.value,
            }
            missing = sorted(required - set(counts))
            if missing:
                raise ValueError(f"复杂问答评估集缺少类型: {missing}")
        summary = {"case_count": len(cases), "query_type_counts": counts}
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a complex science QA JSONL dataset.")
    parser.add_argument("--dataset", default="data/eval/complex_science_sample.jsonl")
    parser.add_argument("--require-all-types", action="store_true")
    args = parser.parse_args()
    ValidateComplexDataset(Path(args.dataset), require_all_types=args.require_all_types).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/validate_complex_dataset.py --require-all-types
''',
    )

    _write(
        "tests/test_complex_question_router.py",
        '''"""验证 C0-C6 统一路由和子问题规划。"""

from coal_kb.complex_qa.planning import build_complex_spec
from coal_kb.complex_qa.router import route_question


def test_router_covers_milestone_c_types() -> None:
    cases = {
        "比较蒸汽气化与CO2气化的差异": "comparison",
        "高温为什么通过水煤气反应提高H2产率": "multi_hop",
        "平均气化温度是多少": "aggregation",
        "表格中H2对应的数值是多少": "table",
        "多篇文献对压力影响的主要共识是什么": "cross_document",
        "未公开的私人实验压力是多少": "unanswerable",
        "煤气化的主要气化剂有哪些": "fact",
    }
    for query, expected in cases.items():
        assert route_question(query)[0] == expected


def test_comparison_and_multi_hop_plans_are_replayable() -> None:
    comparison = build_complex_spec(
        "比较蒸汽气化与CO2气化对H2产率的影响",
        max_subqueries=4,
        max_multi_hop_steps=3,
    )
    assert comparison.query_type == "comparison"
    assert len(comparison.comparison_entities) == 2
    assert len(comparison.subqueries) == 2

    multi_hop = build_complex_spec(
        "催化剂如何通过焦油裂解路径降低焦油产率",
        max_subqueries=4,
        max_multi_hop_steps=3,
    )
    assert multi_hop.query_type == "multi_hop"
    assert multi_hop.subqueries[1].depends_on == ["hop_1"]


def test_aggregation_plan_extracts_operation_field_and_top_k() -> None:
    spec = build_complex_spec(
        "列出H2产率最高的前5条实验记录",
        max_subqueries=4,
        max_multi_hop_steps=3,
    )
    assert spec.query_type == "aggregation"
    assert spec.aggregation is not None
    assert spec.aggregation.operation == "top_k"
    assert spec.aggregation.field == "pollutants.H2"
    assert spec.aggregation.top_k == 5
''',
    )

    _write(
        "tests/test_complex_query_planner.py",
        '''"""验证 QueryPlanner 同时生成领域约束和复杂问答计划。"""

from coal_kb.infra.config import AppConfig
from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.retrieval.query import FilterParser, QueryPlanner


def test_planner_builds_comparison_plan_and_two_stage_retrieval() -> None:
    cfg = AppConfig()
    planner = QueryPlanner(FilterParser(Ontology.load("configs/schema.yaml")))
    plan = planner.build_plan("比较蒸汽气化与CO2气化对H2产率的影响", cfg)
    assert plan.complex.query_type == "comparison"
    assert len(plan.complex.subqueries) == 2
    assert [step.level for step in plan.retrieval_steps] == ["parent", "child"]
    assert plan.rerank.enabled is True


def test_planner_keeps_domain_constraints_for_complex_query() -> None:
    cfg = AppConfig()
    planner = QueryPlanner(FilterParser(Ontology.load("configs/schema.yaml")))
    plan = planner.build_plan("只考虑1200K蒸汽气化条件下NH3生成的机制链", cfg)
    fields = {item.field for item in plan.query.hard_constraints + plan.query.soft_constraints}
    assert "stage" in fields
    assert "targets" in fields
    assert "T_range_K" in fields
    assert plan.complex.query_type == "multi_hop"
''',
    )

    _write(
        "tests/test_complex_question_service.py",
        '''"""验证比较、多跳和跨文档执行器复用正式 Retriever。"""

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
        if "co2" in query or "二氧化碳" in query:
            source = "co2.pdf"
        elif "相反" in query or "冲突" in query:
            source = "conflict.pdf"
        elif "条件差异" in query:
            source = "conditions.pdf"
        elif "蒸汽" in query:
            source = "steam.pdf"
        else:
            source = f"{abs(hash(query)) % 3}.pdf"
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
''',
    )

    _write(
        "tests/test_complex_aggregation.py",
        '''"""验证统计聚合由 Python 基于结构化记录确定性计算。"""

from __future__ import annotations

from coal_kb.complex_qa.aggregation import AggregationExecutor, AggregationRepository
from coal_kb.core.models.query import AggregationSpec, ComplexQuestionSpec, QueryPlan, QueryUnderstanding
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
''',
    )

    _write(
        "tests/test_complex_tables.py",
        '''"""验证标准表格 JSONL 的行检索和精确证据元数据。"""

from __future__ import annotations

import json

from coal_kb.complex_qa.tables import TableExecutor, TableRepository
from coal_kb.core.models.query import ComplexQuestionSpec, QueryPlan, QueryUnderstanding


class EmptyRetriever:
    def execute(self, plan, trace=None):
        return []


def test_table_executor_returns_matching_row(tmp_path) -> None:
    path = tmp_path / "tables.jsonl"
    payload = {
        "table_id": "table_001",
        "source_file": "paper.pdf",
        "page": 7,
        "caption": "不同温度下的H2产率",
        "headers": ["T_K", "H2"],
        "rows": [{"T_K": 1000, "H2": 30.0}, {"T_K": 1200, "H2": 42.1}],
        "nearby_text": "蒸汽气化实验",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    plan = QueryPlan(
        query=QueryUnderstanding(raw="表格中1200K的H2产率", normalized="表格中1200K的H2产率"),
        complex=ComplexQuestionSpec(query_type="table", require_table=True),
    )
    result = TableExecutor(TableRepository(str(path)), EmptyRetriever(), 5).process(plan)
    assert result.documents
    assert result.documents[0].metadata["table_id"] == "table_001"
    assert result.documents[0].metadata["page"] == 7
    assert result.trace["table_matches"] >= 1
''',
    )

    _write(
        "tests/test_complex_evaluation.py",
        '''"""验证复杂问答数据字段和专项指标。"""

from coal_kb.evaluation.metrics import complex_question_metrics
from coal_kb.evaluation.models import EvaluationCase, QueryType, RetrievedEvidence


def test_complex_case_parses_milestone_c_fields() -> None:
    case = EvaluationCase.from_dict(
        {
            "id": "case_1",
            "query": "比较A与B",
            "query_type": "comparison",
            "expected_subqueries": ["A证据", "B证据"],
            "expected_min_sources": 2,
        },
        row_number=1,
    )
    assert case.query_type == QueryType.COMPARISON
    assert case.expected_subqueries == ("A证据", "B证据")


def test_comparison_and_cross_document_metrics_use_actual_trace() -> None:
    comparison = EvaluationCase(
        case_id="comparison",
        query="比较A与B",
        query_type=QueryType.COMPARISON,
        expected_subqueries=("A证据", "B证据"),
    )
    trace = {
        "complex_route": {
            "query_type": "comparison",
            "subqueries": [{"query": "A证据"}, {"query": "B证据"}],
        },
        "complex_execution": {"query_type": "comparison"},
    }
    retrieved = (
        RetrievedEvidence(1, "a.pdf", None, 1, None, "a", "", metadata={"complex_role": "comparison_1"}),
        RetrievedEvidence(2, "b.pdf", None, 1, None, "b", "", metadata={"complex_role": "comparison_2"}),
    )
    values = complex_question_metrics(comparison, trace, retrieved)
    assert values["route_accuracy"] == 1.0
    assert values["subquery_recall"] == 1.0
    assert values["comparison_side_coverage"] == 1.0

    cross = EvaluationCase(
        case_id="cross",
        query="多篇文献共识",
        query_type=QueryType.CROSS_DOCUMENT,
        expected_min_sources=2,
    )
    cross_trace = {"complex_route": {"query_type": "cross_document"}}
    cross_values = complex_question_metrics(cross, cross_trace, retrieved)
    assert cross_values["cross_document_coverage"] == 1.0
''',
    )

    quality_path = ROOT / "scripts/quality/config.sh"
    quality_text = quality_path.read_text(encoding="utf-8")
    if '"$REPO_ROOT/src/coal_kb/complex_qa"' not in quality_text:
        quality_text = quality_text.replace(
            '  "$REPO_ROOT/src/coal_kb/core"\n',
            '  "$REPO_ROOT/src/coal_kb/core"\n  "$REPO_ROOT/src/coal_kb/complex_qa"\n',
        )
    additions = [
        '  "$REPO_ROOT/tests/test_complex_question_router.py"',
        '  "$REPO_ROOT/tests/test_complex_query_planner.py"',
        '  "$REPO_ROOT/tests/test_complex_question_service.py"',
        '  "$REPO_ROOT/tests/test_complex_aggregation.py"',
        '  "$REPO_ROOT/tests/test_complex_tables.py"',
        '  "$REPO_ROOT/tests/test_complex_evaluation.py"',
    ]
    marker = '  "$REPO_ROOT/tests/test_provider_runtime.py"\n'
    if additions[0] not in quality_text:
        quality_text = quality_text.replace(marker, marker + "\n".join(additions) + "\n")
    quality_path.write_text(quality_text, encoding="utf-8")


if __name__ == "__main__":
    process()
