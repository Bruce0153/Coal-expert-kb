"""验证复杂问答数据字段和专项指标。"""

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
