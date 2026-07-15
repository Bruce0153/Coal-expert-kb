"""验证 C0-C6 统一路由和子问题规划。"""

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
