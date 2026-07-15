"""验证 QueryPlanner 同时生成领域约束和复杂问答计划。"""

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
