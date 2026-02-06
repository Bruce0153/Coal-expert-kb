from coal_kb.metadata.normalize import Ontology
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.settings import load_config


def test_planner_builds_two_stage_plan():
    cfg = load_config()
    onto = Ontology.load("configs/schema.yaml")
    planner = QueryPlanner(filter_parser=FilterParser(onto=onto))
    plan = planner.build_plan("只考虑蒸汽气化 1200K NH3 HCN", cfg)
    assert plan.retrieval_steps[0].level == "parent"
    assert plan.retrieval_steps[1].level == "child"
    fields = {c.field for c in (plan.query.hard_constraints + plan.query.soft_constraints)}
    assert "stage" in fields
    assert "targets" in fields
