from __future__ import annotations

from coal_kb.metadata.normalize import Ontology
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.settings import load_config


def test_query_planner_builds_two_stage_plan() -> None:
    cfg = load_config()
    cfg.backend = "elastic"
    onto = Ontology.load("configs/schema.yaml")
    planner = QueryPlanner(cfg=cfg, parser=FilterParser(onto=onto))
    plan = planner.build_plan("steam gasification at 1200K")
    assert plan.query_text
    assert plan.stage1.name == "parents"
    assert plan.stage2.name == "children"
    assert plan.two_stage_enabled is True
