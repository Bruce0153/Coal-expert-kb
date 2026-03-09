from __future__ import annotations

from coal_kb.metadata.normalize import Ontology
from coal_kb.retrieval.constraint_policy import build_plan
from coal_kb.retrieval.filter_parser import FilterParser


def test_constraint_policy_relax_steps() -> None:
    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)
    constraints = parser.parse("必须 1200K 2MPa steam gasification")
    plan = build_plan(constraints, max_relax_steps=2)
    assert plan.relax_steps
    assert any(c.priority == "hard" for c in constraints.constraints)
