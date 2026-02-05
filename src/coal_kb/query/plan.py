from __future__ import annotations

"""Query planning primitives for modern RAG execution.

Planner decides *what to do*; retriever executor only performs the plan.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coal_kb.retrieval.constraints import Constraint


@dataclass
class RetrievalStep:
    name: str
    enabled: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    k_candidates: int = 50
    k_final: int = 10
    notes: str = ""


@dataclass
class RelaxPolicy:
    allow_relax: bool = True
    dropped_filters: List[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    user_query: str
    query_text: str
    hard_constraints: List[Constraint] = field(default_factory=list)
    soft_constraints: List[Constraint] = field(default_factory=list)
    stage1: RetrievalStep = field(default_factory=lambda: RetrievalStep(name="parents"))
    stage2: RetrievalStep = field(default_factory=lambda: RetrievalStep(name="children"))
    baseline: RetrievalStep = field(default_factory=lambda: RetrievalStep(name="baseline_children", enabled=False))
    two_stage_enabled: bool = True
    rerank_enabled: bool = True
    rerank_top_n: int = 10
    neighbor_expand_n: int = 1
    token_budget: int = 2200
    relax_policy: RelaxPolicy = field(default_factory=RelaxPolicy)
    debug: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "query_text": self.query_text,
            "hard_constraints": [c.__dict__ for c in self.hard_constraints],
            "soft_constraints": [c.__dict__ for c in self.soft_constraints],
            "two_stage_enabled": self.two_stage_enabled,
            "stage1": self.stage1.__dict__,
            "stage2": self.stage2.__dict__,
            "baseline": self.baseline.__dict__,
            "rerank_enabled": self.rerank_enabled,
            "rerank_top_n": self.rerank_top_n,
            "neighbor_expand_n": self.neighbor_expand_n,
            "token_budget": self.token_budget,
            "relax_policy": self.relax_policy.__dict__,
            "debug": self.debug,
        }
