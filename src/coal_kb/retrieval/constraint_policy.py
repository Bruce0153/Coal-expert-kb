from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .constraints import Constraint, ConstraintSet


@dataclass
class RetrievalPlan:
    hard_where: Dict[str, object]
    soft_constraints: List[Constraint]
    relax_steps: List[str] = field(default_factory=list)


def build_plan(
    constraint_set: ConstraintSet,
    *,
    max_relax_steps: int = 2,
    range_expand_schedule: List[float] | None = None,
) -> RetrievalPlan:
    hard_where = constraint_set.to_where_hard()
    soft_constraints = constraint_set.soft_features()
    relax_steps: List[str] = []

    schedule = range_expand_schedule or [0.05, 0.1, 0.2]
    steps = min(max_relax_steps, len(schedule))
    for i in range(steps):
        relax_steps.append(f"expand_numeric_range={int(schedule[i] * 100)}%")
    return RetrievalPlan(
        hard_where=hard_where,
        soft_constraints=soft_constraints,
        relax_steps=relax_steps,
    )
