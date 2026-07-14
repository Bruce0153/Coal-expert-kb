"""构造保持原语义的硬过滤和软约束计划。"""

from __future__ import annotations

from dataclasses import dataclass, field

from coal_kb.retrieval.constraints import config
from coal_kb.retrieval.constraints.models import Constraint, ConstraintSet


@dataclass
class RetrievalPlan:
    hard_where: dict[str, object]
    soft_constraints: list[Constraint]
    relax_steps: list[str] = field(default_factory=list)


def build_plan(
    constraint_set: ConstraintSet,
    *,
    max_relax_steps: int = config.DEFAULT_MAX_RELAX_STEPS,
    range_expand_schedule: list[float] | None = None,
) -> RetrievalPlan:
    hard_where = constraint_set.to_where_hard()
    soft_constraints = constraint_set.soft_features()
    schedule = range_expand_schedule or config.DEFAULT_RANGE_EXPAND_SCHEDULE
    steps = min(max_relax_steps, len(schedule))
    relax_steps = [f"expand_numeric_range={int(schedule[index] * 100)}%" for index in range(steps)]
    return RetrievalPlan(
        hard_where=hard_where,
        soft_constraints=soft_constraints,
        relax_steps=relax_steps,
    )
