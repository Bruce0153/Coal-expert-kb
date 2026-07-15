"""构造硬过滤、软约束和范围放宽计划。"""

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
    """从唯一约束模型生成检索执行计划。"""
    schedule = range_expand_schedule or config.DEFAULT_RANGE_EXPAND_SCHEDULE
    steps = min(max_relax_steps, len(schedule))
    return RetrievalPlan(
        hard_where=constraint_set.to_where_hard(),
        soft_constraints=constraint_set.soft_constraints,
        relax_steps=[
            f"expand_numeric_range={int(schedule[index] * 100)}%"
            for index in range(steps)
        ],
    )
