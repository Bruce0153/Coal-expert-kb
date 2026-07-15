"""定义检索层内部的硬约束和软约束。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Constraint:
    name: str
    ctype: str
    value: Any
    confidence: float
    source: str
    priority: str = "soft"


@dataclass
class ConstraintSet:
    constraints: list[Constraint] = field(default_factory=list)

    @property
    def hard_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if constraint.priority == "hard"]

    @property
    def soft_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if constraint.priority != "hard"]

    def to_where_hard(self) -> dict[str, Any]:
        return {
            constraint.name: constraint.value
            for constraint in self.hard_constraints
        }
