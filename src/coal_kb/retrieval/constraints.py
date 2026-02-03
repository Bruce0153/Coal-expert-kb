from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Constraint:
    name: str
    ctype: str
    value: Any
    confidence: float
    source: str
    priority: str = "soft"  # "hard" or "soft"


@dataclass
class ConstraintSet:
    constraints: List[Constraint] = field(default_factory=list)
    compat_where: Dict[str, Any] = field(default_factory=dict)

    @property
    def hard_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if c.priority == "hard"]

    @property
    def soft_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if c.priority != "hard"]

    def to_where_hard(self) -> Dict[str, Any]:
        where: Dict[str, Any] = {}
        for c in self.hard_constraints:
            where[c.name] = c.value
        return where

    def soft_features(self) -> List[Constraint]:
        return self.soft_constraints
