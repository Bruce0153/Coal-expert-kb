"""Backward-compatible import path for query plan models.

Canonical definitions live in :mod:`coal_kb.core.models.query`.
"""

from coal_kb.core.models.query import (
    AnswerSpec,
    Constraint,
    ContextSpec,
    DiversitySpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RelaxRule,
    RerankSpec,
    RetrievalStep,
)

__all__ = [
    "AnswerSpec",
    "Constraint",
    "ContextSpec",
    "DiversitySpec",
    "QueryPlan",
    "QueryUnderstanding",
    "RelaxPolicy",
    "RelaxRule",
    "RerankSpec",
    "RetrievalStep",
]
