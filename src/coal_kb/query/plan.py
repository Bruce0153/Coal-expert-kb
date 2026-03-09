from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Constraint(BaseModel):
    field: str
    op: str = "eq"
    value: Any
    priority: Literal["hard", "soft"] = "soft"
    confidence: float = 0.5
    source: str = "rule"
    note: Optional[str] = None


class QueryUnderstanding(BaseModel):
    raw: str
    normalized: str
    language: str = "zh"
    rewritten: Optional[str] = None
    rewrite_reason: Optional[str] = None
    hard_constraints: List[Constraint] = Field(default_factory=list)
    soft_constraints: List[Constraint] = Field(default_factory=list)


class RetrievalStep(BaseModel):
    name: str
    level: Literal["parent", "child", "single"]
    fusion_mode: Literal["vector", "bm25", "rrf"] = "rrf"
    k_candidates: int
    k_final: int
    where_mode: Literal["hard_only", "full"] = "hard_only"
    enable_relax: bool = False


class RelaxRule(BaseModel):
    drop_fields: List[str] = Field(default_factory=list)
    widen_ranges: Dict[str, float] = Field(default_factory=dict)
    soften_priority: bool = True


class RelaxPolicy(BaseModel):
    max_steps: int = 2
    rules: List[RelaxRule] = Field(default_factory=list)


class RerankSpec(BaseModel):
    enabled: bool = False
    top_n: int = 10


class NeighborSpec(BaseModel):
    enabled: bool = False
    window: int = 1


class DiversitySpec(BaseModel):
    max_per_source: int = 2


class ContextSpec(BaseModel):
    max_context_tokens: int = 1800
    max_evidence_chunks: int = 8
    group_by_heading: bool = True
    deduplicate: bool = True
    dedup_mode: Literal["text", "embedding", "off"] = "text"


class AnswerSpec(BaseModel):
    require_citations: bool = True
    refuse_threshold: float = 0.3
    min_evidence: int = 2
    output_format: Literal["markdown", "json"] = "markdown"


class ObservabilitySpec(BaseModel):
    trace_id: str
    log_plan: bool = True
    debug: bool = False


class QueryPlan(BaseModel):
    query: QueryUnderstanding
    retrieval_steps: List[RetrievalStep] = Field(default_factory=list)
    relax_policy: RelaxPolicy = Field(default_factory=RelaxPolicy)
    rerank: RerankSpec = Field(default_factory=RerankSpec)
    neighbor: NeighborSpec = Field(default_factory=NeighborSpec)
    diversity: DiversitySpec = Field(default_factory=DiversitySpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    answer: AnswerSpec = Field(default_factory=AnswerSpec)
    observability: ObservabilitySpec

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
