"""定义查询规划、复杂问答路线、检索策略与回答约束。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

QuestionType = Literal[
    "fact",
    "comparison",
    "multi_hop",
    "aggregation",
    "table",
    "cross_document",
    "unanswerable",
]
AggregationOperation = Literal["count", "sum", "average", "min", "max", "median", "group_by", "top_k"]


class Constraint(BaseModel):
    """表示查询中的一个硬约束或软约束。"""

    field: str
    op: str = "eq"
    value: Any
    priority: Literal["hard", "soft"] = "soft"
    confidence: float = 0.5
    source: str = "rule"
    note: Optional[str] = None


class QueryUnderstanding(BaseModel):
    """保存原始问题、标准化问题和领域约束。"""

    raw: str
    normalized: str
    language: str = "zh"
    rewritten: Optional[str] = None
    rewrite_reason: Optional[str] = None
    hard_constraints: List[Constraint] = Field(default_factory=list)
    soft_constraints: List[Constraint] = Field(default_factory=list)


class SubQuerySpec(BaseModel):
    """表示复杂问题中的一个可回放子问题。"""

    subquery_id: str
    query: str
    purpose: str
    depends_on: List[str] = Field(default_factory=list)


class AggregationSpec(BaseModel):
    """表示结构化记录上的确定性聚合请求。"""

    operation: AggregationOperation = "count"
    field: Optional[str] = None
    group_by: Optional[str] = None
    top_k: int = 10


class ComplexQuestionSpec(BaseModel):
    """保存复杂科学问答的统一路由结果。"""

    query_type: QuestionType = "fact"
    confidence: float = 1.0
    reason: str = "默认事实检索"
    subqueries: List[SubQuerySpec] = Field(default_factory=list)
    comparison_entities: List[str] = Field(default_factory=list)
    aggregation: Optional[AggregationSpec] = None
    require_table: bool = False
    require_cross_document: bool = False


class RetrievalStep(BaseModel):
    """表示一个检索阶段。"""

    name: str
    level: Literal["parent", "child", "single"]
    fusion_mode: Literal["vector", "bm25", "rrf"] = "rrf"
    k_candidates: int
    k_final: int
    where_mode: Literal["hard_only", "full"] = "hard_only"
    enable_relax: bool = False


class RelaxRule(BaseModel):
    """表示一次约束放宽规则。"""

    drop_fields: List[str] = Field(default_factory=list)
    widen_ranges: Dict[str, float] = Field(default_factory=dict)
    soften_priority: bool = True


class RelaxPolicy(BaseModel):
    """定义最多允许执行的约束放宽步骤。"""

    max_steps: int = 2
    rules: List[RelaxRule] = Field(default_factory=list)


class RerankSpec(BaseModel):
    """定义重排序开关和输出数量。"""

    enabled: bool = False
    top_n: int = 10


class NeighborSpec(BaseModel):
    """定义相邻证据扩展。"""

    enabled: bool = False
    window: int = 1


class DiversitySpec(BaseModel):
    """定义单一来源的最大证据数量。"""

    max_per_source: int = 2


class ContextSpec(BaseModel):
    """定义上下文预算和去重方式。"""

    max_context_tokens: int = 1800
    max_evidence_chunks: int = 8
    group_by_heading: bool = True
    deduplicate: bool = True
    dedup_mode: Literal["text", "embedding", "off"] = "text"


class AnswerSpec(BaseModel):
    """定义回答引用和拒答要求。"""

    require_citations: bool = True
    refuse_threshold: float = 0.3
    min_evidence: int = 2
    output_format: Literal["markdown", "json"] = "markdown"


class ObservabilitySpec(BaseModel):
    """定义 Trace 和调试配置。"""

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    log_plan: bool = True
    debug: bool = False


class QueryPlan(BaseModel):
    """统一表示事实检索和复杂科学问答计划。"""

    query: QueryUnderstanding
    complex: ComplexQuestionSpec = Field(default_factory=ComplexQuestionSpec)
    retrieval_steps: List[RetrievalStep] = Field(default_factory=list)
    relax_policy: RelaxPolicy = Field(default_factory=RelaxPolicy)
    rerank: RerankSpec = Field(default_factory=RerankSpec)
    neighbor: NeighborSpec = Field(default_factory=NeighborSpec)
    diversity: DiversitySpec = Field(default_factory=DiversitySpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    answer: AnswerSpec = Field(default_factory=AnswerSpec)
    observability: ObservabilitySpec = Field(default_factory=ObservabilitySpec)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
