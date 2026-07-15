"""构建事实检索与复杂科学问答共用的 QueryPlan。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from coal_kb.complex_qa.planning import build_complex_spec
from coal_kb.core.models.query import (
    AnswerSpec,
    Constraint,
    ContextSpec,
    DiversitySpec,
    NeighborSpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RerankSpec,
    RetrievalStep,
)
from coal_kb.retrieval.query.filter_parser import FilterParser


@dataclass
class QueryPlanner:
    """持有领域过滤解析器并生成完整 QueryPlan。"""

    filter_parser: FilterParser

    def build_plan(
        self,
        query: str,
        cfg: Any,
        *,
        enable_llm: bool = False,
        llm_config: Any = None,
    ) -> QueryPlan:
        del enable_llm, llm_config
        normalized = " ".join(query.strip().split())
        parsed = self.filter_parser.parse(normalized)
        hard = [self._constraint(value) for value in parsed.hard_constraints]
        soft = [self._constraint(value) for value in parsed.soft_constraints]
        complex_spec = build_complex_spec(
            normalized,
            max_subqueries=cfg.complex_qa.max_subqueries,
            max_multi_hop_steps=cfg.complex_qa.max_multi_hop_steps,
        )
        retrieval_steps = self._retrieval_steps(cfg)
        context_tokens, evidence_chunks = self._context_budget(complex_spec.query_type, cfg)
        min_evidence = 1 if complex_spec.query_type in {"aggregation", "table"} else 2
        if complex_spec.query_type == "cross_document":
            min_evidence = cfg.complex_qa.cross_document_min_sources
        return QueryPlan(
            query=QueryUnderstanding(raw=query, normalized=normalized, hard_constraints=hard, soft_constraints=soft),
            complex=complex_spec,
            retrieval_steps=retrieval_steps,
            relax_policy=RelaxPolicy(max_steps=cfg.retrieval.max_relax_steps),
            rerank=RerankSpec(enabled=cfg.retrieval.rerank_enabled, top_n=cfg.retrieval.rerank_top_n),
            neighbor=NeighborSpec(enabled=complex_spec.query_type in {"comparison", "multi_hop", "table", "cross_document"}, window=1),
            diversity=DiversitySpec(
                max_per_source=(
                    cfg.complex_qa.cross_document_max_per_source
                    if complex_spec.query_type == "cross_document"
                    else cfg.retrieval.max_per_source
                )
            ),
            context=ContextSpec(max_context_tokens=context_tokens, max_evidence_chunks=evidence_chunks),
            answer=AnswerSpec(min_evidence=min_evidence),
        )

    @staticmethod
    def _constraint(value: Any) -> Constraint:
        return Constraint(
            field=value.name,
            op=value.ctype,
            value=value.value,
            priority=value.priority,
            confidence=value.confidence,
            source=value.source,
        )

    @staticmethod
    def _retrieval_steps(cfg: Any) -> list[RetrievalStep]:
        if cfg.retrieval.two_stage.enabled:
            return [
                RetrievalStep(
                    name="parent_recall",
                    level="parent",
                    fusion_mode="rrf",
                    k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
                    k_final=cfg.retrieval.two_stage.parent_k_final,
                    where_mode="hard_only",
                    enable_relax=False,
                ),
                RetrievalStep(
                    name="child_recall",
                    level="child",
                    fusion_mode="rrf",
                    k_candidates=cfg.retrieval.two_stage.child_k_candidates,
                    k_final=cfg.retrieval.two_stage.child_k_final,
                    where_mode="hard_only",
                    enable_relax=cfg.retrieval.two_stage.allow_relax_in_stage2,
                ),
            ]
        return [
            RetrievalStep(
                name="single_recall",
                level="single",
                fusion_mode="rrf",
                k_candidates=cfg.retrieval.k,
                k_final=cfg.retrieval.k,
                where_mode="hard_only",
                enable_relax=True,
            )
        ]

    @staticmethod
    def _context_budget(query_type: str, cfg: Any) -> tuple[int, int]:
        base_tokens = cfg.complex_qa.base_context_tokens
        base_chunks = cfg.complex_qa.base_evidence_chunks
        multipliers = {
            "comparison": 1.6,
            "multi_hop": 1.8,
            "aggregation": 1.2,
            "table": 1.4,
            "cross_document": 2.0,
        }
        multiplier = multipliers.get(query_type, 1.0)
        return int(base_tokens * multiplier), max(base_chunks, int(base_chunks * multiplier))
