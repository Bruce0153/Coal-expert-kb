from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.settings import AppConfig

from .plan import (
    AnswerSpec,
    Constraint,
    ContextSpec,
    DiversitySpec,
    NeighborSpec,
    ObservabilitySpec,
    QueryPlan,
    QueryUnderstanding,
    RelaxPolicy,
    RelaxRule,
    RerankSpec,
    RetrievalStep,
)


@dataclass
class QueryPlanner:
    filter_parser: FilterParser

    def build_plan(self, question: str, config: AppConfig, *, enable_llm: bool = False, llm_config=None) -> QueryPlan:
        parsed = self.filter_parser.parse(question)
        rewrite = rewrite_query(
            question,
            enable_llm=(config.query_rewrite.enable_llm and enable_llm),
            llm_config=llm_config,
        )

        def _to_constraint(c) -> Constraint:
            return Constraint(
                field=c.name,
                op=c.ctype,
                value=c.value,
                priority=c.priority,
                confidence=c.confidence,
                source=c.source,
            )

        q = QueryUnderstanding(
            raw=question,
            normalized=question.strip(),
            rewritten=rewrite.query,
            rewrite_reason=rewrite.reason,
            language="zh" if any("\u4e00" <= ch <= "\u9fff" for ch in question) else "en",
            hard_constraints=[_to_constraint(c) for c in parsed.hard_constraints],
            soft_constraints=[_to_constraint(c) for c in parsed.soft_constraints],
        )

        steps = [
            RetrievalStep(
                name="stage1_parent",
                level="parent",
                fusion_mode="rrf",
                k_candidates=config.retrieval.two_stage.parent_k_candidates,
                k_final=config.retrieval.two_stage.parent_k_final,
                where_mode="full",
                enable_relax=False,
            ),
            RetrievalStep(
                name="stage2_child",
                level="child",
                fusion_mode="rrf",
                k_candidates=config.retrieval.two_stage.child_k_candidates,
                k_final=config.retrieval.two_stage.child_k_final,
                where_mode="full",
                enable_relax=config.retrieval.two_stage.allow_relax_in_stage2,
            ),
        ]

        relax = RelaxPolicy(
            max_steps=config.retrieval.max_relax_steps,
            rules=[
                RelaxRule(
                    drop_fields=["flags"],
                    widen_ranges={"T_range_K": x, "P_range_MPa": x},
                    soften_priority=True,
                )
                for x in config.retrieval.range_expand_schedule
            ],
        )

        return QueryPlan(
            query=q,
            retrieval_steps=steps,
            relax_policy=relax,
            rerank=RerankSpec(enabled=config.retrieval.rerank_enabled, top_n=config.retrieval.rerank_top_n),
            neighbor=NeighborSpec(enabled=False, window=1),
            diversity=DiversitySpec(max_per_source=config.retrieval.max_per_source),
            context=ContextSpec(),
            answer=AnswerSpec(),
            observability=ObservabilitySpec(trace_id=str(uuid4()), log_plan=True, debug=False),
        )
