from __future__ import annotations

from dataclasses import dataclass

from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.settings import AppConfig

from .plan import (
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


@dataclass
class QueryPlanner:
    filter_parser: FilterParser

    def build_plan(
        self,
        question: str,
        config: AppConfig,
        *,
        enable_llm: bool = False,
        llm_config=None,
    ) -> QueryPlan:
        parsed = self.filter_parser.parse(question)
        rewrite = rewrite_query(
            question,
            enable_llm=(config.query_rewrite.enable_llm and enable_llm),
            llm_config=llm_config,
        )

        def _to_constraint(c) -> Constraint:
            return Constraint(
                field=c.name, op=c.ctype, value=c.value,
                priority=c.priority, confidence=c.confidence, source=c.source,
            )

        q = QueryUnderstanding(
            raw=question,
            normalized=question.strip(),
            rewritten=rewrite.query,
            rewrite_reason=rewrite.reason,
            language="zh" if any("一" <= ch <= "鿿" for ch in question) else "en",
            hard_constraints=[_to_constraint(c) for c in parsed.hard_constraints],
            soft_constraints=[_to_constraint(c) for c in parsed.soft_constraints],
        )

        steps = [
            RetrievalStep(
                name="stage1_parent", level="parent", fusion_mode="rrf",
                k_candidates=max(config.retrieval.two_stage.parent_k_candidates, 200),
                k_final=max(config.retrieval.two_stage.parent_k_final, 80),
                where_mode="hard_only", enable_relax=False,
            ),
            RetrievalStep(
                name="stage2_child", level="child", fusion_mode="rrf",
                k_candidates=max(config.retrieval.two_stage.child_k_candidates, 200),
                k_final=max(config.retrieval.two_stage.child_k_final, 60),
                where_mode="hard_only",
                enable_relax=config.retrieval.two_stage.allow_relax_in_stage2,
            ),
        ]

        schedule = list(config.retrieval.range_expand_schedule or [0.05, 0.1, 0.2])

        rules = []
        if len(schedule) >= 1:
            rules.append(RelaxRule(
                drop_fields=["flags"],
                widen_ranges={"T_range_K": schedule[0], "P_range_MPa": schedule[0]},
                soften_priority=True,
            ))
        if len(schedule) >= 2:
            rules.append(RelaxRule(
                drop_fields=["flags", "targets"],
                widen_ranges={"T_range_K": schedule[1], "P_range_MPa": schedule[1]},
                soften_priority=True,
            ))
        if len(schedule) >= 3:
            rules.append(RelaxRule(
                drop_fields=["flags", "targets", "stage"],
                widen_ranges={"T_range_K": schedule[2], "P_range_MPa": schedule[2]},
                soften_priority=True,
            ))

        relax = RelaxPolicy(max_steps=config.retrieval.max_relax_steps, rules=rules)

        return QueryPlan(
            query=q,
            retrieval_steps=steps,
            relax_policy=relax,
            rerank=RerankSpec(
                enabled=config.retrieval.rerank_enabled,
                top_n=config.retrieval.rerank_top_n,
            ),
            diversity=DiversitySpec(max_per_source=config.retrieval.max_per_source),
            context=ContextSpec(
                max_context_tokens=4000, max_evidence_chunks=16,
            ),
            answer=AnswerSpec(require_citations=True, min_evidence=1),
        )
