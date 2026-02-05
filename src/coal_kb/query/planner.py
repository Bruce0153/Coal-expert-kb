from __future__ import annotations

"""Query planner that converts user query to executable QueryPlan."""

from dataclasses import dataclass
from typing import Optional

from coal_kb.query.plan import QueryPlan, RelaxPolicy, RetrievalStep
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.settings import AppConfig


@dataclass
class QueryPlanner:
    cfg: AppConfig
    parser: FilterParser

    def build_plan(self, user_query: str) -> QueryPlan:
        parsed = self.parser.parse(user_query)
        rewrite = rewrite_query(
            user_query,
            enable_llm=self.cfg.query_rewrite.enable_llm,
            llm_config=self.cfg.llm,
        )
        hard = parsed.hard_constraints
        soft = parsed.soft_constraints
        hard_where = {c.name: c.value for c in hard}

        two_stage_cfg = self.cfg.retrieval.two_stage
        two_stage = bool(self.cfg.backend == "elastic" and two_stage_cfg.enabled)
        stage1 = RetrievalStep(
            name="parents",
            enabled=two_stage,
            filters=dict(hard_where),
            k_candidates=two_stage_cfg.parent_k_candidates,
            k_final=two_stage_cfg.parent_k_final,
            notes="stage1 parents strict hard constraints",
        )
        stage2_filters = dict(hard_where)
        stage2 = RetrievalStep(
            name="children",
            enabled=True,
            filters=stage2_filters,
            k_candidates=two_stage_cfg.child_k_candidates,
            k_final=max(two_stage_cfg.child_k_final, self.cfg.retrieval.k),
            notes="stage2 children scoped by parent_ids",
        )
        baseline = RetrievalStep(
            name="baseline_children",
            enabled=True,
            filters=dict(hard_where),
            k_candidates=two_stage_cfg.child_k_candidates,
            k_final=max(two_stage_cfg.child_k_final, self.cfg.retrieval.k),
            notes="single-stage fallback",
        )
        relax = RelaxPolicy(
            allow_relax=two_stage_cfg.allow_relax_in_stage2,
            dropped_filters=["T_range_K", "P_range_MPa"] if two_stage_cfg.allow_relax_in_stage2 else [],
        )
        return QueryPlan(
            user_query=user_query,
            query_text=rewrite.query,
            hard_constraints=hard,
            soft_constraints=soft,
            stage1=stage1,
            stage2=stage2,
            baseline=baseline,
            two_stage_enabled=two_stage,
            rerank_enabled=self.cfg.retrieval.rerank_enabled,
            rerank_top_n=self.cfg.retrieval.rerank_top_n,
            neighbor_expand_n=1,
            token_budget=2200,
            relax_policy=relax,
            debug={"rewrite_reason": rewrite.reason, "compat_where": parsed.compat_where},
        )
