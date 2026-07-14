"""编排约束、召回、软排序、重排和多样性处理。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from coal_kb.core.models.query import Constraint as PlanConstraint
from coal_kb.core.models.query import QueryPlan
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig, make_embeddings
from coal_kb.ingestion.chunking.sectioner import is_reference_like
from coal_kb.recall import DenseRecall, ParentChildRecall
from coal_kb.reranking import RerankingService
from coal_kb.retrieval import config
from coal_kb.retrieval.constraints import Constraint, ConstraintSet, build_plan

logger = logging.getLogger(__name__)


def _doc_range_overlap(meta: dict, query_range: Optional[List[float]], *, key_point: str, key_min: str, key_max: str) -> bool:
    if query_range is None:
        return True
    qlo, qhi = float(query_range[0]), float(query_range[1])
    dmin = meta.get(key_min)
    dmax = meta.get(key_max)
    if dmin is not None and dmax is not None:
        return max(float(dmin), qlo) <= min(float(dmax), qhi)
    x = meta.get(key_point)
    return x is not None and qlo <= float(x) <= qhi


def _doc_key(d: Document) -> str:
    m = d.metadata or {}
    return str(m.get("chunk_id") or f'{m.get("source_file","")}|{m.get("page","")}')


@dataclass
class ExpertRetriever:
    vector_retriever_factory: Any
    k: int = 6

    rerank_enabled: bool = False
    rerank_top_n: int = 10
    reranker: Optional[Any] = None

    max_per_source: int = 2
    max_relax_steps: int = 2
    range_expand_schedule: Optional[List[float]] = None
    mode: str = "balanced"
    drop_sections: Optional[List[str]] = None
    drop_reference_like: bool = True
    use_fuse: bool = True
    where_full: bool = False

    two_stage_enabled: bool = True
    parent_k_candidates: int = 200
    parent_k_final: int = 60
    max_parents: int = 60
    child_k_candidates: int = 300
    child_k_final: int = 30
    allow_relax_in_stage2: bool = True
    elastic_store: Optional[ElasticStore] = None
    elastic_index: Optional[str] = None
    embeddings_cfg: Optional[EmbeddingsConfig] = None
    elastic_use_icu: bool = False
    tenant_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._embeddings = None
        self._dense_recall = DenseRecall(self.vector_retriever_factory)
        self._parent_child_recall = None
        self._reranking_service = RerankingService(self.reranker) if self.reranker is not None else None
        if self.two_stage_enabled and self.elastic_store and self.embeddings_cfg and self.elastic_index:
            self._embeddings = make_embeddings(self.embeddings_cfg)
            self._parent_child_recall = ParentChildRecall(
                elastic_store=self.elastic_store,
                elastic_index=self.elastic_index,
                embeddings=self._embeddings,
                tenant_id=self.tenant_id,
                use_icu=self.elastic_use_icu,
            )

    def execute(self, plan: QueryPlan, trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        query = plan.query.rewritten or plan.query.normalized
        constraint_set = self._constraintset_from_plan(plan)

        if not self._two_stage_available():
            return self._retrieve_single_stage(query, constraint_set, trace)

        s1 = next((s for s in plan.retrieval_steps if s.level == "parent"), None)
        s2 = next((s for s in plan.retrieval_steps if s.level == "child"), None)
        if s1 is None or s2 is None:
            return self._retrieve_single_stage(query, constraint_set, trace)

        return self._do_two_stage(
            query, constraint_set, trace,
            parent_k_candidates=s1.k_candidates, parent_k_final=s1.k_final,
            child_k_candidates=s2.k_candidates, child_k_final=max(s2.k_final, self.k),
            enable_relax=s2.enable_relax,
            use_plan_soft_constraints=plan.query.soft_constraints,
            plan_for_trace=plan,
        )

    def retrieve(self, query: str, parsed_filter: Union[Dict[str, Any], ConstraintSet], trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        constraint_set = parsed_filter if isinstance(parsed_filter, ConstraintSet) else ConstraintSet(constraints=[], compat_where=parsed_filter)
        if not self._two_stage_available():
            return self._retrieve_single_stage(query, constraint_set, trace)

        return self._do_two_stage(
            query, constraint_set, trace,
            parent_k_candidates=self.parent_k_candidates, parent_k_final=self.parent_k_final,
            child_k_candidates=self.child_k_candidates, child_k_final=max(self.child_k_final, self.k),
            enable_relax=self.allow_relax_in_stage2,
        )

    def _two_stage_available(self) -> bool:
        return bool(self.two_stage_enabled and self._parent_child_recall is not None)

    # ------------------------------------------------------------------
    # unified two-stage retrieval
    # ------------------------------------------------------------------
    def _do_two_stage(
        self,
        query: str,
        constraint_set: ConstraintSet,
        trace: Optional[Dict[str, Any]],
        *,
        parent_k_candidates: int,
        parent_k_final: int,
        child_k_candidates: int,
        child_k_final: int,
        enable_relax: bool = True,
        use_plan_soft_constraints: Optional[List[PlanConstraint]] = None,
        plan_for_trace: Optional[QueryPlan] = None,
    ) -> List[Document]:
        where = self._build_where(constraint_set)
        assert self._parent_child_recall is not None
        recall_result = self._parent_child_recall.process(
            query,
            where=where,
            parent_k_candidates=parent_k_candidates,
            parent_k_final=parent_k_final,
            max_parents=self.max_parents,
            child_k_candidates=child_k_candidates,
            child_k_final=child_k_final,
            final_k=self.k,
            enable_relax=enable_relax,
        )
        parents = recall_result.parents
        children = recall_result.children
        parent_ids = recall_result.parent_ids
        relax_steps = recall_result.relax_steps

        if recall_result.fallback_mode == "parent_as_evidence":
            final_docs = parents[:self.k]
            if trace is not None:
                trace.update({
                    "stage1_parent_hits": len(parents),
                    "stage1_parent_ids": parent_ids,
                    "stage2_hits": 0,
                    "relax_steps": relax_steps,
                    "postfiltered_count": len(final_docs),
                    "fallback_mode": recall_result.fallback_mode,
                })
            return final_docs

        # soft ranking
        if use_plan_soft_constraints:
            soft_constraints = [self._to_retrieval_constraint(c) for c in use_plan_soft_constraints]
        else:
            soft_constraints = build_plan(
                constraint_set, max_relax_steps=self.max_relax_steps,
                range_expand_schedule=self.range_expand_schedule or config.DEFAULT_RANGE_EXPAND_SCHEDULE,
            ).soft_constraints

        filtered, score_map = self._soft_rank(children, soft_constraints)

        # rerank
        if self.rerank_enabled and filtered and self._reranking_service is not None:
            filtered = self._reranking_service.process(query, filtered, top_k=self.k)

        final_docs = self._apply_diversity(filtered, max_per_source=self.max_per_source)[:self.k]

        if trace is not None:
            trace.update({
                "stage1_parent_hits": len(parents), "stage1_parent_ids": parent_ids,
                "stage2_hits": len(children), "relax_steps": relax_steps,
                "postfiltered_count": len(filtered),
                "condition_score_top3": [
                    {"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)}
                    for d in filtered[:3]
                ],
                "final_top_citations": [self._format_citation(d) for d in final_docs[:3]],
                "source_distribution": self._distribution(final_docs, "source_file"),
                "heading_distribution": self._distribution(final_docs, "heading_path"),
            })
            if plan_for_trace:
                trace["plan"] = plan_for_trace.to_dict()
            if relax_steps > 0:
                trace["two_stage_fallback"] = True

        return final_docs

    # ------------------------------------------------------------------
    # single-stage fallback (Chroma / non-Elastic)
    # ------------------------------------------------------------------
    def _retrieve_single_stage(self, query: str, constraint_set: ConstraintSet, trace: Optional[Dict[str, Any]]) -> List[Document]:
        where = self._build_where(constraint_set)
        plan = build_plan(constraint_set, max_relax_steps=self.max_relax_steps, range_expand_schedule=self.range_expand_schedule or config.DEFAULT_RANGE_EXPAND_SCHEDULE)
        docs = self._dense_recall.process(query, k=self.k, where=where)
        if not docs:
            return []
        filtered, score_map = self._soft_rank(docs, plan.soft_constraints)
        if self.rerank_enabled and filtered and self._reranking_service is not None:
            filtered = self._reranking_service.process(query, filtered, top_k=self.k)
        final_docs = self._apply_diversity(filtered)[:self.k]
        if trace is not None:
            trace.update({
                "where": where, "vector_candidates": len(docs), "postfiltered_count": len(filtered),
                "condition_score_top3": [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]],
                "final_top_citations": [self._format_citation(d) for d in final_docs[:3]],
            })
        return final_docs

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _distribution(self, docs: List[Document], key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in docs:
            v = str((d.metadata or {}).get(key) or "unknown")
            out[v] = out.get(v, 0) + 1
        return out

    def _to_retrieval_constraint(self, c: PlanConstraint) -> Constraint:
        ctype = c.op if c.op in {"range", "enum", "set", "text"} else "enum"
        return Constraint(name=c.field, ctype=ctype, value=c.value, confidence=c.confidence, source=c.source, priority=c.priority)

    def _constraintset_from_plan(self, plan: QueryPlan) -> ConstraintSet:
        constraints = [self._to_retrieval_constraint(c) for c in (plan.query.hard_constraints + plan.query.soft_constraints)]
        return ConstraintSet(constraints=constraints, compat_where=self._where_from_plan(plan))

    def _where_from_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        return {c.field: c.value for c in plan.query.hard_constraints}

    def _build_where(self, constraint_set: ConstraintSet) -> Dict[str, Any]:
        where = {c.name: c.value for c in constraint_set.hard_constraints}
        if self.where_full and not constraint_set.constraints:
            for key, value in (constraint_set.compat_where or {}).items():
                if key not in where and value is not None:
                    where[key] = value
        return where

    def _format_citation(self, d: Document) -> str:
        m = d.metadata or {}
        src = m.get("source_file", "unknown")
        heading = m.get("heading_path")
        chunk_id = m.get("chunk_id", "")
        if heading:
            return f"{src} [{heading}] #{chunk_id}"
        return f"{src} #{chunk_id}"

    def _soft_rank(self, docs: List[Document], constraints: List[Constraint]) -> Tuple[List[Document], Dict[str, float]]:
        drop_sections = {s.lower().strip() for s in (self.drop_sections or [])}
        scores: Dict[str, float] = {}
        kept: List[Document] = []

        for idx, d in enumerate(docs):
            meta = d.metadata or {}
            if drop_sections and str(meta.get("section", "unknown")).lower().strip() in drop_sections:
                continue
            if self.drop_reference_like and is_reference_like(d.page_content or ""):
                continue

            score = sum(self._constraint_score(meta, c) for c in constraints if self._constraint_score(meta, c) > 0)
            score += (1.0 / (idx + 1))
            scores[_doc_key(d)] = score
            kept.append(d)

        ranked = sorted(kept, key=lambda d: scores.get(_doc_key(d), 0.0), reverse=True)
        return ranked, scores

    def _constraint_score(self, meta: Dict[str, Any], c: Constraint) -> float:
        weight = max(0.1, c.confidence)
        if c.ctype == "range":
            if _doc_range_overlap(meta, c.value or [],
                                  key_point="T_K" if c.name == "T_range_K" else "P_MPa",
                                  key_min="T_min_K" if c.name == "T_range_K" else "P_min_MPa",
                                  key_max="T_max_K" if c.name == "T_range_K" else "P_max_MPa"):
                return 1.0 * weight
            return 0.0
        if c.ctype == "enum":
            return (1.0 if str(meta.get(c.name, "")).lower() == str(c.value).lower() else 0.0) * weight
        if c.ctype == "set":
            values = c.value or []
            hits = sum(1 for v in values if meta.get(f"has_{v}" if c.name == "targets" else f"gas_{str(v).lower()}"))
            return ((hits / max(len(values), 1)) if hits else 0.0) * weight
        if c.ctype == "text":
            return (0.5 if str(c.value).lower() in str(meta.get(c.name) or "").lower() else 0.0) * weight
        return 0.0

    def _apply_diversity(self, docs: List[Document], max_per_source: Optional[int] = None) -> List[Document]:
        limit = self.max_per_source if max_per_source is None else max_per_source
        if not docs or limit <= 0:
            return docs
        counts: Dict[str, int] = {}
        out: List[Document] = []
        for d in docs:
            src = str((d.metadata or {}).get("source_file", "unknown"))
            if counts.get(src, 0) >= limit:
                continue
            counts[src] = counts.get(src, 0) + 1
            out.append(d)
        return out
