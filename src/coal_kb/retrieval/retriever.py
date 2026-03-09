from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings
from coal_kb.query.plan import Constraint as PlanConstraint
from coal_kb.query.plan import QueryPlan
from coal_kb.store.elastic_store import ElasticStore
from .bm25 import bm25_rank, rrf_fuse
from .constraint_policy import build_plan
from .constraints import Constraint, ConstraintSet
from ..chunking.sectioner import is_reference_like

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
        if self.two_stage_enabled and self.elastic_store and self.embeddings_cfg:
            self._embeddings = make_embeddings(self.embeddings_cfg)

    def execute(self, plan: QueryPlan, trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        if not (self.two_stage_enabled and self.elastic_store and self.elastic_index and self._embeddings is not None):
            return self._retrieve_single_stage(plan.query.rewritten or plan.query.normalized, self._constraintset_from_plan(plan), trace)

        query = plan.query.rewritten or plan.query.normalized
        where = self._where_from_plan(plan)
        qvec = self._embeddings.embed_query(query)
        s1 = next((s for s in plan.retrieval_steps if s.level == "parent"), None)
        s2 = next((s for s in plan.retrieval_steps if s.level == "child"), None)
        if s1 is None or s2 is None:
            return self._retrieve_single_stage(query, self._constraintset_from_plan(plan), trace)

        stage1_filters = dict(where)
        if self.tenant_id:
            stage1_filters["tenant_id"] = self.tenant_id
        parents = self.elastic_store.search_parents(
            index=self.elastic_index,
            query_embedding=qvec,
            query_text=query,
            filters=stage1_filters,
            k_candidates=s1.k_candidates,
            k_final=s1.k_final,
            use_icu=self.elastic_use_icu,
            fusion_mode=s1.fusion_mode,
        )
        parent_ids = [str((d.metadata or {}).get("chunk_id")) for d in parents if (d.metadata or {}).get("chunk_id")][: self.max_parents]
        parent_heading = {str((d.metadata or {}).get("chunk_id")): str((d.metadata or {}).get("heading_path") or "") for d in parents}

        child_filters = dict(where)
        if self.tenant_id:
            child_filters["tenant_id"] = self.tenant_id
        if parent_ids:
            child_filters["parent_ids"] = parent_ids

        children = self.elastic_store.search_children(
            index=self.elastic_index,
            query_embedding=qvec,
            query_text=query,
            filters=child_filters,
            k_candidates=s2.k_candidates,
            k_final=max(s2.k_final, self.k),
            use_icu=self.elastic_use_icu,
            fusion_mode=s2.fusion_mode,
        )

        relax_steps = 0
        if not parent_ids or not children:
            fallback_filters = dict(where)
            if self.tenant_id:
                fallback_filters["tenant_id"] = self.tenant_id
            for rule in plan.relax_policy.rules[: plan.relax_policy.max_steps]:
                for f in rule.drop_fields:
                    fallback_filters.pop(f, None)
                relax_steps += 1
            children = self.elastic_store.search_children(
                index=self.elastic_index,
                query_embedding=qvec,
                query_text=query,
                filters=fallback_filters,
                k_candidates=s2.k_candidates,
                k_final=max(s2.k_final, self.k),
                use_icu=self.elastic_use_icu,
                fusion_mode=s2.fusion_mode,
            )
            if trace is not None:
                trace["two_stage_fallback"] = True

        for d in children:
            meta = d.metadata or {}
            pid = str(meta.get("parent_id") or "")
            if pid in parent_heading:
                meta["heading_path"] = parent_heading[pid]
            d.metadata = meta

        filtered, score_map = self._soft_rank(children, [self._to_retrieval_constraint(c) for c in plan.query.soft_constraints])
        if plan.rerank.enabled and filtered and self.reranker is not None:
            candidate_k = min(self.k, len(filtered))
            reranked = self.reranker.rerank(query, filtered[:candidate_k], top_k=candidate_k)
            filtered = reranked + [d for d in filtered if _doc_key(d) not in {_doc_key(x) for x in reranked}]

        final_docs = self._apply_diversity(filtered, max_per_source=plan.diversity.max_per_source)[: self.k]
        if trace is not None:
            trace["plan"] = plan.to_dict()
            trace["stage1_parent_hits"] = len(parents)
            trace["stage1_parent_ids"] = parent_ids
            trace["stage2_hits"] = len(children)
            trace["relax_steps"] = relax_steps
            trace["postfiltered_count"] = len(filtered)
            trace["condition_score_top3"] = [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]]
            trace["final_top_citations"] = [self._format_citation(d) for d in final_docs[:3]]
            trace["source_distribution"] = self._distribution(final_docs, "source_file")
            trace["heading_distribution"] = self._distribution(final_docs, "heading_path")
        return final_docs

    def retrieve(self, query: str, parsed_filter: Union[Dict[str, Any], ConstraintSet], trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        constraint_set = parsed_filter if isinstance(parsed_filter, ConstraintSet) else ConstraintSet(constraints=[], compat_where=parsed_filter)
        if self.two_stage_enabled and self.elastic_store and self.elastic_index and self._embeddings is not None:
            return self._retrieve_two_stage(query, constraint_set, trace)
        return self._retrieve_single_stage(query, constraint_set, trace)

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
        where = {}
        for c in plan.query.hard_constraints:
            where[c.field] = c.value
        for c in plan.query.soft_constraints:
            if c.field in {"stage", "gas_agent", "targets", "T_range_K", "P_range_MPa", "coal_name", "flags"} and c.field not in where:
                where[c.field] = c.value
        return where

    def _retrieve_two_stage(self, query: str, constraint_set: ConstraintSet, trace: Optional[Dict[str, Any]]) -> List[Document]:
        where = self._build_where(constraint_set)
        qvec = self._embeddings.embed_query(query)

        stage1_filters = dict(where)
        if self.tenant_id:
            stage1_filters["tenant_id"] = self.tenant_id
        parents = self.elastic_store.search_parents(index=self.elastic_index, query_embedding=qvec, query_text=query, filters=stage1_filters, k_candidates=self.parent_k_candidates, k_final=self.parent_k_final, use_icu=self.elastic_use_icu)
        parent_ids = [str((d.metadata or {}).get("chunk_id")) for d in parents if (d.metadata or {}).get("chunk_id")][: self.max_parents]
        parent_heading = {str((d.metadata or {}).get("chunk_id")): str((d.metadata or {}).get("heading_path") or "") for d in parents}

        if trace is not None:
            trace["stage1_parent_hits"] = len(parents)
            trace["stage1_parent_ids"] = parent_ids[:5]

        child_filters = dict(where)
        if self.tenant_id:
            child_filters["tenant_id"] = self.tenant_id
        if parent_ids:
            child_filters["parent_ids"] = parent_ids

        children = self.elastic_store.search_children(index=self.elastic_index, query_embedding=qvec, query_text=query, filters=child_filters, k_candidates=self.child_k_candidates, k_final=max(self.child_k_final, self.k), use_icu=self.elastic_use_icu)

        if not parent_ids or not children:
            if trace is not None:
                trace["two_stage_fallback"] = True
            fallback_filters = dict(where)
            if self.tenant_id:
                fallback_filters["tenant_id"] = self.tenant_id
            if self.allow_relax_in_stage2:
                fallback_filters.pop("T_range_K", None)
                fallback_filters.pop("P_range_MPa", None)
            children = self.elastic_store.search_children(index=self.elastic_index, query_embedding=qvec, query_text=query, filters=fallback_filters, k_candidates=self.child_k_candidates, k_final=max(self.child_k_final, self.k), use_icu=self.elastic_use_icu)

        for d in children:
            meta = d.metadata or {}
            pid = str(meta.get("parent_id") or "")
            if pid in parent_heading:
                meta["heading_path"] = parent_heading[pid]
            d.metadata = meta

        filtered, score_map = self._soft_rank(children, build_plan(constraint_set, max_relax_steps=self.max_relax_steps, range_expand_schedule=self.range_expand_schedule or [0.05, 0.1, 0.2]).soft_constraints)
        if self.rerank_enabled and filtered and self.reranker is not None:
            candidate_k = min(self.k, len(filtered))
            reranked = self.reranker.rerank(query, filtered[:candidate_k], top_k=candidate_k)
            filtered = reranked + [d for d in filtered if _doc_key(d) not in {_doc_key(x) for x in reranked}]

        final_docs = self._apply_diversity(filtered)[: self.k]
        if trace is not None:
            trace["postfiltered_count"] = len(filtered)
            trace["condition_score_top3"] = [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]]
            trace["final_top_citations"] = [self._format_citation(d) for d in final_docs[:3]]
        return final_docs

    def _retrieve_single_stage(self, query: str, constraint_set: ConstraintSet, trace: Optional[Dict[str, Any]]) -> List[Document]:
        where = self._build_where(constraint_set)
        plan = build_plan(constraint_set, max_relax_steps=self.max_relax_steps, range_expand_schedule=self.range_expand_schedule or [0.05, 0.1, 0.2])
        vec_retriever = self.vector_retriever_factory(k=self.k, where=where)
        vector_docs = vec_retriever.get_relevant_documents(query) if hasattr(vec_retriever, "get_relevant_documents") else vec_retriever.invoke(query)
        if not vector_docs:
            return []
        fused_docs = rrf_fuse(vector_docs, [d for d, _ in bm25_rank(query, vector_docs)], k=60) if self.use_fuse else vector_docs
        filtered, score_map = self._soft_rank(fused_docs, plan.soft_constraints)
        if self.rerank_enabled and filtered and self.reranker is not None:
            candidate_k = min(self.k, len(filtered))
            reranked = self.reranker.rerank(query, filtered[:candidate_k], top_k=candidate_k)
            filtered = reranked + [d for d in filtered if _doc_key(d) not in {_doc_key(x) for x in reranked}]
        final_docs = self._apply_diversity(filtered)[: self.k]
        if trace is not None:
            trace["where"] = where
            trace["vector_candidates"] = len(vector_docs)
            trace["fused_candidates"] = len(fused_docs)
            trace["postfiltered_count"] = len(filtered)
            trace["condition_score_top3"] = [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]]
            trace["final_top_citations"] = [self._format_citation(d) for d in final_docs[:3]]
        return final_docs

    def _format_citation(self, d: Document) -> str:
        m = d.metadata or {}
        src = m.get("source_file", "unknown")
        heading = m.get("heading_path")
        chunk_id = m.get("chunk_id", "")
        if heading:
            return f"{src} [{heading}] #{chunk_id}"
        return f"{src} #{chunk_id}"

    def _build_where(self, constraint_set: ConstraintSet) -> Dict[str, Any]:
        where = {c.name: c.value for c in constraint_set.hard_constraints}
        if self.where_full and not constraint_set.constraints:
            for key, value in (constraint_set.compat_where or {}).items():
                if key not in where and value is not None:
                    where[key] = value
        return where

    def _soft_rank(self, docs: List[Document], constraints: List[Constraint]) -> Tuple[List[Document], Dict[str, float]]:
        drop_sections = {s.lower().strip() for s in (self.drop_sections or [])}
        scores: Dict[str, float] = {}
        kept_docs: List[Document] = []
        for idx, d in enumerate(docs):
            meta = d.metadata or {}
            if drop_sections and str(meta.get("section", "unknown")).lower().strip() in drop_sections:
                continue
            if self.drop_reference_like and is_reference_like(d.page_content or ""):
                continue
            score = sum(self._constraint_score(meta, c) for c in constraints)
            scores[_doc_key(d)] = score + (1.0 / (idx + 1))
            kept_docs.append(d)
        ranked = sorted(kept_docs, key=lambda d: scores.get(_doc_key(d), 0.0), reverse=True)
        return ranked, scores

    def _constraint_score(self, meta: Dict[str, Any], c: Constraint) -> float:
        weight = max(0.1, c.confidence)
        if c.ctype == "range":
            if _doc_range_overlap(meta, c.value or [], key_point="T_K" if c.name == "T_range_K" else "P_MPa", key_min="T_min_K" if c.name == "T_range_K" else "P_min_MPa", key_max="T_max_K" if c.name == "T_range_K" else "P_max_MPa"):
                return 1.0 * weight
            return -0.3 * weight
        if c.ctype == "enum":
            return (1.0 if str(meta.get(c.name, "")).lower() == str(c.value).lower() else -0.3) * weight
        if c.ctype == "set":
            values = c.value or []
            hits = 0
            for v in values:
                key = f"has_{str(v)}" if c.name == "targets" else f"gas_{str(v).lower()}"
                if meta.get(key):
                    hits += 1
            return ((hits / max(len(values), 1)) if hits else -0.2) * weight
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
