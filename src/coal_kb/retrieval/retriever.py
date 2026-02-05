from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings
from coal_kb.query.plan import QueryPlan
from coal_kb.store.elastic_store import ElasticStore
from .bm25 import bm25_rank, rrf_fuse
from .constraint_policy import build_plan
from .constraints import Constraint, ConstraintSet
from ..chunking.sectioner import is_reference_like

logger = logging.getLogger(__name__)


def _doc_key(d: Document) -> str:
    m = d.metadata or {}
    return str(m.get("chunk_id") or f'{m.get("source_file","")}|{m.get("page","")}')


def _doc_range_overlap(meta: dict, query_range: Optional[List[float]], *, key_point: str, key_min: str, key_max: str) -> bool:
    if query_range is None:
        return True
    qlo, qhi = float(query_range[0]), float(query_range[1])
    if meta.get(key_min) is not None and meta.get(key_max) is not None:
        return max(float(meta[key_min]), qlo) <= min(float(meta[key_max]), qhi)
    x = meta.get(key_point)
    return x is not None and qlo <= float(x) <= qhi


@dataclass
class ExpertRetriever:
    """Retriever executor.

    New API: `execute(plan)`.
    Backward compatibility: `retrieve(query, parsed_filter)` remains as baseline path.
    """

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

    # two-stage elastic
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
        if self.elastic_store and self.embeddings_cfg:
            self._embeddings = make_embeddings(self.embeddings_cfg)

    def execute(self, plan: QueryPlan, trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        if plan.two_stage_enabled and self.elastic_store and self.elastic_index and self._embeddings is not None:
            return self._execute_two_stage(plan, trace)
        return self._execute_baseline(query=plan.query_text, hard_where=plan.stage2.filters, soft_constraints=plan.soft_constraints, trace=trace)

    def retrieve(self, query: str, parsed_filter: Union[Dict[str, Any], ConstraintSet], trace: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Backward compatible entry used by existing tests/legacy scripts."""
        constraint_set = parsed_filter if isinstance(parsed_filter, ConstraintSet) else ConstraintSet(constraints=[], compat_where=parsed_filter)
        hard_where = self._build_where(constraint_set)
        soft = build_plan(
            constraint_set,
            max_relax_steps=self.max_relax_steps,
            range_expand_schedule=self.range_expand_schedule or [0.05, 0.1, 0.2],
        ).soft_constraints
        if self.two_stage_enabled and self.elastic_store and self.elastic_index and self._embeddings is not None:
            from coal_kb.query.plan import QueryPlan, RetrievalStep, RelaxPolicy

            plan = QueryPlan(
                user_query=query,
                query_text=query,
                hard_constraints=constraint_set.hard_constraints,
                soft_constraints=soft,
                stage1=RetrievalStep(name="parents", enabled=True, filters=dict(hard_where), k_candidates=self.parent_k_candidates, k_final=self.parent_k_final),
                stage2=RetrievalStep(name="children", enabled=True, filters=dict(hard_where), k_candidates=self.child_k_candidates, k_final=max(self.child_k_final, self.k)),
                baseline=RetrievalStep(name="baseline_children", enabled=True, filters=dict(hard_where), k_candidates=self.child_k_candidates, k_final=max(self.child_k_final, self.k)),
                two_stage_enabled=True,
                rerank_enabled=self.rerank_enabled,
                rerank_top_n=self.rerank_top_n,
                neighbor_expand_n=1,
                token_budget=2200,
                relax_policy=RelaxPolicy(allow_relax=self.allow_relax_in_stage2, dropped_filters=["T_range_K", "P_range_MPa"] if self.allow_relax_in_stage2 else []),
            )
            return self._execute_two_stage(plan, trace)
        return self._execute_baseline(query=query, hard_where=hard_where, soft_constraints=soft, trace=trace)

    def _execute_two_stage(self, plan: QueryPlan, trace: Optional[Dict[str, Any]]) -> List[Document]:
        qvec = self._embeddings.embed_query(plan.query_text)
        stage1_filters = dict(plan.stage1.filters)
        if self.tenant_id:
            stage1_filters["tenant_id"] = self.tenant_id
        parents = self.elastic_store.search_parents(
            index=self.elastic_index,
            query_embedding=qvec,
            query_text=plan.query_text,
            filters=stage1_filters,
            k_candidates=plan.stage1.k_candidates,
            k_final=plan.stage1.k_final,
            use_icu=self.elastic_use_icu,
        )
        parent_ids = [str((d.metadata or {}).get("chunk_id")) for d in parents if (d.metadata or {}).get("chunk_id")][: self.max_parents]
        parent_heading = {str((d.metadata or {}).get("chunk_id")): str((d.metadata or {}).get("heading_path") or "") for d in parents}

        stage2_filters = dict(plan.stage2.filters)
        if self.tenant_id:
            stage2_filters["tenant_id"] = self.tenant_id
        if parent_ids:
            stage2_filters["parent_ids"] = parent_ids

        children = self.elastic_store.search_children(
            index=self.elastic_index,
            query_embedding=qvec,
            query_text=plan.query_text,
            filters=stage2_filters,
            k_candidates=plan.stage2.k_candidates,
            k_final=max(plan.stage2.k_final, self.k),
            use_icu=self.elastic_use_icu,
        )

        fallback = False
        if not parent_ids or not children:
            fallback = True
            baseline_filters = dict(plan.baseline.filters)
            if self.tenant_id:
                baseline_filters["tenant_id"] = self.tenant_id
            if plan.relax_policy.allow_relax:
                for f in plan.relax_policy.dropped_filters:
                    baseline_filters.pop(f, None)
            children = self.elastic_store.search_children(
                index=self.elastic_index,
                query_embedding=qvec,
                query_text=plan.query_text,
                filters=baseline_filters,
                k_candidates=plan.baseline.k_candidates,
                k_final=max(plan.baseline.k_final, self.k),
                use_icu=self.elastic_use_icu,
            )

        # neighbor expansion
        expanded = self._expand_neighbors(children, n=max(0, plan.neighbor_expand_n))
        for d in expanded:
            meta = d.metadata or {}
            pid = str(meta.get("parent_id") or "")
            if pid in parent_heading:
                meta["heading_path"] = parent_heading[pid]
            d.metadata = meta

        filtered, score_map = self._soft_rank(expanded, plan.soft_constraints)
        final_docs = self._post_rank(plan.query_text, filtered)[: self.k]

        final_docs = self._apply_diversity(filtered)[: self.k]
        if trace is not None:
            trace["stage1_parent_hits"] = len(parents)
            trace["stage1_parent_ids"] = parent_ids[:8]
            trace["stage2_child_hits"] = len(children)
            trace["neighbors_added"] = max(0, len(expanded) - len(children))
            trace["two_stage_fallback"] = fallback
            trace["postfiltered_count"] = len(filtered)
            trace["condition_score_top3"] = [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]]
        return final_docs

    def _expand_neighbors(self, docs: List[Document], n: int) -> List[Document]:
        if n <= 0:
            return docs
        out: Dict[str, Document] = {_doc_key(d): d for d in docs}
        for d in list(docs):
            m = d.metadata or {}
            pid = m.get("parent_id")
            pos = m.get("position_start")
            if pid is None or pos is None:
                continue
            for offset in range(1, n + 1):
                for np in (int(pos) - offset, int(pos) + offset):
                    if np < 0:
                        continue
                    nm = dict(m)
                    nm["neighbor"] = True
                    nm["position_start"] = np
                    nid = f"{m.get('chunk_id')}::nb::{np}"
                    if nid in out:
                        continue
                    out[nid] = Document(page_content=d.page_content, metadata=nm)
        return list(out.values())

    def _execute_baseline(self, *, query: str, hard_where: Dict[str, Any], soft_constraints: List[Constraint], trace: Optional[Dict[str, Any]]) -> List[Document]:
        vec_retriever = self.vector_retriever_factory(k=self.k, where=hard_where)
        vector_docs = vec_retriever.get_relevant_documents(query) if hasattr(vec_retriever, "get_relevant_documents") else vec_retriever.invoke(query)
        if not vector_docs:
            return []
        fused = rrf_fuse(vector_docs, [d for d, _ in bm25_rank(query, vector_docs)], k=60) if self.use_fuse else vector_docs
        filtered, score_map = self._soft_rank(fused, soft_constraints)
        final_docs = self._post_rank(query, filtered)[: self.k]
        if trace is not None:
            trace["where"] = hard_where
            trace["vector_candidates"] = len(vector_docs)
            trace["fused_candidates"] = len(fused)
            trace["postfiltered_count"] = len(filtered)
            trace["condition_score_top3"] = [{"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)} for d in filtered[:3]]
        return final_docs

    def _post_rank(self, query: str, docs: List[Document]) -> List[Document]:
        if self.rerank_enabled and docs and self.reranker is not None:
            top_k = min(self.k, len(docs))
            reranked = self.reranker.rerank(query, docs[:top_k], top_k=top_k)
            keys = {_doc_key(x) for x in reranked}
            docs = reranked + [d for d in docs if _doc_key(d) not in keys]
        return self._apply_diversity(docs)

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
        kept: List[Document] = []
        for i, d in enumerate(docs):
            m = d.metadata or {}
            if drop_sections and str(m.get("section", "unknown")).lower().strip() in drop_sections:
                continue
            if self.drop_reference_like and is_reference_like(d.page_content or ""):
                continue
            score = 0.0
            for c in constraints:
                score += self._constraint_score(m, c)
            scores[_doc_key(d)] = score + (1.0 / (i + 1))
            kept.append(d)
        ranked = sorted(kept, key=lambda d: scores.get(_doc_key(d), 0.0), reverse=True)
        return ranked, scores

    def _constraint_score(self, meta: Dict[str, Any], c: Constraint) -> float:
        weight = max(0.1, c.confidence)
        if c.ctype == "range":
            ok = _doc_range_overlap(
                meta,
                c.value or [],
                key_point="T_K" if c.name == "T_range_K" else "P_MPa",
                key_min="T_min_K" if c.name == "T_range_K" else "P_min_MPa",
                key_max="T_max_K" if c.name == "T_range_K" else "P_max_MPa",
            )
            return (1.0 if ok else -0.3) * weight
        if c.ctype == "enum":
            return (1.0 if str(meta.get(c.name, "")).lower() == str(c.value).lower() else -0.3) * weight
        if c.ctype == "set":
            values = c.value or []
            hits = 0
            for v in values:
                key = f"has_{str(v)}" if c.name == "targets" else f"gas_{str(v).lower()}"
                if meta.get(key):
                    hits += 1
            return ((hits / max(1, len(values))) if hits else -0.2) * weight
        if c.ctype == "text":
            return (0.5 if str(c.value).lower() in str(meta.get(c.name) or "").lower() else 0.0) * weight
        return 0.0

    def _apply_diversity(self, docs: List[Document]) -> List[Document]:
        if not docs or self.max_per_source <= 0:
            return docs
        counts: Dict[str, int] = {}
        out: List[Document] = []
        for d in docs:
            src = str((d.metadata or {}).get("source_file", "unknown"))
            if counts.get(src, 0) >= self.max_per_source:
                continue
            counts[src] = counts.get(src, 0) + 1
            out.append(d)
        return out
