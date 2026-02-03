from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from .bm25 import bm25_rank, rrf_fuse
from .constraints import Constraint, ConstraintSet
from .constraint_policy import build_plan
from .rerank import CrossEncoderReranker
from ..chunking.sectioner import is_reference_like

logger = logging.getLogger(__name__)


def _doc_range_overlap(meta: dict, query_range: Optional[List[float]], *, key_point: str, key_min: str, key_max: str) -> bool:
    """
    If doc has [min,max], check overlap with query range.
    Else fallback to point value.
    """
    if query_range is None:
        return True

    qlo, qhi = float(query_range[0]), float(query_range[1])

    dmin = meta.get(key_min)
    dmax = meta.get(key_max)
    if dmin is not None and dmax is not None:
        dlo, dhi = float(dmin), float(dmax)
        # overlap: max(lo) <= min(hi)
        return max(dlo, qlo) <= min(dhi, qhi)

    x = meta.get(key_point)
    if x is None:
        return False
    return qlo <= float(x) <= qhi



def _doc_key(d: Document) -> str:
    m = d.metadata or {}
    return str(m.get("chunk_id") or f'{m.get("source_file","")}|{m.get("page","")}|{d.page_content[:60]}')


def _has_flag(meta: Dict[str, Any], flag: str) -> bool:
    return bool(meta.get(flag) is True)


def _or_match_flags(meta: Dict[str, Any], flags: List[str]) -> Tuple[bool, int]:
    """
    OR match with count of matched flags.
    """
    if not flags:
        return True, 0
    c = sum(1 for f in flags if _has_flag(meta, f))
    return (c > 0), c


@dataclass
class ExpertRetriever:
    """
    Retrieval strategy (production-friendly, stable across vectorstores):
    1) Use minimal vectorstore filter: stage only (fast & safe).
    2) Retrieve candidates via vector similarity.
    3) Build BM25 lexical ranking on the candidate set.
    4) Fuse rankings with RRF.
    5) Post-filter:
       - numeric ranges (T/P)
       - OR semantics for gas_agent and targets (priority)
       - optional coal_name contains match (if provided)
    6) Sort by (match strength, fused order), output top-k.
    """

    vector_retriever_factory: Any  # e.g. ChromaStore.as_retriever
    k: int = 6
    k_candidates: int = 40
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 50
    rerank_candidates: int = 50
    rerank_device: str = "auto"
    max_per_source: int = 2
    max_relax_steps: int = 2
    range_expand_schedule: List[float] = None
    mode: str = "balanced"
    drop_sections: Optional[List[str]] = None
    drop_reference_like: bool = True
    use_fuse: bool = True
    where_full: bool = False

    def retrieve(
        self,
        query: str,
        parsed_filter: Union[Dict[str, Any], ConstraintSet],
        trace: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        constraint_set = (
            parsed_filter
            if isinstance(parsed_filter, ConstraintSet)
            else ConstraintSet(constraints=[], compat_where=parsed_filter)
        )
        if self.range_expand_schedule is None:
            self.range_expand_schedule = [0.05, 0.1, 0.2]
        where = self._build_where(constraint_set)
        plan = build_plan(
            constraint_set,
            max_relax_steps=self.max_relax_steps,
            range_expand_schedule=self.range_expand_schedule,
        )
        logger.info("Retrieval: vector search | where=%s k_candidates=%d", where, self.k_candidates)
        vec_retriever = self.vector_retriever_factory(k=self.k_candidates, where=where)

        if hasattr(vec_retriever, "get_relevant_documents"):
            vector_docs: List[Document] = vec_retriever.get_relevant_documents(query)
        else:
            vector_docs = vec_retriever.invoke(query)

        vector_top = [self._format_citation(d) for d in vector_docs[:3]]
        if trace is not None:
            trace["where"] = where
            trace["vector_candidates"] = len(vector_docs)
            trace["vector_top_citations"] = vector_top
        if not vector_docs:
            logger.info("Retrieval: no vector candidates.")
            return []

        if self.use_fuse:
            # Step 2: BM25 on candidate set + RRF fuse
            logger.info("Retrieval: BM25+RRF fuse | candidates=%d", len(vector_docs))
            bm25_ranked = [d for d, _s in bm25_rank(query, vector_docs)]
            fused_docs = rrf_fuse(vector_docs, bm25_ranked, k=60)
        else:
            fused_docs = vector_docs
        if trace is not None:
            trace["fused_candidates"] = len(fused_docs)

        # Step 1: adaptive soft scoring
        logger.info("Retrieval: soft-scoring | candidates=%d", len(fused_docs))
        filtered, score_map = self._soft_rank(fused_docs, plan.soft_constraints)
        if trace is not None:
            trace["postfiltered_count"] = len(filtered)
            trace["applied_hard_filters"] = plan.hard_where
            trace["soft_constraints_used"] = [
                {
                    "name": c.name,
                    "value": c.value,
                    "confidence": c.confidence,
                    "priority": c.priority,
                }
                for c in plan.soft_constraints
            ]
            trace["condition_score_top3"] = [
                {"chunk_id": (d.metadata or {}).get("chunk_id"), "score": score_map.get(_doc_key(d), 0.0)}
                for d in filtered[:3]
            ]

        if self.rerank_enabled and filtered:
            logger.info("Retrieval: rerank | top_n=%d model=%s", self.rerank_top_n, self.rerank_model)
            candidate_k = min(self.rerank_candidates, len(filtered))
            reranker = CrossEncoderReranker(model_name=self.rerank_model, device=self.rerank_device)
            reranked = reranker.rerank(query, filtered[:candidate_k], top_k=candidate_k)
            top_n = min(self.rerank_top_n, len(reranked))
            reranked = reranked[:top_n]

            reranked_keys = {_doc_key(d) for d in reranked}
            remainder = [d for d in filtered if _doc_key(d) not in reranked_keys]
            filtered = reranked + remainder

        filtered = self._apply_diversity(filtered)

        final_docs = filtered[: self.k]
        if trace is not None:
            trace["final_top_citations"] = [self._format_citation(d) for d in final_docs[:3]]
            trace["relax_steps_taken"] = plan.relax_steps
            trace["diversity@k"] = len({(d.metadata or {}).get("source_file") for d in final_docs})
        logger.info("Retrieval: done | final=%d", len(final_docs))
        return final_docs

    def _format_citation(self, d: Document) -> str:
        m = d.metadata or {}
        src = m.get("source_file", "unknown")
        page_label = m.get("page_label")
        page = m.get("page")
        chunk_id = m.get("chunk_id", "")
        if page_label:
            return f"{src} ({page_label}) #{chunk_id}"
        if isinstance(page, int):
            return f"{src} (page {page + 1}) #{chunk_id}"
        return f"{src} #{chunk_id}"

    def _build_where(self, constraint_set: ConstraintSet) -> Dict[str, Any]:
        """
        Keep vectorstore filter minimal: only hard constraints.
        """
        where: Dict[str, Any] = {}
        for constraint in constraint_set.hard_constraints:
            where[constraint.name] = constraint.value
        if self.where_full and not constraint_set.constraints:
            for key, value in (constraint_set.compat_where or {}).items():
                if key not in where and value is not None:
                    where[key] = value
        return where

    def _soft_rank(
        self, docs: List[Document], constraints: List[Constraint]
    ) -> Tuple[List[Document], Dict[str, float]]:
        if not docs:
            return [], {}
        drop_sections = {s.lower() for s in (self.drop_sections or [])}
        scores: Dict[str, float] = {}
        for idx, d in enumerate(docs):
            m = d.metadata or {}
            section = str(m.get("section", "unknown")).lower()
            if drop_sections and section in drop_sections:
                scores[_doc_key(d)] = -10.0
                continue
            if self.drop_reference_like and is_reference_like(d.page_content or ""):
                scores[_doc_key(d)] = -10.0
                continue
            score = 0.0
            for c in constraints:
                score += self._constraint_score(m, c)
            scores[_doc_key(d)] = score + (1.0 / (idx + 1))
        ranked = sorted(docs, key=lambda d: scores.get(_doc_key(d), 0.0), reverse=True)
        return ranked, scores

    def _constraint_score(self, meta: Dict[str, Any], c: Constraint) -> float:
        weight = max(0.1, c.confidence)
        if c.ctype == "range":
            value = c.value or []
            if not value:
                return 0.0
            if _doc_range_overlap(meta, value, key_point="T_K" if c.name == "T_range_K" else "P_MPa",
                                  key_min="T_min_K" if c.name == "T_range_K" else "P_min_MPa",
                                  key_max="T_max_K" if c.name == "T_range_K" else "P_max_MPa"):
                return 1.0 * weight
            if meta.get("T_K") is None and meta.get("P_MPa") is None:
                return -0.1 * weight
            return -0.5 * weight
        if c.ctype == "enum":
            val = str(meta.get(c.name, "")).lower()
            return (1.0 if val == str(c.value).lower() else -0.3) * weight
        if c.ctype == "set":
            values = c.value or []
            if not values:
                return 0.0
            hits = 0
            for v in values:
                if c.name == "targets":
                    key = f"has_{str(v)}"
                else:
                    key = f"gas_{str(v).lower()}"
                if meta.get(key):
                    hits += 1
            if not hits:
                return -0.2 * weight
            return (hits / max(len(values), 1)) * weight
        if c.ctype == "text":
            text = str(meta.get(c.name) or "").lower()
            return (0.5 if str(c.value).lower() in text else 0.0) * weight
        return 0.0

    def _apply_diversity(self, docs: List[Document]) -> List[Document]:
        if not docs or self.max_per_source <= 0:
            return docs
        counts: Dict[str, int] = {}
        output: List[Document] = []
        for d in docs:
            src = str((d.metadata or {}).get("source_file", "unknown"))
            counts.setdefault(src, 0)
            if counts[src] >= self.max_per_source:
                continue
            counts[src] += 1
            output.append(d)
        return output
