from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .bm25 import bm25_rank, rrf_fuse
from .rerank import CrossEncoderReranker

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
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20

    def retrieve(
        self,
        query: str,
        parsed_filter: Dict[str, Any],
        trace: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        where = self._build_where_minimal(parsed_filter)
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

        # Step 2: BM25 on candidate set + RRF fuse
        logger.info("Retrieval: BM25+RRF fuse | candidates=%d", len(vector_docs))
        bm25_ranked = [d for d, _s in bm25_rank(query, vector_docs)]
        fused_docs = rrf_fuse(vector_docs, bm25_ranked, k=60)
        if trace is not None:
            trace["fused_candidates"] = len(fused_docs)

        # Step 1: OR-first post-filtering
        logger.info("Retrieval: post-filtering | candidates=%d", len(fused_docs))
        filtered = self._post_filter_and_rank(fused_docs, parsed_filter)
        if trace is not None:
            trace["postfiltered_count"] = len(filtered)

        if self.rerank_enabled and filtered:
            logger.info("Retrieval: rerank | top_k=%d model=%s", self.rerank_top_k, self.rerank_model)
            top_k = min(self.rerank_top_k, len(filtered))
            reranker = CrossEncoderReranker(model_name=self.rerank_model)
            reranked = reranker.rerank(query, filtered[:top_k], top_k=top_k)

            reranked_keys = {_doc_key(d) for d in reranked}
            remainder = [d for d in filtered[top_k:] if _doc_key(d) not in reranked_keys]
            filtered = reranked + remainder

        final_docs = filtered[: self.k]
        if trace is not None:
            trace["final_top_citations"] = [self._format_citation(d) for d in final_docs[:3]]
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

    def _build_where_minimal(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep vectorstore filter minimal to avoid backend-specific boolean expressions.
        """
        where: Dict[str, Any] = {}
        stage = f.get("stage")
        if stage and stage != "unknown":
            where["stage"] = str(stage).lower()
        return where

    def _post_filter_and_rank(self, docs: List[Document], f: Dict[str, Any]) -> List[Document]:
        T_range = f.get("T_range_K")
        P_range = f.get("P_range_MPa")

        gas = f.get("gas_agent") or []
        targets = f.get("targets") or []
        coal = f.get("coal_name")

        # OR flags (from flatten_for_filtering)
        gas_flags = [f"gas_{str(g).lower()}" for g in gas] if isinstance(gas, list) else []
        target_flags = [f"has_{t}" for t in targets] if isinstance(targets, list) else []

        kept: List[Tuple[Document, int, int, int]] = []
        # (doc, gas_match_count, target_match_count, coal_match)
        for d in docs:
            m = d.metadata or {}

            # numeric post-filter
            if not _doc_range_overlap(m, T_range, key_point="T_K", key_min="T_min_K", key_max="T_max_K"):
                continue
            if not _doc_range_overlap(m, P_range, key_point="P_MPa", key_min="P_min_MPa", key_max="P_max_MPa"):
                continue

            gas_ok, gas_cnt = _or_match_flags(m, gas_flags)
            if not gas_ok:
                continue

            tgt_ok, tgt_cnt = _or_match_flags(m, target_flags)
            if not tgt_ok:
                continue

            coal_match = 0
            if coal:
                cn = str(m.get("coal_name") or "").lower()
                if coal.lower() in cn and coal.strip():
                    coal_match = 1
                else:
                    # OR priority：coal_name 作为“软条件”，不强行过滤，只影响排序
                    coal_match = 0

            kept.append((d, gas_cnt, tgt_cnt, coal_match))

        if not kept:
            return []

        # Preserve fused order as tiebreaker
        fused_rank: Dict[str, int] = {_doc_key(d): i for i, d in enumerate(docs)}

        # Rank by match strength first (more matches = more “expert-fit”), then by fused order
        kept.sort(
            key=lambda x: (
                -(x[1] + x[2] + x[3]),   # total match count
                -x[3],                   # coal match bonus
                fused_rank.get(_doc_key(x[0]), 10**9),
            )
        )

        return [d for d, *_ in kept]

