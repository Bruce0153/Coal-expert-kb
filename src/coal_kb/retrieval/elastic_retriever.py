from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings
from coal_kb.retrieval.bm25 import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class ElasticRetriever:
    client: Any
    index: str
    embeddings_cfg: EmbeddingsConfig
    k: int = 6
    candidates: int = 50
    rrf_k: int = 60
    tenant_id: Optional[str] = None
    where: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._embeddings = make_embeddings(self.embeddings_cfg)

    def invoke(self, query: str) -> List[Document]:
        query_vec = self._embeddings.embed_query(query)
        filters = self._build_filters()
        top_n = max(self.candidates, self.k)

        bm25_body = {
            "size": top_n,
            "query": {
                "bool": {
                    "filter": filters,
                    "must": [{"match": {"text": {"query": query}}}],
                }
            },
        }
        bm25_rsp = self.client.search(index=self.index, body=bm25_body)
        bm25_hits = bm25_rsp.get("hits", {}).get("hits", [])
        bm25_docs = [self._hit_to_doc(hit) for hit in bm25_hits]

        knn_body = {
            "size": top_n,
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": top_n,
                "num_candidates": max(top_n * 4, 20),
                "filter": filters,
            },
        }
        knn_rsp = self.client.search(index=self.index, body=knn_body)
        knn_hits = knn_rsp.get("hits", {}).get("hits", [])
        knn_docs = [self._hit_to_doc(hit) for hit in knn_hits]

        if not bm25_docs and not knn_docs:
            return []

        fused = rrf_fuse(bm25_docs, knn_docs, k=self.rrf_k)
        return fused[: self.k]

    def _build_filters(self) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        if self.tenant_id:
            filters.append({"term": {"tenant_id": self.tenant_id}})
        where = self.where or {}
        if where.get("stage"):
            filters.append({"term": {"stage": where["stage"]}})
        if where.get("document_id"):
            filters.append({"term": {"document_id": where["document_id"]}})
        if where.get("gas_agent"):
            filters.append({"terms": {"gas_agent": where["gas_agent"]}})
        if where.get("targets"):
            filters.append({"terms": {"targets": where["targets"]}})
        if where.get("T_range_K"):
            qlo, qhi = where["T_range_K"]
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"range": {"T_K": {"gte": qlo, "lte": qhi}}},
                            {
                                "bool": {
                                    "must": [
                                        {"range": {"T_min_K": {"lte": qhi}}},
                                        {"range": {"T_max_K": {"gte": qlo}}},
                                    ]
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if where.get("P_range_MPa"):
            qlo, qhi = where["P_range_MPa"]
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"range": {"P_MPa": {"gte": qlo, "lte": qhi}}},
                            {
                                "bool": {
                                    "must": [
                                        {"range": {"P_min_MPa": {"lte": qhi}}},
                                        {"range": {"P_max_MPa": {"gte": qlo}}},
                                    ]
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        return filters

    def _hit_to_doc(self, hit: Dict[str, Any]) -> Document:
        src = hit.get("_source", {})
        text = src.get("text", "")
        meta = {
            "chunk_id": src.get("chunk_id"),
            "document_id": src.get("document_id"),
            "source_file": src.get("source_file"),
            "page": src.get("page"),
            "page_label": src.get("page_label"),
            "section": src.get("section"),
            "stage": src.get("stage"),
            "gas_agent": src.get("gas_agent"),
            "targets": src.get("targets"),
            "T_K": src.get("T_K"),
            "T_min_K": src.get("T_min_K"),
            "T_max_K": src.get("T_max_K"),
            "P_MPa": src.get("P_MPa"),
            "P_min_MPa": src.get("P_min_MPa"),
            "P_max_MPa": src.get("P_max_MPa"),
            "coal_name": src.get("coal_name"),
        }
        for key, value in src.items():
            if key.startswith("gas_") or key.startswith("has_"):
                meta[key] = value
        return Document(page_content=text, metadata=meta)


def make_elastic_retriever_factory(
    *,
    client: Any,
    index: str,
    embeddings_cfg: EmbeddingsConfig,
    candidates: int = 50,
    rrf_k: int = 60,
    tenant_id: Optional[str] = None,
):
    def factory(k: int, where: Optional[Dict[str, Any]] = None):
        retriever = ElasticRetriever(
            client=client,
            index=index,
            embeddings_cfg=embeddings_cfg,
            k=k,
            candidates=candidates,
            rrf_k=rrf_k,
            tenant_id=tenant_id,
            where=where,
        )
        return retriever

    return factory
