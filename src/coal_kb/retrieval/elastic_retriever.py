from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings

logger = logging.getLogger(__name__)


@dataclass
class ElasticRetriever:
    client: Any
    index: str
    embeddings_cfg: EmbeddingsConfig
    k: int = 6
    tenant_id: Optional[str] = None
    where: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self._embeddings = make_embeddings(self.embeddings_cfg)

    def invoke(self, query: str) -> List[Document]:
        query_vec = self._embeddings.embed_query(query)
        filters = self._build_filters()

        body = {
            "size": self.k,
            "query": {
                "bool": {
                    "filter": filters,
                    "should": [{"match": {"text": query}}],
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": self.k,
                "num_candidates": max(self.k * 4, 20),
            },
        }
        rsp = self.client.search(index=self.index, body=body)
        hits = rsp.get("hits", {}).get("hits", [])
        return [self._hit_to_doc(hit) for hit in hits]

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
    tenant_id: Optional[str] = None,
):
    def factory(k: int, where: Optional[Dict[str, Any]] = None):
        retriever = ElasticRetriever(
            client=client,
            index=index,
            embeddings_cfg=embeddings_cfg,
            k=k,
            tenant_id=tenant_id,
            where=where,
        )
        return retriever

    return factory
