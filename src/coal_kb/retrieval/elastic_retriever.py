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

    def __post_init__(self) -> None:
        self._embeddings = make_embeddings(self.embeddings_cfg)

    def invoke(self, query: str) -> List[Document]:
        query_vec = self._embeddings.embed_query(query)
        filters = []
        if self.tenant_id:
            filters.append({"term": {"tenant_id": self.tenant_id}})

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
        )
        if where:
            retriever = _FilteredElasticRetriever(retriever=retriever, where=where)
        return retriever

    return factory


@dataclass
class _FilteredElasticRetriever:
    retriever: ElasticRetriever
    where: Dict[str, Any]

    def invoke(self, query: str) -> List[Document]:
        docs = self.retriever.invoke(query)
        stage = (self.where or {}).get("stage")
        if stage:
            stage = str(stage).lower()
            docs = [d for d in docs if (d.metadata or {}).get("stage") == stage]
        return docs
