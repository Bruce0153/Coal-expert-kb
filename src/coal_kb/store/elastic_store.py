from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ElasticStore:
    host: str
    verify_certs: bool = False
    timeout_s: int = 60

    def __post_init__(self) -> None:
        from elasticsearch import Elasticsearch, helpers

        self._helpers = helpers
        self._client = Elasticsearch(
            self.host,
            verify_certs=self.verify_certs,
            request_timeout=self.timeout_s,
        )

    @property
    def client(self) -> Any:
        return self._client

    def build_index_name(self, *, index_prefix: str, embedding_version: str, schema_hash: str) -> str:
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"{index_prefix}__emb{embedding_version}__schema{schema_hash}__{stamp}"

    def create_index(self, index_name: str, dims: int, *, enable_icu_analyzer: bool = False) -> None:
        if self._client.indices.exists(index=index_name):
            return
        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic": True,
                "properties": {
                    "text": {"type": "text"},
                    "heading_path_text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    "parent_id": {"type": "keyword"},
                    "heading_path": {"type": "keyword"},
                    "chunk_level": {"type": "short"},
                    "position_start": {"type": "integer"},
                    "position_end": {"type": "integer"},
                    "page": {"type": "integer"},
                    "page_label": {"type": "keyword"},
                    "section": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "stage": {"type": "keyword"},
                    "gas_agent": {"type": "keyword"},
                    "targets": {"type": "keyword"},
                    "T_K": {"type": "float"},
                    "T_min_K": {"type": "float"},
                    "T_max_K": {"type": "float"},
                    "P_MPa": {"type": "float"},
                    "P_min_MPa": {"type": "float"},
                    "P_max_MPa": {"type": "float"},
                    "coal_name": {"type": "keyword"},
                    "metadata_json": {"type": "text"},
                    "tenant_id": {"type": "keyword"},
                }
            },
        }
        if enable_icu_analyzer:
            body["mappings"]["properties"]["text"]["fields"] = {
                "icu": {"type": "text", "analyzer": "icu_analyzer"}
            }
            body["settings"]["analysis"] = {
                "analyzer": {"icu_analyzer": {"type": "icu_analyzer"}}
            }
            self._ensure_icu_plugin()
        self._client.indices.create(index=index_name, body=body)
        logger.info("Created Elasticsearch index: %s", index_name)

    def _ensure_icu_plugin(self) -> None:
        plugins = self._client.cat.plugins(format="json")
        has_icu = any("analysis-icu" in (p.get("component") or "") for p in plugins)
        if not has_icu:
            raise RuntimeError("ICU analyzer requested but analysis-icu is not installed.")

    def bulk_upsert_chunks(self, index_name: str, docs: Iterable[Dict[str, Any]]) -> None:
        actions = []
        for doc in docs:
            chunk_id = doc["chunk_id"]
            actions.append({"_op_type": "index", "_index": index_name, "_id": chunk_id, "_source": doc})
        if actions:
            self._helpers.bulk(self._client, actions)

    def delete_by_document_id(self, index_name_or_alias: str, document_id: str) -> None:
        self._client.delete_by_query(
            index=index_name_or_alias,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed",
        )

    def _build_filters(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        filters = dict(filters or {})
        clauses: List[Dict[str, Any]] = []
        for key in ("stage", "document_id", "is_parent", "parent_id", "chunk_level", "tenant_id"):
            if key in filters and filters[key] is not None:
                clauses.append({"term": {key: filters.pop(key)}})
        for key in ("gas_agent", "targets"):
            if key in filters and filters[key]:
                vals = filters.pop(key)
                if not isinstance(vals, list):
                    vals = [vals]
                clauses.append({"terms": {key: vals}})
        parent_ids = filters.pop("parent_ids", None)
        if parent_ids:
            clauses.append({"terms": {"parent_id": parent_ids}})

        if filters.get("T_range_K"):
            qlo, qhi = filters.pop("T_range_K")
            clauses.append({"bool": {"should": [{"range": {"T_K": {"gte": qlo, "lte": qhi}}}, {"bool": {"must": [{"range": {"T_min_K": {"lte": qhi}}}, {"range": {"T_max_K": {"gte": qlo}}}]}}], "minimum_should_match": 1}})
        if filters.get("P_range_MPa"):
            qlo, qhi = filters.pop("P_range_MPa")
            clauses.append({"bool": {"should": [{"range": {"P_MPa": {"gte": qlo, "lte": qhi}}}, {"bool": {"must": [{"range": {"P_min_MPa": {"lte": qhi}}}, {"range": {"P_max_MPa": {"gte": qlo}}}]}}], "minimum_should_match": 1}})

        for k, v in filters.items():
            if v is not None:
                clauses.append({"term": {k: v}})
        return clauses

    def _hit_to_doc(self, hit: Dict[str, Any]) -> Document:
        src = hit.get("_source", {})
        text = src.get("text", "")
        meta = {k: v for k, v in src.items() if k != "embedding" and k != "text"}
        return Document(page_content=text, metadata=meta)

    def _search_hybrid(
        self,
        *,
        index: str,
        query_text: str,
        query_embedding: List[float],
        filters: Dict[str, Any],
        k_candidates: int,
        k_final: int,
        use_icu: bool = False,
        heading_boost: bool = False,
        fusion_mode: str = "rrf",
    ) -> List[Document]:
        text_field = "text.icu" if use_icu else "text"
        filter_clauses = self._build_filters(filters)
        must = [{"match": {text_field: {"query": query_text}}}]
        if heading_boost:
            must.append({"match": {"heading_path_text": {"query": query_text}}})
        bm25 = self._client.search(
            index=index,
            body={"size": k_candidates, "query": {"bool": {"filter": filter_clauses, "must": must[:1]}}},
        ).get("hits", {}).get("hits", [])
        knn = self._client.search(
            index=index,
            body={
                "size": k_candidates,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": k_candidates,
                    "num_candidates": max(k_candidates * 4, 20),
                    "filter": filter_clauses,
                },
            },
        ).get("hits", {}).get("hits", [])

        score_map: Dict[str, float] = {}
        if fusion_mode in {"rrf", "bm25"}:
            for r, hit in enumerate(bm25, 1):
                score_map[hit["_id"]] = score_map.get(hit["_id"], 0.0) + 1.0 / (60 + r)
        if fusion_mode in {"rrf", "vector"}:
            for r, hit in enumerate(knn, 1):
                score_map[hit["_id"]] = score_map.get(hit["_id"], 0.0) + 1.0 / (60 + r)

        hit_map = {h["_id"]: h for h in bm25 + knn}
        ranked_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)[:k_final]
        return [self._hit_to_doc(hit_map[i]) for i in ranked_ids if i in hit_map]

    def search_parents(self, *, index: str, query_embedding: List[float], query_text: str, filters: Dict[str, Any], k_candidates: int, k_final: int, use_icu: bool = False, fusion_mode: str = "rrf") -> List[Document]:
        f = dict(filters)
        f["is_parent"] = True
        f["chunk_level"] = 0
        return self._search_hybrid(index=index, query_text=query_text, query_embedding=query_embedding, filters=f, k_candidates=k_candidates, k_final=k_final, use_icu=use_icu, heading_boost=True, fusion_mode=fusion_mode)

    def search_children(self, *, index: str, query_embedding: List[float], query_text: str, filters: Dict[str, Any], k_candidates: int, k_final: int, use_icu: bool = False, fusion_mode: str = "rrf") -> List[Document]:
        f = dict(filters)
        f["is_parent"] = False
        f["chunk_level"] = 1
        return self._search_hybrid(index=index, query_text=query_text, query_embedding=query_embedding, filters=f, k_candidates=k_candidates, k_final=k_final, use_icu=use_icu, fusion_mode=fusion_mode)

    def get_parents_by_ids(self, *, index: str, parent_ids: List[str]) -> Dict[str, Document]:
        if not parent_ids:
            return {}
        rsp = self._client.search(
            index=index,
            body={"size": len(parent_ids), "query": {"bool": {"filter": [{"terms": {"chunk_id": parent_ids}}, {"term": {"is_parent": True}}]}}},
        )
        hits = rsp.get("hits", {}).get("hits", [])
        return {h["_source"].get("chunk_id"): self._hit_to_doc(h) for h in hits}

    def switch_alias(self, *, alias_current: str, alias_prev: str, new_index: str) -> None:
        actions = []
        current = self.resolve_current_index(alias_current)
        if current:
            actions.append({"remove": {"index": current, "alias": alias_current}})
            actions.append({"add": {"index": current, "alias": alias_prev}})
        actions.append({"add": {"index": new_index, "alias": alias_current}})
        self._client.indices.update_aliases(body={"actions": actions})

    def rollback(self, *, alias_current: str, alias_prev: str) -> None:
        prev = self.resolve_current_index(alias_prev)
        if not prev:
            raise RuntimeError("No previous alias target to roll back.")
        current = self.resolve_current_index(alias_current)
        actions = []
        if current:
            actions.append({"remove": {"index": current, "alias": alias_current}})
        actions.append({"add": {"index": prev, "alias": alias_current}})
        self._client.indices.update_aliases(body={"actions": actions})

    def resolve_current_index(self, alias: str) -> Optional[str]:
        if self._client.indices.exists(index=alias):
            return alias
        if self._client.indices.exists_alias(name=alias):
            data = self._client.indices.get_alias(name=alias)
            return next(iter(data.keys()))
        return None
