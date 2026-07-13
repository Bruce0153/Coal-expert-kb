"""实现 Elasticsearch 索引、检索与别名管理适配器。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ElasticStore:
    host: str
    verify_certs: bool = False
    timeout_s: int = 60

    def _normalize_doc_for_indexing(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a chunk document before indexing into Elasticsearch.

        Goal:
        - Keep original fields (text/source_file/page/embedding...)
        - Parse metadata_json (if present) and *promote* important keys to top-level,
          so they can be filtered and searched efficiently (stage/gas_agent/has_NH3/...).
        - Fix gas_agent: if stored as a JSON-string like "[\"co2\",\"o2\"]", convert to list.
        - Ensure chunk_id exists and is consistent.
        - Rebuild metadata_json to a valid JSON string (for debugging / export), using normalized values.
        """
        out = dict(doc)

        # 1) Parse metadata_json if it's a string
        meta: Dict[str, Any] = {}
        raw_mj = out.get("metadata_json")

        if isinstance(raw_mj, str) and raw_mj.strip():
            try:
                meta = json.loads(raw_mj)
            except Exception:
                # Some metadata_json may be malformed; do not crash ingestion.
                meta = {}
        elif isinstance(raw_mj, dict):
            meta = dict(raw_mj)

        # 2) Promote selected keys from meta to top-level (only if not already present)
        # Add keys you rely on for filtering / retrieval.
        promote_keys = [
            "chunk_id",
            "document_id",
            "source_file",
            "page",
            "page_label",
            "section",
            "stage",
            "coal_name",
            "tenant_id",
            "parent_id",
            "heading_path",
            "chunk_level",
            "chunk_index",
            "position_start",
            "position_end",
            "T_K",
            "T_min_K",
            "T_max_K",
            "P_MPa",
            "P_min_MPa",
            "P_max_MPa",
            "targets",
            "gas_agent",
            "has_NH3", "has_HCN", "has_H2S", "has_SO2", "has_NOx", "has_COS",
            "gas_o2", "gas_co2", "gas_steam", "gas_n2", "gas_air",
        ]

        for k in promote_keys:
            if k in meta and out.get(k) is None:
                out[k] = meta.get(k)

        # 3) Normalize gas_agent into a real list (preferred for ES terms filter)
        gas_agent = out.get("gas_agent", None)
        if isinstance(gas_agent, str):
            s = gas_agent.strip()
            # cases:
            # - '["co2","o2"]' (json list as string)
            # - "co2,o2" or "co2; o2" (csv-like)
            if (s.startswith("[") and s.endswith("]")) or (s.startswith('"[') and s.endswith(']"')):
                try:
                    out["gas_agent"] = json.loads(s.strip('"'))
                except Exception:
                    # fallback: extract tokens
                    out["gas_agent"] = [x for x in re.split(r"[,\s;]+", s.strip("[]\" ")) if x]
            else:
                out["gas_agent"] = [x for x in re.split(r"[,\s;]+", s) if x]
        elif gas_agent is None:
            # keep as None (no field)
            pass
        else:
            # list already ok
            pass

        # 4) Normalize targets into list if it's a scalar
        targets = out.get("targets", None)
        if isinstance(targets, str):
            out["targets"] = [targets]
        elif targets is None:
            pass

        # 5) Ensure is_parent exists (needed by two-stage retrieval)
        # If upstream didn't send it, infer from chunk_level when possible.
        if out.get("is_parent") is None:
            cl = out.get("chunk_level", None)
            if cl is not None:
                out["is_parent"] = bool(int(cl) == 0)
            else:
                # fallback: if it has parent_id, treat as child; else treat as child by default
                out["is_parent"] = False

        # 6) Ensure chunk_level exists (keep consistent with is_parent)
        if out.get("chunk_level") is None:
            out["chunk_level"] = 0 if out.get("is_parent") else 1

        # 7) Ensure chunk_id exists (use meta chunk_id if present)
        if out.get("chunk_id") is None and isinstance(meta, dict) and meta.get("chunk_id"):
            out["chunk_id"] = meta.get("chunk_id")

        # 8) Rebuild metadata_json using normalized fields (especially gas_agent list)
        # Keep all meta keys, but override with normalized versions for important ones.
        if isinstance(meta, dict):
            meta2 = dict(meta)
        else:
            meta2 = {}

        # Always keep these consistent
        for k in [
            "chunk_id",
            "document_id",
            "source_file",
            "page",
            "page_label",
            "section",
            "stage",
            "coal_name",
            "tenant_id",
            "parent_id",
            "heading_path",
            "chunk_level",
            "chunk_index",
            "position_start",
            "position_end",
            "T_K",
            "T_min_K",
            "T_max_K",
            "P_MPa",
            "P_min_MPa",
            "P_max_MPa",
            "targets",
            "gas_agent",
            "is_parent",
        ]:
            if out.get(k) is not None:
                meta2[k] = out.get(k)

        out["metadata_json"] = json.dumps(meta2, ensure_ascii=False)

        return out


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
            norm = self._normalize_doc_for_indexing(doc)

            chunk_id = norm.get("chunk_id")
            if not chunk_id:
                raise ValueError("Missing chunk_id for ES indexing (after normalization).")

            actions.append(
                {"_op_type": "index", "_index": index_name, "_id": chunk_id, "_source": norm}
            )
        if actions:
            success, errors = self._helpers.bulk(
                self._client,
                actions,
                raise_on_error=False,
                stats_only=False,
            )

            if errors:
                import json
                print("bulk errors sample:")
                for e in errors[:5]:
                    print(json.dumps(e, ensure_ascii=False, indent=2))
                raise RuntimeError(f"{len(errors)} docs failed in bulk indexing")

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

    def make_retriever_factory(
        self,
        *,
        index: str,
        embeddings_cfg: Any = None,
        candidates: int = 50,
        rrf_k: int = 60,
        use_icu: bool = False,
        tenant_id: Optional[str] = None,
    ) -> Callable:
        """Return a factory compatible with ExpertRetriever's vector_retriever_factory."""
        from coal_kb.infra.providers.embeddings import make_embeddings

        _embeddings = make_embeddings(embeddings_cfg) if embeddings_cfg else None

        def factory(k: int, where: Optional[Dict[str, Any]] = None):
            class _Retriever:
                def invoke(_self, query: str) -> List[Document]:
                    query_vec = _embeddings.embed_query(query) if _embeddings else None
                    return self._search_hybrid(
                        index=index, query_text=query, query_embedding=query_vec,
                        filters=where or {}, k_candidates=max(candidates, k), k_final=k,
                        use_icu=use_icu,
                    )
            return _Retriever()
        return factory
