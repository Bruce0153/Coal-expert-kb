from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ElasticStore:
    host: str
    verify_certs: bool = False

    def __post_init__(self) -> None:
        from elasticsearch import Elasticsearch, helpers

        self._helpers = helpers
        self._client = Elasticsearch(self.host, verify_certs=self.verify_certs)

    @property
    def client(self) -> Any:
        return self._client

    def build_index_name(self, *, index_prefix: str, embedding_version: str, schema_hash: str) -> str:
        stamp = datetime.utcnow().strftime("%Y%m%d%H%M")
        return f"{index_prefix}__emb{embedding_version}__schema{schema_hash}__{stamp}"

    def create_index(self, index_name: str, dims: int) -> None:
        if self._client.indices.exists(index=index_name):
            return
        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source_file": {"type": "keyword"},
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
        self._client.indices.create(index=index_name, body=body)
        logger.info("Created Elasticsearch index: %s", index_name)

    def bulk_upsert_chunks(self, index_name: str, docs: Iterable[Dict[str, Any]]) -> None:
        actions = []
        for doc in docs:
            chunk_id = doc["chunk_id"]
            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": chunk_id,
                    "_source": doc,
                }
            )
        if not actions:
            return
        self._helpers.bulk(self._client, actions)

    def delete_by_document_id(self, index_name_or_alias: str, document_id: str) -> None:
        self._client.delete_by_query(
            index=index_name_or_alias,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed",
        )

    def switch_alias(self, *, alias_current: str, alias_prev: str, new_index: str) -> None:
        actions = []
        current = self.resolve_current_index(alias_current)
        if current:
            actions.append({"remove": {"index": current, "alias": alias_current}})
            actions.append({"add": {"index": current, "alias": alias_prev}})
        actions.append({"add": {"index": new_index, "alias": alias_current}})
        self._client.indices.update_aliases({"actions": actions})
        logger.info("Alias switched | current=%s prev=%s new=%s", alias_current, alias_prev, new_index)

    def rollback(self, *, alias_current: str, alias_prev: str) -> None:
        prev = self.resolve_current_index(alias_prev)
        if not prev:
            raise RuntimeError("No previous alias target to roll back.")
        current = self.resolve_current_index(alias_current)
        actions = []
        if current:
            actions.append({"remove": {"index": current, "alias": alias_current}})
        actions.append({"add": {"index": prev, "alias": alias_current}})
        self._client.indices.update_aliases({"actions": actions})
        logger.info("Alias rollback | current=%s prev=%s", alias_current, alias_prev)

    def resolve_current_index(self, alias: str) -> Optional[str]:
        if not self._client.indices.exists_alias(name=alias):
            return None
        data = self._client.indices.get_alias(name=alias)
        return next(iter(data.keys()), None)
