from __future__ import annotations

from pathlib import Path

from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.elastic_validation import validate_index
from coal_kb.utils.hash import stable_chunk_id


class IndexManager:
    def __init__(self, store: ElasticStore) -> None:
        self.store = store

    def build_index_name(self, cfg) -> str:
        schema_sig = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))
        return self.store.build_index_name(
            index_prefix=cfg.elastic.index_prefix,
            embedding_version=cfg.model_versions.embedding_version,
            schema_hash=schema_sig[:8],
        )

    def validate(self, cfg, index_name: str, dims: int) -> dict:
        return validate_index(
            client=self.store.client,
            index_or_alias=index_name,
            embeddings_cfg=cfg.embeddings,
            expected_dims=dims,
            query_text="validation probe",
        )
