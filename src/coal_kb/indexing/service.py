"""编排 Elasticsearch 索引构建、验证、别名切换和回滚。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coal_kb.indexing import config
from coal_kb.indexing.validation import validate_index
from coal_kb.infra.config import AppConfig
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig, make_embeddings
from coal_kb.ingestion.pipeline import IngestPipeline
from coal_kb.utils.hash import stable_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class IndexService:
    """保存索引构建所需配置，并提供统一的 process() 编排入口。"""

    cfg: AppConfig

    def _resolve_dims(self) -> int:
        dimensions = self.cfg.embeddings.dimensions or 0
        if dimensions:
            return int(dimensions)
        embeddings = make_embeddings(EmbeddingsConfig(**self.cfg.embeddings.model_dump()))
        return len(embeddings.embed_query("dimension probe"))

    def _build_store(self) -> ElasticStore:
        return ElasticStore(
            host=self.cfg.elastic.host,
            verify_certs=self.cfg.elastic.verify_certs,
            timeout_s=self.cfg.elastic.timeout_s,
        )

    def _build(
        self,
        elastic_store: ElasticStore,
        *,
        embedding_version: str | None,
        resume_index: str | None,
    ) -> dict[str, Any]:
        if embedding_version:
            self.cfg.model_versions.embedding_version = embedding_version
        self.cfg.backend = "elastic"

        dimensions = self._resolve_dims()
        schema_signature = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))
        schema_hash = schema_signature[:8]

        if resume_index:
            index_name = resume_index
            if not elastic_store.client.indices.exists(index=index_name):
                raise SystemExit(f"Resume index not found: {index_name}")
            logger.info("Resuming existing index: %s", index_name)
            rebuild = False
        else:
            index_name = elastic_store.build_index_name(
                index_prefix=self.cfg.elastic.index_prefix,
                embedding_version=self.cfg.model_versions.embedding_version,
                schema_hash=schema_hash,
            )
            elastic_store.create_index(
                index_name,
                dimensions,
                enable_icu_analyzer=self.cfg.elastic.enable_icu_analyzer,
            )
            rebuild = True

        stats = IngestPipeline(cfg=self.cfg).process(
            rebuild=rebuild,
            elastic_index_override=index_name,
        )
        validation = validate_index(
            client=elastic_store.client,
            index_or_alias=index_name,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            expected_dims=dimensions,
            query_text=config.VALIDATION_QUERY,
        )
        if not validation["ok"]:
            logger.error("Index validation failed. Alias not switched.")
            for error in validation["errors"]:
                logger.error("Validation error: %s", error)
            raise SystemExit(1)

        elastic_store.switch_alias(
            alias_current=self.cfg.elastic.alias_current,
            alias_prev=self.cfg.elastic.alias_prev,
            new_index=index_name,
        )
        logger.info("Index build complete: %s", index_name)
        return {"index": index_name, "stats": stats, "validation": validation}

    def _switch(self, elastic_store: ElasticStore, *, index_name: str) -> dict[str, str]:
        elastic_store.switch_alias(
            alias_current=self.cfg.elastic.alias_current,
            alias_prev=self.cfg.elastic.alias_prev,
            new_index=index_name,
        )
        logger.info("Switched alias to %s", index_name)
        return {"alias_current": self.cfg.elastic.alias_current, "new_index": index_name}

    def _rollback(self, elastic_store: ElasticStore) -> dict[str, str]:
        elastic_store.rollback(
            alias_current=self.cfg.elastic.alias_current,
            alias_prev=self.cfg.elastic.alias_prev,
        )
        logger.info("Rollback complete.")
        return {
            "alias_current": self.cfg.elastic.alias_current,
            "alias_prev": self.cfg.elastic.alias_prev,
        }

    def process(
        self,
        command: str,
        *,
        embedding_version: str | None = None,
        resume_index: str | None = None,
        index_name: str | None = None,
    ) -> dict[str, Any]:
        elastic_store = self._build_store()
        if command == "build":
            return self._build(
                elastic_store,
                embedding_version=embedding_version,
                resume_index=resume_index,
            )
        if command == "switch":
            assert index_name is not None
            return self._switch(elastic_store, index_name=index_name)
        if command == "rollback":
            return self._rollback(elastic_store)
        raise ValueError(f"Unsupported command: {command}")
