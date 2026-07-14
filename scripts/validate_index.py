"""验证 Elasticsearch 索引映射、向量维度与基础检索能力。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from coal_kb.indexing import config
from coal_kb.indexing.validation import validate_index
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.interfaces.cli.ui import print_banner, print_stats_table

logger = logging.getLogger(__name__)


@dataclass
class ValidateIndex:
    cfg: AppConfig
    index_name: str
    query: str

    def process(self) -> dict:
        print_banner("Coal KB Index Validation", f"index={self.index_name}")
        elastic_store = ElasticStore(
            host=self.cfg.elastic.host,
            verify_certs=self.cfg.elastic.verify_certs,
            timeout_s=self.cfg.elastic.timeout_s,
        )
        logger.info("Stage: validate_index | index=%s", self.index_name)
        result = validate_index(
            client=elastic_store.client,
            index_or_alias=self.index_name,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            expected_dims=self.cfg.embeddings.dimensions,
            query_text=self.query,
        )
        print_stats_table(
            "Validation Summary",
            [
                ("index", str(result["index_name"])),
                ("doc_count", str(result["doc_count"])),
                ("embedding_dims", str(result["embedding_dims"])),
                ("expected_dims", str(result["expected_dims"])),
                ("ok", str(result["ok"])),
            ],
        )
        for error in result.get("errors") or []:
            logger.error("Validation error: %s", error)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Elasticsearch index readiness.")
    parser.add_argument("--index", required=True, help="Index name or alias to validate.")
    parser.add_argument("--query", default=config.VALIDATION_QUERY, help="Query text for self-check.")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    result = ValidateIndex(cfg=cfg, index_name=args.index, query=args.query).process()
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# 运行命令：python scripts/validate_index.py --index coal_kb_chunks_current
