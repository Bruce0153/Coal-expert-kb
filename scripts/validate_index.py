from __future__ import annotations

import argparse
import logging
import sys

from coal_kb.cli_ui import print_banner, print_stats_table
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.logging import setup_logging
from coal_kb.settings import load_config
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.elastic_validation import validate_index

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Elasticsearch index readiness.")
    parser.add_argument("--index", required=True, help="Index name or alias to validate.")
    parser.add_argument("--query", default="validation probe", help="Query text for self-check.")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    print_banner("Coal KB Index Validation", f"index={args.index}")

    elastic_store = ElasticStore(
        host=cfg.elastic.host,
        verify_certs=cfg.elastic.verify_certs,
        timeout_s=cfg.elastic.timeout_s,
    )

    logger.info("Stage: validate_index | index=%s", args.index)
    result = validate_index(
        client=elastic_store.client,
        index_or_alias=args.index,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
        expected_dims=cfg.embeddings.dimensions,
        query_text=args.query,
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
    errors = result.get("errors") or []
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
