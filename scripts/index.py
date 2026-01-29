from __future__ import annotations

import argparse
import logging
from pathlib import Path

from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings
from coal_kb.cli_ui import print_banner, print_kv, print_stats_table, progress_status
from coal_kb.logging import setup_logging
from coal_kb.pipelines.ingest_pipeline import IngestPipeline
from coal_kb.settings import load_config
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.utils.hash import stable_chunk_id

logger = logging.getLogger(__name__)


def _resolve_dims(cfg) -> int:
    dims = cfg.embeddings.dimensions or 0
    if dims:
        return dims
    embeddings = make_embeddings(EmbeddingsConfig(**cfg.embeddings.model_dump()))
    return len(embeddings.embed_query("dimension probe"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Elasticsearch index versions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Create new index and ingest with elastic backend.")
    build.add_argument("--embedding-version", default=None, help="Override embedding version.")

    switch = sub.add_parser("switch", help="Switch alias_current to a specific index.")
    switch.add_argument("--index", required=True, help="Target index name.")

    sub.add_parser("rollback", help="Rollback alias_current to alias_prev.")

    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    print_banner("Coal KB Index Manager", f"backend={cfg.backend}")

    elastic_store = ElasticStore(
        host=cfg.elastic.host,
        verify_certs=cfg.elastic.verify_certs,
        timeout_s=cfg.elastic.timeout_s,
    )

    if args.cmd == "build":
        if args.embedding_version:
            cfg.model_versions.embedding_version = args.embedding_version
        cfg.backend = "elastic"
        print_kv(
            "Index Build",
            {
                "embedding_version": cfg.model_versions.embedding_version,
                "index_prefix": cfg.elastic.index_prefix,
                "alias_current": cfg.elastic.alias_current,
                "alias_prev": cfg.elastic.alias_prev,
            },
        )
        dims = _resolve_dims(cfg)
        schema_sig = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))
        schema_hash = schema_sig[:8]
        index_name = elastic_store.build_index_name(
            index_prefix=cfg.elastic.index_prefix,
            embedding_version=cfg.model_versions.embedding_version,
            schema_hash=schema_hash,
        )
        elastic_store.create_index(index_name, dims)
        elastic_store.switch_alias(
            alias_current=cfg.elastic.alias_current,
            alias_prev=cfg.elastic.alias_prev,
            new_index=index_name,
        )
        pipe = IngestPipeline(cfg=cfg)
        with progress_status("Building index"):
            stats = pipe.run(rebuild=True, elastic_index_override=index_name)
        print_stats_table(
            "Build Summary",
            [
                ("index", index_name),
                ("indexed", str(stats.get("indexed"))),
                ("chunks", str(stats.get("chunks"))),
                ("elapsed_s", str(stats.get("elapsed_s"))),
            ],
        )
        logger.info("Index build complete: %s", index_name)
        return

    if args.cmd == "switch":
        elastic_store.switch_alias(
            alias_current=cfg.elastic.alias_current,
            alias_prev=cfg.elastic.alias_prev,
            new_index=args.index,
        )
        print_stats_table(
            "Alias Switch",
            [
                ("alias_current", cfg.elastic.alias_current),
                ("new_index", args.index),
            ],
        )
        logger.info("Switched alias to %s", args.index)
        return

    if args.cmd == "rollback":
        elastic_store.rollback(
            alias_current=cfg.elastic.alias_current,
            alias_prev=cfg.elastic.alias_prev,
        )
        print_stats_table(
            "Alias Rollback",
            [
                ("alias_current", cfg.elastic.alias_current),
                ("alias_prev", cfg.elastic.alias_prev),
            ],
        )
        logger.info("Rollback complete.")


if __name__ == "__main__":
    main()
