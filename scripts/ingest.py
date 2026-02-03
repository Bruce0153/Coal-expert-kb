from __future__ import annotations

import argparse
import logging
import time

from coal_kb.logging import setup_logging
from coal_kb.cli_ui import print_banner, print_kv, print_stats_table, progress_status
from coal_kb.pipelines.ingest_pipeline import IngestPipeline
from coal_kb.settings import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the expert KB.")
    parser.add_argument("--tables", action="store_true", help="Enable optional table extraction (Camelot).")
    parser.add_argument("--table-flavor", default="lattice", choices=["lattice", "stream", "auto"])
    parser.add_argument(
        "--llm-metadata",
        action="store_true",
        help="Enable LLM metadata augmentation (LLM is configured in configs/app.yaml).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the KB (clear vectorstore + manifest) before ingest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if manifest signatures mismatch (not recommended).",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    print_banner("Coal KB Ingest", f"backend={cfg.backend}")
    print_kv(
        "Config",
        {
            "raw_pdfs_dir": cfg.paths.raw_pdfs_dir,
            "raw_docs_dir": cfg.paths.raw_docs_dir,
            "chroma_dir": cfg.paths.chroma_dir,
            "registry_db": cfg.registry.sqlite_path,
            "embedding_model": cfg.embeddings.model,
            "chunk_size": str(cfg.chunking.chunk_size),
            "chunk_overlap": str(cfg.chunking.chunk_overlap),
            "drop_sections": ",".join(cfg.ingestion.drop_sections),
        },
    )
    logger.info(
        "Ingest config | raw_pdfs_dir=%s raw_docs_dir=%s chroma_dir=%s interim_dir=%s",
        cfg.paths.raw_pdfs_dir,
        cfg.paths.raw_docs_dir,
        cfg.paths.chroma_dir,
        cfg.paths.interim_dir,
    )
    logger.info(
        "Embeddings | provider=%s model=%s",
        cfg.embeddings.provider,
        cfg.embeddings.model,
    )
    logger.info(
        "Chunking | size=%d overlap=%d | tables=%s llm_metadata=%s",
        cfg.chunking.chunk_size,
        cfg.chunking.chunk_overlap,
        args.tables,
        args.llm_metadata,
    )
    logger.info(
        "Backend | mode=%s registry_db=%s",
        cfg.backend,
        cfg.registry.sqlite_path,
    )

    start = time.monotonic()
    pipe = IngestPipeline(
        cfg=cfg,
        enable_table_extraction=args.tables,
        table_flavor=args.table_flavor,
        enable_llm_metadata=args.llm_metadata,
    )
    with progress_status("Ingesting documents"):
        stats = pipe.run(rebuild=args.rebuild, force=args.force)
    elapsed = stats.get("elapsed_s", round(time.monotonic() - start, 2))
    logger.info(
        "Ingest summary | scanned=%s changed=%s removed=%s pages=%s chunks=%s indexed=%s dropped=%s elapsed=%.2fs",
        stats.get("docs_scanned"),
        stats.get("docs_changed"),
        stats.get("docs_removed"),
        stats.get("pages_parsed"),
        stats.get("chunks"),
        stats.get("indexed"),
        stats.get("dropped_chunks"),
        elapsed,
    )
    print_stats_table(
        "Ingest Summary",
        [
            ("docs_scanned", str(stats.get("docs_scanned"))),
            ("docs_changed", str(stats.get("docs_changed"))),
            ("pages_parsed", str(stats.get("pages_parsed"))),
            ("chunks", str(stats.get("chunks"))),
            ("indexed", str(stats.get("indexed"))),
            ("dropped_chunks", str(stats.get("dropped_chunks"))),
            ("doc_type_counts", str(stats.get("doc_type_counts"))),
            ("language_counts", str(stats.get("language_counts"))),
            ("elapsed_s", str(elapsed)),
        ],
    )
    print(stats)


if __name__ == "__main__":
    main()
