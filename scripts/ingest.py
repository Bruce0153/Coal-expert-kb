from __future__ import annotations

import argparse
import logging
import time

from coal_kb.logging import setup_logging
from coal_kb.pipelines.ingest_pipeline import IngestPipeline
from coal_kb.settings import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma expert KB.")
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
    logger.info(
        "Ingest config | raw_dir=%s chroma_dir=%s interim_dir=%s",
        cfg.paths.raw_pdfs_dir,
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

    start = time.monotonic()
    pipe = IngestPipeline(
        cfg=cfg,
        enable_table_extraction=args.tables,
        table_flavor=args.table_flavor,
        enable_llm_metadata=args.llm_metadata,
    )
    stats = pipe.run(rebuild=args.rebuild, force=args.force)
    elapsed = stats.get("elapsed_s", round(time.monotonic() - start, 2))
    logger.info(
        "Ingest summary | scanned=%s changed=%s removed=%s pages=%s chunks=%s indexed=%s elapsed=%.2fs",
        stats.get("pdfs_scanned"),
        stats.get("pdfs_changed"),
        stats.get("pdfs_removed"),
        stats.get("pages_parsed"),
        stats.get("chunks"),
        stats.get("indexed"),
        elapsed,
    )
    print(stats)


if __name__ == "__main__":
    main()
