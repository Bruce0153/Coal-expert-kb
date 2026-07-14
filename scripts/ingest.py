"""扫描并摄入知识库文档，保持原索引与元数据逻辑不变。"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.ingestion.pipeline import IngestPipeline
from coal_kb.interfaces.cli.ui import print_banner, print_kv, print_stats_table

logger = logging.getLogger(__name__)


@dataclass
class Ingest:
    cfg: AppConfig
    rebuild: bool
    force: bool
    enable_tables: bool
    table_flavor: str

    def process(self) -> dict:
        print_banner("Coal KB Ingest", f"backend={self.cfg.backend}")
        print_kv(
            "Config",
            {
                "raw_pdfs_dir": self.cfg.paths.raw_pdfs_dir,
                "raw_docs_dir": self.cfg.paths.raw_docs_dir,
                "chroma_dir": self.cfg.paths.chroma_dir,
                "registry_db": self.cfg.registry.sqlite_path,
                "embedding_model": self.cfg.embeddings.model,
                "chunk_size": str(self.cfg.chunking.chunk_size),
                "chunk_overlap": str(self.cfg.chunking.chunk_overlap),
                "drop_sections": ",".join(self.cfg.ingestion.drop_sections),
            },
        )
        logger.info(
            "Ingest config | raw_pdfs_dir=%s raw_docs_dir=%s chroma_dir=%s interim_dir=%s",
            self.cfg.paths.raw_pdfs_dir,
            self.cfg.paths.raw_docs_dir,
            self.cfg.paths.chroma_dir,
            self.cfg.paths.interim_dir,
        )
        logger.info("Embeddings | provider=%s model=%s", self.cfg.embeddings.provider, self.cfg.embeddings.model)
        logger.info(
            "Chunking | size=%d overlap=%d | tables=%s",
            self.cfg.chunking.chunk_size,
            self.cfg.chunking.chunk_overlap,
            self.enable_tables,
        )
        logger.info("Backend | mode=%s registry_db=%s", self.cfg.backend, self.cfg.registry.sqlite_path)
        started_at = time.monotonic()
        pipeline = IngestPipeline(
            cfg=self.cfg,
            enable_table_extraction=self.enable_tables,
            table_flavor=self.table_flavor,
        )
        stats = pipeline.process(rebuild=self.rebuild, force=self.force)
        elapsed = stats.get("elapsed_s", round(time.monotonic() - started_at, 2))
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
        return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the expert KB.")
    parser.add_argument("--tables", action="store_true", help="Enable optional table extraction (Camelot).")
    parser.add_argument("--table-flavor", default="lattice", choices=["lattice", "stream", "auto"])
    parser.add_argument("--rebuild", action="store_true", help="Clear vectorstore and manifest before ingest.")
    parser.add_argument("--force", action="store_true", help="Continue when recoverable batch failures occur.")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    step = Ingest(
        cfg=cfg,
        rebuild=args.rebuild,
        force=args.force,
        enable_tables=args.tables,
        table_flavor=args.table_flavor,
    )
    print(step.process())


if __name__ == "__main__":
    main()

# 运行命令：python scripts/ingest.py
