from __future__ import annotations

import argparse
import logging

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

    pipe = IngestPipeline(
        cfg=cfg,
        enable_table_extraction=args.tables,
        table_flavor=args.table_flavor,
        enable_llm_metadata=args.llm_metadata,
    )
    stats = pipe.run(rebuild=args.rebuild, force=args.force)
    print(stats)


if __name__ == "__main__":
    main()
