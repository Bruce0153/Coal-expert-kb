from __future__ import annotations

import argparse
import logging

from coal_kb.logging import setup_logging
from coal_kb.pipelines.record_pipeline import RecordPipeline
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ExperimentRecords into SQLite.")
    parser.add_argument("--limit", type=int, default=200, help="Max chunks to sample for extraction.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM record extraction.")
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai"])
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    # Pull chunks from vectorstore (simple scan by querying empty string)
    store = ChromaStore(
        persist_dir=cfg.paths.chroma_dir,
        collection_name=cfg.chroma.collection_name,
        embedding_model=cfg.embedding.model_name,
    )

    retriever = store.as_retriever(k=args.limit, where=None)

    # A common trick: query a generic token to get "some" docs.
    # For full extraction, you'd implement a real scan iterator over Chroma collection.
    docs = retriever.get_relevant_documents("gasification pyrolysis experimental conditions table results")

    pipe = RecordPipeline(cfg=cfg, enable_llm_records=args.llm, llm_provider=args.llm_provider)
    stats = pipe.run(docs)
    print(stats)


if __name__ == "__main__":
    main()
