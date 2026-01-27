from __future__ import annotations

import argparse
import logging

from coal_kb.logging import setup_logging
from coal_kb.metadata.normalize import Ontology
from coal_kb.qa.rag_answer import RAGAnswerer
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with metadata-aware retrieval.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai"])
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)

    store = ChromaStore(
        persist_dir=cfg.paths.chroma_dir,
        collection_name=cfg.chroma.collection_name,
        embedding_model=cfg.embedding.model_name,
    )

    expert = ExpertRetriever(vector_retriever_factory=store.as_retriever, k=args.k)
    answerer = RAGAnswerer(enable_llm=args.llm, llm_provider=args.llm_provider)

    while True:
        q = input("\n你的问题> ").strip()
        if not q:
            continue
        f = parser_.parse(q)
        docs = expert.retrieve(q, f)
        ans = answerer.answer(q, docs)
        print("\n" + ans)


if __name__ == "__main__":
    main()
