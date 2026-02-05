from __future__ import annotations

import argparse

from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.evaluation.runner import EvalRunner, make_default_planner
from coal_kb.logging import setup_logging
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.elastic_store import ElasticStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint.")
    parser.add_argument("--task", default="retrieval", choices=["retrieval", "faithfulness", "extraction"])
    parser.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    store = ElasticStore(host=cfg.elastic.host, verify_certs=cfg.elastic.verify_certs, timeout_s=cfg.elastic.timeout_s)
    planner = make_default_planner(cfg)
    retriever = ExpertRetriever(
        vector_retriever_factory=lambda k, where=None: None,
        k=args.k,
        rerank_enabled=cfg.retrieval.rerank_enabled,
        rerank_top_n=cfg.retrieval.rerank_top_n,
        max_per_source=cfg.retrieval.max_per_source,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        two_stage_enabled=cfg.retrieval.two_stage.enabled,
        elastic_store=store,
        elastic_index=cfg.elastic.alias_current,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
        elastic_use_icu=cfg.elastic.enable_icu_analyzer,
    )
    runner = EvalRunner(planner=planner, retriever=retriever)
    result = runner.run(task=args.task, gold_path=args.gold, k=args.k)
    print(result)


if __name__ == "__main__":
    main()
