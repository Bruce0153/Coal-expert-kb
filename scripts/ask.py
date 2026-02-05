from __future__ import annotations

import argparse
import json
import logging
import time

from coal_kb.cli_ui import print_banner, print_kv, print_stats_table
from coal_kb.context.builder import ContextBuilder
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.generation.answerer import Answerer
from coal_kb.llm.factory import LLMConfig
from coal_kb.logging import setup_logging
from coal_kb.metadata.normalize import Ontology
from coal_kb.observability.query_log import QueryLog
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.rerank import make_reranker
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with QueryPlan + two-stage retrieval.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai", "openai_compatible", "dashscope"])
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)

    if cfg.backend != "elastic":
        raise RuntimeError("Modern ask path requires backend=elastic")

    k = int(args.k or cfg.retrieval.k)
    print_banner("Coal KB Ask", f"backend={cfg.backend}")
    print_kv("Config", {"k": str(k), "two_stage": str(cfg.retrieval.two_stage.enabled), "index": cfg.elastic.alias_current})

    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)
    planner = QueryPlanner(cfg=cfg, parser=parser_)

    elastic_store = ElasticStore(host=cfg.elastic.host, verify_certs=cfg.elastic.verify_certs, timeout_s=cfg.elastic.timeout_s)
    reranker = make_reranker(cfg) if cfg.retrieval.rerank_enabled else None

    retriever = ExpertRetriever(
        vector_retriever_factory=lambda k, where=None: None,
        k=k,
        rerank_enabled=cfg.retrieval.rerank_enabled,
        rerank_top_n=cfg.retrieval.rerank_top_n,
        reranker=reranker,
        max_per_source=cfg.retrieval.max_per_source,
        max_relax_steps=cfg.retrieval.max_relax_steps,
        range_expand_schedule=cfg.retrieval.range_expand_schedule,
        mode=cfg.retrieval.mode,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=True,
        where_full=True,
        two_stage_enabled=cfg.retrieval.two_stage.enabled,
        parent_k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
        parent_k_final=cfg.retrieval.two_stage.parent_k_final,
        max_parents=cfg.retrieval.two_stage.max_parents,
        child_k_candidates=cfg.retrieval.two_stage.child_k_candidates,
        child_k_final=cfg.retrieval.two_stage.child_k_final,
        allow_relax_in_stage2=cfg.retrieval.two_stage.allow_relax_in_stage2,
        elastic_store=elastic_store,
        elastic_index=cfg.elastic.alias_current,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
        elastic_use_icu=cfg.elastic.enable_icu_analyzer,
        tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
    )

    context_builder = ContextBuilder()

    llm_provider = args.llm_provider if args.llm_provider != "none" else cfg.llm.provider
    llm_cfg = LLMConfig(**{**cfg.llm.model_dump(), "provider": llm_provider})
    answerer = Answerer(enable_llm=args.llm, llm_provider=llm_provider, llm_config=llm_cfg)

    registry = RegistrySQLite(cfg.registry.sqlite_path)

    while True:
        q = input("\n你的问题> ").strip()
        if not q:
            continue

        plan = planner.build_plan(q)
        trace: dict = {}
        start = time.monotonic()
        docs = retriever.execute(plan, trace=trace)
        latency_ms = (time.monotonic() - start) * 1000

        context_pkg = context_builder.build(plan, docs)
        result = answerer.answer(q, context_pkg)

        print_stats_table("Query Stats", [("docs", str(len(docs))), ("latency_ms", f"{latency_ms:.2f}"), ("token_estimate", str(context_pkg.token_estimate))])
        if docs:
            print("\nTop evidence headings:")
            for i, d in enumerate(docs[: min(5, len(docs))], start=1):
                m = d.metadata or {}
                print(f"  {i}. {m.get('heading_path') or 'N/A'} | {m.get('source_file')}#{m.get('chunk_id')}")

        print("\n引用映射:")
        print(json.dumps(result.citations, ensure_ascii=False, indent=2))
        print("\n" + result.answer_text)

        qlog = QueryLog(
            query=q,
            plan=plan.as_dict(),
            filters=plan.stage2.filters,
            k=k,
            latency_ms=round(latency_ms, 2),
            relax_steps=trace.get("relax_steps_taken", []),
            top_sources=[(d.metadata or {}).get("source_file") for d in docs],
            top_chunk_ids=[(d.metadata or {}).get("chunk_id") for d in docs],
            answer_stats={"uncertain": result.uncertain, "used_docs": result.used_docs},
        )

        registry.log_query(
            query=qlog.query,
            filters=qlog.filters,
            constraints={"plan": qlog.plan},
            top_chunk_ids=qlog.top_chunk_ids,
            top_source_files=qlog.top_sources,
            latency_ms=qlog.latency_ms,
            backend=cfg.backend,
            tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
            embedding_version=cfg.model_versions.embedding_version,
            rerank_enabled=cfg.retrieval.rerank_enabled,
            mode=cfg.retrieval.mode,
            relax_steps=trace.get("relax_steps_taken"),
            diversity_k=trace.get("diversity@k"),
        )


if __name__ == "__main__":
    main()
