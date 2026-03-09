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
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.bm25 import rrf_fuse
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.rerank import make_reranker
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with metadata-aware retrieval.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--show-plan", action="store_true", help="Print QueryPlan JSON.")
    parser.add_argument("--save-trace", action="store_true", help="Persist plan+retrieval trace in registry log.")
    parser.add_argument("--rerank", action="store_false", help="Enable reranking.")
    parser.add_argument("--rerank-model", default=None, help="Local rerank model name (fallback).")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai", "openai_compatible", "dashscope"])
    parser.add_argument("--backend", default=None, choices=["chroma", "elastic", "both"])
    parser.add_argument("--mode", default=None, choices=["strict", "balanced", "broad"])
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    print_banner("Coal KB Ask", f"backend={cfg.backend}")

    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)
    planner = QueryPlanner(filter_parser=parser_)

    backend = args.backend or cfg.backend
    k = int(args.k or cfg.retrieval.k)
    rerank_enabled = bool(args.rerank or cfg.retrieval.rerank_enabled)
    if args.rerank_model:
        cfg.retrieval.rerank_model = args.rerank_model
    rerank_top_n = int(args.rerank_top_k or cfg.retrieval.rerank_top_n)
    mode = args.mode or cfg.retrieval.mode

    print_kv("Retrieval Config", {"backend": backend, "k": str(k), "rerank_enabled": str(rerank_enabled), "rerank_top_n": str(rerank_top_n), "max_per_source": str(cfg.retrieval.max_per_source), "mode": mode})

    registry = RegistrySQLite(cfg.registry.sqlite_path)

    chroma_factory = None
    elastic_factory = None
    elastic_store = None
    if backend in {"chroma", "both"}:
        store = ChromaStore(persist_dir=cfg.paths.chroma_dir, collection_name=cfg.chroma.collection_name, embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()), embedding_model=cfg.embedding.model_name)
        chroma_factory = store.as_retriever

    if backend in {"elastic", "both"}:
        elastic_store = ElasticStore(host=cfg.elastic.host, verify_certs=cfg.elastic.verify_certs, timeout_s=cfg.elastic.timeout_s)
        elastic_factory = make_elastic_retriever_factory(client=elastic_store.client, index=cfg.elastic.alias_current, embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()), candidates=k, rrf_k=cfg.retrieval.rrf_k, use_icu=cfg.elastic.enable_icu_analyzer, tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None)

    if backend == "both":
        vector_factory = _combine_factories(chroma_factory, elastic_factory, rrf_k=cfg.retrieval.rrf_k)
    elif backend == "elastic":
        vector_factory = elastic_factory
    else:
        vector_factory = chroma_factory

    reranker = make_reranker(cfg) if rerank_enabled else None
    expert = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=k,
        rerank_enabled=rerank_enabled,
        rerank_top_n=rerank_top_n,
        reranker=reranker,
        max_per_source=cfg.retrieval.max_per_source,
        max_relax_steps=cfg.retrieval.max_relax_steps,
        range_expand_schedule=cfg.retrieval.range_expand_schedule,
        mode=mode,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=(backend != "elastic"),
        where_full=(backend == "elastic"),
        two_stage_enabled=(backend == "elastic" and cfg.retrieval.two_stage.enabled),
        parent_k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
        parent_k_final=cfg.retrieval.two_stage.parent_k_final,
        max_parents=cfg.retrieval.two_stage.max_parents,
        child_k_candidates=cfg.retrieval.two_stage.child_k_candidates,
        child_k_final=cfg.retrieval.two_stage.child_k_final,
        allow_relax_in_stage2=cfg.retrieval.two_stage.allow_relax_in_stage2,
        elastic_store=elastic_store if backend == "elastic" else None,
        elastic_index=cfg.elastic.alias_current if backend == "elastic" else None,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()) if backend == "elastic" else None,
        elastic_use_icu=cfg.elastic.enable_icu_analyzer,
        tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
    )

    llm_provider = args.llm_provider
    if args.llm and llm_provider == "none":
        llm_provider = cfg.llm.provider
    provider = llm_provider if llm_provider != "none" else cfg.llm.provider
    llm_cfg = LLMConfig(**{**cfg.llm.model_dump(), "provider": provider})

    context_builder = ContextBuilder()
    answerer = Answerer()

    while True:
        q = input("\n你的问题> ").strip()
        if not q:
            continue

        plan = planner.build_plan(q, cfg, enable_llm=args.llm, llm_config=llm_cfg)
        if args.show_plan:
            print("\nQueryPlan:")
            print(plan.to_json())

        trace: dict = {}
        start = time.monotonic()
        docs = expert.execute(plan, trace=trace)
        ctx = context_builder.build(plan, docs)
        result = answerer.answer(plan, ctx)
        latency_ms = (time.monotonic() - start) * 1000

        _print_trace(trace, docs)
        print_stats_table("Query Stats", [("docs", str(len(docs))), ("latency_ms", f"{latency_ms:.2f}")])
        print("\n" + result.answer_text)
        if result.citations:
            print("\n引用列表:")
            for sid, item in result.citations.items():
                print(f"- [{sid}] {item['source_file']} | page={item.get('page')} | heading={item.get('heading_path')} | chunk={item['chunk_id']}")

        registry.log_query(
            query=plan.query.rewritten or plan.query.normalized,
            filters=trace.get("where") or {},
            constraints={"plan": plan.to_dict(), "retrieval_trace": trace, "citations": result.citations} if args.save_trace else {"plan": plan.to_dict()},
            top_chunk_ids=[d.metadata.get("chunk_id") for d in docs],
            top_source_files=[d.metadata.get("source_file") for d in docs],
            latency_ms=round(latency_ms, 2),
            backend=backend,
            tenant_id=None,
            embedding_version=cfg.model_versions.embedding_version,
            rerank_enabled=expert.rerank_enabled,
            mode=mode,
            relax_steps=trace.get("relax_steps") if isinstance(trace.get("relax_steps"), list) else None,
            diversity_k=trace.get("diversity@k"),
        )

        if args.save_trace:
            print(f"trace_id: {plan.observability.trace_id}")


def _print_trace(trace: dict, docs: list) -> None:
    if not trace:
        return
    print("\nRetrieval trace:")
    print(f"  stage1_parent_hits={trace.get('stage1_parent_hits', 0)}")
    print(f"  stage2_hits={trace.get('stage2_hits', 0)}")
    print(f"  relax_steps={trace.get('relax_steps', 0)}")
    if docs:
        print("  source_distribution:", trace.get("source_distribution", {}))
        print("  heading_distribution:", trace.get("heading_distribution", {}))


def _combine_factories(chroma_factory, elastic_factory, *, rrf_k: int = 60):
    def factory(k: int, where=None):
        chroma = chroma_factory(k=k, where=where) if chroma_factory else None
        elastic = elastic_factory(k=k, where=where) if elastic_factory else None
        return _CombinedRetriever(chroma=chroma, elastic=elastic, k=k, rrf_k=rrf_k)

    return factory


class _CombinedRetriever:
    def __init__(self, chroma, elastic, k: int, rrf_k: int = 60):
        self._chroma = chroma
        self._elastic = elastic
        self._k = k
        self._rrf_k = rrf_k

    def invoke(self, query: str):
        chroma_docs = []
        elastic_docs = []
        if self._chroma is not None:
            chroma_docs = self._chroma.get_relevant_documents(query) if hasattr(self._chroma, "get_relevant_documents") else self._chroma.invoke(query)
        if self._elastic is not None:
            elastic_docs = self._elastic.invoke(query)
        if not chroma_docs and not elastic_docs:
            return []
        fused = rrf_fuse(elastic_docs, chroma_docs, k=self._rrf_k)
        return fused[: self._k]


if __name__ == "__main__":
    main()
