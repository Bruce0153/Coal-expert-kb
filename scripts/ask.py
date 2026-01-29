from __future__ import annotations

import argparse
import json
import logging
import time

from coal_kb.logging import setup_logging
from coal_kb.cli_ui import print_banner, print_kv, print_stats_table
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.llm.factory import LLMConfig
from coal_kb.metadata.normalize import Ontology
from coal_kb.qa.rag_answer import RAGAnswerer
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with metadata-aware retrieval.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Cross-encoder model name (overrides config).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help="How many candidates to rerank (overrides config).",
    )
    parser.add_argument(
        "--llm-provider",
        default="none",
        choices=["none", "openai", "openai_compatible", "dashscope"],
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["chroma", "elastic", "both"],
        help="Override backend config (chroma|elastic|both).",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    print_banner("Coal KB Ask", f"backend={cfg.backend}")

    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)

    backend = args.backend or cfg.backend
    if backend not in {"chroma", "elastic", "both"}:
        raise ValueError(f"Unsupported backend: {backend}")

    registry = RegistrySQLite(cfg.registry.sqlite_path)
    chroma_factory = None
    elastic_factory = None
    if backend in {"chroma", "both"}:
        store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            embedding_model=cfg.embedding.model_name,
        )
        chroma_factory = store.as_retriever

    if backend in {"elastic", "both"}:
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        elastic_factory = make_elastic_retriever_factory(
            client=elastic_store.client,
            index=cfg.elastic.alias_current,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
        )

    if backend == "both":
        vector_factory = _combine_factories(chroma_factory, elastic_factory)
    elif backend == "elastic":
        vector_factory = elastic_factory
    else:
        vector_factory = chroma_factory
    if vector_factory is None:
        raise RuntimeError("No retriever factory configured.")

    rerank_enabled = args.rerank or cfg.retrieval.rerank_enabled
    rerank_model = args.rerank_model or cfg.retrieval.rerank_model
    rerank_top_n = args.rerank_top_k or cfg.retrieval.rerank_top_n

    print_kv(
        "Retrieval Config",
        {
            "backend": backend,
            "k": str(args.k or cfg.retrieval.k),
            "candidates": str(cfg.retrieval.candidates),
            "rerank_enabled": str(rerank_enabled),
            "max_per_source": str(cfg.retrieval.max_per_source),
        },
    )

    expert = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=args.k or cfg.retrieval.k,
        rerank_enabled=rerank_enabled,
        rerank_model=rerank_model,
        rerank_top_n=rerank_top_n,
        rerank_candidates=cfg.retrieval.candidates,
        rerank_device=cfg.retrieval.rerank_device,
        max_per_source=cfg.retrieval.max_per_source,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=(backend != "elastic"),
        where_full=(backend == "elastic"),
    )
    llm_provider = args.llm_provider
    if args.llm and llm_provider == "none":
        llm_provider = cfg.llm.provider

    provider = llm_provider if llm_provider != "none" else cfg.llm.provider
    llm_cfg = LLMConfig(**{**cfg.llm.model_dump(), "provider": provider})
    answerer = RAGAnswerer(
        enable_llm=args.llm,
        llm_provider=llm_provider,
        llm_config=llm_cfg,
    )

    while True:
        q = input("\n你的问题> ").strip()
        if not q:
            continue
        f = parser_.parse(q)
        rewrite = rewrite_query(
            q,
            enable_llm=cfg.query_rewrite.enable_llm and args.llm,
            llm_config=llm_cfg,
        )
        print("\n解析到的过滤条件:")
        print(json.dumps(f, ensure_ascii=False, indent=2))
        if rewrite.reason:
            print(f"查询扩展: {rewrite.reason}")
            print(rewrite.query)
        trace: dict = {}
        start = time.monotonic()
        docs = expert.retrieve(rewrite.query, f, trace=trace)
        latency_ms = (time.monotonic() - start) * 1000
        _print_trace(trace, docs)
        print_stats_table(
            "Query Stats",
            [
                ("docs", str(len(docs))),
                ("latency_ms", f"{latency_ms:.2f}"),
            ],
        )
        registry.log_query(
            query=rewrite.query,
            filters=f,
            top_chunk_ids=[d.metadata.get("chunk_id") for d in docs],
            top_source_files=[d.metadata.get("source_file") for d in docs],
            latency_ms=round(latency_ms, 2),
            backend=backend,
            tenant_id=None,
            embedding_version=cfg.model_versions.embedding_version,
            rerank_enabled=expert.rerank_enabled,
        )
        ans = answerer.answer(q, docs)
        print("\n" + ans)


if __name__ == "__main__":
    main()


def _print_trace(trace: dict, docs: list) -> None:
    if not trace:
        return
    where = trace.get("where") or {}
    counts = {
        "vector": trace.get("vector_candidates", 0),
        "fused": trace.get("fused_candidates", 0),
        "postfiltered": trace.get("postfiltered_count", 0),
    }
    vector_cites = trace.get("vector_top_citations", [])[:3]
    final_cites = trace.get("final_top_citations", [])[:3]

    print("\nRetrieval trace:")
    print(f"  where: {where}")
    print(f"  counts: vector={counts['vector']} fused={counts['fused']} postfiltered={counts['postfiltered']}")
    if vector_cites:
        print("  top vector candidates:")
        for c in vector_cites:
            print(f"    - {c}")
    else:
        print("  top vector candidates: (none)")
    if docs:
        if final_cites:
            print("  top evidence citations:")
            for c in final_cites:
                print(f"    - {c}")
    else:
        print("  top evidence citations: (none)")


def _combine_factories(chroma_factory, elastic_factory):
    def factory(k: int, where=None):
        chroma = chroma_factory(k=k, where=where) if chroma_factory else None
        elastic = elastic_factory(k=k, where=where) if elastic_factory else None
        return _CombinedRetriever(chroma=chroma, elastic=elastic)

    return factory


class _CombinedRetriever:
    def __init__(self, chroma, elastic):
        self._chroma = chroma
        self._elastic = elastic

    def invoke(self, query: str):
        docs = []
        if self._chroma is not None:
            if hasattr(self._chroma, "get_relevant_documents"):
                docs.extend(self._chroma.get_relevant_documents(query))
            else:
                docs.extend(self._chroma.invoke(query))
        if self._elastic is not None:
            docs.extend(self._elastic.invoke(query))
        seen = set()
        unique = []
        for d in docs:
            key = d.metadata.get("chunk_id")
            if key in seen:
                continue
            seen.add(key)
            unique.append(d)
        return unique
