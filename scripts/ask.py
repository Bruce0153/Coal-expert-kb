"""启动命令行交互式检索与证据回答会话。"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from coal_kb.interfaces.cli.ui import print_banner, print_kv, print_stats_table

from coal_kb.answering import Answerer
from coal_kb.context import ContextBuilder
from coal_kb.infra.config import AppConfig, load_config
from coal_kb.infra.observability.logging import setup_logging
from coal_kb.infra.persistence.registry import RegistrySQLite
from coal_kb.infra.persistence.search import ElasticStore
from coal_kb.infra.persistence.vector import ChromaStore
from coal_kb.infra.providers.embeddings import EmbeddingsConfig
from coal_kb.infra.providers.llm import LLMConfig
from coal_kb.infra.providers.rerank import make_reranker
from coal_kb.ingestion.metadata.normalize import Ontology
from coal_kb.recall import rrf_fuse
from coal_kb.retrieval.query.filter_parser import FilterParser
from coal_kb.retrieval.query.planner import QueryPlanner
from coal_kb.retrieval.service import ExpertRetriever

logger = logging.getLogger(__name__)


class _CombinedRetriever:
    def __init__(self, chroma: Any, elastic: Any, k: int, rrf_k: int = 60) -> None:
        self._chroma = chroma
        self._elastic = elastic
        self._k = k
        self._rrf_k = rrf_k

    def invoke(self, query: str) -> list[Any]:
        chroma_documents: list[Any] = []
        elastic_documents: list[Any] = []
        if self._chroma is not None:
            chroma_documents = (
                self._chroma.get_relevant_documents(query)
                if hasattr(self._chroma, "get_relevant_documents")
                else self._chroma.invoke(query)
            )
        if self._elastic is not None:
            elastic_documents = self._elastic.invoke(query)
        if not chroma_documents and not elastic_documents:
            return []
        return rrf_fuse(elastic_documents, chroma_documents, k=self._rrf_k)[: self._k]


@dataclass
class Ask:
    cfg: AppConfig
    args: argparse.Namespace

    @staticmethod
    def _combine_factories(
        chroma_factory: Callable[..., Any] | None,
        elastic_factory: Callable[..., Any] | None,
        *,
        rrf_k: int,
    ) -> Callable[..., _CombinedRetriever]:
        def factory(k: int, where: dict[str, Any] | None = None) -> _CombinedRetriever:
            chroma = chroma_factory(k=k, where=where) if chroma_factory else None
            elastic = elastic_factory(k=k, where=where) if elastic_factory else None
            return _CombinedRetriever(chroma=chroma, elastic=elastic, k=k, rrf_k=rrf_k)

        return factory

    @staticmethod
    def _print_trace(trace: dict[str, Any], documents: list[Any]) -> None:
        if not trace:
            return
        print("\nRetrieval trace:")
        print(f"  stage1_parent_hits={trace.get('stage1_parent_hits', 0)}")
        print(f"  stage2_hits={trace.get('stage2_hits', 0)}")
        print(f"  relax_steps={trace.get('relax_steps', 0)}")
        if trace.get("fallback_mode"):
            print(f"  fallback_mode={trace.get('fallback_mode')}")
        if documents:
            print("  source_distribution:", trace.get("source_distribution", {}))
            print("  heading_distribution:", trace.get("heading_distribution", {}))

    def _build_expert_retriever(self) -> tuple[ExpertRetriever, str, int]:
        backend = self.args.backend or self.cfg.backend
        k = int(self.args.k or self.cfg.retrieval.k)
        rerank_enabled = bool(self.args.rerank or self.cfg.retrieval.rerank_enabled)
        if self.args.rerank_model:
            self.cfg.retrieval.rerank_model = self.args.rerank_model
        rerank_top_n = int(self.args.rerank_top_k or self.cfg.retrieval.rerank_top_n)
        mode = self.args.mode or self.cfg.retrieval.mode

        print_kv(
            "Retrieval Config",
            {
                "backend": backend,
                "k": str(k),
                "rerank_enabled": str(rerank_enabled),
                "rerank_top_n": str(rerank_top_n),
                "max_per_source": str(self.cfg.retrieval.max_per_source),
                "mode": mode,
            },
        )

        chroma_factory = None
        elastic_factory = None
        elastic_store = None
        if backend in {"chroma", "both"}:
            chroma_store = ChromaStore(
                persist_dir=self.cfg.paths.chroma_dir,
                collection_name=self.cfg.chroma.collection_name,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                embedding_model=self.cfg.embeddings.model,
            )
            chroma_factory = chroma_store.as_retriever

        if backend in {"elastic", "both"}:
            elastic_store = ElasticStore(
                host=self.cfg.elastic.host,
                verify_certs=self.cfg.elastic.verify_certs,
                timeout_s=self.cfg.elastic.timeout_s,
            )
            elastic_factory = elastic_store.make_retriever_factory(
                index=self.cfg.elastic.alias_current,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                candidates=k,
                rrf_k=self.cfg.retrieval.rrf_k,
                use_icu=self.cfg.elastic.enable_icu_analyzer,
                tenant_id=self.cfg.tenancy.default_tenant_id if self.cfg.tenancy.enabled else None,
            )

        if backend == "both":
            vector_factory = self._combine_factories(
                chroma_factory,
                elastic_factory,
                rrf_k=self.cfg.retrieval.rrf_k,
            )
        elif backend == "elastic":
            vector_factory = elastic_factory
        else:
            vector_factory = chroma_factory
        if vector_factory is None:
            raise RuntimeError("No retrieval backend is available.")

        reranker = make_reranker(self.cfg) if rerank_enabled else None
        expert = ExpertRetriever(
            vector_retriever_factory=vector_factory,
            k=k,
            rerank_enabled=rerank_enabled,
            rerank_top_n=rerank_top_n,
            reranker=reranker,
            max_per_source=self.cfg.retrieval.max_per_source,
            max_relax_steps=self.cfg.retrieval.max_relax_steps,
            range_expand_schedule=self.cfg.retrieval.range_expand_schedule,
            mode=mode,
            drop_sections=self.cfg.retrieval.drop_sections,
            drop_reference_like=self.cfg.retrieval.drop_reference_like,
            use_fuse=(backend != "elastic"),
            where_full=(backend == "elastic"),
            two_stage_enabled=(backend == "elastic" and self.cfg.retrieval.two_stage.enabled),
            parent_k_candidates=self.cfg.retrieval.two_stage.parent_k_candidates,
            parent_k_final=self.cfg.retrieval.two_stage.parent_k_final,
            max_parents=self.cfg.retrieval.two_stage.max_parents,
            child_k_candidates=self.cfg.retrieval.two_stage.child_k_candidates,
            child_k_final=self.cfg.retrieval.two_stage.child_k_final,
            allow_relax_in_stage2=self.cfg.retrieval.two_stage.allow_relax_in_stage2,
            elastic_store=elastic_store if backend == "elastic" else None,
            elastic_index=self.cfg.elastic.alias_current if backend == "elastic" else None,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()) if backend == "elastic" else None,
            elastic_use_icu=self.cfg.elastic.enable_icu_analyzer,
            tenant_id=self.cfg.tenancy.default_tenant_id if self.cfg.tenancy.enabled else None,
        )
        return expert, backend, k

    def process(self) -> None:
        print_banner("Coal KB Ask", f"backend={self.cfg.backend}")
        planner = QueryPlanner(filter_parser=FilterParser(onto=Ontology.load("configs/schema.yaml")))
        expert, backend, _ = self._build_expert_retriever()
        registry = RegistrySQLite(self.cfg.registry.sqlite_path)

        llm_provider = self.args.llm_provider
        if self.args.llm and llm_provider == "none":
            llm_provider = self.cfg.llm.provider
        provider = llm_provider if llm_provider != "none" else self.cfg.llm.provider
        llm_config = LLMConfig(**{**self.cfg.llm.model_dump(), "provider": provider})
        context_builder = ContextBuilder()
        answerer = Answerer(
            enable_llm=self.args.llm,
            llm_config=llm_config if self.args.llm else None,
        )

        while True:
            question = input("\n你的问题> ").strip()
            if not question:
                continue
            try:
                plan = planner.build_plan(
                    question,
                    self.cfg,
                    enable_llm=self.args.llm,
                    llm_config=llm_config,
                )
                if self.args.show_plan:
                    print("\nQueryPlan:")
                    print(plan.to_json())

                trace: dict[str, Any] = {}
                started_at = time.monotonic()
                documents = expert.execute(plan, trace=trace)
                context = context_builder.build(plan, documents)
                result = answerer.answer(plan, context)
                latency_ms = (time.monotonic() - started_at) * 1000

                self._print_trace(trace, documents)
                print_stats_table(
                    "Query Stats",
                    [("docs", str(len(documents))), ("latency_ms", f"{latency_ms:.2f}")],
                )
                print("\n" + result.answer_text)
                if result.citations:
                    print("\n引用列表:")
                    for source_id, item in result.citations.items():
                        print(
                            f"- [{source_id}] {item['source_file']} | page={item.get('page')} | "
                            f"heading={item.get('heading_path')} | chunk={item['chunk_id']}"
                        )

                registry.log_query(
                    query=plan.query.rewritten or plan.query.normalized,
                    filters=trace.get("where") or {},
                    constraints={
                        "plan": plan.to_dict(),
                        "retrieval_trace": trace,
                        "citations": result.citations,
                    }
                    if self.args.save_trace
                    else {"plan": plan.to_dict()},
                    top_chunk_ids=[document.metadata.get("chunk_id") for document in documents],
                    top_source_files=[document.metadata.get("source_file") for document in documents],
                    latency_ms=round(latency_ms, 2),
                    backend=backend,
                    tenant_id=None,
                    embedding_version=self.cfg.model_versions.embedding_version,
                    rerank_enabled=expert.rerank_enabled,
                    mode=expert.mode,
                    relax_steps=trace.get("relax_steps") if isinstance(trace.get("relax_steps"), list) else None,
                    diversity_k=trace.get("diversity@k"),
                )
                if self.args.save_trace:
                    print(f"trace_id: {plan.observability.trace_id}")
            except Exception as error:
                print(f"\n检索或回答失败: {type(error).__name__}: {error}")
                logger.exception("Ask loop failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with metadata-aware retrieval.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--show-plan", action="store_true", help="Print QueryPlan JSON.")
    parser.add_argument("--save-trace", action="store_true", help="Persist plan and retrieval trace.")
    parser.add_argument("--rerank", action="store_false", help="Enable reranking.")
    parser.add_argument("--rerank-model", default=None, help="Local rerank model name.")
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--llm-provider", default="none", choices=["none", "openai", "openai_compatible", "dashscope"])
    parser.add_argument("--backend", default=None, choices=["chroma", "elastic", "both"])
    parser.add_argument("--mode", default=None, choices=["strict", "balanced", "broad"])
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg, logger_name=__name__)
    Ask(cfg=cfg, args=args).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/ask.py
