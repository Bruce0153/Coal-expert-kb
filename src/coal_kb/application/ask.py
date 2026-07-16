"""编排单轮问答运行时、研究路线、检索执行和响应格式化。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document

from coal_kb.application import config
from coal_kb.infra.observability.trace import build_retrieval_trace_summary

if TYPE_CHECKING:
    from coal_kb.answering import Answerer, AnswerResult
    from coal_kb.complex_qa import ComplexQuestionService
    from coal_kb.context import ContextBuilder
    from coal_kb.core.models.query import QueryPlan
    from coal_kb.infra.config import AppConfig
    from coal_kb.infra.persistence.registry import RegistrySQLite
    from coal_kb.infra.providers.llm import LLMConfig
    from coal_kb.research import ResearchRouteService
    from coal_kb.retrieval.query.planner import QueryPlanner
    from coal_kb.retrieval.service import ExpertRetriever


@dataclass
class AskRuntime:
    cfg: AppConfig
    backend: str
    k: int
    mode: str
    planner: QueryPlanner
    retriever: ExpertRetriever
    context_builder: ContextBuilder
    complex_question_service: ComplexQuestionService
    research_route_service: ResearchRouteService
    answerer: Answerer
    registry: RegistrySQLite
    llm_config: LLMConfig | None


@dataclass
class AskExecution:
    query: str
    retrieval_query: str
    plan: QueryPlan
    docs: list[Document]
    trace: dict[str, Any]
    context_debug: dict[str, Any]
    result: AnswerResult
    timings_ms: dict[str, float]
    research_route: str = "standard"
    history_used: bool = False
    history_reason: str = "standalone_query"


def normalize_query(query: str) -> str:
    """折叠查询中的多余空白。"""
    return " ".join(query.strip().split())


def parse_command(query: str) -> str | None:
    """解析交互式 CLI 的内置命令。"""
    normalized = query.strip().lower()
    if normalized in {"exit", "quit"}:
        return "exit"
    if normalized in {"help", ":help"}:
        return "help"
    if normalized in {"debug", ":debug"}:
        return "debug"
    return None


class _CombinedRetriever:
    def __init__(self, chroma: Any, elastic: Any, k: int, rrf_k: int) -> None:
        self._chroma = chroma
        self._elastic = elastic
        self._k = k
        self._rrf_k = rrf_k

    def invoke(self, query: str) -> list[Document]:
        chroma_documents = self._invoke(self._chroma, query)
        elastic_documents = self._invoke(self._elastic, query)
        if not chroma_documents and not elastic_documents:
            return []
        from coal_kb.recall import rrf_fuse

        return rrf_fuse(elastic_documents, chroma_documents, k=self._rrf_k)[: self._k]

    @staticmethod
    def _invoke(retriever: Any, query: str) -> list[Document]:
        if retriever is None:
            return []
        if hasattr(retriever, "invoke"):
            return list(retriever.invoke(query))
        return list(retriever.get_relevant_documents(query))


def _combine_factories(chroma_factory: Any, elastic_factory: Any, *, rrf_k: int):
    def factory(k: int, where: dict[str, Any] | None = None) -> _CombinedRetriever:
        chroma = chroma_factory(k=k, where=where) if chroma_factory else None
        elastic = elastic_factory(k=k, where=where) if elastic_factory else None
        return _CombinedRetriever(chroma, elastic, k, rrf_k)

    return factory


def build_runtime(
    cfg: AppConfig,
    *,
    backend: str | None = None,
    k: int | None = None,
    rerank_enabled: bool | None = None,
    rerank_top_n: int | None = None,
    mode: str | None = None,
    enable_llm: bool = False,
    llm_provider: str = "none",
) -> AskRuntime:
    """从应用配置组装唯一问答运行时。"""
    from coal_kb.answering import Answerer
    from coal_kb.complex_qa import ComplexQuestionService
    from coal_kb.context import ContextBuilder
    from coal_kb.infra.persistence.registry import RegistrySQLite
    from coal_kb.infra.persistence.search import ElasticStore
    from coal_kb.infra.persistence.vector import ChromaStore
    from coal_kb.infra.providers.rerank import make_reranker
    from coal_kb.infra.providers.tokenizers import make_tokenizer
    from coal_kb.ingestion.metadata.normalize import Ontology
    from coal_kb.research import GraphRoute, ResearchRouteService
    from coal_kb.retrieval.query import FilterParser
    from coal_kb.retrieval.query.planner import QueryPlanner
    from coal_kb.retrieval.service import ExpertRetriever

    active_backend = backend or cfg.backend
    active_k = int(k or cfg.retrieval.k)
    active_mode = mode or cfg.retrieval.mode
    active_rerank = cfg.retrieval.rerank_enabled if rerank_enabled is None else rerank_enabled
    active_rerank_top_n = int(rerank_top_n or cfg.retrieval.rerank_top_n)
    planner = QueryPlanner(filter_parser=FilterParser(onto=Ontology.load(config.ONTOLOGY_PATH)))
    registry = RegistrySQLite(cfg.registry.sqlite_path)
    chroma_factory = None
    elastic_factory = None
    elastic_store = None

    if active_backend in {"chroma", "both"}:
        chroma_store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=cfg.embeddings,
            embedding_model=cfg.embeddings.model,
        )
        chroma_factory = chroma_store.as_retriever
    if active_backend in {"elastic", "both"}:
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        elastic_factory = elastic_store.make_retriever_factory(
            index=cfg.elastic.alias_current,
            embeddings_cfg=cfg.embeddings,
            candidates=active_k,
            rrf_k=cfg.retrieval.rrf_k,
            use_icu=cfg.elastic.enable_icu_analyzer,
            tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
        )
    if active_backend == "both":
        vector_factory = _combine_factories(
            chroma_factory,
            elastic_factory,
            rrf_k=cfg.retrieval.rrf_k,
        )
    elif active_backend == "elastic":
        vector_factory = elastic_factory
    elif active_backend == "chroma":
        vector_factory = chroma_factory
    else:
        raise ValueError(f"Unsupported retrieval backend: {active_backend}")
    if vector_factory is None:
        raise RuntimeError(f"Retrieval backend is unavailable: {active_backend}")

    retriever = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=active_k,
        rerank_enabled=active_rerank,
        rerank_top_n=active_rerank_top_n,
        reranker=make_reranker(cfg.rerank) if active_rerank else None,
        max_per_source=cfg.retrieval.max_per_source,
        max_relax_steps=cfg.retrieval.max_relax_steps,
        range_expand_schedule=cfg.retrieval.range_expand_schedule,
        mode=active_mode,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        two_stage_enabled=active_backend == "elastic" and cfg.retrieval.two_stage.enabled,
        parent_k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
        parent_k_final=cfg.retrieval.two_stage.parent_k_final,
        max_parents=cfg.retrieval.two_stage.max_parents,
        child_k_candidates=cfg.retrieval.two_stage.child_k_candidates,
        child_k_final=cfg.retrieval.two_stage.child_k_final,
        allow_relax_in_stage2=cfg.retrieval.two_stage.allow_relax_in_stage2,
        elastic_store=elastic_store if active_backend == "elastic" else None,
        elastic_index=cfg.elastic.alias_current if active_backend == "elastic" else None,
        embeddings_cfg=cfg.embeddings if active_backend == "elastic" else None,
        elastic_use_icu=cfg.elastic.enable_icu_analyzer,
        tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
    )
    complex_service = ComplexQuestionService(
        retriever=retriever,
        sqlite_path=cfg.paths.sqlite_path,
        table_records_path=cfg.complex_qa.table_records_path,
        comparison_k_per_side=cfg.complex_qa.comparison_k_per_side,
        max_multi_hop_steps=cfg.complex_qa.max_multi_hop_steps,
        aggregation_record_limit=cfg.complex_qa.aggregation_record_limit,
        aggregation_evidence_limit=cfg.complex_qa.aggregation_evidence_limit,
        table_top_k=cfg.complex_qa.table_top_k,
        cross_document_min_sources=cfg.complex_qa.cross_document_min_sources,
        cross_document_max_per_source=cfg.complex_qa.cross_document_max_per_source,
    )
    llm_config = None
    if enable_llm:
        llm_config = cfg.llm.model_copy(deep=True)
        if llm_provider != "none":
            llm_config.active.provider = llm_provider

    return AskRuntime(
        cfg=cfg,
        backend=active_backend,
        k=active_k,
        mode=active_mode,
        planner=planner,
        retriever=retriever,
        context_builder=ContextBuilder(token_counter=make_tokenizer(cfg.tokenizer).count_tokens),
        complex_question_service=complex_service,
        research_route_service=ResearchRouteService(
            standard_service=complex_service,
            graph_route=GraphRoute(),
        ),
        answerer=Answerer(
            enable_llm=enable_llm and llm_config is not None,
            llm_config=llm_config,
        ),
        registry=registry,
        llm_config=llm_config,
    )


def execute_query(
    runtime: AskRuntime,
    raw_query: str,
    *,
    enable_llm: bool = False,
    original_query: str | None = None,
    research_route: str = "standard",
    history_used: bool = False,
    history_reason: str = "standalone_query",
) -> AskExecution:
    """执行规划、研究路线、上下文和回答链路。"""
    query = normalize_query(original_query or raw_query)
    retrieval_query = normalize_query(raw_query)
    if not retrieval_query:
        raise ValueError("query is empty")
    trace: dict[str, Any] = {}
    started = time.monotonic()
    plan = runtime.planner.build_plan(
        retrieval_query,
        runtime.cfg,
        enable_llm=False,
        llm_config=None,
    )
    plan_ms = (time.monotonic() - started) * 1000
    started = time.monotonic()
    documents = runtime.research_route_service.process(
        plan,
        route=research_route,
        trace=trace,
    )
    retrieve_ms = (time.monotonic() - started) * 1000
    started = time.monotonic()
    context_package = runtime.context_builder.build(plan, documents)
    context_ms = (time.monotonic() - started) * 1000
    started = time.monotonic()
    result = runtime.answerer.answer(plan, context_package, enable_llm=enable_llm)
    answer_ms = (time.monotonic() - started) * 1000
    return AskExecution(
        query=query,
        retrieval_query=retrieval_query,
        plan=plan,
        docs=documents,
        trace=trace,
        context_debug=context_package.debug,
        result=result,
        timings_ms={
            "plan": round(plan_ms, 2),
            "retrieve": round(retrieve_ms, 2),
            "context": round(context_ms, 2),
            "answer": round(answer_ms, 2),
            "total": round(plan_ms + retrieve_ms + context_ms + answer_ms, 2),
        },
        research_route=research_route,
        history_used=history_used,
        history_reason=history_reason,
    )


def retrieval_diagnostics(
    execution: AskExecution,
    *,
    limit: int = config.DEFAULT_DIAGNOSTIC_LIMIT,
) -> list[dict[str, Any]]:
    """生成检索结果摘要。"""
    score_map = {
        item.get("chunk_id"): item.get("score")
        for item in execution.trace.get("condition_score_top3", [])
        if isinstance(item, dict)
    }
    return [
        {
            "chunk_id": metadata.get("chunk_id"),
            "source_file": metadata.get("source_file"),
            "title": metadata.get("title"),
            "page": metadata.get("page"),
            "heading_path": metadata.get("heading_path"),
            "score": score_map.get(metadata.get("chunk_id")),
        }
        for document in execution.docs[:limit]
        for metadata in [document.metadata or {}]
    ]


def ordered_citations(execution: AskExecution) -> list[dict[str, Any]]:
    """按回答实际引用顺序组织引用。"""
    citations = execution.result.citations
    preferred = execution.result.referenced_labels or list(citations)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label in preferred + list(citations):
        if label in seen or label not in citations:
            continue
        seen.add(label)
        item = dict(citations[label])
        item["referenced_in_answer"] = label in execution.result.referenced_labels
        ordered.append(item)
    return ordered


def _trace_summary(execution: AskExecution) -> dict[str, Any]:
    summary = build_retrieval_trace_summary(
        retrieval_query=execution.retrieval_query,
        history_used=execution.history_used,
        history_reason=execution.history_reason,
        trace=execution.trace,
    )
    summary["research_route"] = execution.research_route
    return summary


def build_response_payload(
    execution: AskExecution,
    *,
    include_debug: bool = False,
) -> dict[str, Any]:
    """构建 CLI 与 API 共用的响应字典。"""
    diagnostics: dict[str, Any] = {
        "retrieval": retrieval_diagnostics(execution),
        "context": execution.context_debug,
    }
    if include_debug:
        diagnostics.update(
            {
                "trace": execution.trace,
                "answer_debug": execution.result.debug,
            }
        )
    return {
        "query": execution.query,
        "retrieval_query": execution.retrieval_query,
        "answer": execution.result.answer_text,
        "referenced_labels": execution.result.referenced_labels,
        "rendered_citations": execution.result.rendered_citations,
        "citations": ordered_citations(execution),
        "used_chunks": execution.result.used_chunks,
        "evidence_items": execution.result.evidence_items,
        "source_cards": execution.result.source_cards,
        "claim_items": execution.result.claim_items,
        "retrieval_trace_summary": _trace_summary(execution),
        "evidence_sufficiency": execution.result.evidence_sufficiency,
        "confidence_score": execution.result.confidence_score,
        "timings_ms": execution.timings_ms,
        "diagnostics": diagnostics,
    }


def log_query(
    runtime: AskRuntime,
    execution: AskExecution,
    *,
    save_trace: bool = False,
) -> None:
    """将查询、证据和可选 Trace 写入注册库。"""
    constraints: dict[str, Any] = {
        "plan": execution.plan.to_dict(),
        "research_route": execution.research_route,
    }
    if save_trace:
        constraints.update(
            {
                "retrieval_trace": execution.trace,
                "retrieval_trace_summary": _trace_summary(execution),
                "citations": execution.result.citations,
                "referenced_labels": execution.result.referenced_labels,
                "context_debug": execution.context_debug,
                "evidence_sufficiency": execution.result.evidence_sufficiency,
                "confidence_score": execution.result.confidence_score,
            }
        )
    runtime.registry.log_query(
        query=execution.plan.query.rewritten or execution.plan.query.normalized,
        filters=execution.trace.get("where") or {},
        constraints=constraints,
        top_chunk_ids=[document.metadata.get("chunk_id") for document in execution.docs],
        top_source_files=[document.metadata.get("source_file") for document in execution.docs],
        latency_ms=execution.timings_ms["total"],
        backend=runtime.backend,
        tenant_id=None,
        embedding_version=runtime.cfg.model_versions.embedding_version,
        rerank_enabled=runtime.retriever.rerank_enabled,
        mode=runtime.mode,
        relax_steps=execution.trace.get("relax_steps"),
        diversity_k=execution.trace.get("diversity@k"),
    )
