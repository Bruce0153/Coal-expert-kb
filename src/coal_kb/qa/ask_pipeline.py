from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from coal_kb.context.builder import ContextBuilder
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.generation.answerer import AnswerResult, Answerer
from coal_kb.llm.factory import LLMConfig
from coal_kb.metadata.normalize import Ontology
from coal_kb.query.plan import QueryPlan
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.bm25 import rrf_fuse
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.rerank import make_reranker
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import AppConfig
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite

logger = logging.getLogger(__name__)

HELP_TEXT = """Commands:
  help            Show help
  debug           Toggle debug output
  exit / quit     Exit
Type any question to start retrieval.
"""


@dataclass
class AskRuntime:
    cfg: AppConfig
    backend: str
    k: int
    mode: str
    planner: QueryPlanner
    retriever: ExpertRetriever
    context_builder: ContextBuilder
    answerer: Answerer
    registry: RegistrySQLite
    llm_config: Optional[LLMConfig]


@dataclass
class AskExecution:
    query: str
    retrieval_query: str
    plan: QueryPlan
    docs: List[Document]
    trace: Dict[str, Any]
    context_debug: Dict[str, Any]
    result: AnswerResult
    timings_ms: Dict[str, float]
    history_used: bool = False
    history_reason: str = "standalone_query"


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def parse_command(query: str) -> Optional[str]:
    normalized = query.strip().lower()
    if normalized in {"exit", "quit"}:
        return "exit"
    if normalized in {"help", ":help"}:
        return "help"
    if normalized in {"debug", ":debug"}:
        return "debug"
    return None


def build_runtime(
    cfg: AppConfig,
    *,
    backend: Optional[str] = None,
    k: Optional[int] = None,
    rerank_enabled: Optional[bool] = None,
    rerank_top_n: Optional[int] = None,
    rerank_model: Optional[str] = None,
    mode: Optional[str] = None,
    enable_llm: bool = False,
    llm_provider: str = "none",
) -> AskRuntime:
    onto = Ontology.load("configs/schema.yaml")
    planner = QueryPlanner(filter_parser=FilterParser(onto=onto))

    active_backend = backend or cfg.backend
    active_k = int(k or cfg.retrieval.k)
    active_mode = mode or cfg.retrieval.mode
    active_rerank = cfg.retrieval.rerank_enabled if rerank_enabled is None else rerank_enabled
    active_rerank_top_n = int(rerank_top_n or cfg.retrieval.rerank_top_n)

    if rerank_model:
        cfg.retrieval.rerank_model = rerank_model

    registry = RegistrySQLite(cfg.registry.sqlite_path)

    chroma_factory = None
    elastic_factory = None
    elastic_store = None

    if active_backend in {"chroma", "both"}:
        store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            embedding_model=cfg.embedding.model_name,
        )
        chroma_factory = store.as_retriever

    if active_backend in {"elastic", "both"}:
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=cfg.elastic.timeout_s,
        )
        elastic_factory = make_elastic_retriever_factory(
            client=elastic_store.client,
            index=cfg.elastic.alias_current,
            embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
            candidates=active_k,
            rrf_k=cfg.retrieval.rrf_k,
            use_icu=cfg.elastic.enable_icu_analyzer,
            tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
        )

    if active_backend == "both":
        vector_factory = _combine_factories(chroma_factory, elastic_factory, rrf_k=cfg.retrieval.rrf_k)
    elif active_backend == "elastic":
        vector_factory = elastic_factory
    else:
        vector_factory = chroma_factory

    reranker = make_reranker(cfg) if active_rerank else None
    retriever = ExpertRetriever(
        vector_retriever_factory=vector_factory,
        k=active_k,
        rerank_enabled=active_rerank,
        rerank_top_n=active_rerank_top_n,
        reranker=reranker,
        max_per_source=cfg.retrieval.max_per_source,
        max_relax_steps=cfg.retrieval.max_relax_steps,
        range_expand_schedule=cfg.retrieval.range_expand_schedule,
        mode=active_mode,
        drop_sections=cfg.retrieval.drop_sections,
        drop_reference_like=cfg.retrieval.drop_reference_like,
        use_fuse=(active_backend != "elastic"),
        where_full=(active_backend == "elastic"),
        two_stage_enabled=(active_backend == "elastic" and cfg.retrieval.two_stage.enabled),
        parent_k_candidates=cfg.retrieval.two_stage.parent_k_candidates,
        parent_k_final=cfg.retrieval.two_stage.parent_k_final,
        max_parents=cfg.retrieval.two_stage.max_parents,
        child_k_candidates=cfg.retrieval.two_stage.child_k_candidates,
        child_k_final=cfg.retrieval.two_stage.child_k_final,
        allow_relax_in_stage2=cfg.retrieval.two_stage.allow_relax_in_stage2,
        elastic_store=elastic_store if active_backend == "elastic" else None,
        elastic_index=cfg.elastic.alias_current if active_backend == "elastic" else None,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()) if active_backend == "elastic" else None,
        elastic_use_icu=cfg.elastic.enable_icu_analyzer,
        tenant_id=cfg.tenancy.default_tenant_id if cfg.tenancy.enabled else None,
    )

    final_provider = llm_provider
    if enable_llm and final_provider == "none":
        final_provider = cfg.llm.provider
    llm_config = None
    if enable_llm and final_provider != "none":
        llm_config = LLMConfig(**{**cfg.llm.model_dump(), "provider": final_provider})

    return AskRuntime(
        cfg=cfg,
        backend=active_backend,
        k=active_k,
        mode=active_mode,
        planner=planner,
        retriever=retriever,
        context_builder=ContextBuilder(),
        answerer=Answerer(),
        registry=registry,
        llm_config=llm_config,
    )


def execute_query(
    runtime: AskRuntime,
    raw_query: str,
    *,
    enable_llm: bool = False,
    original_query: Optional[str] = None,
    conversation_context: Optional[str] = None,
    history_used: bool = False,
    history_reason: str = "standalone_query",
) -> AskExecution:
    query = normalize_query(original_query or raw_query)
    retrieval_query = normalize_query(raw_query)
    if not retrieval_query:
        raise ValueError("query is empty")

    trace: Dict[str, Any] = {}

    started = time.monotonic()
    plan = runtime.planner.build_plan(retrieval_query, runtime.cfg, enable_llm=False, llm_config=None)
    plan_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    docs = runtime.retriever.execute(plan, trace=trace)
    retrieve_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    context_package = runtime.context_builder.build(plan, docs)
    context_ms = (time.monotonic() - started) * 1000

    started = time.monotonic()
    result = runtime.answerer.answer(
        plan,
        context_package,
        query=query,
        enable_llm=enable_llm,
        llm_config=runtime.llm_config,
        conversation_context=conversation_context,
    )
    answer_ms = (time.monotonic() - started) * 1000

    return AskExecution(
        query=query,
        retrieval_query=retrieval_query,
        plan=plan,
        docs=docs,
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
        history_used=history_used,
        history_reason=history_reason,
    )


def retrieval_diagnostics(execution: AskExecution, *, limit: int = 5) -> List[Dict[str, Any]]:
    score_map = {
        item.get("chunk_id"): item.get("score")
        for item in execution.trace.get("condition_score_top3", [])
        if isinstance(item, dict)
    }
    diagnostics: List[Dict[str, Any]] = []
    for doc in execution.docs[:limit]:
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id")
        diagnostics.append(
            {
                "chunk_id": chunk_id,
                "source_file": meta.get("source_file"),
                "title": meta.get("title"),
                "page": meta.get("page"),
                "heading_path": meta.get("heading_path"),
                "score": score_map.get(chunk_id),
            }
        )
    return diagnostics


def ordered_citations(execution: AskExecution) -> List[Dict[str, Any]]:
    citations = execution.result.citations
    preferred = execution.result.referenced_labels or list(citations.keys())
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for label in preferred + list(citations.keys()):
        if label in seen or label not in citations:
            continue
        seen.add(label)
        item = dict(citations[label])
        item["referenced_in_answer"] = label in execution.result.referenced_labels
        ordered.append(item)
    return ordered


def format_sources(execution: AskExecution) -> str:
    lines: List[str] = []
    for item in ordered_citations(execution):
        page = item.get("page")
        heading = item.get("heading_path")
        page_text = f" | page={page}" if page is not None else ""
        heading_text = f" | heading={heading}" if heading else ""
        lines.append(f"[{item['label']}] {item.get('source_file', 'unknown')}{page_text}{heading_text}")
        lines.append(f"  {item.get('snippet', '')}")
    return "\n".join(lines)


def retrieval_trace_summary(execution: AskExecution) -> Dict[str, Any]:
    return {
        "retrieval_query": execution.retrieval_query,
        "history_used": execution.history_used,
        "history_reason": execution.history_reason,
        "vector_candidates": execution.trace.get("vector_candidates"),
        "postfiltered_count": execution.trace.get("postfiltered_count"),
        "source_distribution": execution.trace.get("source_distribution"),
        "heading_distribution": execution.trace.get("heading_distribution"),
    }


def build_response_payload(execution: AskExecution, *, include_debug: bool = False) -> Dict[str, Any]:
    payload = {
        "query": execution.query,
        "retrieval_query": execution.retrieval_query,
        "answer": execution.result.answer_text,
        "referenced_labels": execution.result.referenced_labels,
        "citations": ordered_citations(execution),
        "used_chunks": execution.result.used_chunks,
        "evidence_items": execution.result.evidence_items,
        "retrieval_trace_summary": retrieval_trace_summary(execution),
        "evidence_sufficiency": execution.result.evidence_sufficiency,
        "confidence_score": execution.result.confidence_score,
        "timings_ms": execution.timings_ms,
        "diagnostics": {
            "retrieval": retrieval_diagnostics(execution),
            "context": execution.context_debug,
            "trace": execution.trace,
            "answer_debug": execution.result.debug,
        }
        if include_debug
        else {
            "retrieval": retrieval_diagnostics(execution),
            "context": execution.context_debug,
        },
    }
    return payload


def format_debug_info(execution: AskExecution) -> str:
    return json.dumps(build_response_payload(execution, include_debug=True), ensure_ascii=False, indent=2)


def log_query(runtime: AskRuntime, execution: AskExecution, *, save_trace: bool = False) -> None:
    runtime.registry.log_query(
        query=execution.plan.query.rewritten or execution.plan.query.normalized,
        filters=execution.trace.get("where") or {},
        constraints={
            "plan": execution.plan.to_dict(),
            "retrieval_trace": execution.trace,
            "retrieval_trace_summary": retrieval_trace_summary(execution),
            "citations": execution.result.citations,
            "referenced_labels": execution.result.referenced_labels,
            "context_debug": execution.context_debug,
            "evidence_sufficiency": execution.result.evidence_sufficiency,
            "confidence_score": execution.result.confidence_score,
        }
        if save_trace
        else {"plan": execution.plan.to_dict()},
        top_chunk_ids=[doc.metadata.get("chunk_id") for doc in execution.docs],
        top_source_files=[doc.metadata.get("source_file") for doc in execution.docs],
        latency_ms=execution.timings_ms["total"],
        backend=runtime.backend,
        tenant_id=None,
        embedding_version=runtime.cfg.model_versions.embedding_version,
        rerank_enabled=runtime.retriever.rerank_enabled,
        mode=runtime.mode,
        relax_steps=execution.trace.get("relax_steps"),
        diversity_k=execution.trace.get("diversity@k"),
    )


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
