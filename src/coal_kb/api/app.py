from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coal_kb.api.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    ClaimResponse,
    SettingsDefaultsResponse,
    SourceCardResponse,
)
from coal_kb.api.runtime_overrides import apply_runtime_overrides, build_settings_defaults
from coal_kb.api.routes_chat import build_chat_router
from coal_kb.conversation.service import ConversationService
from coal_kb.conversation.store import ConversationStore
from coal_kb.logging import setup_logging
from coal_kb.qa.ask_pipeline import build_response_payload, build_runtime, execute_query, log_query
from coal_kb.settings import load_config


def create_app() -> FastAPI:
    cfg = load_config()
    setup_logging(cfg, logger_name="coal_kb.api")

    app = FastAPI(
        title="Coal Expert KB",
        version="0.2.0",
        description="Conversation-capable evidence-grounded RAG API for coal pyrolysis and gasification literature.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    conversation_service = ConversationService(ConversationStore(cfg.registry.sqlite_path))
    app.include_router(build_chat_router(cfg, conversation_service))

    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/settings/defaults", response_model=SettingsDefaultsResponse)
    def settings_defaults() -> SettingsDefaultsResponse:
        return build_settings_defaults(cfg)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        runtime_cfg = apply_runtime_overrides(cfg, payload)
        runtime = build_runtime(
            runtime_cfg,
            backend=payload.backend,
            k=payload.k,
            rerank_enabled=payload.rerank,
            mode=payload.mode,
            enable_llm=payload.llm,
            llm_provider=payload.llm_provider,
        )
        execution = execute_query(runtime, query, enable_llm=payload.llm)
        log_query(runtime, execution, save_trace=payload.debug)
        response_payload = build_response_payload(execution, include_debug=payload.debug)
        citations = [CitationResponse.model_validate(item) for item in response_payload["citations"]]
        source_cards = [SourceCardResponse.model_validate(item) for item in response_payload["source_cards"]]
        claim_items = [ClaimResponse.model_validate(item) for item in response_payload["claim_items"]]
        return AskResponse(
            query=response_payload["query"],
            retrieval_query=response_payload["retrieval_query"],
            answer=response_payload["answer"],
            referenced_labels=response_payload["referenced_labels"],
            rendered_citations=response_payload["rendered_citations"],
            citations=citations,
            used_chunks=response_payload["used_chunks"],
            evidence_items=response_payload["evidence_items"],
            source_cards=source_cards,
            claim_items=claim_items,
            retrieval_trace_summary=response_payload["retrieval_trace_summary"],
            evidence_sufficiency=response_payload["evidence_sufficiency"],
            confidence_score=response_payload["confidence_score"],
            timings_ms=response_payload["timings_ms"],
            diagnostics=response_payload["diagnostics"],
        )

    return app


app = create_app()
