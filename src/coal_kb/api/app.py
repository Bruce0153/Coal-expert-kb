from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coal_kb.api.models import AskRequest, AskResponse, CitationResponse
from coal_kb.logging import setup_logging
from coal_kb.qa.ask_pipeline import build_response_payload, build_runtime, execute_query, log_query
from coal_kb.settings import load_config


def create_app() -> FastAPI:
    cfg = load_config()
    setup_logging(cfg, logger_name="coal_kb.api")

    app = FastAPI(
        title="Coal Expert KB",
        version="0.1.0",
        description="Evidence-grounded RAG demo API for coal pyrolysis and gasification literature.",
    )

    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        runtime = build_runtime(
            cfg,
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
        return AskResponse(
            query=response_payload["query"],
            answer=response_payload["answer"],
            referenced_labels=response_payload["referenced_labels"],
            citations=citations,
            timings_ms=response_payload["timings_ms"],
            diagnostics=response_payload["diagnostics"],
        )

    return app


app = create_app()
