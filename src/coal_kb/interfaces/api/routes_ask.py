"""定义单轮问答 HTTP 路由。"""

from fastapi import APIRouter, HTTPException

from coal_kb.application.ask import build_response_payload, build_runtime, execute_query, log_query
from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.interfaces.api.models import AskRequest, AskResponse
from coal_kb.interfaces.api.presenters import build_ask_response
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides


def build_ask_router(configs: RuntimeConfigStore) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["ask"])

    @router.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        runtime_cfg = apply_runtime_overrides(configs.snapshot(), payload)
        runtime = build_runtime(
            runtime_cfg,
            backend=payload.backend,
            k=payload.k,
            rerank_enabled=payload.rerank,
            mode=payload.mode,
            enable_llm=payload.llm,
            llm_provider=payload.llm_provider,
        )
        execution = execute_query(
            runtime,
            query,
            enable_llm=payload.llm,
            research_route=payload.research_route,
        )
        log_query(runtime, execution, save_trace=payload.debug)
        response_payload = build_response_payload(execution, include_debug=payload.debug)
        return build_ask_response(response_payload)

    return router
