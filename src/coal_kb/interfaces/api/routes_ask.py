"""定义单轮问答 HTTP 路由。"""

from fastapi import APIRouter, Depends, HTTPException

from coal_kb.application.ask import build_response_payload, build_runtime, execute_query, log_query
from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.infra.security import PublicRequestGuard, PublicSecurityPolicy
from coal_kb.interfaces.api.models import AskRequest, AskResponse
from coal_kb.interfaces.api.presenters import build_ask_response
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides


def build_ask_router(
    configs: RuntimeConfigStore,
    policy: PublicSecurityPolicy | None = None,
    guard: PublicRequestGuard | None = None,
) -> APIRouter:
    policy = policy or PublicSecurityPolicy.from_env()
    guard = guard or PublicRequestGuard(policy)
    router = APIRouter(prefix="/api", tags=["ask"])

    @router.post("/ask", response_model=AskResponse, dependencies=[Depends(guard.protect)])
    def ask(payload: AskRequest) -> AskResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        if len(query) > policy.max_query_chars:
            raise HTTPException(status_code=413, detail="问题过长，请缩短后重试。")

        cfg = configs.snapshot()
        if policy.public_mode:
            if payload.debug:
                raise HTTPException(status_code=403, detail="公网模式不允许启用 debug。")
            if payload.research_route not in policy.allowed_research_routes:
                raise HTTPException(status_code=403, detail="当前公网部署未开放该研究路线。")
            runtime_cfg = cfg.model_copy(deep=True)
            backend = cfg.backend
            k = cfg.retrieval.k
            rerank_enabled = cfg.retrieval.rerank_enabled
            mode = cfg.retrieval.mode
            enable_llm = True
            llm_provider = "none"
            debug = False
        else:
            runtime_cfg = apply_runtime_overrides(cfg, payload)
            backend = payload.backend
            k = payload.k
            rerank_enabled = payload.rerank
            mode = payload.mode
            enable_llm = payload.llm
            llm_provider = payload.llm_provider
            debug = payload.debug

        runtime = build_runtime(
            runtime_cfg,
            backend=backend,
            k=k,
            rerank_enabled=rerank_enabled,
            mode=mode,
            enable_llm=enable_llm,
            llm_provider=llm_provider,
        )
        execution = execute_query(
            runtime,
            query,
            enable_llm=enable_llm,
            research_route=payload.research_route,
        )
        log_query(runtime, execution, save_trace=debug)
        response_payload = build_response_payload(execution, include_debug=debug)
        return build_ask_response(response_payload)

    return router
