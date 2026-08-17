"""定义会话与多轮问答 HTTP 路由。"""

from fastapi import APIRouter, Depends, HTTPException, Request

from coal_kb.application.ask import build_runtime
from coal_kb.application.chat import ChatOrchestrator
from coal_kb.application.runtime_config import RuntimeConfigStore
from coal_kb.conversation.service import ConversationService
from coal_kb.infra.security import PublicRequestGuard, PublicSecurityPolicy
from coal_kb.interfaces.api.models import (
    ChatRequest,
    ChatResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    MessageResponse,
)
from coal_kb.interfaces.api.runtime_overrides import apply_runtime_overrides


def build_chat_router(
    configs: RuntimeConfigStore,
    conversations: ConversationService,
    policy: PublicSecurityPolicy | None = None,
    guard: PublicRequestGuard | None = None,
) -> APIRouter:
    policy = policy or PublicSecurityPolicy.from_env()
    guard = guard or PublicRequestGuard(policy)
    router = APIRouter(prefix="/api", tags=["chat"])

    def scoped(request: Request) -> ConversationService:
        session_id = str(getattr(request.state, "session_id", "legacy"))
        return conversations.for_session(session_id)

    @router.post("/conversations", response_model=ConversationSummaryResponse)
    def create_conversation(payload: CreateConversationRequest, request: Request) -> ConversationSummaryResponse:
        conversation = scoped(request).create_conversation(title=payload.title)
        return ConversationSummaryResponse.model_validate(conversation.model_dump())

    @router.get("/conversations", response_model=list[ConversationSummaryResponse])
    def list_conversations(request: Request) -> list[ConversationSummaryResponse]:
        return [
            ConversationSummaryResponse.model_validate(item.model_dump())
            for item in scoped(request).list_conversations()
        ]

    @router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
    def list_messages(conversation_id: str, request: Request) -> list[MessageResponse]:
        try:
            messages = scoped(request).list_messages(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [MessageResponse.model_validate(message.model_dump()) for message in messages]

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str, request: Request) -> dict[str, object]:
        if not scoped(request).delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"deleted": True, "conversation_id": conversation_id}

    @router.post("/chat", response_model=ChatResponse, dependencies=[Depends(guard.protect)])
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        query = payload.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Message must not be empty.")
        if len(query) > policy.max_query_chars:
            raise HTTPException(status_code=413, detail="消息过长，请缩短后重试。")

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
        orchestrator = ChatOrchestrator(conversations=scoped(request), runtime=runtime)
        try:
            result = orchestrator.chat(
                query=query,
                conversation_id=payload.conversation_id,
                enable_llm=enable_llm,
                research_route=payload.research_route,
                save_trace=debug,
                debug=debug,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ChatResponse.model_validate(result.response)

    return router
