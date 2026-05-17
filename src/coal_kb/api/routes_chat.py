from __future__ import annotations

from fastapi import APIRouter, HTTPException

from coal_kb.api.models import (
    ChatRequest,
    ChatResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    MessageResponse,
)
from coal_kb.api.runtime_overrides import apply_runtime_overrides
from coal_kb.chat.orchestrator import ChatOrchestrator
from coal_kb.conversation.service import ConversationService
from coal_kb.qa.ask_pipeline import build_runtime
from coal_kb.settings import AppConfig


def build_chat_router(cfg: AppConfig, conversations: ConversationService) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.post("/conversations", response_model=ConversationSummaryResponse)
    def create_conversation(payload: CreateConversationRequest) -> ConversationSummaryResponse:
        conversation = conversations.create_conversation(title=payload.title)
        return ConversationSummaryResponse.model_validate(conversation.model_dump())

    @router.get("/conversations", response_model=list[ConversationSummaryResponse])
    def list_conversations() -> list[ConversationSummaryResponse]:
        return [
            ConversationSummaryResponse.model_validate(item.model_dump())
            for item in conversations.list_conversations()
        ]

    @router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
    def list_messages(conversation_id: str) -> list[MessageResponse]:
        try:
            messages = conversations.list_messages(conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [MessageResponse.model_validate(message.model_dump()) for message in messages]

    @router.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict:
        if not conversations.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return {"deleted": True, "conversation_id": conversation_id}

    @router.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
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
        orchestrator = ChatOrchestrator(conversations=conversations, runtime=runtime)
        try:
            result = orchestrator.chat(
                query=payload.message,
                conversation_id=payload.conversation_id,
                enable_llm=payload.llm,
                save_trace=payload.debug,
                debug=payload.debug,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ChatResponse.model_validate(result.response)

    return router
