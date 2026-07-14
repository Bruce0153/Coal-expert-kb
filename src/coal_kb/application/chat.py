"""编排多轮会话、历史上下文和单轮问答用例。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from coal_kb.application.ask import AskRuntime, build_response_payload, execute_query, log_query
from coal_kb.conversation.history import PreparedHistory, prepare_history_context
from coal_kb.conversation.models import ConversationMessage, ConversationSummary
from coal_kb.conversation.service import ConversationService


@dataclass
class ChatTurnResult:
    conversation: ConversationSummary
    user_message: ConversationMessage
    assistant_message: ConversationMessage
    response: Dict[str, Any]
    prepared_history: PreparedHistory


class ChatOrchestrator:
    def __init__(self, *, conversations: ConversationService, runtime: AskRuntime) -> None:
        self.conversations = conversations
        self.runtime = runtime

    def chat(
        self,
        *,
        query: str,
        conversation_id: Optional[str] = None,
        enable_llm: bool = False,
        save_trace: bool = False,
        debug: bool = False,
    ) -> ChatTurnResult:
        conversation = self.conversations.ensure_conversation(conversation_id, title_hint=query)
        prior_messages = self.conversations.list_messages(conversation.conversation_id)
        prepared_history = prepare_history_context(prior_messages, query)

        user_message = self.conversations.add_message(
            conversation_id=conversation.conversation_id,
            role="user",
            content=query,
            metadata={
                "history_used": prepared_history.used_history,
                "history_reason": prepared_history.reason,
            },
        )

        execution = execute_query(
            self.runtime,
            prepared_history.retrieval_query,
            enable_llm=enable_llm,
            original_query=query,
            conversation_context=prepared_history.answer_history,
            history_used=prepared_history.used_history,
            history_reason=prepared_history.reason,
        )
        log_query(self.runtime, execution, save_trace=save_trace or debug)
        response = build_response_payload(execution, include_debug=debug)
        response["conversation_id"] = conversation.conversation_id

        assistant_message = self.conversations.add_message(
            conversation_id=conversation.conversation_id,
            role="assistant",
            content=response["answer"],
            metadata={
                "citations": response["citations"],
                "used_chunks": response["used_chunks"],
                "evidence_items": response["evidence_items"],
                "source_cards": response["source_cards"],
                "claim_items": response["claim_items"],
                "rendered_citations": response["rendered_citations"],
                "retrieval_trace_summary": response["retrieval_trace_summary"],
                "evidence_sufficiency": response["evidence_sufficiency"],
                "confidence_score": response["confidence_score"],
                "diagnostics": response["diagnostics"],
                "timings_ms": response["timings_ms"],
            },
        )
        response["message_id"] = assistant_message.message_id
        return ChatTurnResult(
            conversation=self.conversations.get_state(conversation.conversation_id).conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            response=response,
            prepared_history=prepared_history,
        )
