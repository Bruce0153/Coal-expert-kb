from __future__ import annotations

from .models import ConversationMessage, ConversationState, ConversationSummary
from .store import ConversationStore


class ConversationService:
    def __init__(self, store: ConversationStore) -> None:
        self.store = store

    def create_conversation(self, *, title: str | None = None) -> ConversationSummary:
        return self.store.create_conversation(title=title)

    def ensure_conversation(self, conversation_id: str | None, *, title_hint: str | None = None) -> ConversationSummary:
        if conversation_id:
            conversation = self.store.get_conversation(conversation_id)
            if conversation is not None:
                return conversation
            raise KeyError(f"Conversation not found: {conversation_id}")
        title = self._title_from_hint(title_hint) if title_hint else None
        return self.store.create_conversation(title=title)

    def list_conversations(self, *, limit: int = 50) -> list[ConversationSummary]:
        return self.store.list_conversations(limit=limit)

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.store.delete_conversation(conversation_id)

    def get_state(self, conversation_id: str) -> ConversationState:
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError(f"Conversation not found: {conversation_id}")
        messages = self.store.list_messages(conversation_id)
        return ConversationState(conversation=conversation, messages=messages)

    def list_messages(self, conversation_id: str) -> list[ConversationMessage]:
        return self.get_state(conversation_id).messages

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> ConversationMessage:
        message = self.store.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata,
        )
        conversation = self.store.get_conversation(conversation_id)
        if conversation and conversation.title == "New conversation" and role == "user":
            self.store.update_title(conversation_id, self._title_from_hint(content))
        return message

    def _title_from_hint(self, text: str | None) -> str:
        if not text:
            return "New conversation"
        normalized = " ".join(text.split())
        return (normalized[:60] + "...") if len(normalized) > 60 else normalized
