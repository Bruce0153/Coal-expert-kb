from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationMessage(BaseModel):
    message_id: str
    conversation_id: str
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ConversationState(BaseModel):
    conversation: ConversationSummary
    messages: list[ConversationMessage] = Field(default_factory=list)
    recent_summary: Optional[str] = None
