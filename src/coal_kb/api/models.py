"""兼容旧 API 模型导入路径。"""

from coal_kb.interfaces.api.models import (
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ClaimResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    MessageResponse,
    RuntimeSettingsRequest,
    SettingsDefaultsResponse,
    SourceCardResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "ChatRequest",
    "ChatResponse",
    "CitationResponse",
    "ClaimResponse",
    "ConversationSummaryResponse",
    "CreateConversationRequest",
    "MessageResponse",
    "RuntimeSettingsRequest",
    "SettingsDefaultsResponse",
    "SourceCardResponse",
]
