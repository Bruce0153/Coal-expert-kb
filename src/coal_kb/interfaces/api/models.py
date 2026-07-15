"""定义 HTTP 请求与响应协议，字段保持向后兼容。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RuntimeSettingsRequest(BaseModel):
    llm: bool = False
    debug: bool = False
    backend: Optional[str] = Field(default=None, pattern="^(chroma|elastic|both)?$")
    mode: Optional[str] = Field(default=None, pattern="^(strict|balanced|broad)?$")
    k: Optional[int] = Field(default=None, ge=1, le=50)
    rerank: bool = True
    tokenizer_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")
    tokenizer_provider: Optional[str] = None
    tokenizer_base_url: Optional[str] = None
    tokenizer_api_key: Optional[str] = None
    tokenizer_model: Optional[str] = None
    embedding_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")
    embedding_provider: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    rerank_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")
    rerank_provider: Optional[str] = None
    rerank_base_url: Optional[str] = None
    rerank_api_key: Optional[str] = None
    rerank_model: Optional[str] = None
    llm_mode: Optional[str] = Field(default=None, pattern="^(remote|local)?$")
    llm_provider: str = "none"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None


class AskRequest(RuntimeSettingsRequest):
    query: str = Field(..., min_length=1)


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class ChatRequest(RuntimeSettingsRequest):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1)


class CitationResponse(BaseModel):
    label: str
    source_file: str
    title: Optional[str] = None
    page: Optional[int] = None
    heading_path: Optional[str] = None
    chunk_id: str
    snippet: str = ""
    source_display: str
    referenced_in_answer: bool = False


class ClaimResponse(BaseModel):
    claim_id: str
    text: str
    citations: List[str] = Field(default_factory=list)
    support: str = "direct"


class SourceCardResponse(BaseModel):
    source_id: str
    source_file: str
    title: str
    pages: List[int] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)
    evidence_labels: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    snippet_preview: str = ""


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AskResponse(BaseModel):
    query: str
    retrieval_query: str
    answer: str
    referenced_labels: List[str] = Field(default_factory=list)
    rendered_citations: List[str] = Field(default_factory=list)
    citations: List[CitationResponse] = Field(default_factory=list)
    used_chunks: List[str] = Field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = Field(default_factory=list)
    source_cards: List[SourceCardResponse] = Field(default_factory=list)
    claim_items: List[ClaimResponse] = Field(default_factory=list)
    retrieval_trace_summary: Dict[str, Any] = Field(default_factory=dict)
    evidence_sufficiency: str = "insufficient"
    confidence_score: float = 0.0
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(AskResponse):
    conversation_id: str
    message_id: str


class SettingsDefaultsResponse(BaseModel):
    api_base_url: str = ""
    tokenizer: Dict[str, Any]
    embeddings: Dict[str, Any]
    rerank_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    backend: str
    mode: str
    k: int
    rerank: bool
    llm: bool = False
    debug: bool = False
    backend_options: List[str] = Field(default_factory=list)
    mode_options: List[str] = Field(default_factory=list)
    remote_provider_options: List[str] = Field(default_factory=list)
    local_provider_options: Dict[str, List[str]] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
