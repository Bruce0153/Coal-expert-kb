from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    llm: bool = False
    debug: bool = False
    backend: Optional[str] = Field(default=None, pattern="^(chroma|elastic|both)?$")
    mode: Optional[str] = Field(default=None, pattern="^(strict|balanced|broad)?$")
    k: Optional[int] = Field(default=None, ge=1, le=50)
    rerank: bool = True
    llm_provider: str = "none"


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


class AskResponse(BaseModel):
    query: str
    answer: str
    referenced_labels: List[str] = Field(default_factory=list)
    citations: List[CitationResponse] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
