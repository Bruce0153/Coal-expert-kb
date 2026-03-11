from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    label: str
    source_file: str
    title: Optional[str] = None
    page: Optional[int] = None
    heading_path: Optional[str] = None
    chunk_id: str
    snippet: str = ""
    source_display: str
    source_id: str
    rank: int = 0


class SourceCard(BaseModel):
    source_id: str
    source_file: str
    title: str
    pages: List[int] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)
    evidence_labels: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    snippet_preview: str = ""


class ContextPackage(BaseModel):
    markdown: str
    citations: Dict[str, CitationItem] = Field(default_factory=dict)
    evidence_items: List[CitationItem] = Field(default_factory=list)
    source_cards: List[SourceCard] = Field(default_factory=list)
    used_chunks: List[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
