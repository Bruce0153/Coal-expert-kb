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


class ContextPackage(BaseModel):
    markdown: str
    citations: Dict[str, CitationItem] = Field(default_factory=dict)
    evidence_items: List[CitationItem] = Field(default_factory=list)
    used_chunks: List[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
