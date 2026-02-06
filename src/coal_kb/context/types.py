from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    sid: str
    source_file: str
    page: int | None = None
    heading_path: str | None = None
    chunk_id: str


class ContextPackage(BaseModel):
    markdown: str
    citations: Dict[str, CitationItem] = Field(default_factory=dict)
    used_chunks: List[str] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
