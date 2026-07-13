"""兼容旧文档切分导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.chunking.markdown_hierarchical_semantic import (
    AtomicUnit,
    ChunkingParams,
    SectionNode,
    parse_markdown_sections,
    split_docs_markdown_hierarchical_semantic,
)

__all__ = [
    "AtomicUnit",
    "ChunkingParams",
    "SectionNode",
    "parse_markdown_sections",
    "split_docs_markdown_hierarchical_semantic",
]
