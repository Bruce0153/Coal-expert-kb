"""兼容旧文档切分导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.chunking.splitter import (
    split_docs_markdown_hierarchical_semantic,
    split_page_docs,
)

__all__ = [
    "split_docs_markdown_hierarchical_semantic",
    "split_page_docs",
]
