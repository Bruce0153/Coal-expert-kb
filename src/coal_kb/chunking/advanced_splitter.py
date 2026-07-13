"""兼容旧文档切分导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.chunking.advanced_splitter import (
    split_page_docs_section_aware,
)

__all__ = [
    "split_page_docs_section_aware",
]
