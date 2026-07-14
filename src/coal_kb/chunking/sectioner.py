"""兼容旧文档切分导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.chunking.sectioner import (
    infer_section,
    infer_section_with_debug,
    is_reference_like,
)

__all__ = [
    "infer_section",
    "infer_section_with_debug",
    "is_reference_like",
]
