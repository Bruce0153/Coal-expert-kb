"""兼容旧文档切分导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.chunking.sentence_split import (
    split_sentences,
)

__all__ = [
    "split_sentences",
]
