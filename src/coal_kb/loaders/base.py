"""兼容旧加载器模块。"""

from __future__ import annotations

from coal_kb.ingestion.loaders.base import (
    BaseLoader,
    detect_language,
    normalize_text,
)

__all__ = [
    "BaseLoader",
    "detect_language",
    "normalize_text",
]
