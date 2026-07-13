"""提供摄入阶段的loaders实现。"""

from __future__ import annotations

# Register builtin loaders
from . import (  # noqa: F401
    csv_loader,
    docx_loader,
    html_loader,
    json_loader,
    pdf_loader,
    pptx_loader,
    text_loader,
    xlsx_loader,
)
from .base import BaseLoader, detect_language, normalize_text
from .registry import get_loader_for_path, load_any, register_loader

__all__ = [
    "BaseLoader",
    "detect_language",
    "normalize_text",
    "register_loader",
    "get_loader_for_path",
    "load_any",
]
