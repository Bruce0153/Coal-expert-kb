from __future__ import annotations

from .base import BaseLoader, detect_language, normalize_text
from .registry import get_loader_for_path, load_any, register_loader

# Register builtin loaders
from . import csv_loader, docx_loader, html_loader, json_loader, pdf_loader, pptx_loader, text_loader, xlsx_loader  # noqa: F401

__all__ = [
    "BaseLoader",
    "detect_language",
    "normalize_text",
    "register_loader",
    "get_loader_for_path",
    "load_any",
]
