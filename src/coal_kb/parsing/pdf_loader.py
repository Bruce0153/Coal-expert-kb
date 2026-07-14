"""兼容旧 PDF 解析导入路径。"""

from __future__ import annotations

from coal_kb.ingestion.parsing.pdf_loader import (
    load_pdf_pages,
    load_pdfs_from_dir,
)

__all__ = [
    "load_pdf_pages",
    "load_pdfs_from_dir",
]
