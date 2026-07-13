"""兼容旧 PDF 加载模块，并保持 monkeypatch 行为。"""

from __future__ import annotations

import sys

from coal_kb.ingestion.loaders import pdf_loader as _implementation

sys.modules[__name__] = _implementation
