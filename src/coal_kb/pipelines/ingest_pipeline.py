"""兼容旧摄入流水线模块，并保持 monkeypatch 行为。"""

from __future__ import annotations

import sys

from coal_kb.ingestion import pipeline as _implementation

sys.modules[__name__] = _implementation
