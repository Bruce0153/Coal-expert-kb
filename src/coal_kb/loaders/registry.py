"""兼容旧加载器模块。"""

from __future__ import annotations

from coal_kb.ingestion.loaders.registry import (
    get_loader_for_path,
    load_any,
    register_loader,
)

__all__ = [
    "get_loader_for_path",
    "load_any",
    "register_loader",
]
