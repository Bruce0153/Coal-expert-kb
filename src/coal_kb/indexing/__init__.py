"""提供索引构建、验证和版本管理边界。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IndexService",
    "Manifest",
    "ManifestEntry",
    "resolve_index_name",
    "validate_index",
]


def __getattr__(name: str) -> Any:
    if name == "IndexService":
        from .service import IndexService

        return IndexService
    if name in {"Manifest", "ManifestEntry"}:
        from .manifest import Manifest, ManifestEntry

        return {"Manifest": Manifest, "ManifestEntry": ManifestEntry}[name]
    if name in {"resolve_index_name", "validate_index"}:
        from .validation import resolve_index_name, validate_index

        return {
            "resolve_index_name": resolve_index_name,
            "validate_index": validate_index,
        }[name]
    raise AttributeError(name)
