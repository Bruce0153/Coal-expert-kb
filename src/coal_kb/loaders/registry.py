from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Type

from langchain_core.documents import Document

from .base import BaseLoader

logger = logging.getLogger(__name__)


_REGISTRY: Dict[str, Type[BaseLoader]] = {}


def register_loader(ext: str, loader_cls: Type[BaseLoader]) -> None:
    _REGISTRY[ext.lower().lstrip(".")] = loader_cls


def get_loader_for_path(path: str) -> Optional[BaseLoader]:
    ext = Path(path).suffix.lower().lstrip(".")
    loader_cls = _REGISTRY.get(ext)
    if not loader_cls:
        return None
    try:
        return loader_cls()
    except Exception as exc:
        logger.warning(
            "Loader unavailable for .%s: %s. Install extras (e.g., `pip install .[docs]`) to enable.",
            ext,
            exc,
        )
        return None


def load_any(path: str) -> list[Document]:
    loader = get_loader_for_path(path)
    if loader is None:
        logger.warning("No loader registered for %s", path)
        return []
    return loader.load(path)
