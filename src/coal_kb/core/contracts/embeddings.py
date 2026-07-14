from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Minimal embedding capability required by the application."""

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
