from __future__ import annotations

from typing import Any, Protocol, Sequence


class RerankerProvider(Protocol):
    """Minimal reranking capability required by retrieval orchestration."""

    def rerank(
        self, query: str, documents: Sequence[Any], *, top_k: int
    ) -> list[Any]: ...
