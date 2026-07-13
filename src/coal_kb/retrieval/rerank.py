"""Backward-compatible reranker provider imports."""

from coal_kb.infra.providers.rerank import (
    CrossEncoderReranker,
    DashScopeReranker,
    make_reranker,
)

__all__ = ["CrossEncoderReranker", "DashScopeReranker", "make_reranker"]
