"""Backward-compatible embedding provider imports."""

from coal_kb.infra.providers.embeddings import EmbeddingsConfig, make_embeddings

__all__ = ["EmbeddingsConfig", "make_embeddings"]
