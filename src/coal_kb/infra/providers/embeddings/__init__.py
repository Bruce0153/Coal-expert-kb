"""导出 Embedding Provider 正式入口。"""

from coal_kb.infra.providers.config import EmbeddingsProviderConfig as EmbeddingsConfig
from coal_kb.infra.providers.embeddings.factory import make_embeddings

__all__ = ["EmbeddingsConfig", "make_embeddings"]
