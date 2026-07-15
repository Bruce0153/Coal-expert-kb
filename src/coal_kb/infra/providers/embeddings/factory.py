"""根据显式模式创建远程或本地 Embedding。"""

from __future__ import annotations

from coal_kb.infra.providers.config import EmbeddingsProviderConfig
from coal_kb.infra.providers.embeddings.local.huggingface import make_local_embeddings
from coal_kb.infra.providers.embeddings.remote.openai_compatible import make_remote_embeddings

EmbeddingsConfig = EmbeddingsProviderConfig


def make_embeddings(config: EmbeddingsProviderConfig):
    """按配置模式创建 Embedding，禁止隐式跨模式回退。"""
    if config.mode == "remote":
        return make_remote_embeddings(config.remote)
    if config.mode == "local":
        return make_local_embeddings(config.local)
    raise ValueError(f"Unsupported embedding mode: {config.mode}")
