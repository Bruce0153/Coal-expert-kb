"""根据显式模式创建远程或本地 Reranker。"""

from __future__ import annotations

from coal_kb.infra.providers.config import RerankProviderConfig
from coal_kb.infra.providers.rerank.local.cross_encoder import LocalCrossEncoderReranker
from coal_kb.infra.providers.rerank.remote.http import RemoteReranker


def make_reranker(config: RerankProviderConfig):
    """按配置模式创建 Reranker，失败时不静默改变执行模式。"""
    if config.mode == "remote":
        return RemoteReranker(config.remote)
    if config.mode == "local":
        return LocalCrossEncoderReranker(config.local)
    raise ValueError(f"Unsupported rerank mode: {config.mode}")
