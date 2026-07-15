"""通过 OpenAI 兼容远程 API 创建 Embedding。"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from coal_kb.infra.providers.config import RemoteProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key

SUPPORTED_REMOTE_EMBEDDING_PROVIDERS = {
    "openai",
    "openai_compatible",
    "dashscope",
    "deepseek",
    "siliconflow",
    "moonshot",
    "zhipu",
}


def make_remote_embeddings(config: RemoteProviderConfig) -> OpenAIEmbeddings:
    """创建需要 API Key 的远程 Embedding 客户端。"""
    if config.provider not in SUPPORTED_REMOTE_EMBEDDING_PROVIDERS:
        raise ValueError(f"Unsupported remote embedding provider: {config.provider}")
    kwargs: dict[str, object] = {}
    if config.dimensions is not None:
        kwargs["dimensions"] = config.dimensions
    if config.provider == "dashscope":
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 10
    return OpenAIEmbeddings(
        model=config.model,
        api_key=resolve_remote_api_key(api_key=config.api_key, api_key_env=config.api_key_env),
        base_url=config.base_url,
        **kwargs,
    )
