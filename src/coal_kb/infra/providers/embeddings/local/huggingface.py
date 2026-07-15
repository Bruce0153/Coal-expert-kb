"""从本地 HuggingFace 模型目录创建 Embedding。"""

from __future__ import annotations

from coal_kb.infra.providers.config import LocalProviderConfig


def make_local_embeddings(config: LocalProviderConfig):
    """创建不读取远程 API Key 的本地 Embedding。"""
    if config.provider != "huggingface":
        raise ValueError(f"Unsupported local embedding provider: {config.provider}")
    model_name = config.model_path or config.model
    model_kwargs = {
        "device": config.device,
        "trust_remote_code": config.trust_remote_code,
    }
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
