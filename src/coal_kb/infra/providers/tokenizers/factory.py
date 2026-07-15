"""根据显式模式创建远程或本地 Tokenizer。"""

from __future__ import annotations

from coal_kb.infra.providers.config import TokenizerProviderConfig
from coal_kb.infra.providers.tokenizers.local.huggingface import LocalTokenizer
from coal_kb.infra.providers.tokenizers.remote.http import RemoteTokenizer


def make_tokenizer(config: TokenizerProviderConfig):
    """按配置模式创建 Tokenizer。"""
    if config.mode == "remote":
        return RemoteTokenizer(config.remote)
    if config.mode == "local":
        return LocalTokenizer(config.local)
    raise ValueError(f"Unsupported tokenizer mode: {config.mode}")
