"""根据显式模式创建远程或本地 LLM。"""

from __future__ import annotations

from coal_kb.infra.providers.config import LLMProviderConfig
from coal_kb.infra.providers.llm.local.openai_compatible import make_local_chat_llm
from coal_kb.infra.providers.llm.remote.openai_compatible import make_remote_chat_llm

LLMConfig = LLMProviderConfig


def make_chat_llm(config: LLMProviderConfig):
    """按配置模式创建 LLM，禁止远程失败后切换本地。"""
    if config.mode == "remote":
        return make_remote_chat_llm(config)
    if config.mode == "local":
        return make_local_chat_llm(config)
    raise ValueError(f"Unsupported llm mode: {config.mode}")
