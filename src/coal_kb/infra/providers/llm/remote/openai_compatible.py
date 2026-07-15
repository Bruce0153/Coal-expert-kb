"""通过 OpenAI 兼容远程 API 创建 LLM。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from coal_kb.infra.providers.config import LLMProviderConfig
from coal_kb.infra.providers.utils.http import resolve_remote_api_key

SUPPORTED_REMOTE_LLM_PROVIDERS = {
    "openai",
    "openai_compatible",
    "dashscope",
    "deepseek",
    "siliconflow",
    "moonshot",
    "zhipu",
}


def make_remote_chat_llm(config: LLMProviderConfig) -> ChatOpenAI:
    """创建需要 API Key 的远程聊天模型。"""
    remote = config.remote
    if remote.provider not in SUPPORTED_REMOTE_LLM_PROVIDERS:
        raise ValueError(f"Unsupported remote llm provider: {remote.provider}")
    return ChatOpenAI(
        model=remote.model,
        api_key=resolve_remote_api_key(api_key=remote.api_key, api_key_env=remote.api_key_env),
        base_url=remote.base_url,
        temperature=config.temperature,
        timeout=remote.timeout,
    )
