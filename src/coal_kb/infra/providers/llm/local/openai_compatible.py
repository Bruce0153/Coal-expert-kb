"""连接 vLLM、Ollama 等本地 OpenAI 兼容 LLM 服务。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from coal_kb.infra.providers.config import LLMProviderConfig


def make_local_chat_llm(config: LLMProviderConfig) -> ChatOpenAI:
    """创建不读取远程 API Key 的本地聊天模型客户端。"""
    local = config.local
    if local.provider not in {"openai_compatible", "vllm", "ollama"}:
        raise ValueError(f"Unsupported local llm provider: {local.provider}")
    if not local.base_url:
        raise ValueError("Local LLM base_url is required")
    return ChatOpenAI(
        model=local.model,
        api_key="local-model-no-remote-key",
        base_url=local.base_url,
        temperature=config.temperature,
        timeout=local.timeout,
    )
