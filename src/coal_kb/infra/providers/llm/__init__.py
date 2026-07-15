"""导出 LLM Provider 正式入口。"""

from coal_kb.infra.providers.config import LLMProviderConfig as LLMConfig
from coal_kb.infra.providers.llm.factory import make_chat_llm

__all__ = ["LLMConfig", "make_chat_llm"]
